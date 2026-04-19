"""FeatureCollection — a GeoDataFrame with pyramids-specific GIS methods.

ARC-10 split the original ``feature.py`` into a subpackage:

- :mod:`pyramids.feature.collection` (this module) — the
  :class:`FeatureCollection` class.
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors (``create_polygon``, ``create_point``, ``get_coords``,
  ``explode_gdf``, ``multi_geom_handler``, …).
- :mod:`pyramids.feature.crs` — CRS/EPSG/reprojection helpers
  (``get_epsg_from_prj``, ``reproject_coordinates``,
  ``create_sr_from_proj``).
- :mod:`pyramids.feature._ogr` — private OGR bridge.

``FeatureCollection`` is a direct subclass of
:class:`geopandas.GeoDataFrame` (ARC-1a). Every GeoDataFrame method
is inherited; pyramids adds rasterization, Dataset interop, vertex
extraction, and CRS-helper delegates on top. ``ogr.DataSource`` is
internal only (ARC-1b); see :mod:`pyramids.feature._ogr`.
"""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids import _io as _pyramids_io
from pyramids.base._errors import CRSError, FeatureError
from pyramids.base._utils import Catalog
from pyramids.feature import crs as _crs
from pyramids.feature import geometry as _geom

CATALOG = Catalog(raster_driver=False)
gdal.UseExceptions()


class FeatureCollection(GeoDataFrame):
    """A :class:`geopandas.GeoDataFrame` with pyramids-specific GIS methods.

    ``FeatureCollection`` *is a* ``GeoDataFrame`` — ``isinstance(fc,
    GeoDataFrame)`` is ``True`` — so every geopandas method is
    available directly. Pyramids adds rasterization, Dataset interop,
    vertex extraction, and CRS helpers on top.

    The OGR/GDAL backend is internal only; see
    :mod:`pyramids.feature._ogr`.
    """

    @property
    def _constructor(self):
        """Return the type pandas uses when constructing new frames."""
        return FeatureCollection

    # ARC-31 — merge with GeoDataFrame._metadata instead of replacing it.
    # The parent class lists ``_geometry_column_name`` (the name of the
    # active geometry column); overriding _metadata with just our own
    # entries drops that attribute on pickle / copy / concat, and the
    # restored object can no longer find its geometry column. Always
    # splat the parent's list first.
    _metadata: list[str] = [
        *GeoDataFrame._metadata,
        "_epsg_cache_crs",
        "_epsg_cache_value",
    ]
    """Instance attributes pandas must preserve across copy/slice/pickle.

    Holds:

    * ``GeoDataFrame._metadata`` (currently ``_geometry_column_name``)
      — required for pickle round-trips to remember which column is
      the active geometry column.
    * ``_epsg_cache_crs`` / ``_epsg_cache_value`` — the ARC-6 EPSG
      cache.
    """

    def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Construct a FeatureCollection.

        Accepts anything :class:`geopandas.GeoDataFrame` accepts.
        Rejects ``ogr.DataSource`` / ``gdal.Dataset`` with a clear error
        (ARC-1b).
        """
        if isinstance(data, (ogr.DataSource, gdal.Dataset)):
            raise TypeError(
                "FeatureCollection no longer accepts ogr.DataSource or "
                "gdal.Dataset objects. OGR is an internal implementation "
                "detail. Use FeatureCollection.read_file(path) to load a "
                "file, or pass a GeoDataFrame."
            )
        super().__init__(data, *args, **kwargs)

    def __enter__(self) -> FeatureCollection:
        """Enter a context-managed block (ARC-5). Returns ``self``."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit the context-managed block (ARC-5). Calls :meth:`close`."""
        self.close()
        return False

    def close(self) -> None:
        """Release resources held by this FeatureCollection (ARC-5).

        No-op today (the OGR bridge is self-cleaning). Exists so future
        resource-holding features have an idiomatic release point.
        """
        return None

    @classmethod
    def from_features(
        cls,
        features: Iterable[Any],
        *,
        crs: Any = None,
        columns: list[str] | None = None,
    ) -> FeatureCollection:
        """Build a FeatureCollection from GeoJSON-like feature dicts (ARC-28).

        Delegates to :meth:`geopandas.GeoDataFrame.from_features` and
        wraps the result. Accepts any of the shapes that method
        accepts — a list of GeoJSON feature dicts, any object
        implementing ``__geo_interface__``, or an iterator that yields
        such objects.

        Args:
            features (Iterable):
                Feature dicts of the form
                ``{"type": "Feature", "geometry": {...}, "properties": {...}}``,
                or any ``__geo_interface__`` provider. Also accepts a
                bare ``FeatureCollection`` dict.
            crs:
                CRS to attach to the result (EPSG int, ``"EPSG:4326"``,
                WKT, Proj, or a :class:`pyproj.CRS`). ``None`` leaves
                the CRS unset.
            columns (list[str] | None):
                Explicit column order for the output. When ``None``,
                geopandas infers columns from the first feature.

        Returns:
            FeatureCollection: A new FC backed by the supplied features.

        Examples:
            - Build from a list of feature dicts:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> feats = [
                ...     {"type": "Feature",
                ...      "geometry": {"type": "Point", "coordinates": [0, 0]},
                ...      "properties": {"name": "a"}},
                ...     {"type": "Feature",
                ...      "geometry": {"type": "Point", "coordinates": [1, 1]},
                ...      "properties": {"name": "b"}},
                ... ]
                >>> fc = FeatureCollection.from_features(feats, crs=4326)
                >>> len(fc)
                2
                >>> fc.epsg
                4326

                ```
        """
        gdf = gpd.GeoDataFrame.from_features(
            features, crs=crs, columns=columns
        )
        return cls(gdf)

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict],
        *,
        geometry: str = "geometry",
        crs: Any = None,
    ) -> FeatureCollection:
        """Build a FeatureCollection from dict records (ARC-28).

        Each record is a dict whose keys become column names; the
        ``geometry`` key (name configurable) must hold a shapely
        geometry. Useful for ingesting rows from an API response that
        doesn't emit GeoJSON but already has shapely geoms.

        Args:
            records (Iterable[dict]):
                Iterable of ``{column: value, ..., geometry: <shapely>}``
                dicts.
            geometry (str):
                Name of the key holding the shapely geometry. Default
                ``"geometry"``.
            crs:
                CRS to attach (same forms as :meth:`from_features`).

        Returns:
            FeatureCollection: A new FC with one row per record.

        Raises:
            FeatureError: If a record is missing the ``geometry``
                column.
        """
        records_list = list(records)
        if not records_list:
            return cls(
                gpd.GeoDataFrame({geometry: []}, geometry=geometry, crs=crs)
            )
        df = pd.DataFrame.from_records(records_list)
        if geometry not in df.columns:
            raise FeatureError(
                f"records missing required geometry column {geometry!r}; "
                f"columns present: {list(df.columns)}"
            )
        return cls(gpd.GeoDataFrame(df, geometry=geometry, crs=crs))

    _VALID_TILE_STRATEGIES: tuple[str, ...] = (
        "auto",
        "rtree",
        "row_group",
        "none",
    )

    @classmethod
    def iter_features(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        where: str | None = None,
        chunksize: int | None = None,
        tile_strategy: str = "auto",
    ) -> Any:
        """Stream features from ``path`` without materializing the full file.

        ARC-25 + ARC-34. Two orthogonal knobs:

        * **Chunk shape**. ``chunksize=None`` yields one GeoJSON-style
          dict per row (fiona idiom). ``chunksize=N`` yields
          :class:`FeatureCollection` batches of up to N rows each so
          batched pipelines get a DataFrame-shaped payload.
        * **Tile strategy** (ARC-34). Controls whether the ``bbox``
          filter is pushed into the format's spatial index (rtree on
          GPKG, row-group statistics on Parquet, …) or applied after
          a full scan. Pass one of:

          - ``"auto"`` (default) — let pyogrio pick. For a GPKG,
            pyogrio queries the ``rtree_<layer>_geom`` companion
            table automatically. For a Parquet file, pyogrio /
            pyarrow push the bbox down to the row-group statistics
            and skip non-matching groups. For formats without a
            spatial index (GeoJSON, Shapefile without a ``.qix``)
            this falls back to a full scan in the driver.
          - ``"rtree"`` — same as ``"auto"``; kept as an explicit
            name so pipeline code can document intent.
          - ``"row_group"`` — same as ``"auto"``; explicit name for
            the Parquet case.
          - ``"none"`` — disable index pushdown; read whole chunks
            from the driver and apply the bbox filter in Python.
            Useful when the on-disk spatial index is stale or
            suspected wrong; also exercises the "slow path" in
            tests.

        ``bbox`` / ``where`` compose with any tile_strategy. Paths run
        through :func:`pyramids._io._parse_path` so cloud URLs and
        archive paths work the same way as in :meth:`read_file`.

        Args:
            path (str | Path): File path, URL, archive path.
            layer (str | int | None): Layer selector for multi-layer
                formats.
            bbox: ``(minx, miny, maxx, maxy)`` filter.
            where (str | None): OGR SQL predicate.
            chunksize (int | None): ``None`` yields dicts, an ``int``
                yields ``FeatureCollection`` chunks.
            tile_strategy (str): One of ``"auto"``, ``"rtree"``,
                ``"row_group"``, ``"none"``. Default ``"auto"``.

        Yields:
            dict | FeatureCollection: Per-feature dicts when
            ``chunksize`` is ``None``; FeatureCollection chunks
            otherwise.

        Raises:
            ValueError: If ``chunksize`` is given but ``< 1``, or if
                ``tile_strategy`` is not one of the accepted values.
        """
        if chunksize is not None and chunksize < 1:
            raise ValueError(
                f"chunksize must be >= 1 when supplied; got {chunksize}."
            )
        if tile_strategy not in cls._VALID_TILE_STRATEGIES:
            raise ValueError(
                f"tile_strategy must be one of "
                f"{cls._VALID_TILE_STRATEGIES}; got {tile_strategy!r}."
            )

        resolved = str(_pyramids_io._parse_path(path))

        # Determine how many features are in the layer so we can
        # iterate in fixed-size batches via skip_features / max_features.
        # pyogrio's read_info is O(1) per call.
        info_kwargs: dict[str, Any] = {}
        if layer is not None:
            info_kwargs["layer"] = layer
        info = pyogrio.read_info(resolved, **info_kwargs)
        total = int(info["features"])

        if chunksize is None:
            batch_size = 1000
        else:
            batch_size = int(chunksize)

        read_kwargs: dict[str, Any] = {}
        if layer is not None:
            read_kwargs["layer"] = layer
        if where is not None:
            read_kwargs["where"] = where

        # ARC-34: when tile_strategy is "auto"/"rtree"/"row_group",
        # forward the bbox to pyogrio which transparently uses the
        # format's spatial index. When "none", hold the bbox back
        # and apply it in Python after each chunk loads.
        pushdown_bbox = bbox if tile_strategy != "none" else None
        python_bbox = bbox if tile_strategy == "none" else None
        if pushdown_bbox is not None:
            read_kwargs["bbox"] = pushdown_bbox

        for start in range(0, total, batch_size):
            gdf_chunk = gpd.read_file(
                resolved,
                skip_features=start,
                max_features=batch_size,
                **read_kwargs,
            )
            if python_bbox is not None and len(gdf_chunk) > 0:
                xmin, ymin, xmax, ymax = python_bbox
                mask = gdf_chunk.intersects(
                    box(xmin, ymin, xmax, ymax)
                )
                gdf_chunk = gdf_chunk[mask]
            if chunksize is None:
                for feat in gdf_chunk.iterfeatures(na="null"):
                    yield feat
            else:
                yield cls(gdf_chunk)

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        bbox: tuple[float, float, float, float] | Any = None,
        mask: Any = None,
        rows: slice | int | None = None,
        columns: list[str] | None = None,
        where: str | None = None,
        backend: str = "pandas",
        npartitions: int | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> "FeatureCollection | Any":
        """Read a vector file into a FeatureCollection.

        ARC-23: path is first routed through
        :func:`pyramids._io._parse_path`, which handles:

        * Cloud-URL rewriting (``s3://``, ``gs://``, ``az://``,
          ``abfs://``, ``http(s)://``, ``file://`` → GDAL ``/vsi*/``
          form). ARC-22 verified end-to-end through an HTTP test.
          For AWS / GCS / Azure credentials either set the standard
          environment variables (``AWS_ACCESS_KEY_ID``,
          ``AWS_SECRET_ACCESS_KEY``, ``GOOGLE_APPLICATION_CREDENTIALS``,
          ``AZURE_STORAGE_CONNECTION_STRING``, …) or scope them via
          :class:`pyramids.base.remote.CloudConfig` as a context
          manager around the ``read_file`` call.
        * Compressed-archive dispatch for ``.zip``, ``.tar``, ``.tar.gz``,
          ``.gz`` on **local** paths — the returned path is a
          ``/vsizip/``, ``/vsitar/`` or ``/vsigzip/`` string that
          :func:`geopandas.read_file` (via GDAL's virtual filesystem)
          can open directly. You can either pass just the archive
          path (first contained file wins) or
          ``archive.zip/inner.geojson`` to target a specific member.
          Cloud + archive chaining (``http://host/x.zip``) is not
          automatic today — if you need it, stage the archive
          locally first or use ``CloudConfig`` with an explicit
          ``/vsizip//vsicurl/...`` path.

        ARC-24: filter kwargs are pushed down to fiona/pyogrio so the
        dataset never fully materializes when only a subset is needed.

        Args:
            path (str | Path):
                File path, URL, archive path, or
                ``archive.ext/inner-file`` form.
            layer (str | int | None):
                Layer name or index for multi-layer formats
                (GeoPackage, GDB, KML, …). ``None`` reads the first /
                default layer.
            bbox:
                ``(minx, miny, maxx, maxy)`` tuple, or a
                ``GeoDataFrame`` / ``GeoSeries`` / shapely geometry
                whose total bounds are used. Only features
                intersecting the bbox are loaded.
            mask:
                A shapely geometry (or mapping / GeoSeries /
                GeoDataFrame) whose geometries are used as a mask —
                only features intersecting the mask are loaded. Finer
                than ``bbox`` (actual geometry intersection, not just
                envelope). Mutually exclusive with ``bbox``.
            rows (slice | int | None):
                ``int`` — read at most N rows. ``slice`` — read the
                given range of rows. Useful for sampling.
            columns (list[str] | None):
                Restrict loaded attribute columns. Geometry is
                always loaded. ``None`` loads every column.
            where (str | None):
                OGR SQL ``WHERE``-clause predicate pushed down to the
                driver (e.g. ``"population > 10000"``). Avoids loading
                non-matching features.
            **kwargs:
                Forwarded to :func:`geopandas.read_file` verbatim for
                engine-specific options (``engine="pyogrio"``,
                ``use_arrow=True``, driver-specific creation options).

        Returns:
            FeatureCollection | LazyFeatureCollection: When
            ``backend="pandas"`` (default), an eager
            :class:`FeatureCollection`. When ``backend="dask"``, a
            :class:`~pyramids.feature.LazyFeatureCollection` (a
            :class:`dask_geopandas.GeoDataFrame` subclass). Call
            ``.compute()`` on the lazy return to materialize back to an
            eager :class:`FeatureCollection`.

            ``backend="dask"`` requires the optional ``[parquet-lazy]``
            extra. It does NOT honor the ``bbox`` / ``mask`` / ``rows`` /
            ``columns`` / ``where`` / ``layer`` filter kwargs —
            dask-geopandas has no pushdown for them — so supplying any
            of those with ``backend="dask"`` raises :class:`ValueError`.

        Examples:
            - Load a GeoJSON file:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.read_file("tests/data/coello-gauges.geojson")
                >>> len(fc) > 0
                True

                ```
        """
        resolved = _pyramids_io._parse_path(path)
        if backend == "dask":
            # M7: dask_geopandas.read_file does NOT forward pyogrio
            # filter kwargs (bbox / mask / rows / columns / where) —
            # silently dropping them was the bug. Raise a clear
            # ValueError instead so users know to either pre-filter
            # or call .compute() and filter eagerly.
            unsupported = {
                "bbox": bbox, "mask": mask, "rows": rows,
                "columns": columns, "where": where, "layer": layer,
            }
            supplied = [k for k, v in unsupported.items() if v is not None]
            if supplied:
                raise ValueError(
                    f"backend='dask' does not support filter kwargs "
                    f"{supplied}. dask_geopandas.read_file has no "
                    "pushdown story for these. Either omit them and "
                    "filter post-load via .clip / .loc / .compute, or "
                    "switch to read_parquet(backend='dask', filters=...)"
                )
            try:
                import dask_geopandas
            except ImportError as exc:
                raise ImportError(
                    "backend='dask' requires the optional "
                    "'dask-geopandas' dependency. Install with: "
                    "pip install 'pyramids-gis[parquet-lazy]'"
                ) from exc
            partition_kwargs: dict[str, Any] = {}
            if npartitions is not None:
                partition_kwargs["npartitions"] = npartitions
            if chunksize is not None:
                partition_kwargs["chunksize"] = chunksize
            if not partition_kwargs:
                partition_kwargs["npartitions"] = 1
            # DASK-22F: wrap the lazy return as a LazyFeatureCollection so the
            # dask branch stays inside the pyramids type system. The inline
            # import of LazyFeatureCollection is intentional — the import of
            # dask-geopandas above has already gated us on the [parquet-lazy]
            # extra, and moving _lazy_collection to the module top would
            # force that extra onto every minimal install.
            from pyramids.feature._lazy_collection import LazyFeatureCollection

            dask_gdf = dask_geopandas.read_file(resolved, **partition_kwargs)
            return LazyFeatureCollection._from_dask_gdf(dask_gdf)
        if backend != "pandas":
            raise ValueError(
                f"backend must be 'pandas' or 'dask', got {backend!r}"
            )
        passthrough: dict[str, Any] = {}
        if layer is not None:
            passthrough["layer"] = layer
        if bbox is not None:
            passthrough["bbox"] = bbox
        if mask is not None:
            passthrough["mask"] = mask
        if rows is not None:
            passthrough["rows"] = rows
        if columns is not None:
            passthrough["columns"] = columns
        if where is not None:
            passthrough["where"] = where
        passthrough.update(kwargs)
        gdf = gpd.read_file(resolved, **passthrough)
        return cls(gdf)

    @property
    def epsg(self) -> int | None:
        """EPSG code of this FeatureCollection's CRS (ARC-6 cached).

        The value is cached per CRS-object identity so repeated access
        on hot paths skips the ``pyproj.CRS.to_epsg`` call. The cache
        auto-invalidates whenever ``self.crs`` is replaced.
        """
        crs = self.crs
        cached_crs = getattr(self, "_epsg_cache_crs", None)
        if cached_crs is crs:
            return getattr(self, "_epsg_cache_value", None)
        if crs is None:
            value: int | None = None
        else:
            code = crs.to_epsg()
            value = int(code) if code is not None else None
        object.__setattr__(self, "_epsg_cache_crs", crs)
        object.__setattr__(self, "_epsg_cache_value", value)
        return value

    @property
    def top_left_corner(self) -> list[Number]:
        """Top-left corner ``[xmin, ymax]`` of the total bounds."""
        bounds = self.total_bounds.tolist()
        return [bounds[0], bounds[3]]

    @property
    def column(self) -> list[str]:
        """Deprecated alias for :attr:`columns` returning a ``list[str]``."""
        return self.columns.tolist()

    def __str__(self) -> str:
        """Return a short, pyramids-branded summary of the collection."""
        n = len(self)
        cols = self.columns.tolist()
        epsg = self.epsg
        return (
            f"FeatureCollection({n} features, "
            f"columns={cols}, epsg={epsg})"
        )

    def __repr__(self) -> str:
        """Return a pyramids-branded repr."""
        return (
            f"FeatureCollection(n_features={len(self)}, "
            f"columns={self.columns.tolist()}, epsg={self.epsg})"
        )

    @property
    def schema(self) -> dict:
        """Fiona-style schema: geometry type + field-type dict (ARC-27).

        Returns a dict shaped like fiona's ``schema`` attribute so
        callers migrating from ``fiona.open(path).schema`` can consume
        this without rewriting. The dict has two keys:

        * ``"geometry"``: single string (``"Point"``, ``"Polygon"``,
          …) when every row has the same geom type, otherwise
          ``"Unknown"``.
        * ``"properties"``: ``{column_name: dtype_string}`` for every
          non-geometry column.

        Empty FeatureCollections (``len(self) == 0``) report
        ``"Unknown"`` for the geometry type.

        Returns:
            dict: Two-key dict with ``"geometry"`` and ``"properties"``.
        """
        geom_types = {
            g.geom_type
            for g in self.geometry
            if g is not None
        }
        if len(geom_types) == 1:
            (geom_type,) = geom_types
        else:
            geom_type = "Unknown"
        properties = {
            col: str(dt)
            for col, dt in self.dtypes.items()
            if col != "geometry"
        }
        return {"geometry": geom_type, "properties": properties}

    @classmethod
    def list_layers(cls, path: str | Path) -> list[str]:
        """List every vector-layer name in ``path`` (ARC-27).

        Routes through :func:`pyramids._io._parse_path` so the same
        cloud-URL / archive rewriting that :meth:`read_file` uses
        applies here too. Uses :func:`pyogrio.list_layers` under the
        hood (geopandas' default engine).

        Args:
            path (str | Path):
                File path, URL, or archive path. Single-layer formats
                like GeoJSON return one name; multi-layer formats
                (GPKG, GDB, KML) return every layer.

        Returns:
            list[str]: Layer names in the order the driver reports them.
        """
        resolved = _pyramids_io._parse_path(path)
        arr = pyogrio.list_layers(str(resolved))
        # pyogrio returns an ndarray of shape (N, 2): (name, geom_type).
        # We only expose names here; callers who want geom_types too
        # can open individual layers and inspect ``schema``.
        return [str(row[0]) for row in arr]

    @classmethod
    def open_arrow(
        cls,
        path: str | Path,
        *,
        layer: str | int | None = None,
        columns: list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        where: str | None = None,
        batch_size: int | None = None,
    ) -> Any:
        """Open a vector file as a streaming :class:`pyarrow.RecordBatchReader`.

        Thin wrapper over :func:`pyogrio.raw.open_arrow` that
        surfaces the underlying Arrow RecordBatch iterator. Rows are
        yielded in batches, so callers can iterate through multi-GB
        datasets without materializing the whole table in memory —
        useful for building custom dask partitioners.

        Args:
            path: Vector file path (Shapefile, GPKG, FlatGeobuf,
                GeoJSON, GeoParquet, ...). Routed through
                :func:`pyramids._io._parse_path` so cloud URLs work.
            layer: Layer name or index for multi-layer formats.
            columns: Attribute columns to load (``geometry`` is
                always included).
            bbox: ``(minx, miny, maxx, maxy)`` filter.
            where: OGR SQL ``WHERE`` predicate pushed down to the
                driver.
            batch_size: Requested RecordBatch size in rows. ``None``
                uses the driver default.

        Returns:
            pyarrow.RecordBatchReader: A streaming reader.
            Call ``.read_all()`` to materialise, or iterate for
            row-batch consumption.

        Raises:
            ImportError: If :mod:`pyogrio` is not installed.
        """
        try:
            from pyogrio.raw import open_arrow
        except ImportError as exc:
            raise ImportError(
                "open_arrow requires the optional 'pyogrio' dependency. "
                "Install with: pip install pyogrio"
            ) from exc
        resolved = _pyramids_io._parse_path(path)
        kwargs: dict[str, Any] = {}
        if layer is not None:
            kwargs["layer"] = layer
        if columns is not None:
            kwargs["columns"] = columns
        if bbox is not None:
            kwargs["bbox"] = bbox
        if where is not None:
            kwargs["where"] = where
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        return open_arrow(resolved, **kwargs)

    @classmethod
    def read_parquet(
        cls,
        path: str | Path,
        *,
        columns: list[str] | None = None,
        backend: str = "pandas",
        split_row_groups: bool | None = None,
        filters: list | None = None,
        blocksize: int | str | None = None,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> "FeatureCollection | Any":
        """Read a GeoParquet file into a FeatureCollection (ARC-32).

        GeoParquet is a cloud-native columnar vector format (OGC-
        adopted December 2024) — faster to scan than GeoJSON, smaller
        than Shapefile, and partitioned in a way that suits distributed
        compute. This method is a thin wrapper around
        :func:`geopandas.read_parquet`; the path is first routed
        through :func:`pyramids._io._parse_path` so cloud URLs
        (``s3://``, ``gs://``, ``http(s)://``, …) resolve the same way
        they do in :meth:`read_file`.

        Requires the optional :mod:`pyarrow` dependency. Install with
        ``pip install pyramids-gis[parquet]`` or
        ``pixi add pyarrow``.

        Args:
            path (str | Path):
                Local path, cloud URL, or any form
                :func:`pyramids._io._parse_path` accepts.
            columns (list[str] | None):
                Project a subset of columns — Parquet's columnar
                layout makes this a true I/O win, unlike row-oriented
                formats. ``geometry`` is always loaded. ``None``
                loads every column.
            **kwargs:
                Forwarded to :func:`geopandas.read_parquet`
                (``storage_options=`` for fsspec, etc.).

        Returns:
            FeatureCollection | LazyFeatureCollection: When
            ``backend="pandas"`` (default), an eager
            :class:`FeatureCollection`. When ``backend="dask"``, a
            :class:`~pyramids.feature.LazyFeatureCollection` whose
            :attr:`spatial_partitions` survives from the underlying
            dask-geopandas frame. Call ``.compute()`` on the lazy return
            to materialize back to an eager :class:`FeatureCollection`.

        Raises:
            ImportError: If :mod:`pyarrow` is not installed, or if
                ``backend="dask"`` is requested without the
                ``[parquet-lazy]`` extra.
        """
        resolved = _pyramids_io._parse_path(path)
        if backend == "dask":
            try:
                import dask_geopandas
            except ImportError as exc:
                raise ImportError(
                    "backend='dask' requires the optional "
                    "'dask-geopandas' dependency. Install with: "
                    "pip install 'pyramids-gis[parquet-lazy]'"
                ) from exc
            dask_kwargs: dict[str, Any] = {}
            if columns is not None:
                dask_kwargs["columns"] = columns
            if split_row_groups is not None:
                dask_kwargs["split_row_groups"] = split_row_groups
            if filters is not None:
                dask_kwargs["filters"] = filters
            if blocksize is not None:
                dask_kwargs["blocksize"] = blocksize
            if storage_options is not None:
                dask_kwargs["storage_options"] = storage_options
            dask_kwargs.update(kwargs)
            # DASK-23F: wrap the lazy return as a LazyFeatureCollection so the
            # dask branch stays inside the pyramids type system. Same
            # inline-import exception as DASK-22F.
            from pyramids.feature._lazy_collection import LazyFeatureCollection

            dask_gdf = dask_geopandas.read_parquet(resolved, **dask_kwargs)
            return LazyFeatureCollection._from_dask_gdf(dask_gdf)
        if backend != "pandas":
            raise ValueError(
                f"backend must be 'pandas' or 'dask', got {backend!r}"
            )
        passthrough: dict[str, Any] = dict(kwargs)
        if columns is not None:
            passthrough["columns"] = columns
        if storage_options is not None:
            passthrough["storage_options"] = storage_options
        gdf = gpd.read_parquet(resolved, **passthrough)
        return cls(gdf)

    def to_parquet(
        self,
        path: str | Path,
        *,
        compression: str = "snappy",
        index: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this FeatureCollection to GeoParquet (ARC-32).

        Thin wrapper around :meth:`geopandas.GeoDataFrame.to_parquet`
        that defaults :param:`compression` to ``"snappy"`` — the
        format-standard tradeoff between speed and size.

        Requires the optional :mod:`pyarrow` dependency. Install with
        ``pip install pyramids-gis[parquet]`` or
        ``pixi add pyarrow``.

        Args:
            path (str | Path):
                Destination file path.
            compression (str):
                Parquet compression codec — ``"snappy"`` (default),
                ``"gzip"``, ``"brotli"``, ``"lz4"``, ``"zstd"``, or
                ``"none"``. ``"snappy"`` is the GeoParquet-spec
                recommended default.
            index (bool | None):
                Whether to include the pandas index as a column.
                ``None`` (default) uses geopandas' default behavior:
                preserve a non-default index, drop the default
                ``RangeIndex``.
            **kwargs:
                Forwarded to :meth:`geopandas.GeoDataFrame.to_parquet`.

        Raises:
            ImportError: If :mod:`pyarrow` is not installed.
        """
        super().to_parquet(
            path, compression=compression, index=index, **kwargs
        )

    def to_file(
        self,
        path: str | Path,
        driver: str = "geojson",
        *,
        layer: str | None = None,
        mode: str = "w",
        **creation_options: Any,
    ) -> None:
        """Write this FeatureCollection to a vector file.

        ARC-26: ``layer``, ``mode``, and arbitrary driver creation
        options are now first-class kwargs. Previously callers had to
        rely on implicit ``**kwargs`` forwarding, which hurt
        discoverability.

        Args:
            path (str | Path):
                Destination file path.
            driver (str):
                Driver alias (e.g. ``"geojson"``, ``"gpkg"``) or
                literal GDAL driver name (``"GeoJSON"``, ``"GPKG"``,
                ``"ESRI Shapefile"``). Resolved via :class:`Catalog`.
            layer (str | None):
                Layer name for multi-layer drivers (GPKG, GDB, …).
                Writing two layers into the same GPKG is the canonical
                use case. ``None`` defers to the driver default.
            mode (str):
                ``"w"`` (default) overwrites; ``"a"`` appends to an
                existing layer. Append support depends on the driver
                — GPKG and Shapefile accept it, GeoJSON does not.
            **creation_options:
                Driver-specific creation options, forwarded to the
                underlying engine (pyogrio / fiona). Examples:

                * GPKG: ``SPATIAL_INDEX="YES"``, ``FID="id"``.
                * Shapefile: ``ENCODING="UTF-8"``.
                * GeoJSON: ``COORDINATE_PRECISION=6``, ``RFC7946=YES``.

                Keys are case-preserving and passed verbatim to the
                driver; consult the GDAL driver docs for the full
                list.

        Raises:
            ValueError: If ``mode`` isn't ``"w"`` or ``"a"``.

        Examples:
            - Write a GeoJSON (default driver):
                ```python
                fc.to_file("out.geojson")
                ```
            - Write a GPKG with spatial index + named layer:
                ```python
                fc.to_file(
                    "out.gpkg", driver="gpkg",
                    layer="rivers", SPATIAL_INDEX="YES",
                )
                ```
            - Append to an existing layer:
                ```python
                fc.to_file("out.gpkg", driver="gpkg",
                           layer="rivers", mode="a")
                ```
        """
        if mode not in ("w", "a"):
            raise ValueError(
                f"mode must be 'w' (write) or 'a' (append); got {mode!r}."
            )
        try:
            resolved = CATALOG.get_gdal_name(driver) or driver
        except AttributeError:
            resolved = driver

        passthrough: dict[str, Any] = {"driver": resolved, "mode": mode}
        if layer is not None:
            passthrough["layer"] = layer
        passthrough.update(creation_options)
        super().to_file(path, **passthrough)

    # ARC-4: FeatureCollection.to_dataset was moved to
    # Dataset.from_features(features, ...) to break the circular import
    # that used to force a CLAUDE.md-violating inline
    # ``from pyramids.dataset import Dataset`` inside the method body.
    # Callers should migrate:
    #     fc.to_dataset(dataset=ds, column_name="pop")
    #         → Dataset.from_features(fc, template=ds, column_name="pop")
    #     fc.to_dataset(cell_size=10)
    #         → Dataset.from_features(fc, cell_size=10)

    # Static delegates to helper modules kept for backward compatibility
    # (callers may do ``FeatureCollection.create_polygon(...)``).

    @staticmethod
    def _create_sr_from_proj(
        prj: str, string_type: str | None = None
    ) -> osr.SpatialReference:
        """Delegate to :func:`pyramids.feature.crs.create_sr_from_proj`."""
        return _crs.create_sr_from_proj(prj, string_type)

    @staticmethod
    def get_epsg_from_prj(prj: str) -> int:
        """Delegate to :func:`pyramids.feature.crs.get_epsg_from_prj`."""
        return _crs.get_epsg_from_prj(prj)

    @staticmethod
    def reproject_coordinates(
        x: list[float],
        y: list[float],
        *,
        from_crs: Any = 4326,
        to_crs: Any = 3857,
        precision: int | None = 6,
    ) -> tuple[list[float], list[float]]:
        """Delegate to :func:`pyramids.feature.crs.reproject_coordinates`.

        ARC-14: canonical replacement for the deleted
        ``reproject_points`` / ``reproject_points2`` pair. Argument and
        return order is ``(x, y)`` throughout. ``from_crs`` / ``to_crs``
        accept any form :meth:`pyproj.Transformer.from_crs` understands
        (EPSG int, authority string, WKT, Proj4, :class:`pyproj.CRS`).

        Examples:
            - Reproject a single WGS84 point into Web Mercator:
                ```python
                >>> x, y = FeatureCollection.reproject_coordinates(
                ...     [31.0], [30.0], from_crs=4326, to_crs=3857
                ... )
                >>> round(x[0])
                3450904
                >>> round(y[0])
                3503550

                ```
        """
        return _crs.reproject_coordinates(
            x, y, from_crs=from_crs, to_crs=to_crs, precision=precision
        )

    @staticmethod
    def _get_xy_coords(geometry: Any, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_xy_coords`."""
        return _geom.get_xy_coords(geometry, coord_type)

    @staticmethod
    def _get_point_coords(geometry: Point, coord_type: str) -> float | int:
        """Delegate to :func:`pyramids.feature.geometry.get_point_coords`."""
        return _geom.get_point_coords(geometry, coord_type)

    @staticmethod
    def _get_line_coords(geometry: LineString, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_line_coords`."""
        return _geom.get_line_coords(geometry, coord_type)

    @staticmethod
    def _get_poly_coords(geometry: Polygon, coord_type: str) -> list:
        """Delegate to :func:`pyramids.feature.geometry.get_poly_coords`."""
        return _geom.get_poly_coords(geometry, coord_type)

    @staticmethod
    def _explode_gdf(
        gdf: GeoDataFrame, geometry: str = "multipolygon"
    ) -> GeoDataFrame:
        """Delegate to :func:`pyramids.feature.geometry.explode_gdf`."""
        return _geom.explode_gdf(gdf, geometry)

    @staticmethod
    def _multi_geom_handler(
        multi_geometry: MultiPolygon | MultiPoint | MultiLineString,
        coord_type: str,
        geom_type: str,
    ) -> list:
        """Delegate to :func:`pyramids.feature.geometry.multi_geom_handler`."""
        return _geom.multi_geom_handler(multi_geometry, coord_type, geom_type)

    @staticmethod
    def _geometry_collection(geom: Any, coord_type: str) -> list[Any]:
        """Delegate to :func:`pyramids.feature.geometry.geometry_collection_coords`."""
        return _geom.geometry_collection_coords(geom, coord_type)

    @staticmethod
    def _get_coords(row: Any, geom_col: str, coord_type: str) -> Any:
        """Delegate to :func:`pyramids.feature.geometry.get_coords`."""
        return _geom.get_coords(row, geom_col, coord_type)

    @staticmethod
    def create_polygon(
        coords: list[tuple[float, float]], wkt: bool = False
    ) -> str | Polygon:
        """Create a Polygon (or its WKT) from coordinates.

        ARC-15: the ``wkt=True`` form is deprecated in favor of
        :meth:`polygon_wkt`. ``wkt=False`` (the default) now delegates
        to the new unconditional :func:`pyramids.feature.geometry.create_polygon`
        which always returns a ``Polygon``.
        """
        if wkt:
            return _geom.create_polygon_legacy(coords, wkt=True)
        return _geom.create_polygon(coords)

    @staticmethod
    def polygon_wkt(coords: list[tuple[float, float]]) -> str:
        """Return the WKT for a polygon built from ``coords`` (ARC-15).

        Delegates to :func:`pyramids.feature.geometry.polygon_wkt`.
        """
        return _geom.polygon_wkt(coords)

    @staticmethod
    def create_point(
        coords: Iterable[tuple[float, ...]], epsg: int | None = None
    ) -> list[Point] | FeatureCollection:
        """Create shapely Points (or a FeatureCollection wrapper).

        ARC-15: the polymorphic return is deprecated. Use
        :meth:`create_points` for the list form and
        :meth:`point_collection` for the FeatureCollection form. This
        method kept for back-compat; it emits DeprecationWarning when
        ``epsg`` is provided.
        """
        if epsg is not None:
            result = _geom.create_point_legacy(coords, epsg=epsg)
            return FeatureCollection(result)
        return _geom.create_points(coords)

    @staticmethod
    def create_points(coords: Iterable[tuple[float, ...]]) -> list[Point]:
        """Return a list of shapely Points from ``coords`` (ARC-15).

        Delegates to :func:`pyramids.feature.geometry.create_points`.
        """
        return _geom.create_points(coords)

    @staticmethod
    def point_collection(
        coords: Iterable[tuple[float, ...]], crs: Any
    ) -> FeatureCollection:
        """Return a FeatureCollection of points with the given CRS (ARC-15).

        Delegates to :func:`pyramids.feature.geometry.point_collection`
        and wraps the result as a ``FeatureCollection``.
        """
        return FeatureCollection(_geom.point_collection(coords, crs=crs))

    def with_coordinates(self) -> FeatureCollection:
        """Return a new FeatureCollection with per-vertex ``x`` and ``y`` columns.

        ARC-16: non-mutating replacement for the old ``xy()`` method
        (which has been deleted). Matches pandas / geopandas
        convention — data-transformation methods return a new object.
        The ``with_`` prefix follows the stdlib/pandas pattern for
        "return a copy with this change applied" (e.g.
        :meth:`pathlib.Path.with_suffix`).

        Explodes MultiPolygon and GeometryCollection geometries into
        their parts first, then attaches ``x`` and ``y`` columns
        containing the coordinate sequences of each row.

        Returns:
            FeatureCollection: A new FeatureCollection (``self`` is
            not modified) with the original columns plus ``x`` and
            ``y`` per-vertex coordinate lists.
        """
        gdf = _geom.explode_gdf(
            gpd.GeoDataFrame(self, copy=True), geometry="multipolygon"
        )
        gdf = _geom.explode_gdf(gdf, geometry="geometrycollection")

        fc = FeatureCollection(gdf)
        fc["x"] = fc.apply(
            _geom.get_coords, geom_col="geometry", coord_type="x", axis=1
        )
        fc["y"] = fc.apply(
            _geom.get_coords, geom_col="geometry", coord_type="y", axis=1
        )
        fc.reset_index(drop=True, inplace=True)
        return fc

    def plot(
        self,
        column: str | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot features, optionally on a web-tile basemap.

        Delegates to :meth:`geopandas.GeoDataFrame.plot` and, when
        ``basemap`` is truthy, adds an OSM (or named provider) tile
        layer underneath.

        Raises:
            ValueError: If ``basemap`` is requested but the FC has no CRS.
        """
        ax = super().plot(column=column, **kwargs)

        if basemap:
            if self.epsg is None:
                raise CRSError(
                    "FeatureCollection must have a CRS (epsg) to use basemap."
                )
            from pyramids.basemap.basemap import add_basemap

            source = basemap if isinstance(basemap, str) else None
            add_basemap(ax, crs=self.epsg, source=source)

        return ax

    def concat(self, other: GeoDataFrame) -> FeatureCollection:
        """Concatenate another GeoDataFrame onto this FeatureCollection.

        ARC-16: mirrors :func:`pandas.concat` — returns a new
        ``FeatureCollection`` and never mutates ``self``. No
        ``inplace`` kwarg (pandas' ``pd.concat`` has never had one;
        follow the convention).

        Equivalent to ``pd.concat([fc, other])`` which also works
        directly and returns a ``FeatureCollection`` via the
        ``_constructor`` hook (ARC-1a).

        Args:
            other (GeoDataFrame): The rows to append.

        Returns:
            FeatureCollection: A new FC containing ``self``'s rows
            followed by ``other``'s rows, with ``self``'s CRS and a
            freshly-reset index.
        """
        combined = gpd.GeoDataFrame(pd.concat([self, other]))
        combined.index = list(range(len(combined)))
        combined.crs = self.crs
        return FeatureCollection(combined)

    def with_centroid(self) -> FeatureCollection:
        """Return a new FC with per-feature center-point columns attached.

        ARC-16: non-mutating replacement for the old ``center_point()``
        method (which has been deleted). The ``with_`` prefix mirrors
        stdlib / pandas conventions for "return a copy with this
        change applied".

        Computes average x/y per feature (after
        :meth:`with_coordinates`) and attaches three columns:
        ``avg_x``, ``avg_y`` and ``center_point`` (shapely ``Point``).

        Returns:
            FeatureCollection: A new FeatureCollection (``self`` is
            not modified) with ``x``, ``y``, ``avg_x``, ``avg_y``,
            ``center_point`` columns added.
        """
        fc = self.with_coordinates()
        for i, row_i in fc.iterrows():
            fc.loc[i, "avg_x"] = np.mean(row_i["x"])
            fc.loc[i, "avg_y"] = np.mean(row_i["y"])

        coords_list = zip(fc["avg_x"].tolist(), fc["avg_y"].tolist())
        fc["center_point"] = _geom.create_points(coords_list)
        return fc
