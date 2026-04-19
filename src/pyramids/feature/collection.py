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

import functools
from numbers import Number
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids.base._utils import Catalog
from pyramids.feature import crs as _crs
from pyramids.feature import geometry as _geom

CATALOG = Catalog(raster_driver=False)
gdal.UseExceptions()


# C15: module-level LRU cache backing ``FeatureCollection.list_layers``.
# Keyed on the already-resolved ``str`` path (post ``_parse_path``). The
# tuple return type plays nicely with ``functools.lru_cache`` (lists are
# unhashable and would break LRU internals if returned directly).
@functools.lru_cache(maxsize=128)
def _list_layers_cached(resolved_path: str) -> tuple[str, ...]:
    """Return a tuple of layer names for a resolved path (memoised)."""
    import pyogrio

    arr = pyogrio.list_layers(resolved_path)
    return tuple(str(row[0]) for row in arr)


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
    # C3 — dedupe via ``dict.fromkeys`` so that if a future geopandas
    # release adds one of our own names to its own ``_metadata`` list,
    # the pyramids subclass does not carry a duplicate entry. Python
    # preserves insertion order in dicts since 3.7, so the parent's
    # ordering is preserved.
    _metadata: list[str] = list(dict.fromkeys([
        *GeoDataFrame._metadata,
        "_epsg_cache_crs",
        "_epsg_cache_value",
    ]))
    """Instance attributes pandas must preserve across copy/slice/pickle.

    Holds:

    * ``GeoDataFrame._metadata`` (currently ``_geometry_column_name``)
      — required for pickle round-trips to remember which column is
      the active geometry column.
    * ``_epsg_cache_crs`` / ``_epsg_cache_value`` — the ARC-6 EPSG
      cache.

    The list is wrapped in ``list(dict.fromkeys(...))`` so that a
    future geopandas release adding one of our own names to its own
    ``_metadata`` list does not produce a duplicate entry. ``dict``
    preserves insertion order since Python 3.7, so the parent's
    ordering is preserved (C3).
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

        Raises:
            ValueError: If ``features`` is empty or exhausted before any
                feature is consumed (C9). An empty GeoDataFrame from
                ``from_features`` has no ``geometry`` column, which
                breaks downstream pyramids methods that assume the
                column exists. Fail fast instead.

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
        # C9: materialise an iterator so we can detect the empty case
        # before handing off to geopandas. ``geopandas.from_features([])``
        # returns a GeoDataFrame with no ``geometry`` column, which
        # breaks every pyramids op that assumes the column exists.
        features_list = list(features)
        if not features_list:
            raise ValueError(
                "from_features requires at least one feature. An empty "
                "iterable would produce a GeoDataFrame with no geometry "
                "column, which breaks downstream pyramids methods."
            )
        gdf = gpd.GeoDataFrame.from_features(
            features_list, crs=crs, columns=columns
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
        import pandas as pd

        from pyramids.base._errors import FeatureError

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
        include_index: bool = False,
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
            include_index (bool): When ``True``, each yielded dict gets
                an additional ``"id"`` key whose value is the
                0-based file-row index of that feature. The chunked
                form (``chunksize=N``) attaches the same index as a
                ``"_row_index"`` column on the yielded FC. The indices
                stay aligned with the on-disk rows even when a
                Python-side bbox filter (``tile_strategy="none"``)
                drops some rows — only the surviving features are
                yielded, and their ids match the positions they had
                in the source file. Defaults to ``False`` for
                back-compat with the fiona idiom.

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

        import pyogrio

        from pyramids import _io as _pyramids_io

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

        # D-M3: pin the engine to pyogrio. ``skip_features`` /
        # ``max_features`` are pyogrio-specific (geopandas' fiona
        # engine silently ignores them, which would turn every chunk
        # into a full scan). Pinning the engine makes the contract
        # explicit and fails fast if pyogrio is absent.
        read_kwargs: dict[str, Any] = {"engine": "pyogrio"}
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
            # C14: remember the absolute row indices before any
            # bbox-based masking so callers can map yielded features
            # back to their source rows even after a Python-side filter
            # has dropped some of them.
            if include_index:
                row_indices = list(range(start, start + len(gdf_chunk)))
            if python_bbox is not None and len(gdf_chunk) > 0:
                xmin, ymin, xmax, ymax = python_bbox
                mask = gdf_chunk.intersects(
                    box(xmin, ymin, xmax, ymax)
                )
                if include_index:
                    row_indices = [
                        ri for ri, keep in zip(row_indices, mask) if keep
                    ]
                gdf_chunk = gdf_chunk[mask]
            if chunksize is None:
                iterator = gdf_chunk.iterfeatures(na="null")
                if include_index:
                    for ri, feat in zip(row_indices, iterator):
                        feat["id"] = ri
                        yield feat
                else:
                    for feat in iterator:
                        yield feat
            else:
                chunk_fc = cls(gdf_chunk)
                if include_index:
                    chunk_fc["_row_index"] = row_indices
                yield chunk_fc

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
        **kwargs: Any,
    ) -> FeatureCollection:
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
            FeatureCollection: The (possibly filtered) features
            wrapped as a FeatureCollection.

        Examples:
            - Load a GeoJSON file:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.read_file("tests/data/coello-gauges.geojson")
                >>> len(fc) > 0
                True

                ```
        """
        from pyramids import _io as _pyramids_io

        resolved = _pyramids_io._parse_path(path)
        # Only pass kwargs that were actually supplied — passing the
        # defaults (None) is fine for some geopandas engines but
        # confuses others. Build a clean kwargs dict.
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

        C11: identity-miss falls back to equality. If ``self.crs`` has
        been reassigned to a different CRS object that nevertheless
        compares equal to the cached one (e.g. ``fc.crs = pyproj.CRS(
        "EPSG:4326")`` on a frame already in EPSG:4326), we adopt the
        new object as the cache key and skip the ``.to_epsg()`` call.
        Only when the value really differs do we recompute.
        """
        crs = self.crs
        cached_crs = getattr(self, "_epsg_cache_crs", None)
        if cached_crs is crs:
            return getattr(self, "_epsg_cache_value", None)
        # C11: try equality before falling back to a fresh to_epsg() call.
        # pyproj.CRS comparison is cheaper than a full re-parse, and the
        # common "reassign an equivalent CRS" case (e.g. set_crs chain)
        # should stay in the fast path.
        if cached_crs is not None and crs is not None:
            try:
                equivalent = cached_crs == crs
            except (TypeError, ValueError):
                equivalent = False
            if equivalent:
                object.__setattr__(self, "_epsg_cache_crs", crs)
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

        C15: results are memoised behind a 128-entry LRU cache keyed on
        the resolved ``str`` path. Re-calling ``list_layers`` on the
        same cloud URL or local path in a loop now costs one hash
        lookup instead of one datasource open. Call
        :meth:`list_layers_cache_clear` to invalidate after an
        out-of-band write.

        Args:
            path (str | Path):
                File path, URL, or archive path. Single-layer formats
                like GeoJSON return one name; multi-layer formats
                (GPKG, GDB, KML) return every layer.

        Returns:
            list[str]: Layer names in the order the driver reports them.
        """
        from pyramids import _io as _pyramids_io

        resolved = str(_pyramids_io._parse_path(path))
        return list(_list_layers_cached(resolved))

    @classmethod
    def list_layers_cache_clear(cls) -> None:
        """Clear the C15 LRU cache backing :meth:`list_layers`.

        Call this after writing a new layer to an existing multi-layer
        file (e.g. a GPKG) if you then want :meth:`list_layers` to see
        the new layer. Otherwise the 128-entry LRU cache is self-
        managing and callers do not need to touch it.
        """
        _list_layers_cached.cache_clear()

    @classmethod
    def read_parquet(
        cls,
        path: str | Path,
        *,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> FeatureCollection:
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
            FeatureCollection: The file's features wrapped as a
            FeatureCollection.

        Raises:
            ImportError: If :mod:`pyarrow` is not installed.
        """
        from pyramids import _io as _pyramids_io

        resolved = _pyramids_io._parse_path(path)
        passthrough: dict[str, Any] = dict(kwargs)
        if columns is not None:
            passthrough["columns"] = columns
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

                C8: pyogrio (the default geopandas engine on 1.0+)
                raises :class:`ValueError` with the message
                ``"unrecognized option '<name>' for driver '<driver>'"``
                when a supplied option is neither in the driver's
                dataset nor its layer creation-option list. This
                surfaces typos (``SPATIAL_INDX`` vs ``SPATIAL_INDEX``)
                at write-time rather than silently producing a
                different file. Some drivers may still accept options
                that pyogrio does not list — verify against the
                driver's docs when in doubt.

        Raises:
            ValueError: If ``mode`` isn't ``"w"`` or ``"a"``, or if a
                supplied creation option is not recognised by the
                driver (raised by pyogrio — see the ``**creation_options``
                note above).

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
    def create_polygon(coords: list[tuple[float, float]]) -> Polygon:
        """Create a :class:`shapely.Polygon` from coordinates.

        Delegates to :func:`pyramids.feature.geometry.create_polygon`.
        For the WKT-string form use :meth:`polygon_wkt` instead.

        D-H2: the ARC-15 ``wkt=True`` polymorphic kwarg was removed
        outright (no deprecation shim) — a polymorphic return type
        was the whole motivation for splitting ARC-15 in the first
        place.
        """
        return _geom.create_polygon(coords)

    @staticmethod
    def polygon_wkt(coords: list[tuple[float, float]]) -> str:
        """Return the WKT for a polygon built from ``coords`` (ARC-15).

        Delegates to :func:`pyramids.feature.geometry.polygon_wkt`.
        """
        return _geom.polygon_wkt(coords)

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
                from pyramids.base._errors import CRSError

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
        import pandas as pd

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

        C18: feeding a degenerate or empty geometry (for example an
        empty ``Point``, or a ``Polygon`` whose ring has zero area)
        produces ``(NaN, NaN)`` averages. The method emits a single
        ``UserWarning`` listing the row indices whose ``avg_x`` /
        ``avg_y`` could not be computed so downstream code can guard
        against the NaN centroids instead of silently consuming them.
        The ``center_point`` value at those rows is an empty
        ``shapely.Point`` (``Point.is_empty is True``) rather than a
        ``(NaN, NaN)`` point.

        Returns:
            FeatureCollection: A new FeatureCollection (``self`` is
            not modified) with ``x``, ``y``, ``avg_x``, ``avg_y``,
            ``center_point`` columns added.
        """
        fc = self.with_coordinates()
        for i, row_i in fc.iterrows():
            fc.loc[i, "avg_x"] = np.mean(row_i["x"])
            fc.loc[i, "avg_y"] = np.mean(row_i["y"])

        # C18: detect rows whose averaged coordinate could not be
        # computed (empty geometry, all-NaN rings, etc.). Emit a single
        # summary warning and substitute an empty Point so the column
        # does not expose a ``(NaN, NaN)`` Point that would then crash
        # downstream reprojections.
        avg_x = fc["avg_x"].to_numpy()
        avg_y = fc["avg_y"].to_numpy()
        bad_mask = np.isnan(avg_x) | np.isnan(avg_y)
        if bad_mask.any():
            bad_idx = [int(i) for i, is_bad in enumerate(bad_mask) if is_bad]
            import warnings

            warnings.warn(
                f"with_centroid: {len(bad_idx)} row(s) yielded NaN centroids "
                f"(rows {bad_idx}). Their ``center_point`` is an empty "
                f"shapely.Point. Drop or repair those rows before running "
                f"a method that requires a valid centroid (e.g. reproject, "
                f"distance).",
                UserWarning,
                stacklevel=2,
            )

        coords_list = []
        for idx, (ax, ay) in enumerate(zip(avg_x.tolist(), avg_y.tolist())):
            if bad_mask[idx]:
                coords_list.append((float("nan"), float("nan")))
            else:
                coords_list.append((ax, ay))
        points = _geom.create_points(coords_list)
        # Substitute empty Points for the NaN rows so the column's
        # invariant is "every entry is a non-NaN shapely Point OR is
        # Point.is_empty".
        cleaned: list[Any] = []
        for idx, pt in enumerate(points):
            cleaned.append(Point() if bad_mask[idx] else pt)
        fc["center_point"] = cleaned
        return fc
