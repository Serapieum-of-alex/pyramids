"""FeatureCollection — a GeoDataFrame with pyramids-specific GIS methods.

``FeatureCollection`` is a direct subclass of :class:`geopandas.GeoDataFrame`.
It is an *extension* of GeoDataFrame, not a replacement: every GeoDataFrame
method (``to_crs``, ``clip``, ``buffer``, ``dissolve``, ``overlay``,
``sjoin``, ``sindex``, ``query``, ``explode``, ``centroid``, ``area``,
``length``, ``make_valid``, ``plot``, …) is inherited and works directly on
a ``FeatureCollection`` instance. On top of that, FeatureCollection adds
pyramids-specific capabilities that no pure GeoDataFrame offers:

- Rasterization to a :class:`pyramids.dataset.Dataset` (:meth:`to_dataset`).
- Pyramids-specific spatial metadata (:attr:`epsg`, :attr:`top_left_corner`).
- Geometry-factory helpers (:meth:`create_polygon`, :meth:`create_point`).
- Vertex extraction (:meth:`xy`, :meth:`center_point`).
- EPSG / SR helpers (:meth:`get_epsg_from_prj`, :meth:`reproject_points`,
  :meth:`reproject_points2`).

The ``ogr.DataSource`` object is **not** part of the public API of this
module. It is used only inside methods that must call GDAL/OGR
(rasterization), and only via the private bridge in
:mod:`pyramids.feature._ogr`. It is never accepted by a public
constructor, never returned, and never stored as instance state.
"""

from __future__ import annotations

import warnings
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pyramids.dataset import Dataset

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform as _pyproj_transform
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids.base._errors import DriverNotExistError
from pyramids.base._utils import Catalog
from pyramids.feature import _ogr

CATALOG = Catalog(raster_driver=False)
gdal.UseExceptions()


class FeatureCollection(GeoDataFrame):
    """A :class:`geopandas.GeoDataFrame` with pyramids-specific GIS methods.

    ``FeatureCollection`` *is a* ``GeoDataFrame`` — ``isinstance(fc,
    GeoDataFrame)`` is ``True`` — so every geopandas method is available
    directly. Pyramids adds rasterization, Dataset interop, vertex
    extraction, and CRS helpers on top.

    The OGR/GDAL backend is internal only; see :mod:`pyramids.feature._ogr`.
    """

    # ── pandas / geopandas subclass contract ──
    # Override ``_constructor`` so pandas returns FeatureCollection (not
    # plain GeoDataFrame) through slicing, filtering, copy(), concat(),
    # groupby().apply(), etc. Do NOT override ``_constructor_sliced``:
    # GeoDataFrame's implementation already returns GeoSeries for the
    # geometry column and Series for non-geometry columns. Replacing it
    # with a hardcoded GeoSeries breaks internals like ``self.dtypes``
    # which builds a non-geometry Series via ``_constructor_sliced``.

    @property
    def _constructor(self):
        """Return the type pandas uses when constructing new frames."""
        return FeatureCollection

    _metadata: list[str] = []
    """Instance attributes pandas must preserve across copy/slice.

    Currently empty — ``FeatureCollection`` keeps no per-instance state
    beyond the GeoDataFrame data itself. Extend this list if you add
    an instance attribute that must survive pandas operations.
    """

    # ── construction ──

    def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Construct a FeatureCollection.

        Accepts anything :class:`geopandas.GeoDataFrame` accepts. Rejects
        ``ogr.DataSource`` / ``gdal.Dataset`` with a clear error — those
        objects are pyramids-internal and must not appear in public APIs.

        Args:
            data:
                A ``GeoDataFrame`` / ``DataFrame`` / dict / records
                iterable — anything ``GeoDataFrame.__init__`` accepts.
            *args:
                Positional args forwarded to ``GeoDataFrame.__init__``.
            **kwargs:
                Keyword args (``geometry=``, ``crs=``, ``columns=``, …)
                forwarded to ``GeoDataFrame.__init__``.

        Raises:
            TypeError: If ``data`` is an ``ogr.DataSource`` or a
                ``gdal.Dataset``. These types are internal-only; use
                :meth:`read_file` or a ``GeoDataFrame`` instead.
        """
        if isinstance(data, (ogr.DataSource, gdal.Dataset)):
            raise TypeError(
                "FeatureCollection no longer accepts ogr.DataSource or "
                "gdal.Dataset objects. OGR is an internal implementation "
                "detail. Use FeatureCollection.read_file(path) to load a "
                "file, or pass a GeoDataFrame."
            )
        super().__init__(data, *args, **kwargs)

    @classmethod
    def read_file(cls, path: str | Path) -> FeatureCollection:
        """Read a vector file into a FeatureCollection.

        Delegates to :func:`geopandas.read_file` and wraps the result.

        Args:
            path (str | Path):
                Path to any format :mod:`fiona` / :mod:`geopandas`
                supports (GeoJSON, Shapefile, GeoPackage, …).

        Returns:
            FeatureCollection: A new FeatureCollection with the file's
            features, columns, and CRS.

        Examples:
            - Load a GeoJSON file:
                ```python
                >>> from pyramids.feature import FeatureCollection
                >>> fc = FeatureCollection.read_file("tests/data/coello-gauges.geojson")
                >>> len(fc) > 0
                True

                ```
        """
        gdf = gpd.read_file(path)
        return cls(gdf)

    # ── pyramids-specific spatial metadata ──

    @property
    def epsg(self) -> int | None:
        """EPSG code of this FeatureCollection's CRS.

        Returns:
            int | None: EPSG code (e.g. 4326) or ``None`` when the CRS
            is unset or cannot be resolved to an EPSG authority code.
        """
        if self.crs is None:
            return None
        code = self.crs.to_epsg()
        return int(code) if code is not None else None

    @property
    def top_left_corner(self) -> list[Number]:
        """Top-left corner ``[xmin, ymax]`` of the total bounds.

        Returns:
            list[Number]: A two-element list ``[minx, maxy]`` in the
            FeatureCollection's CRS.
        """
        bounds = self.total_bounds.tolist()
        return [bounds[0], bounds[3]]

    @property
    def column(self) -> list[str]:
        """Deprecated alias for :attr:`columns` returning a ``list[str]``.

        ``columns`` is inherited from :class:`pandas.DataFrame` and is an
        :class:`pandas.Index`. This alias returns a plain list, matching
        the original pyramids API.

        Returns:
            list[str]: Column names, including ``"geometry"``.
        """
        return self.columns.tolist()

    # ── dunders (branding only; data-level dunders are inherited) ──

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
        """Return a pyramids-branded repr with feature count, columns, EPSG."""
        return (
            f"FeatureCollection(n_features={len(self)}, "
            f"columns={self.columns.tolist()}, epsg={self.epsg})"
        )

    # ── I/O ──

    def to_file(
        self, path: str | Path, driver: str = "geojson", **kwargs: Any
    ) -> None:
        """Write this FeatureCollection to a vector file.

        Delegates to :meth:`geopandas.GeoDataFrame.to_file` after
        resolving the pyramids driver short name (e.g. ``"geojson"``) to
        the GDAL driver name (``"GeoJSON"``) via :class:`Catalog`.

        Args:
            path (str | Path):
                Destination file path.
            driver (str):
                Driver alias or GDAL driver name. Default ``"geojson"``.
            **kwargs:
                Additional keyword arguments forwarded to
                ``GeoDataFrame.to_file`` (e.g. ``layer=``, ``mode=``,
                driver creation options).

        Raises:
            DriverNotExistError: If the driver alias is unknown.
        """
        try:
            resolved = CATALOG.get_gdal_name(driver) or driver
        except AttributeError:
            # Driver isn't in the Catalog alias map — assume ``driver`` is
            # already a GDAL driver name and pass it through unchanged.
            resolved = driver
        super().to_file(path, driver=resolved, **kwargs)

    # ── rasterization (uses the internal OGR bridge) ──

    def to_dataset(
        self,
        cell_size: Any | None = None,
        dataset: Dataset | None = None,
        column_name: str | list[str] | None = None,
    ) -> Dataset:
        """Rasterize this FeatureCollection to a :class:`pyramids.Dataset`.

        Burns the values from ``column_name`` (or every attribute column
        if ``None``) into a single-band or multi-band raster. When a
        template ``dataset`` is given, the new raster adopts its
        geotransform, cell size, row/column count, and no-data value;
        otherwise the extent is derived from :attr:`total_bounds` and
        ``cell_size`` controls the resolution.

        Args:
            cell_size (int | float | None):
                Cell size for the new raster. Required if ``dataset`` is
                not given; ignored otherwise.
            dataset (Dataset | None):
                Optional template raster. When provided, the output
                raster inherits its geotransform (top-left, cell size,
                row/column count) and no-data value.
            column_name (str | list[str] | None):
                Attribute column(s) to burn as band values. ``None``
                burns every non-geometry column as a separate band.

        Returns:
            Dataset: The burned raster. Single band if ``column_name``
            is a string or a single-entry list; multi-band (one band per
            column) otherwise.

        Raises:
            ValueError: If neither ``cell_size`` nor ``dataset`` is
                given, if the vector CRS disagrees with the raster CRS,
                or if the internal OGR conversion fails.
            TypeError: If ``dataset`` is not a pyramids ``Dataset``.
        """
        from pyramids.dataset import Dataset

        if cell_size is None and dataset is None:
            raise ValueError(
                "You have to enter either cell size or Dataset object."
            )

        ds_epsg = self.epsg
        if dataset is not None:
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    "The dataset parameter must be a pyramids Dataset "
                    "(see pyramids.dataset.Dataset.read_file)."
                )
            if dataset.epsg != ds_epsg:
                raise ValueError(
                    f"Dataset and vector are not the same EPSG. "
                    f"{dataset.epsg} != {ds_epsg}"
                )
            xmin, ymax = dataset.top_left_corner
            no_data_value = (
                dataset.no_data_value[0]
                if dataset.no_data_value[0] is not None
                else np.nan
            )
            rows = dataset.rows
            columns = dataset.columns
            cell_size = dataset.cell_size
        else:
            xmin, ymin, xmax, ymax = self.total_bounds
            no_data_value = Dataset.default_no_data_value
            columns = int(np.ceil((xmax - xmin) / cell_size))
            rows = int(np.ceil((ymax - ymin) / cell_size))

        burn_values = None
        if column_name is None:
            column_name = [c for c in self.columns if c != "geometry"]

        if isinstance(column_name, list):
            numpy_dtype = self.dtypes[column_name[0]]
        else:
            numpy_dtype = self.dtypes[column_name]

        dtype = str(numpy_dtype)
        attribute = column_name
        top_left_corner = (xmin, ymax)
        bands_count = 1 if not isinstance(attribute, list) else len(attribute)
        cell_size_val: int | float = float(cell_size)

        dataset_n = Dataset.create(
            cell_size_val,
            rows,
            columns,
            dtype,
            bands_count,
            top_left_corner,
            ds_epsg,
            no_data_value,
        )

        # The internal OGR bridge yields a gdal.Dataset for the duration
        # of the rasterize call; the /vsimem/ file is unlinked on exit.
        with _ogr.as_datasource(self, gdal_dataset=True) as vector_ds:
            bands = list(range(1, bands_count + 1))
            for ind, band in enumerate(bands):
                rasterize_opts = gdal.RasterizeOptions(
                    bands=[band],
                    burnValues=burn_values,
                    attribute=(
                        attribute[ind]
                        if isinstance(attribute, list)
                        else attribute
                    ),
                    allTouched=True,
                )
                gdal.Rasterize(
                    dataset_n.raster, vector_ds, options=rasterize_opts
                )

        return dataset_n

    # ── SR / EPSG helpers (kept as-is for back-compat) ──

    @staticmethod
    def _create_sr_from_proj(
        prj: str, string_type: str | None = None
    ) -> osr.SpatialReference:
        """Create a :class:`osr.SpatialReference` from a projection string.

        Args:
            prj (str):
                The projection string.
            string_type (str | None):
                One of ``"WKT"``, ``"ESRI wkt"``, ``"PROj4"``, or
                ``None`` for auto-detect (default). Auto-detect uses WKT
                import, and falls back to ESRI WKT or Proj4 based on the
                string's prefix.

        Returns:
            osr.SpatialReference: The constructed spatial reference.
        """
        srs = osr.SpatialReference()
        if string_type is None:
            srs.ImportFromWkt(prj)
        elif prj.startswith("PROJCS") or prj.startswith("GEOGCS"):
            srs.ImportFromESRI([prj])
        else:
            srs.ImportFromProj4(prj)
        return srs

    @staticmethod
    def get_epsg_from_prj(prj: str) -> int:
        """Return the EPSG code identified by a projection string.

        Auto-identifies the EPSG from a WKT / ESRI WKT / Proj4 string.
        For the legacy behavior, an empty input string returns 4326;
        callers that want strict validation should check for empty input
        themselves.

        Args:
            prj (str): Projection string (WKT / ESRI WKT / Proj4).

        Returns:
            int: The resolved EPSG code.
        """
        if prj != "":
            srs = FeatureCollection._create_sr_from_proj(prj)
            try:
                response = srs.AutoIdentifyEPSG()
            except RuntimeError:
                response = 6

            if response == 0:
                epsg = int(srs.GetAuthorityCode(None))
            else:
                epsg = int(srs.GetAttrValue("AUTHORITY", 1))
        else:
            epsg = 4326
        return epsg

    # ── coordinate extraction (used by xy / center_point) ──

    @staticmethod
    def _get_xy_coords(geometry: Any, coord_type: str) -> list:
        """Return x or y coords from a LineString / Polygon boundary.

        Args:
            geometry:
                A geometry exposing ``.coords.xy`` (e.g. LineString,
                LinearRing).
            coord_type (str):
                Either ``"x"`` or ``"y"``.

        Returns:
            list: Coordinate values.

        Raises:
            ValueError: If ``coord_type`` is not ``"x"`` or ``"y"``.
        """
        if coord_type == "x":
            return list(geometry.coords.xy[0].tolist())
        if coord_type == "y":
            return list(geometry.coords.xy[1].tolist())
        raise ValueError("coord_type can only have a value of 'x' or 'y'")

    @staticmethod
    def _get_point_coords(geometry: Point, coord_type: str) -> float | int:
        """Return the x or y coordinate of a shapely :class:`Point`.

        Args:
            geometry (Point):
                A shapely Point.
            coord_type (str):
                ``"x"`` or ``"y"``.

        Returns:
            float | int: The requested coordinate.

        Raises:
            ValueError: If ``coord_type`` is not ``"x"`` or ``"y"``.
        """
        if coord_type == "x":
            return float(geometry.x)
        if coord_type == "y":
            return float(geometry.y)
        raise ValueError("coord_type can only have a value of 'x' or 'y'")

    @staticmethod
    def _get_line_coords(geometry: LineString, coord_type: str) -> list:
        """Return x or y coordinates of a :class:`LineString`."""
        return FeatureCollection._get_xy_coords(geometry, coord_type)

    @staticmethod
    def _get_poly_coords(geometry: Polygon, coord_type: str) -> list:
        """Return x or y coordinates of a :class:`Polygon` exterior."""
        return FeatureCollection._get_xy_coords(geometry.exterior, coord_type)

    @staticmethod
    def _explode_gdf(
        gdf: GeoDataFrame, geometry: str = "multipolygon"
    ) -> GeoDataFrame:
        """Explode multi-geometries into per-row single geometries.

        Rows whose geometry type matches ``geometry`` are expanded so
        that each child geometry becomes its own row.

        Args:
            gdf (GeoDataFrame):
                The GeoDataFrame to process.
            geometry (str):
                The geometry type to explode (``"multipolygon"`` or
                ``"geometrycollection"``). Default ``"multipolygon"``.

        Returns:
            GeoDataFrame: A new GeoDataFrame with exploded rows first
            and preserved (non-matching) rows after.
        """
        new_gdf = gpd.GeoDataFrame()
        to_drop: list[int] = []
        for idx, row in gdf.iterrows():
            geom_type = row.geometry.geom_type.lower()
            if geom_type == geometry:
                n_rows = len(row.geometry.geoms)
                new_gdf = gpd.GeoDataFrame(pd.concat([new_gdf] + [row] * n_rows))
                new_gdf.reset_index(drop=True, inplace=True)
                new_gdf.columns = row.index.values
                for geom in range(n_rows):
                    new_gdf.loc[geom, "geometry"] = row.geometry.geoms[geom]
                to_drop.append(idx)

        gdf.drop(labels=to_drop, axis=0, inplace=True)
        new_gdf = gpd.GeoDataFrame(pd.concat([gdf] + [new_gdf]))
        new_gdf.reset_index(drop=True, inplace=True)
        new_gdf.columns = gdf.columns
        return new_gdf

    @staticmethod
    def _multi_geom_handler(
        multi_geometry: MultiPolygon | MultiPoint | MultiLineString,
        coord_type: str,
        geom_type: str,
    ) -> list:
        """Extract per-part coordinates from a multi-geometry.

        Args:
            multi_geometry:
                A shapely ``MultiPoint`` / ``MultiLineString`` /
                ``MultiPolygon`` instance.
            coord_type (str):
                ``"x"`` or ``"y"``.
            geom_type (str):
                One of ``"multipoint"``, ``"multilinestring"``,
                ``"multipolygon"`` (case-insensitive).

        Returns:
            list: A list of per-part coordinate sequences.
        """
        coord_arrays: list[Any] = []
        geom_type = geom_type.lower()
        if geom_type in ("multipoint", "multilinestring"):
            for part in multi_geometry.geoms:
                if geom_type == "multipoint":
                    coord_arrays.append(
                        FeatureCollection._get_point_coords(part, coord_type)
                    )
                else:
                    coord_arrays.append(
                        FeatureCollection._get_line_coords(part, coord_type)
                    )
        elif geom_type == "multipolygon":
            for part in multi_geometry.geoms:
                coord_arrays.append(
                    FeatureCollection._get_poly_coords(part, coord_type)
                )

        return coord_arrays

    @staticmethod
    def _geometry_collection(geom: Any, coord_type: str) -> list[Any]:
        """Extract coords from every sub-geometry of a GeometryCollection.

        Args:
            geom:
                A shapely ``GeometryCollection``.
            coord_type (str):
                ``"x"`` or ``"y"``.

        Returns:
            list: Merged list of coordinates from Point, LineString, and
            Polygon sub-geometries (in iteration order).
        """
        coords: list[Any] = []
        for sub_geom in geom.geoms:
            gtype = sub_geom.geom_type.lower()
            if gtype == "point":
                coords.append(
                    FeatureCollection._get_point_coords(sub_geom, coord_type)
                )
            elif gtype == "linestring":
                coords.extend(
                    FeatureCollection._get_line_coords(sub_geom, coord_type)
                )
            elif gtype == "polygon":
                coords.extend(
                    FeatureCollection._get_poly_coords(sub_geom, coord_type)
                )
        return coords

    @staticmethod
    def _get_coords(row: Any, geom_col: str, coord_type: str) -> Any:
        """Return coordinates for a row, dispatching by geometry type.

        Returns ``-9999`` for MultiPolygon rows as a sentinel so the
        caller (:meth:`xy`) can drop them after the per-row `.apply()`.

        Args:
            row (pandas.Series):
                A row of the GeoDataFrame.
            geom_col (str):
                Name of the geometry column.
            coord_type (str):
                ``"x"`` or ``"y"``.

        Returns:
            Any: Coordinates as ``list`` / ``float`` / ``int``, or the
            sentinel ``-9999`` for MultiPolygon rows.
        """
        geom = row[geom_col]
        gtype = geom.geom_type.lower()
        if gtype == "point":
            return FeatureCollection._get_point_coords(geom, coord_type)
        if gtype == "linestring":
            return list(FeatureCollection._get_line_coords(geom, coord_type))
        if gtype == "polygon":
            return list(FeatureCollection._get_poly_coords(geom, coord_type))
        if gtype == "multipolygon":
            return -9999
        if gtype == "geometrycollection":
            return FeatureCollection._geometry_collection(geom, coord_type)
        return FeatureCollection._multi_geom_handler(geom, coord_type, gtype)

    def xy(self) -> None:
        """Add per-vertex ``x`` and ``y`` columns to this FeatureCollection.

        Explodes MultiPolygon and GeometryCollection geometries into
        their parts, then assigns ``x`` and ``y`` columns containing the
        coordinate sequences of each row. Mutates ``self`` in place.

        Re-initializing a pandas subclass on itself is hostile to pandas'
        ``_constructor`` plumbing (it causes infinite recursion through
        the subclass contract). Instead this method builds the exploded
        frame separately, computes the coordinate columns on it, drops
        sentinel rows, and swaps the result into ``self`` via
        ``_update_inplace``, the pandas-internal primitive that
        ``DataFrame.sort_values(inplace=True)`` and friends also use.

        Returns:
            None: Mutation only.
        """
        gdf = self._explode_gdf(
            gpd.GeoDataFrame(self, copy=True), geometry="multipolygon"
        )
        gdf = self._explode_gdf(gdf, geometry="geometrycollection")

        # Coerce to a FeatureCollection so ``apply`` and friends see the
        # subclass methods without needing an extra isinstance branch.
        fc = FeatureCollection(gdf)

        fc["x"] = fc.apply(
            fc._get_coords, geom_col="geometry", coord_type="x", axis=1
        )
        fc["y"] = fc.apply(
            fc._get_coords, geom_col="geometry", coord_type="y", axis=1
        )

        to_delete = np.where(fc["x"] == -9999)[0]
        fc.drop(fc.index[to_delete], inplace=True)
        fc.reset_index(drop=True, inplace=True)

        # Replace self's data with fc's — pandas-internal primitive used
        # by DataFrame's own inplace methods.
        self._update_inplace(fc)

    # ── geometry factories ──

    @staticmethod
    def create_polygon(
        coords: list[tuple[float, float]], wkt: bool = False
    ) -> str | Polygon:
        """Build a :class:`shapely.Polygon` (or its WKT) from coordinates.

        Args:
            coords (list[tuple[float, float]]):
                Sequence of ``(x, y)`` tuples forming the ring.
            wkt (bool):
                Return the WKT string form instead of a shapely
                ``Polygon``. Default ``False``.

        Returns:
            str | Polygon: The WKT string if ``wkt=True``, else the
            ``Polygon``.

        Examples:
            - Create a WKT polygon from coordinates:
                ```python
                >>> coords = [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]
                >>> FeatureCollection.create_polygon(coords, wkt=True)
                'POLYGON ((-106.64 24, -106.49 24.05, -106.49 24.01, -106.49 23.98, -106.64 24))'

                ```
        """
        poly = Polygon(coords)
        return poly.wkt if wkt else poly

    @staticmethod
    def create_point(
        coords: Iterable[tuple[float, ...]], epsg: int | None = None
    ) -> list[Point] | FeatureCollection:
        """Build shapely ``Point`` objects (or a FeatureCollection of them).

        Args:
            coords:
                Iterable of ``(x, y)`` tuples.
            epsg (int | None):
                When given, return a FeatureCollection with the supplied
                CRS; otherwise return a plain ``list[Point]``.

        Returns:
            list[Point] | FeatureCollection: A list of shapely Points,
            or a FeatureCollection if ``epsg`` is provided.
        """
        points = list(map(Point, coords))
        if epsg is not None:
            gdf = gpd.GeoDataFrame(
                columns=["geometry"], data=points, crs=epsg
            )
            return FeatureCollection(gdf)
        return points

    # ── plot (adds optional basemap; chart itself is inherited) ──

    def plot(
        self,
        column: str | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot features, optionally on a web-tile basemap.

        Delegates to :meth:`geopandas.GeoDataFrame.plot` and, when
        ``basemap`` is truthy, adds an OpenStreetMap (or named provider)
        tile layer underneath the vector data.

        Args:
            column (str | None):
                Attribute column used to color features. Passed to
                ``GeoDataFrame.plot(column=...)``.
            basemap (bool | str | None):
                ``True`` for OpenStreetMap; a string for a named tile
                provider (e.g. ``"CartoDB.Positron"``); ``None`` or
                ``False`` for no basemap. Requires the ``[viz]`` extra.
            **kwargs:
                Passed through to ``GeoDataFrame.plot()``.

        Returns:
            matplotlib.axes.Axes: The plot axes.

        Raises:
            ValueError: If ``basemap`` is requested but the
                FeatureCollection has no CRS.
        """
        ax = super().plot(column=column, **kwargs)

        if basemap:
            if self.epsg is None:
                raise ValueError(
                    "FeatureCollection must have a CRS (epsg) to use basemap."
                )
            from pyramids.basemap.basemap import add_basemap

            source = basemap if isinstance(basemap, str) else None
            add_basemap(ax, crs=self.epsg, source=source)

        return ax

    # ── concatenation helper (kept for back-compat) ──

    def concate(
        self, gdf: GeoDataFrame, inplace: bool = False
    ) -> GeoDataFrame | None:
        """Concatenate another GeoDataFrame onto this FeatureCollection.

        Under the GeoDataFrame-subclass design you can also use the
        standard idiom ``pd.concat([fc, other])`` which returns a
        ``FeatureCollection`` because of the ``_constructor`` hook.

        Args:
            gdf (GeoDataFrame):
                The rows to append.
            inplace (bool):
                When ``True``, replace the rows of ``self`` with the
                concatenation and return ``None``. When ``False``
                (default), return a new ``GeoDataFrame``.

        Returns:
            GeoDataFrame | None: The concatenation, or ``None`` when
            ``inplace=True``.
        """
        new_gdf = gpd.GeoDataFrame(pd.concat([self, gdf]))
        new_gdf.index = list(range(len(new_gdf)))
        new_gdf.crs = self.crs
        if inplace:
            self._update_inplace(FeatureCollection(new_gdf))
            return None
        return new_gdf

    # ── point reprojection helpers (standalone — no collection needed) ──

    @staticmethod
    def reproject_points(
        lat: list,
        lon: list,
        from_epsg: int = 4326,
        to_epsg: int = 3857,
        precision: int = 6,
    ) -> tuple[list[float], list[float]]:
        """Reproject point coordinates (legacy wrapper).

        Historical API kept for back-compatibility; see
        :meth:`reproject_points2` for the ``osr``-based form. Internally
        suppresses the :class:`pyproj` ``FutureWarning`` emitted by the
        legacy ``Proj(init=...)`` construction.

        Args:
            lat (list): Y-coordinates in the source CRS.
            lon (list): X-coordinates in the source CRS.
            from_epsg (int): Source EPSG code. Default 4326.
            to_epsg (int): Target EPSG code. Default 3857.
            precision (int): Decimal places to round to.

        Returns:
            tuple[list[float], list[float]]: ``(y, x)`` lists in the
            target CRS. Note the ``(y, x)`` ordering (legacy).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            from_epsg_str = "epsg:" + str(from_epsg)
            in_proj = Proj(init=from_epsg_str)
            to_epsg_str = "epsg:" + str(to_epsg)
            out_proj = Proj(init=to_epsg_str)

        x = np.ones(len(lat)) * np.nan
        y = np.ones(len(lat)) * np.nan

        for i in range(len(lat)):
            x[i], y[i] = np.round(
                _pyproj_transform(
                    in_proj, out_proj, lon[i], lat[i], always_xy=True
                ),
                precision,
            )

        return y.tolist(), x.tolist()

    @staticmethod
    def reproject_points2(
        lat: list, lng: list, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> tuple[list[float], list[float]]:
        """Reproject point coordinates using :class:`osr.CoordinateTransformation`.

        Args:
            lat (list): Y-coordinates in the source CRS.
            lng (list): X-coordinates in the source CRS.
            from_epsg (int): Source EPSG code. Default 4326.
            to_epsg (int): Target EPSG code. Default 3857.

        Returns:
            tuple[list[float], list[float]]: ``(x, y)`` lists in the
            target CRS.
        """
        source = osr.SpatialReference()
        source.ImportFromEPSG(from_epsg)

        target = osr.SpatialReference()
        target.ImportFromEPSG(to_epsg)

        coord_transform = osr.CoordinateTransformation(source, target)
        x: list[float] = []
        y: list[float] = []
        for i in range(len(lat)):
            point = ogr.CreateGeometryFromWkt(
                "POINT (" + str(lng[i]) + " " + str(lat[i]) + ")"
            )
            point.Transform(coord_transform)
            x.append(point.GetPoints()[0][0])
            y.append(point.GetPoints()[0][1])
        return x, y

    # ── centroid helper (keeps legacy column-adding behavior) ──

    def center_point(self) -> GeoDataFrame:
        """Compute per-feature centers as extra columns on ``self``.

        Calls :meth:`xy` first, then adds ``avg_x``, ``avg_y`` and
        ``center_point`` columns. Mutates and returns ``self``.

        Returns:
            GeoDataFrame: ``self`` with the added columns.
        """
        self.xy()
        for i, row_i in self.iterrows():
            self.loc[i, "avg_x"] = np.mean(row_i["x"])
            self.loc[i, "avg_y"] = np.mean(row_i["y"])

        coords_list = zip(self["avg_x"].tolist(), self["avg_y"].tolist())
        self["center_point"] = FeatureCollection.create_point(coords_list)
        return self
