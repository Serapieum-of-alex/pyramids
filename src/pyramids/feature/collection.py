"""FeatureCollection — a GeoDataFrame with pyramids-specific GIS methods.

ARC-10 split the original ``feature.py`` into a subpackage:

- :mod:`pyramids.feature.collection` (this module) — the
  :class:`FeatureCollection` class.
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors (``create_polygon``, ``create_point``, ``get_coords``,
  ``explode_gdf``, ``multi_geom_handler``, …).
- :mod:`pyramids.feature.crs` — CRS/EPSG/reprojection helpers
  (``get_epsg_from_prj``, ``reproject_points``,
  ``reproject_points_osr``, ``create_sr_from_proj``).
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
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pyramids.dataset import Dataset

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from osgeo import gdal, ogr, osr
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids.base._errors import DriverNotExistError
from pyramids.base._utils import Catalog
from pyramids.feature import _ogr
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

    _metadata: list[str] = ["_epsg_cache_crs", "_epsg_cache_value"]
    """Instance attributes pandas must preserve across copy/slice.

    Currently holds the per-instance EPSG cache added by ARC-6.
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
    def read_file(cls, path: str | Path) -> FeatureCollection:
        """Read a vector file into a FeatureCollection.

        Delegates to :func:`geopandas.read_file` and wraps the result.

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

    def to_file(
        self, path: str | Path, driver: str = "geojson", **kwargs: Any
    ) -> None:
        """Write this FeatureCollection to a vector file.

        Resolves a pyramids driver alias (e.g. ``"geojson"``) to the
        GDAL driver name via :class:`Catalog`, then delegates to
        :meth:`geopandas.GeoDataFrame.to_file`.
        """
        try:
            resolved = CATALOG.get_gdal_name(driver) or driver
        except AttributeError:
            resolved = driver
        super().to_file(path, driver=resolved, **kwargs)

    def to_dataset(
        self,
        cell_size: Any | None = None,
        dataset: Dataset | None = None,
        column_name: str | list[str] | None = None,
    ) -> Dataset:
        """Rasterize this FeatureCollection to a :class:`pyramids.Dataset`.

        Burns values from ``column_name`` (or every attribute column
        when ``None``) into a raster. When a template ``dataset`` is
        given, the output adopts its geotransform, cell size, and
        no-data value; otherwise ``cell_size`` controls the resolution.

        Raises:
            ValueError: If neither ``cell_size`` nor ``dataset`` is
                given, or if CRSes disagree.
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
    def reproject_points(
        lat: list,
        lon: list,
        from_epsg: int = 4326,
        to_epsg: int = 3857,
        precision: int = 6,
    ) -> tuple[list[float], list[float]]:
        """Delegate to :func:`pyramids.feature.crs.reproject_points`.

        Examples:
            - Reproject two WGS84 points into Web Mercator and inspect
              the magnitudes of the projected coordinates:
                ```python
                >>> lat = [30.0]
                >>> lon = [31.0]
                >>> y, x = FeatureCollection.reproject_points(
                ...     lat, lon, from_epsg=4326, to_epsg=3857
                ... )
                >>> round(x[0])
                3450904
                >>> round(y[0])
                3503550

                ```
        """
        return _crs.reproject_points(lat, lon, from_epsg, to_epsg, precision)

    @staticmethod
    def reproject_points2(
        lat: list, lng: list, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> tuple[list[float], list[float]]:
        """Delegate to :func:`pyramids.feature.crs.reproject_points_osr`."""
        return _crs.reproject_points_osr(lat, lng, from_epsg, to_epsg)

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
        """Delegate to :func:`pyramids.feature.geometry.create_polygon`."""
        return _geom.create_polygon(coords, wkt)

    @staticmethod
    def create_point(
        coords: Iterable[tuple[float, ...]], epsg: int | None = None
    ) -> list[Point] | FeatureCollection:
        """Delegate to :func:`pyramids.feature.geometry.create_point`.

        Wraps the plain-GeoDataFrame result in a FeatureCollection when
        ``epsg`` is supplied (for API continuity).
        """
        result = _geom.create_point(coords, epsg)
        if epsg is not None:
            return FeatureCollection(result)
        return result

    def xy(self) -> None:
        """Add per-vertex ``x`` and ``y`` columns to this FeatureCollection.

        Explodes MultiPolygon and GeometryCollection geometries into
        their parts, then assigns ``x`` and ``y`` columns containing
        the coordinate sequences of each row. Mutates ``self`` in place
        via pandas' ``_update_inplace`` primitive (avoids the
        subclass-re-init recursion trap).

        Returns:
            None: Mutation only.
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
        to_delete = np.where(fc["x"] == -9999)[0]
        fc.drop(fc.index[to_delete], inplace=True)
        fc.reset_index(drop=True, inplace=True)

        self._update_inplace(fc)

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
                raise ValueError(
                    "FeatureCollection must have a CRS (epsg) to use basemap."
                )
            from pyramids.basemap.basemap import add_basemap

            source = basemap if isinstance(basemap, str) else None
            add_basemap(ax, crs=self.epsg, source=source)

        return ax

    def concate(
        self, gdf: GeoDataFrame, inplace: bool = False
    ) -> GeoDataFrame | None:
        """Concatenate another GeoDataFrame onto this FeatureCollection.

        Under the GeoDataFrame-subclass design you can also use the
        standard idiom ``pd.concat([fc, other])`` which returns a
        ``FeatureCollection`` because of the ``_constructor`` hook.
        """
        import pandas as pd

        new_gdf = gpd.GeoDataFrame(pd.concat([self, gdf]))
        new_gdf.index = list(range(len(new_gdf)))
        new_gdf.crs = self.crs
        if inplace:
            self._update_inplace(FeatureCollection(new_gdf))
            return None
        return new_gdf

    def center_point(self) -> GeoDataFrame:
        """Compute per-feature centers as extra columns on ``self``.

        Calls :meth:`xy` first, then adds ``avg_x``, ``avg_y`` and
        ``center_point`` columns. Mutates and returns ``self``.
        """
        self.xy()
        for i, row_i in self.iterrows():
            self.loc[i, "avg_x"] = np.mean(row_i["x"])
            self.loc[i, "avg_y"] = np.mean(row_i["y"])

        coords_list = zip(self["avg_x"].tolist(), self["avg_y"].tolist())
        self["center_point"] = _geom.create_point(coords_list)
        return self
