"""
Dataset module.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values.
"""

from __future__ import annotations

import logging
from numbers import Number
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from osgeo import gdal, osr
from osgeo.osr import SpatialReference

from pyramids import _io
from pyramids.dataset.abstract_dataset import (
    DEFAULT_NO_DATA_VALUE,
    AbstractDataset,
)
from pyramids.base._errors import CRSError
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    numpy_to_gdal_dtype,
)
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr

from pyramids.dataset.ops import (
    Analysis,
    BandMetadata,
    COGMixin,
    Cell,
    IO,
    Spatial,
    Vectorize,
)
from pyramids.dataset.ops._focal import (
    aspect,
    focal_apply,
    focal_mean,
    focal_std,
    hillshade,
    slope,
)
from pyramids.dataset.ops._zarr import (
    read_dataset_from_zarr,
    write_dataset_to_zarr,
)
from pyramids.dataset.ops._zonal import zonal_stats as _zonal_stats

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


class Dataset(  # type: ignore[misc]
    BandMetadata,
    IO,
    COGMixin,
    Spatial,
    Analysis,
    Vectorize,
    Cell,
    AbstractDataset,
):
    """Single-band or multi-band raster dataset (GeoTIFF, etc.).

    Wraps a GDAL dataset with spatial operations (crop, reproject, align,
    mosaic), band-level I/O, and no-data handling.  For NetCDF files use
    the :class:`~pyramids.netcdf.NetCDF` subclass; for temporal stacks of
    rasters use :class:`~pyramids.dataset.DatasetCollection`.
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__."""
        self.logger = logging.getLogger(__name__)
        super().__init__(src, access=access)

        self._no_data_value = [
            src.GetRasterBand(i).GetNoDataValue() for i in range(1, self.band_count + 1)
        ]
        self._band_names = self._get_band_names()
        self._band_units = [
            src.GetRasterBand(i).GetUnitType() for i in range(1, self.band_count + 1)
        ]

    def _update_inplace(self, src: gdal.Dataset, access: str | None = None) -> None:
        """Swap internal state from a new GDAL dataset.

        Creates a fresh Dataset and copies its internal data
        into this instance, similar to pandas' _update_inplace.
        """
        new = Dataset(src, access=access or self._access)
        self.__dict__.update(new.__dict__)

    def focal_mean(self, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_mean`."""
        return focal_mean(self, radius=radius, chunks=chunks, band=band)

    def focal_std(self, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_std`."""
        return focal_std(self, radius=radius, chunks=chunks, band=band)

    def focal_apply(self, func, radius: int = 1, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.focal_apply`."""
        return focal_apply(self, func, radius=radius, chunks=chunks, band=band)

    def slope(self, *, chunks=None, band: int = 0, units: str = "degrees"):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.slope`."""
        return slope(self, chunks=chunks, band=band, units=units)

    def aspect(self, *, chunks=None, band: int = 0):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.aspect`."""
        return aspect(self, chunks=chunks, band=band)

    def hillshade(
        self,
        *,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        chunks=None,
        band: int = 0,
    ):
        """Thin forwarder to :func:`pyramids.dataset.ops._focal.hillshade`."""
        return hillshade(
            self, azimuth=azimuth, altitude=altitude, chunks=chunks, band=band,
        )

    def zonal_stats(
        self,
        fc,
        *,
        stats=("mean",),
        method: str = "rasterize",
        band: int = 0,
    ):
        """Compute zonal statistics of this dataset over a polygon FeatureCollection.

        Thin forwarder to
        :func:`pyramids.dataset.ops._zonal.zonal_stats`; see that
        function for the full argument contract.

        Args:
            fc: A :class:`pyramids.feature.FeatureCollection` of
                polygons sharing this dataset's CRS.
            stats: Sequence of stat names (``"mean"``, ``"sum"``,
                ``"min"``, ``"max"``, ``"std"``, ``"var"``,
                ``"count"``).
            method: ``"rasterize"`` (default) or ``"exactextract"``.
            band: Zero-based band index.

        Returns:
            pandas.DataFrame: Indexed by ``fc.index``; one column per stat.
        """
        return _zonal_stats(self, fc, stats=stats, method=method, band=band)

    def to_zarr(
        self,
        store,
        *,
        compute: bool = True,
        mode: str = "w",
        chunks=None,
        storage_options: dict | None = None,
    ):
        """Serialise this Dataset to a Zarr store (parallel writes per chunk).

        Thin forwarder to
        :func:`pyramids.dataset.ops._zarr.write_dataset_to_zarr`; see
        that function for the full argument contract. Zarr is the
        only raster output format where pyramids can write in true
        parallel — each dask chunk becomes an independent Zarr chunk
        file. Requires the ``[lazy]`` optional extra.

        Args:
            store: Target store (path / fsspec URL / zarr.Store).
            compute: ``True`` writes immediately; ``False`` returns a
                :class:`dask.delayed.Delayed`.
            mode: Zarr open mode, usually ``"w"`` or ``"a"``.
            chunks: Chunk spec forwarded to :meth:`read_array`.
                ``None`` defaults to ``"auto"`` via the zarr helper.
            storage_options: fsspec options for cloud stores.
        """
        resolved_chunks = chunks if chunks is not None else "auto"
        return write_dataset_to_zarr(
            self, store,
            compute=compute, mode=mode, chunks=resolved_chunks,
            storage_options=storage_options,
        )

    @classmethod
    def from_zarr(
        cls,
        store,
        *,
        chunks=None,
        storage_options: dict | None = None,
    ) -> "Dataset":
        """Load a pyramids-written Zarr store into a new :class:`Dataset`.

        Thin forwarder to
        :func:`pyramids.dataset.ops._zarr.read_dataset_from_zarr`.

        Args:
            store: Input store (path / fsspec URL / zarr.Store).
            chunks: If non-None, the loaded Dataset is flagged as
                dask-backed so downstream ``read_array`` calls return
                lazy arrays.
            storage_options: fsspec options for cloud stores.
        """
        return read_dataset_from_zarr(
            store, chunks=chunks, storage_options=storage_options,
        )

    def __str__(self) -> str:
        """__str__."""
        message = f"""
            Top Left Corner: {self.top_left_corner}
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            Number of Bands: {self.band_count}
            Band names: {self.band_names}
            Band colors: {self.band_color}
            Band units: {self.band_units}
            Scale: {self.scale}
            Offset: {self.offset}
            Mask: {self._no_data_value[0]}
            Data type: {self.dtype[0]}
            File: {self.file_name}
        """
        return message

    def __repr__(self) -> str:
        """__repr__."""
        return str(gdal.Info(self.raster))

    @property
    def access(self) -> str:
        """
        Access mode.

        Returns:
            str:
                The access mode of the dataset (read_only/write).
        """
        return str(super().access)

    @property
    def raster(self) -> gdal.Dataset:
        """Base GDAL Dataset (read-only)."""
        return super().raster

    @property
    def rows(self) -> int:
        """Number of rows in the raster array."""
        return int(self._rows)

    @property
    def columns(self) -> int:
        """Number of columns in the raster array."""
        return int(self._columns)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape (bands, rows, columns)."""
        return self.band_count, self.rows, self.columns

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        """WKT projection.

        (top left corner X/lon coordinate, cell_size, 0, top left corner y/lat coordinate, 0, -cell_size).

        See Also:
            - Dataset.top_left_corner: Coordinate of the top left corner of the dataset.
            - Dataset.epsg: EPSG number of the dataset coordinate reference system.
        """
        gt: tuple[float, float, float, float, float, float] = self._geotransform
        return gt

    @property
    def epsg(self) -> int:
        """EPSG number."""
        crs = self.raster.GetProjection()
        # ARC-7: get_epsg_from_prj now raises on empty input; preserve
        # the historical 4326 fallback here so rasters with an empty
        # projection (common for in-memory NetCDF slices) still report
        # a stable EPSG.
        return FeatureCollection.get_epsg_from_prj(crs) if crs else 4326

    @epsg.setter
    def epsg(self, value: int):
        """EPSG number."""
        sr = Dataset._create_sr_from_epsg(value)
        self.raster.SetProjection(sr.ExportToWkt())
        self._update_inplace(self._raster)

    @property
    def crs(self) -> str:
        """Coordinate reference system.

        Returns:
            str:
                the coordinate reference system of the dataset.

        See Also:
            Dataset.set_crs : Set the Coordinate Reference System (CRS).
            Dataset.to_crs : Reproject the dataset to any projection.
            Dataset.epsg : epsg number of the dataset coordinate reference system.
        """
        return self._get_crs()

    @crs.setter
    def crs(self, value: str):
        """Coordinate reference system.

        Args:
            value (str):
                WellKnownText (WKT) string.

        See Also:
            - Dataset.set_crs: Set the Coordinate Reference System (CRS).
            - Dataset.to_crs: Reproject the dataset to any projection.
            - Dataset.epsg: EPSG number of the dataset coordinate reference system.
        """
        self.set_crs(value)

    @property
    def cell_size(self) -> float:
        """Cell size."""
        return float(self._cell_size)

    @property
    def band_count(self) -> int:
        """Number of bands in the raster."""
        return int(self._band_count)

    @property
    def band_names(self) -> list[str]:
        """Band names."""
        return self._get_band_names()

    @band_names.setter
    def band_names(self, name_list: list):
        """Band names."""
        self._set_band_names(name_list)

    @property
    def band_units(self) -> list[str]:
        """Band units."""
        return self._band_units

    @band_units.setter
    def band_units(self, value: list[str]):
        """Band units setter."""
        self._band_units = value
        for i, val in enumerate(value):
            self._iloc(i).SetUnitType(val)

    @property
    def no_data_value(self) -> list:
        """No data value that marks the cells out of the domain."""
        return list(self._no_data_value)

    @no_data_value.setter
    def no_data_value(self, value: list | Number):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes:
            - The setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - Use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - To change the values of the cells, to the new no_data_value, use the `change_no_data_value` method.

        See Also:
            - Dataset.change_no_data_value: Change the No Data Value.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self._change_no_data_value_attr(i, val)
        else:
            self._change_no_data_value_attr(0, value)

    @property
    def meta_data(self):
        """Meta-data."""
        return super().meta_data

    @meta_data.setter
    def meta_data(self, value: dict[str, str]):
        """Meta-data."""
        for key, val in value.items():
            self._raster.SetMetadataItem(key, val)

    @property
    def block_size(self) -> list[tuple[int, int]]:
        """Block Size.

        The block size is the size of the block that the raster is divided into, the block size is used to
        read and write the raster data in blocks.

        See Also:
            - Dataset.get_block_arrangement: Get block arrangement to read the dataset in chunks.
            - Dataset.get_tile: Get tiles.
            - Dataset.read_array: Read the data stored in the dataset bands.
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: list[tuple[int, int]]):
        """Block Size.

        Args:
            value (List[Tuple[int, int]]):
                block size for each band in the raster(512, 512).
        """
        if len(value[0]) != 2:
            raise ValueError("block size should be a tuple of 2 integers")

        self._block_size = value

    @property
    def file_name(self):
        """File name."""
        return super().file_name

    @property
    def driver_type(self):
        """Driver Type."""
        return super().driver_type

    @property
    def scale(self) -> list[float]:
        """Scale.

        The value of the scale is used to convert the pixel values to the real-world values.
        """
        scale_list = []
        for i in range(self.band_count):
            band_scale = self._iloc(i).GetScale()
            scale_list.append(band_scale if band_scale is not None else 1.0)
        return scale_list

    @scale.setter
    def scale(self, value: list[float]):
        """Scale."""
        for i, val in enumerate(value):
            self._iloc(i).SetScale(val)

    @property
    def offset(self):
        """Offset.

        The value of the offset is used to convert the pixel values to the real-world values.
        """
        offset_list = []
        for i in range(self.band_count):
            band_offset = self._iloc(i).GetOffset()
            offset_list.append(band_offset if band_offset is not None else 0)
        return offset_list

    @offset.setter
    def offset(self, value: list[float]):
        """Offset."""
        for i, val in enumerate(value):
            self._iloc(i).SetOffset(val)

    @property
    def top_left_corner(self):
        """Top left corner coordinates.

        See Also:
            - Dataset.geotransform: Dataset geotransform.
        """
        return super().top_left_corner

    @property
    def bounds(self) -> GeoDataFrame:
        """Bounds - the bbox as a geodataframe with a polygon geometry.

        See Also:
            - Dataset.bbox: Dataset bounding box.
        """
        return self._calculate_bounds()

    @property
    def bbox(self) -> list:
        """Bound box [xmin, ymin, xmax, ymax].

        See Also:
            - Dataset.bounds: Dataset bounding polygon.
        """
        return self._calculate_bbox()

    @property
    def total_bounds(self) -> np.ndarray:
        """Bounding box ``[minx, miny, maxx, maxy]`` as a NumPy array.

        ARC-17 introduced this property so that ``Dataset`` and
        :class:`pyramids.feature.FeatureCollection` expose the same
        shape (``GeoDataFrame.total_bounds`` is the geopandas name
        for exactly this array), letting both classes satisfy the
        :class:`pyramids.base.protocols.SpatialObject` protocol.
        """
        return np.asarray(self._calculate_bbox())

    @property
    def lon(self) -> np.ndarray:
        """Longitude coordinates.

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.lat: Dataset latitude.
        """
        x_coords = self.get_x_lon_dimension_array(
            self.top_left_corner[0], self.cell_size, self.columns
        )
        return x_coords

    @property
    def lat(self) -> np.ndarray:
        """Latitude-coordinate.

        See Also:
            - Dataset.x: Dataset x coordinates.
            - Dataset.y: Dataset y coordinates.
            - Dataset.lon: Dataset longitude.
        """
        y_coords = self.get_y_lat_dimension_array(
            self.top_left_corner[1], self.cell_size, self.rows
        )
        return y_coords

    @property
    def x(self) -> np.ndarray:
        """X-coordinate/Longitude.

        See Also:
            - Dataset.lat: Dataset latitude.
            - Dataset.y: Dataset y coordinates.
            - Dataset.lon: Dataset longitude.
        """
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """Y-coordinate/Latitude.

        See Also:
            - Dataset.x: Dataset y coordinates.
            - Dataset.lat: Dataset latitude.
            - Dataset.lon: Dataset longitude.
        """
        # Y_coordinate = upper-left corner y - index * cell size - cell-size/2
        return self.lat

    @property
    def gdal_dtype(self):
        """Data Type."""
        return [
            self.raster.GetRasterBand(i).DataType for i in range(1, self.band_count + 1)
        ]

    @property
    def numpy_dtype(self) -> list[type]:
        """List of the numpy data Type of each band, the data type is a numpy function."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "numpy"].values[0]
            for i in self.gdal_dtype
        ]

    @property
    def dtype(self) -> list[str]:
        """List of the data Type of each band as strings."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "name"].values[0]
            for i in self.gdal_dtype
        ]

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        read_only=True,
        file_i: int = 0,
    ) -> Dataset:
        """read_file.

        Args:
            path (str):
                Path of file to open.
            read_only (bool):
                File mode, set to False, to open in "update" mode.
            file_i (int):
                Index to the file inside the compressed file you want to read, if the compressed file
                has only one file. Default is 0.

        Returns:
            Dataset:
                Opened dataset instance.

        See Also:
            - Dataset.read_array: Read the values stored in a dataset band.
        """
        src = _io.read_file(path, read_only=read_only, file_i=file_i)
        return cls(src, access="read_only" if read_only else "write")

    def copy(self, path: str | Path | None = None) -> Dataset:
        """Deep copy.

        Args:
            path (str, optional):
                Destination path to save the copied dataset. If None is passed, the copied dataset
                will be created in memory.
        """
        if path is None:
            path = ""
            driver = "MEM"
        else:
            driver = "GTiff"

        src = gdal.GetDriverByName(driver).CreateCopy(str(path), self._raster)

        return Dataset(src, access="write")

    def close(self) -> None:
        """Close the dataset.

        Safe to call multiple times — subsequent calls after the first are no-ops.
        """
        if self._raster is not None:
            self._raster.FlushCache()
            self._raster = None

    @staticmethod
    def _create_dataset(
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        driver: str = "MEM",
        path: str | Path | None = None,
    ) -> gdal.Dataset:
        """Create a GDAL driver.

            creates a driver and save it to disk and in memory if the path is not given.

        Args:
            cols (int):
                Number of columns.
            rows (int):
                Number of rows.
            bands (int):
                Number of bands.
            dtype:
                GDAL data type.
            driver (str):
                Driver type ["GTiff", "MEM"].
            path (str):
                Path to save the GTiff driver.

        Returns:
            gdal driver
        """
        if path:
            driver = "GTiff" if driver == "MEM" else driver
            if not isinstance(path, (str, Path)):
                raise TypeError(
                    f"The path input should be string or Path type, given: {type(path)}"
                )
            path = Path(path)
            if driver == "GTiff" and path.suffix != ".tif":
                raise TypeError(
                    "The path to save the created raster should end with .tif"
                )
            # LZW is a lossless compression method achieve the highest compression but with a lot
            # of computations.
            src = gdal.GetDriverByName(driver).Create(
                str(path), cols, rows, bands, dtype, ["COMPRESS=LZW"]
            )
        else:
            # for memory drivers
            driver = "MEM"
            src = gdal.GetDriverByName(driver).Create("", cols, rows, bands, dtype)
        return src

    @classmethod
    def _build_dataset(
        cls,
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        geo: tuple,
        crs: str,
        no_data_value,
        driver: str = "MEM",
        path: str | Path | None = None,
        access: str = "write",
    ) -> Dataset:
        """Create a GDAL dataset, set its spatial metadata, and wrap it as a Dataset.

        Consolidates the repeated pattern of _create_dataset + SetGeoTransform +
        SetProjection + wrap + _set_no_data_value into a single helper.

        Args:
            cols (int): Number of columns.
            rows (int): Number of rows.
            bands (int): Number of bands.
            dtype (int): GDAL data type.
            geo (tuple): Geotransform tuple.
            crs (str): Projection as WKT string.
            no_data_value: No-data value. Scalar (broadcast to all bands) or list (one per band).
            driver (str): Driver type. Default is "MEM".
            path (str | Path | None): Path for disk-based drivers.
            access (str): Access mode for the Dataset wrapper. Default is "write".
                Note: MEM driver datasets can be written to regardless of access mode since
                the access flag is enforced at the pyramids level, not by GDAL.

        Returns:
            Dataset: A fully configured Dataset object.
        """
        dst = cls._create_dataset(cols, rows, bands, dtype, driver=driver, path=path)
        dst.SetGeoTransform(geo)
        dst.SetProjection(crs)
        dst_obj = cls(dst, access=access)
        dst_obj._set_no_data_value(no_data_value=no_data_value)
        return dst_obj

    @classmethod
    def create(
        cls,
        cell_size: int | float,
        rows: int,
        columns: int,
        dtype: str,
        bands: int,
        top_left_corner: tuple,
        epsg: int,
        no_data_value: Any | None = None,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset and fill it with the no_data_value.

        The new dataset will have an array filled with the no_data_value.

        Args:
            cell_size (int|float):
                Cell size.
            rows (int):
                Number of rows.
            columns (int):
                Number of columns.
            dtype (str):
                Data type.
            bands (int|None):
                Number of bands to create in the output raster.
            top_left_corner (Tuple):
                Coordinates of the top left corner point.
            epsg (int):
                EPSG number to identify the projection of the coordinates in the created raster.
            no_data_value (float|None):
                No data value.
            path (str, optional):
                Path on disk; if None, the dataset is created in memory. Default is None.

        Returns:
            Dataset: A new dataset
        """
        # Create the driver.
        gdal_dtype = numpy_to_gdal_dtype(dtype)
        dst = Dataset._create_dataset(columns, rows, bands, gdal_dtype, path=path)
        sr = Dataset._create_sr_from_epsg(epsg)
        geotransform = (
            top_left_corner[0],
            cell_size,
            0,
            top_left_corner[1],
            0,
            -1 * cell_size,
        )
        dst.SetGeoTransform(geotransform)
        # Set the projection.
        dst.SetProjection(sr.ExportToWkt())

        dst = cls(dst, access="write")
        if no_data_value is not None:
            dst._set_no_data_value(no_data_value=no_data_value)

        return dst

    @classmethod
    def from_features(
        cls,
        features: FeatureCollection,
        *,
        cell_size: Any | None = None,
        template: Dataset | None = None,
        column_name: str | list[str] | None = None,
    ) -> Dataset:
        """Rasterize a :class:`FeatureCollection` into a new :class:`Dataset`.

        ARC-4: this classmethod replaces ``FeatureCollection.to_dataset``.
        Moving the method here breaks the circular import that forced
        the old code to do ``from pyramids.dataset import Dataset``
        inside the method body (a CLAUDE.md violation).
        ``pyramids.dataset`` already imports :class:`FeatureCollection`
        at module level, so this direction is cycle-free.

        Burns the values from ``column_name`` (or every attribute
        column if ``None``) into a single-band or multi-band raster.
        When a ``template`` Dataset is given, the output adopts its
        geotransform, cell size, row/column count, and no-data value.
        Otherwise ``cell_size`` controls the resolution and the extent
        is derived from :attr:`FeatureCollection.total_bounds`.

        Args:
            features (FeatureCollection):
                The vector to rasterize.
            cell_size (int | float | None):
                Cell size for the new raster. Required unless
                ``template`` is given.
            template (Dataset | None):
                Optional template raster. When supplied, the output
                inherits its geotransform and no-data value.
            column_name (str | list[str] | None):
                Attribute column(s) to burn as band values. ``None``
                burns every non-geometry column as a separate band.

        Returns:
            Dataset: The burned raster.

        Raises:
            ValueError: If neither ``cell_size`` nor ``template`` is
                given, or if the vector CRS disagrees with the raster
                CRS.
            TypeError: If ``template`` is not a pyramids ``Dataset``.
        """
        if cell_size is None and template is None:
            raise ValueError(
                "You have to enter either cell size or Dataset object."
            )

        ds_epsg = features.epsg
        if template is not None:
            if not isinstance(template, Dataset):
                raise TypeError(
                    "The template parameter must be a pyramids Dataset "
                    "(see pyramids.dataset.Dataset.read_file)."
                )
            if template.epsg != ds_epsg:
                raise CRSError(
                    f"Dataset and vector are not the same EPSG. "
                    f"{template.epsg} != {ds_epsg}"
                )
            xmin, ymax = template.top_left_corner
            no_data_value = (
                template.no_data_value[0]
                if template.no_data_value[0] is not None
                else np.nan
            )
            rows = template.rows
            columns = template.columns
            cell_size = template.cell_size
        else:
            xmin, ymin, xmax, ymax = features.total_bounds
            no_data_value = cls.default_no_data_value
            columns = int(np.ceil((xmax - xmin) / cell_size))
            rows = int(np.ceil((ymax - ymin) / cell_size))

        burn_values = None
        if column_name is None:
            column_name = [c for c in features.columns if c != "geometry"]

        if isinstance(column_name, list):
            numpy_dtype = features.dtypes[column_name[0]]
        else:
            numpy_dtype = features.dtypes[column_name]

        dtype = str(numpy_dtype)
        attribute = column_name
        top_left_corner = (xmin, ymax)
        bands_count = 1 if not isinstance(attribute, list) else len(attribute)
        cell_size_val: int | float = float(cell_size)

        dataset_n = cls.create(
            cell_size_val,
            rows,
            columns,
            dtype,
            bands_count,
            top_left_corner,
            ds_epsg,
            no_data_value,
        )

        with _feature_ogr.as_datasource(features, gdal_dataset=True) as vector_ds:
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

    @classmethod
    def create_from_array(  # type: ignore[override]
        cls,
        arr: np.ndarray,
        top_left_corner: tuple[float, float] | None = None,
        cell_size: int | float | None = None,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int = 4326,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset from an array.

        Args:
            arr (np.ndarray):
                Numpy array.
            top_left_corner (Tuple[float, float], optional):
                The coordinates of the top left corner of the dataset.
            cell_size (int|float, optional):
                Cell size in the same units of the coordinate reference system defined by the `epsg`
                parameter.
            geo (Tuple[float, float, float, float, float, float], optional):
                Geotransform tuple (minimum lon/x, pixel-size, rotation, maximum lat/y, rotation,
                pixel-size).
            epsg (int):
                Integer reference number to the projection (https://epsg.io/).
            no_data_value (Any, optional):
                No data value to mask the cells out of the domain. The default is -9999.
            driver_type (str, optional):
                Driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM".
            path (str, optional):
                Path to save the driver.

        Returns:
            Dataset:
                Dataset object will be returned.
        """
        if geo is None:
            if top_left_corner is None or cell_size is None:
                raise ValueError(
                    "Either top_left_corner and cell_size or geo should be provided."
                )
            geo = (
                top_left_corner[0],
                cell_size,
                0,
                top_left_corner[1],
                0,
                -1 * cell_size,
            )

        if arr.ndim == 2:
            bands = 1
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            bands = arr.shape[0]
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        dst_obj = cls._create_gtiff_from_array(
            arr,
            cols,
            rows,
            bands,
            geo,
            epsg,
            no_data_value,
            driver_type=driver_type,
            path=path,
        )

        return dst_obj

    @staticmethod
    def _create_gtiff_from_array(
        arr: np.ndarray,
        cols: int,
        rows: int,
        bands: int = 1,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int | None = None,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str | Path | None = None,
    ) -> Dataset:
        dtype = numpy_to_gdal_dtype(arr)
        dst_ds = Dataset._create_dataset(
            cols, rows, bands, dtype, driver=driver_type, path=path
        )

        if epsg is None:
            raise ValueError("epsg must be provided")

        srse = Dataset._create_sr_from_epsg(epsg=int(epsg))
        dst_ds.SetProjection(srse.ExportToWkt())
        dst_ds.SetGeoTransform(geo)

        dst_obj = Dataset(dst_ds, access="write")
        dst_obj._set_no_data_value(no_data_value=no_data_value)
        dst_obj._raster.FlushCache()

        if bands == 1:
            dst_obj.raster.GetRasterBand(1).WriteArray(arr)
        else:
            for i in range(bands):
                dst_obj.raster.GetRasterBand(i + 1).WriteArray(arr[i, :, :])

        dst_obj._raster.FlushCache()
        return dst_obj

    @classmethod
    def dataset_like(
        cls,
        src: Dataset,
        array: np.ndarray,
        path: str | Path | None = None,
    ) -> Dataset:
        """Create a new dataset like another dataset.

        dataset_like method creates a Dataset from an array like another source dataset. The new dataset
        will have the same `projection`, `coordinates` or the `top left corner` of the original dataset,
        `cell size`, `no_data_velue`, and number of `rows` and `columns`.
        the array and the source dataset should have the same number of columns and rows

        Args:
            src (Dataset):
                source raster to get the spatial information
            array (ndarray):
                data to store in the new dataset.
            path (str, optional):
                path to save the new dataset, if not given, the method will return in-memory dataset.

        Returns:
            Dataset:
                if the `path` is given, the method will save the new raster to the given path, else the
                method will return an in-memory dataset.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array should be of type numpy array")

        if array.ndim == 2:
            bands = 1
        else:
            bands = array.shape[0]

        dtype = numpy_to_gdal_dtype(array)

        dst_obj = cls._build_dataset(
            src.columns, src.rows, bands, dtype, src.geotransform, src.crs,
            src.no_data_value[0], path=path,
        )

        if bands == 1:
            dst_obj.raster.GetRasterBand(1).WriteArray(array)
        else:
            for band_i in range(bands):
                dst_obj.raster.GetRasterBand(band_i + 1).WriteArray(array[band_i, :, :])

        if path is not None:
            dst_obj.raster.FlushCache()

        return dst_obj

    @staticmethod
    def _create_sr_from_epsg(epsg: int) -> SpatialReference:
        """Create a spatial reference object from EPSG number.

        https://gdal.org/tutorials/osr_api_tut.html

        Args:
            epsg (int):
                EPSG number.

        Returns:
            SpatialReference:
                SpatialReference object.
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(int(epsg))
        return sr
