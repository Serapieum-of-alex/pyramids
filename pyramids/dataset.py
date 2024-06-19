"""
Dataset module.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values. gdal class: https://gdal.org/api/index.html#python-api.
"""

import os
import warnings
import logging
from numbers import Number
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import DataFrame, GeoDataFrame
from loguru import logger
from osgeo import gdal, ogr, osr
from osgeo.osr import SpatialReference

from pyramids._errors import (
    AlignmentError,
    FailedToSaveError,
    NoDataValueError,
    ReadOnlyError,
    OutOfBoundsError,
)
from pyramids._utils import (
    DTYPE_CONVERSION_DF,
    INTERPOLATION_METHODS,
    gdal_to_numpy_dtype,
    gdal_to_ogr_dtype,
    import_cleopatra,
    numpy_to_gdal_dtype,
    color_name_to_gdal_constant,
    gdal_constant_to_color_name,
)

from hpc.indexing import get_pixels, get_indices2, get_pixels2, locate_values
from pyramids.featurecollection import FeatureCollection
from pyramids import _io
from pyramids.abstract_dataset import AbstractDataset
from pyramids.abstract_dataset import (
    DEFAULT_NO_DATA_VALUE,
    CATALOG,
    OVERVIEW_LEVELS,
    RESAMPLING_METHODS,
)


class Dataset(AbstractDataset):
    """Dataset.

    The Dataset class contains methods to deal with raster and netcdf files, change projection and coordinate
    systems.
    """

    default_no_data_value = DEFAULT_NO_DATA_VALUE

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

    def __str__(self):
        """__str__."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            Number of Bands: {self.band_count}
            Band names: {self.band_names}
            Mask: {self._no_data_value[0]}
            Data type: {self.dtype[0]}
            File: {self.file_name}
        """
        return message

    def __repr__(self):
        """__repr__."""
        message = """
            Cell size: {0}
            Dimension: {1} * {2}
            EPSG: {3}
            Number of Bands: {4}
            Band names: {5}
            Mask: {6}
            Data type: {7}
            projection: {8}
            Metadata: {9}
            File: {10}
        """.format(
            self.cell_size,
            self.rows,
            self.columns,
            self.epsg,
            self.band_count,
            self.band_names,
            (
                self._no_data_value
                if self._no_data_value == []
                else self._no_data_value[0]
            ),
            self.dtype if self.dtype == [] else self.dtype[0],
            self.crs,
            self.meta_data,
            self.file_name,
        )
        return message

    @property
    def access(self):
        """open_mode."""
        return super().access

    @property
    def raster(self) -> gdal.Dataset:
        """Base GDAL Dataset."""
        return super().raster

    @raster.setter
    def raster(self, value: gdal.Dataset):
        """Contains GDAL Dataset."""
        self._raster = value

    @property
    def values(self) -> np.ndarray:
        """Values of all the bands.

        Return
        -------
        np.ndarray:
            the values of all the bands in the raster as a 3D numpy array (bands, rows, columns).
        """
        return self.read_array()

    @property
    def rows(self) -> int:
        """Number of rows in the raster array."""
        return self._rows

    @property
    def columns(self) -> int:
        """Number of columns in the raster array."""
        return self._columns

    @property
    def shape(self):
        """Shape (bands, rows, columns)."""
        return self.band_count, self.rows, self.columns

    @property
    def geotransform(self):
        """WKT projection.

        (top left corner X/lon coordinate, cell_size, 0, top left corner y/lat coordinate, 0, -cell_size).
        >>> (432968.120, 4000.0, 0.0, 520007.787, 0.0, -4000)
        """
        return super().geotransform

    @property
    def epsg(self) -> int:
        """EPSG number."""
        return self._epsg

    @property
    def crs(self) -> str:
        """Coordinate reference system.

        Returns
        -------
        str:
            the coordinate reference system of the raster.
            >>> 'PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32618"]]'
        """
        return self._get_crs()

    @crs.setter
    def crs(self, value: str):
        """Coordinate reference system."""
        self.set_crs(value)

    @property
    def cell_size(self) -> int:
        """Cell size."""
        return self._cell_size

    @property
    def band_count(self):
        """Number of bands in the raster."""
        return self._band_count

    @property
    def band_names(self):
        """Band names."""
        return self._get_band_names()

    @band_names.setter
    def band_names(self, name_list: List):
        """Band names."""
        self._set_band_names(name_list)

    @property
    def band_units(self) -> List[str]:
        """Band units."""
        return self._band_units

    @band_units.setter
    def band_units(self, value: List[str]):
        """Band units setter."""
        self._band_units = value
        for i, val in enumerate(value):
            self._iloc(i).SetUnitType(val)

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, value: Union[List, Number]):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes
        -----
            - the setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - to change the values of the cells to the new no_data_value, use the `change_no_data_value` method.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self._change_no_data_value_attr(i, val)
        else:
            self._change_no_data_value_attr(0, value)

    @property
    def meta_data(self):
        """Meta data."""
        return super().meta_data

    @meta_data.setter
    def meta_data(self, value: Dict[str, str]):
        """Meta data.

        Hint
        ----
        - This property does not need the Dataset to be opened in a write mode to be set.
        - The value of the offset will be stored in an xml file by the name of the raster file with the extension of
        .aux.xml, the content of the file will be like the following:
            <PAMDataset>
              <Metadata>
                <MDI key="key">value</MDI>
              </Metadata>
            </PAMDataset>
        """
        for key, value in value.items():
            self._raster.SetMetadataItem(key, value)

    @property
    def block_size(self) -> List[Tuple[int, int]]:
        """Block Size.

        The block size is the size of the block that the raster is divided into, the block size is used to read and
        write the raster data in blocks.

        Examples
        --------
        >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
        >>> size = dataset.block_size
        >>> print(size)
        >>> [(128, 128)]
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: List[Tuple[int, int]]):
        """Block Size.

        Parameters
        ----------
        value : List[Tuple[int, int]]
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
    def scale(self):
        """Scale.

        The value of the scale is used to convert the pixel values to the real-world values.
        """
        scale_list = []
        for i in range(self.band_count):
            band_scale = self._iloc(i).GetScale()
            scale_list.append(band_scale if band_scale is not None else 1.0)
        return scale_list

    @scale.setter
    def scale(self, value: List[float]):
        """Scale.

        The value of the scale is used to convert the pixel values to the real-world values.

        Hint
        ----
        - This property does not need the Dataset to be opened in a write mode to be set.
        - The value of the offset will be stored in an xml file by the name of the raster file with the extension of
        .aux.xml, the content of the file will be like the following:
            <PAMDataset>
              <PAMRasterBand band="1">
                <Description>Band_1</Description>
                <UnitType>m</UnitType>
                <Offset>100</Offset>
                <Scale>2</Scale>
              </PAMRasterBand>
            </PAMDataset>
        """
        for i, val in enumerate(value):
            self._iloc(i).SetScale(val)

    @property
    def offset(self):
        """Offset.

        The value of the offset is used to convert the pixel values to the real-world values.

        Hint
        ----
        - This property does not need the Dataset to be opened in a write mode to be set.
        - The value of the offset will be stored in an xml file by the name of the raster file with the extension of
        .aux.xml, the content of the file will be like the following:
            <PAMDataset>
              <PAMRasterBand band="1">
                <Description>Band_1</Description>
                <UnitType>m</UnitType>
                <Offset>100</Offset>
                <Scale>2</Scale>
              </PAMRasterBand>
            </PAMDataset>
        """
        offset_list = []
        for i in range(self.band_count):
            band_offset = self._iloc(i).GetOffset()
            offset_list.append(band_offset if band_offset is not None else 0)
        return offset_list

    @offset.setter
    def offset(self, value: List[float]):
        for i, val in enumerate(value):
            self._iloc(i).SetOffset(val)

    @classmethod
    def read_file(
        cls,
        path: str,
        read_only=True,
        file_i: int = 0,
    ) -> "Dataset":
        """read_file.

        Parameters
        ----------
        path: [str]
            Path of file to open.
        read_only: [bool]
            File mode, set to False, to open in "update" mode.
        file_i: [int] default is 0
            index to the file inside the compressed file you want to read, if the compressed file have only one file

        Returns
        -------
        Dataset

        Examples
        --------
        Zip files:
            - Internal Zip file path (one/multiple files inside the compressed file):
                if the path contains a zip but does not end with zip (compressed-file-name.zip/1.asc), so the path contains
                    the internal path inside the zip file, so just ad

                >>> from pyramids.dataset import Dataset
                >>> rdir = "tests/data/virtual-file-system"
                >>> dataset = Dataset.read_file(f"{rdir}/multiple_compressed_files.zip/1.asc")
                >>> print(dataset)
                <BLANKLINE>
                            Cell size: 4000.0
                            Dimension: 13 * 14
                            EPSG: 4326
                            Number of Bands: 1
                            Band names: ['Band_1']
                            Mask: -3.4028230607370965e+38
                            Data type: float32
                            File: /vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc
                <BLANKLINE>

            - Only the Zip file path (one/multiple files inside the compressed file):
                If you provide the name of the zip file with multiple files inside it, it will return the path to the first
                file.

                >>> dataset = Dataset.read_file(f"{rdir}/multiple_compressed_files.zip")
                >>> print(dataset)
                <BLANKLINE>
                            Cell size: 4000.0
                            Dimension: 13 * 14
                            EPSG: 4326
                            Number of Bands: 1
                            Band names: ['Band_1']
                            Mask: -3.4028230607370965e+38
                            Data type: float32
                            File: /vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc
                <BLANKLINE>

            - Zip file path and an index (one/multiple files inside the compressed file):
                if you provide the path to the zip file and an index to the file inside the compressed file you want to
                read.

                >>> dataset = Dataset.read_file(f"{rdir}/multiple_compressed_files.zip", file_i=1)
                >>> print(dataset)
                <BLANKLINE>
                            Cell size: 4000.0
                            Dimension: 13 * 14
                            EPSG: 4326
                            Number of Bands: 1
                            Band names: ['Band_1']
                            Mask: -3.4028230607370965e+38
                            Data type: float32
                            File: /vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/2.asc
                <BLANKLINE>

        """
        src = _io.read_file(path, read_only=read_only, file_i=file_i)
        return cls(src, access="read_only" if read_only else "write")

    def read_array(
        self, band: int = None, window: Union[GeoDataFrame, List[int]] = None
    ) -> np.ndarray:
        """Read Array.

            - read the values stored in a given band.

        Data Chuncks/blocks
            When a raster dataset is stored on disk, it might not be stored as one continuous chunk of data. Instead,
            it can be divided into smaller rectangular blocks or tiles. These blocks can be individually accessed,
            which is particularly useful for large datasets:
                Efficiency: Reading or writing small blocks requires less memory than dealing with the entire dataset
                    at once. This is especially beneficial when only a small portion of the data needs to be processed.
                Performance: For certain file formats and operations, working with optimal block sizes can significantly
                    improve performance. For example, if the block size matches the reading or processing window,
                        Pyramids can minimize disk access and data transfer.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data, If None the data of all bands will be read. Default is None
        window: [List/GeoDataFrame]
            List:
                window to specify a block of data to read from the dataset. the window should be a list of 4 integers
                [offset_x, offset_y, window_columns, window_rows].
                - offset_x: x offset of the block.
                - offset_y: y offset of the block.
                - window_columns: number of columns in the block.
                - window_rows: number of rows in the block.
            GeoDataFrame:
                GeoDataFrame with a geometry column, the function will get the total_bounds of the geodataframe and
                use it as a window to read the raster.

        Returns
        -------
        array : [array]
            array with all the values in the raster.

        Examples
        --------
        >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
        >>> arr = dataset.read_array(window=[0, 0, 5, 5])
        >>> print(arr.shape)
        >>> (5, 5)
        """
        if band is None and self.band_count > 1:
            rows = self.rows if window is None else window[3]
            columns = self.columns if window is None else window[2]
            arr = np.ones(
                (
                    self.band_count,
                    rows,
                    columns,
                ),
                dtype=self.numpy_dtype[0],
            )

            for i in range(self.band_count):
                if window is None:
                    # this line could be replaced with the following line
                    # arr[i, :, :] = self._iloc(i).ReadAsArray()
                    arr[i, :, :] = self._raster.GetRasterBand(i + 1).ReadAsArray()
                else:
                    arr[i, :, :] = self._read_block(i, window)
        else:
            # given band number or the raster has only one band
            if band is None:
                band = 0
            else:
                if band > self.band_count - 1:
                    raise ValueError(
                        f"band index should be between 0 and {self.band_count - 1}"
                    )
            if window is None:
                arr = self._iloc(band).ReadAsArray()
            else:
                arr = self._read_block(band, window)

        return arr

    def _read_block(
        self, band: int, window: Union[GeoDataFrame, List[int]]
    ) -> np.ndarray:
        """Read block of data from the dataset.

        Parameters
        ----------
        band : int
            Band index.
        window: [List/GeoDataFrame]
            List:
                window to specify a block of data to read from the dataset. the window should be a list of 4 integers
                [offset_x, offset_y, window_columns, window_rows].
                - offset_x: x offset of the block.
                - offset_y: y offset of the block.
                - window_columns: number of columns in the block.
                - window_rows: number of rows in the block.
            GeoDataFrame:
                GeoDataFrame with a geometry column, the function will get the total_bounds of the geodataframe and
                use it as a window to read the raster.

        Returns
        -------
        np.ndarray[window[2], window[3]]
            array with the values of the block. the shape of the array is (window[2], window[3]), and the location of
            the block in the raster is (window[0], window[1]).
        """
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
        try:
            block = self._iloc(band).ReadAsArray(
                window[0], window[1], window[2], window[3]
            )
        except Exception as e:
            if e.args[0].__contains__("Access window out of range in RasterIO()"):
                raise OutOfBoundsError(
                    f"The window you entered ({window})is out of the raster bounds: {self.rows, self.columns}"
                )
            else:
                raise e
        return block

    def _convert_polygon_to_window(
        self, poly: Union[GeoDataFrame, "FeatureCollection"]
    ) -> List[Any]:
        poly = FeatureCollection(poly)
        bounds = poly.total_bounds
        df = pd.DataFrame(columns=["id", "x", "y"])
        df.loc["top_left", ["x", "y"]] = bounds[0], bounds[3]
        df.loc["bottom_right", ["x", "y"]] = bounds[2], bounds[1]
        arr_indeces = self.map_to_array_coordinates(df)
        xoff = arr_indeces[0, 1]
        yoff = arr_indeces[0, 0]
        x_size = arr_indeces[1, 0] - arr_indeces[0, 0]
        y_size = arr_indeces[1, 1] - arr_indeces[0, 1]
        return [xoff, yoff, x_size, y_size]

    @property
    def pivot_point(self):
        """Top left corner coordinates."""
        return super().pivot_point

    @property
    def bounds(self) -> GeoDataFrame:
        """Bounds - the bbox as a geodataframe with a polygon geometry."""
        return self._calculate_bounds()

    @property
    def bbox(self) -> List:
        """Bound box [xmin, ymin, xmax, ymax]."""
        return self._calculate_bbox()

    @property
    def lon(self):
        """Longitude coordinates."""
        if not hasattr(self, "_lon"):
            pivot_x = self.pivot_point[0]
            cell_size = self.cell_size
            x_coords = [
                pivot_x + i * cell_size + cell_size / 2 for i in range(self.columns)
            ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            x_coords = self._lon
        return np.array(x_coords)

    @property
    def lat(self):
        """Latitude-coordinate."""
        if not hasattr(self, "_lat"):
            pivot_y = self.pivot_point[1]
            cell_size = self.cell_size
            y_coords = [
                pivot_y - i * cell_size - cell_size / 2 for i in range(self.rows)
            ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @property
    def x(self):
        """X-coordinate/Longitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        if not hasattr(self, "_lon"):
            pivot_x = self.pivot_point[0]
            cell_size = self.cell_size
            x_coords = Dataset.get_x_lon_dimension_array(
                pivot_x, cell_size, self.columns
            )
            # x_coords = [
            #     pivot_x + i * cell_size + cell_size / 2 for i in range(self.columns)
            # ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            x_coords = self._lon
        return np.array(x_coords)

    @staticmethod
    def get_x_lon_dimension_array(pivot_x, cell_size, columns) -> List[float]:
        """Get X/Lon coordinates."""
        x_coords = [pivot_x + i * cell_size + cell_size / 2 for i in range(columns)]
        return x_coords

    @property
    def y(self):
        """Y-coordinate/Latitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        if not hasattr(self, "_lat"):
            pivot_y = self.pivot_point[1]
            cell_size = self.cell_size
            # y_coords = [
            #     pivot_y - i * cell_size - cell_size / 2 for i in range(self.rows)
            # ]
            y_coords = Dataset.get_y_lat_dimension_array(pivot_y, cell_size, self.rows)
        else:
            # in case the lat and lon are read from the netcdf file, just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @staticmethod
    def get_y_lat_dimension_array(pivot_y, cell_size, rows) -> List[float]:
        """Get Y/Lat coordinates."""
        y_coords = [pivot_y - i * cell_size - cell_size / 2 for i in range(rows)]
        return y_coords

    @property
    def gdal_dtype(self):
        """Data Type."""
        return [
            self.raster.GetRasterBand(i).DataType for i in range(1, self.band_count + 1)
        ]

    @property
    def numpy_dtype(self) -> List[type]:
        """List of the numpy data Type of each band, the data type is a numpy function."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "numpy"].values[0]
            for i in self.gdal_dtype
        ]

    @property
    def dtype(self) -> List[str]:
        """List of the numpy data Type of each band."""
        return [
            DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == i, "name"].values[0]
            for i in self.gdal_dtype
        ]

    def get_block_arrangement(
        self, band: int = 0, x_block_size: int = None, y_block_size: int = None
    ) -> DataFrame:
        """Get Block Arrangement.

        Parameters
        ----------
        band : int, optional
            band index, by default 0
        x_block_size : int, optional
            x block size, by default None
        y_block_size : int, optional
            y block size, by default None

        Returns
        -------
        DataFrame
            DataFrame with the following columns: [x_offset, y_offset, window_xsize, window_ysize]

        Examples
        --------
        >>> dataset = Dataset.read_file("tests/data/acc4000.tif")
        >>> df = dataset.get_block_arrangement(x_block_size=5, y_block_size=5)
        >>> print(df)
        >>>    x_offset  y_offset  window_xsize  window_ysize
        0         0         0             5             5
        1         5         0             5             5
        2        10         0             4             5
        3         0         5             5             5
        4         5         5             5             5
        5        10         5             4             5
        6         0        10             5             3
        7         5        10             5             3
        8        10        10             4             3

        """
        block_sizes = self.block_size[band]
        x_block_size = block_sizes[0] if x_block_size is None else x_block_size
        y_block_size = block_sizes[1] if y_block_size is None else y_block_size

        df = pd.DataFrame(
            [
                {
                    "x_offset": x,
                    "y_offset": y,
                    "window_xsize": min(x_block_size, self.columns - x),
                    "window_ysize": min(y_block_size, self.rows - y),
                }
                for y in range(0, self.rows, y_block_size)
                for x in range(0, self.columns, x_block_size)
            ],
            columns=["x_offset", "y_offset", "window_xsize", "window_ysize"],
        )
        return df

    def copy(self):
        """Deep copy."""
        src = gdal.GetDriverByName("MEM").CreateCopy("", self._raster)
        return Dataset(src)

    def _iloc(self, i: int) -> gdal.Band:
        """_iloc.

            - Access dataset bands using index.

        Parameters
        ----------
        i: [int]
            index, the index starts from 1.

        Returns
        -------
        Band:
            Gdal Band.
        """
        if i < 0:
            raise IndexError("negative index not supported")

        if i > self.band_count - 1:
            raise IndexError(
                f"index {i} is out of bounds for axis 0 with size {self.band_count}"
            )
        band = self.raster.GetRasterBand(i + 1)
        return band

    def get_attribute_table(self, band: int = 0) -> DataFrame:
        """get_attribute_table.

            - Get the attribute table of a band.

        Parameters
        ----------
        band: [int]
            band index, the index starts from 1.

        Returns
        -------
        DataFrame:
            DataFrame with the attribute table.
        """
        band = self._iloc(band)
        rat = band.GetDefaultRAT()
        if rat is None:
            df = None
        else:
            df = self._attribute_table_to_df(rat)

        return df

    def set_attribute_table(self, df: DataFrame, band: int = None) -> None:
        """set_attribute_table.

            - Set the attribute table for a band.

        Parameters
        ----------
        df: [DataFrame]
            DataFrame with the attribute table.
        band: [int]
            band index.
        """
        rat = self._df_to_attribute_table(df)
        band = self._iloc(band)
        band.SetDefaultRAT(rat)

    @staticmethod
    def _df_to_attribute_table(df: DataFrame) -> gdal.RasterAttributeTable:
        """df_to_attribute_table.

            Convert a DataFrame to a GDAL RasterAttributeTable.

        Parameters
        ----------
        df: [DataFrame]
            DataFrame with columns to be converted to RAT columns.

        Returns
        -------
        gdal.RasterAttributeTable:
            The resulting RasterAttributeTable.
        """
        # Create a new RasterAttributeTable
        rat = gdal.RasterAttributeTable()

        # Create columns in the RAT based on the DataFrame columns
        for column in df.columns:
            dtype = df[column].dtype
            if pd.api.types.is_integer_dtype(dtype):
                rat.CreateColumn(column, gdal.GFT_Integer, gdal.GFU_Generic)
            elif pd.api.types.is_float_dtype(dtype):
                rat.CreateColumn(column, gdal.GFT_Real, gdal.GFU_Generic)
            else:  # Assume string for any other type
                rat.CreateColumn(column, gdal.GFT_String, gdal.GFU_Generic)

        # Populate the RAT with the DataFrame data
        for row_index in range(len(df)):
            for col_index, column in enumerate(df.columns):
                dtype = df[column].dtype
                value = df.iloc[row_index, col_index]
                if pd.api.types.is_integer_dtype(dtype):
                    rat.SetValueAsInt(row_index, col_index, int(value))
                elif pd.api.types.is_float_dtype(dtype):
                    rat.SetValueAsDouble(row_index, col_index, float(value))
                else:  # Assume string for any other type
                    rat.SetValueAsString(row_index, col_index, str(value))

        return rat

    @staticmethod
    def _attribute_table_to_df(rat: gdal.RasterAttributeTable) -> DataFrame:
        """attribute_table_to_df.

        Convert a GDAL RasterAttributeTable to a pandas DataFrame.

        Parameters
        ----------
        rat: [gdal.RasterAttributeTable]
            The RasterAttributeTable to convert.

        Returns
        -------
        pd.DataFrame:
            The resulting DataFrame.
        """
        columns = []
        data = {}

        # Get the column names and create empty lists for data
        for col_index in range(rat.GetColumnCount()):
            col_name = rat.GetNameOfCol(col_index)
            col_type = rat.GetTypeOfCol(col_index)
            columns.append((col_name, col_type))
            data[col_name] = []

        # Get the row count
        row_count = rat.GetRowCount()

        # Populate the data dictionary with RAT values
        for row_index in range(row_count):
            for col_index, (col_name, col_type) in enumerate(columns):
                if col_type == gdal.GFT_Integer:
                    value = rat.GetValueAsInt(row_index, col_index)
                elif col_type == gdal.GFT_Real:
                    value = rat.GetValueAsDouble(row_index, col_index)
                else:  # gdal.GFT_String
                    value = rat.GetValueAsString(row_index, col_index)
                data[col_name].append(value)

        # Create the DataFrame
        df = pd.DataFrame(data)
        return df

    def add_band(
        self,
        array: np.ndarray,
        unit: Any = None,
        attribute_table: DataFrame = None,
        inplace: bool = False,
    ):
        """add_band.

            - Add a new band to the dataset.

        Parameters
        ----------
        array: [np.ndarray]
            2D array to add as a new band.
        unit: [Any] optional
            unit of the values in the new band.
        attribute_table: [DataFrame] optional, Default is None
            attribute_table provides a way to associate tabular data with the values of a raster band.
            This is particularly useful for categorical raster data, such as land cover classifications, where each
            pixel value corresponds to a category that has additional attributes (e.g., class name, color, description).
        inplace: [bool] optional
            if True the new band will be added to the current dataset, if False the new band will be added to a new
            dataset.

        Returns
        -------
        None

        Examples
        --------
        - Example of the attribute_table:
        >>> data = {
        ...     "Value": [1, 2, 3],
        ...     "ClassName": ["Forest", "Water", "Urban"],
        ...     "Color": ["#008000", "#0000FF", "#808080"],
        ... }
        >>> df = pd.DataFrame(data)
        """
        # check the dimensions of the new array
        if array.ndim != 2:
            raise ValueError("The array must be 2D.")
        if array.shape[0] != self.rows or array.shape[1] != self.columns:
            raise ValueError(
                f"The array must have the same dimensions as the raster.{self.rows} {self.columns}"
            )
        # check if the dataset is opened in write mode
        if inplace:
            src = self._raster
            if src.GetRootGroup() is None:
                raise ValueError("The dataset is not opened in write mode.")
        else:
            src = gdal.GetDriverByName("MEM").CreateCopy("", self._raster)

        dtype = numpy_to_gdal_dtype(array.dtype)
        num_bands = src.RasterCount
        src.AddBand(dtype, [])
        band = src.GetRasterBand(num_bands + 1)

        if unit is not None:
            band.SetUnitType(unit)

        if attribute_table is not None:
            # Attach the RAT to the raster band
            rat = Dataset._df_to_attribute_table(attribute_table)
            band.SetDefaultRAT(rat)

        band.WriteArray(array)

        if inplace:
            self.__init__(src)
        else:
            return Dataset(src)

    def stats(self, band: int = None, mask: GeoDataFrame = None) -> DataFrame:
        """stats.

            - Get statistics of a band [Min, max, mean, std].

        Parameters
        ----------
        band: [int]
            band index, if None, the statistics of all bands will be returned.
        mask: [Polygon GeoDataFrame/Dataset object]
            GeodataFrame with a geometry of polygon type

        Returns
        -------
        DataFrame:
            DataFrame of statistics values of each band, the dataframe has the following columns:
            [min, max, mean, std], the index of the dataframe is the band names.
            >>>                Min         max        mean       std
            >>> Band_1  270.369720  270.762299  270.551361  0.154270
            >>> Band_2  269.611938  269.744751  269.673645  0.043788
            >>> Band_3  273.641479  274.168823  273.953979  0.198447
            >>> Band_4  273.991516  274.540344  274.310669  0.205754

        Hint
        ----
        Hint
        ----
        - The value of the stats will be stored in an xml file by the name of the raster file with the extension of
        .aux.xml, the content of the file will be like the following:

        <PAMDataset>
          <PAMRasterBand band="1">
            <Description>Band_1</Description>
            <Metadata>
              <MDI key="RepresentationType">ATHEMATIC</MDI>
              <MDI key="STATISTICS_MAXIMUM">88</MDI>
              <MDI key="STATISTICS_MEAN">7.9662921348315</MDI>
              <MDI key="STATISTICS_MINIMUM">0</MDI>
              <MDI key="STATISTICS_STDDEV">18.294377743948</MDI>
              <MDI key="STATISTICS_VALID_PERCENT">48.9</MDI>
            </Metadata>
          </PAMRasterBand>
        </PAMDataset>
        """
        if mask is not None:
            dst = self.crop(mask, touch=True)

        if band is None:
            df = pd.DataFrame(
                index=self.band_names,
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            for i in range(self.band_count):
                if mask is not None:
                    df.iloc[i, :] = dst._get_stats(i)
                else:
                    df.iloc[i, :] = self._get_stats(i)
        else:
            df = pd.DataFrame(
                index=[self.band_names[band]],
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            if mask is not None:
                df.iloc[0, :] = dst._get_stats(band)
            else:
                df.iloc[0, :] = self._get_stats(band)

        return df

    def _get_stats(self, band: int = None) -> List[float]:
        """_get_stats."""
        band_i = self._iloc(band)
        try:
            vals = band_i.GetStatistics(True, True)
        except RuntimeError:
            # when the GetStatistics gives an error "RuntimeError: Failed to compute statistics, no valid pixels
            # found in sampling."
            vals = [0]

        if sum(vals) == 0:
            warnings.warn(
                f"Band {band} has no statistics, and the statistics are going to be calculate"
            )
            vals = band_i.ComputeStatistics(False)

        return vals

    def plot(
        self,
        band: int = None,
        exclude_value: Any = None,
        rgb: List[int] = None,
        surface_reflectance: int = 10000,
        cutoff: List = None,
        overview: bool = False,
        overview_index: int = 0,
        **kwargs,
    ):
        """Plot.

            - plot the values/overviews of a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0
        exclude_value: [Any]
            value to exclude from the plot. Default is None.
        rgb: [List]
            The `plot` method will check it the rgb bands are defined in the raster file, if all the three bands (
            red, green, blue)) are defined, the method will use them to plot the real image, if not the rgb bands
            will be considered as [2,1,0].
        surface_reflectance: [int]
            Default is 10,000.
        cutoff: [List]
            clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
            and scale them back to between 0 and 1). Default is None.
        overview: [bool]
            True if you want to plot the overview. Default is False.
        overview_index: [int]
            index of the overview. Default is 0.
        **kwargs
            points : [array]
                3 column array with the first column as the value you want to display for the point, the second is the rows
                index of the point in the array, and the third column as the column index in the array.
                - the second and third column tells the location of the point in the array.
            point_color: [str]
                color.
            point_size: [Any]
                size of the point.
            pid_color: [str]
                the color of the annotation of the point. Default is blue.
            pid_size: [Any]
                size of the point annotation.
            figsize: [tuple], optional
                figure size. The default is (8,8).
            title: [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size: [integer], optional
                title size. The default is 15.
            orientation: [string], optional
                orientation of the color bar horizontal/vertical. The default is 'vertical'.
            rotation: [number], optional
                rotation of the color bar label. The default is -90.
            orientation: [string], optional
                orientation of the color bar horizontal/vertical. The default is 'vertical'.
            cbar_length: [float], optional
                ratio to control the height of the color bar. The default is 0.75.
            ticks_spacing: [integer], optional
                Spacing in the color bar ticks. The default is 2.
            cbar_label_size: integer, optional
                size of the color bar label. The default is 12.
            cbar_label: str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale: integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- color_scale 1 is the normal scale
                2- color_scale 2 is the power scale
                3- color_scale 3 is the SymLogNorm scale
                4- color_scale 4 is the PowerNorm scale
                5- color_scale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma: [float], optional
                    value needed for option 2. The default is 1./2.
                line_threshold: [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale: [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint: [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap: [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value: [bool]
                True if you want to display the values of the cells as a text
            num_size: integer, optional
                size of the numbers plotted on top of each cell. The default is 8.
            background_color_threshold: [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black, and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/Serapieum-of-alex/cleopatra"
        )
        from cleopatra.array import Array

        no_data_value = [np.nan if i is None else i for i in self.no_data_value]
        if overview:
            arr = self.read_overview_array(band=band, overview_index=overview_index)
        else:
            arr = self.read_array(band=band)
        # if the raster has three bands or more.
        if self.band_count >= 3:
            if band is None:
                if rgb is None:
                    rgb = [
                        self.get_band_by_color("red"),
                        self.get_band_by_color("green"),
                        self.get_band_by_color("blue"),
                    ]
                    if None in rgb:
                        rgb = [2, 1, 0]
                # first make the band index the first band in the rgb list (red band)
                band = rgb[0]
        # elif self.band_count == 1:
        #     band = 0
        else:
            if band is None:
                band = 0

        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )

        cleo = Array(
            arr,
            exclude_value=exclude_value,
            extent=self.bbox,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            **kwargs,
        )
        fig, ax = cleo.plot(**kwargs)
        return fig, ax

    @staticmethod
    def _create_dataset(
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        driver: str = "MEM",
        path: str = None,
    ) -> gdal.Dataset:
        """Create a GDAL driver.

            creates a driver and save it to disk and in memory if the path is not given.

        Parameters
        ----------
        cols: [int]
            number of columns.
        rows: [int]
            number of rows.
        bands: [int]
            number of bands.
        driver: [str]
            driver type ["GTiff", "MEM"].
        path: [str]
            path to save the GTiff driver.
        dtype:
            gdal data type, use the functions in the utils module to map data types from numpy or ogr to gdal.

            gdal data type, the data type should be one of the following code:
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], which refers to the following data types:.
            GDT_Unknown	0	GDT_UInt32	4	GDT_CInt16	8	GDT_UInt64	12
            GDT_Byte	1	GDT_Int32	5	GDT_CInt32	9	GDT_Int64	13
            GDT_UInt16	2	GDT_Float32 6	GDT_CFloat32 10	GDT_Int8	14
            GDT_Int16	3	GDT_Float64 7	GDT_CFloat64 11	GDT_TypeCount 15

        Returns
        -------
        gdal driver
        """
        if path:
            driver = "GTiff" if driver == "MEM" else driver
            if not isinstance(path, str):
                raise TypeError("The path input should be string")
            if driver == "GTiff":
                if not path.endswith(".tif"):
                    raise TypeError(
                        "The path to save the created raster should end with .tif"
                    )
            # LZW is a lossless compression method achieve the highest compression but with a lot of computations.
            src = gdal.GetDriverByName(driver).Create(
                path, cols, rows, bands, dtype, ["COMPRESS=LZW"]
            )
        else:
            # for memory drivers
            driver = "MEM"
            src = gdal.GetDriverByName(driver).Create("", cols, rows, bands, dtype)
        return src

    @classmethod
    def create(
        cls,
        cell_size: int,
        rows: int,
        columns: int,
        dtype: str,
        bands: int,
        top_left_coords: Tuple,
        epsg: int,
        no_data_value: Any = None,
        path: str = None,
    ) -> "Dataset":
        """Create a new dataset and fill it with the no_data_value.

            - The new dataset will have an array filled with the no_data_value.

        Parameters
        ----------
        cell_size: [Any]
            cell size.
        rows: [int]
            number of rows.
        columns: [int]
            number of columns.
        dtype: [int]
            data type, the data type should be one of the following code:
            None, "byte", "uint16", "int16", "uint32", "int32", "float32", "float64", "complex-int16",
            "complex-int32", "complex-float32", "complex-float64", "uint64", "int64", "int8", "count"
        bands : int or None
            Number of bands to create in the output raster.
        top_left_coords: [Tuple]
            coordinates of the top left corner point.
        epsg: [int]
            epsg number to identify the projection of the coordinates in the created raster.
        no_data_value : float or None
            No data value.
        path: [str] Optional, Default is None
            path on disk, if None the dataset will be created in memory.

        Returns
        -------
        Dataset

        Hints
        -----
        - The no_data_value will be filled in the array of the output dataset.
        - The coordinates of the top left corner point should be in the same projection as the epsg.
        - The cell size should be in the same unit as the coordinates.
        - The number of rows and columns should be positive integers.
        - The dtype should be one of the following code:

        Examples
        --------
        >>> cell_size = 10
        >>> rows = 100
        >>> columns = 100
        >>> dtype = 1
        >>> bands = 1
        >>> top_left_coords = (0, 0)
        >>> epsg = 32618
        >>> no_data_value = -9999
        >>> path = "test.tif"
        >>> dst = Dataset.create(cell_size, rows, columns, dtype, bands, top_left_coords, epsg, no_data_value, path)
        """
        # Create the driver.
        dtype = numpy_to_gdal_dtype(dtype)
        dst = Dataset._create_dataset(columns, rows, bands, dtype, path=path)
        sr = Dataset._create_sr_from_epsg(epsg)
        geotransform = (
            top_left_coords[0],
            cell_size,
            0,
            top_left_coords[1],
            0,
            -1 * cell_size,
        )
        dst.SetGeoTransform(geotransform)
        # Set the projection.
        dst.SetProjection(sr.ExportToWkt())
        if path is None:
            access = "write"
        else:
            access = "read_only"

        dst = cls(dst, access=access)
        if no_data_value is not None:
            dst._set_no_data_value(no_data_value=no_data_value)

        return dst

    @classmethod
    def create_from_array(
        cls,
        arr: np.ndarray,
        geo: Tuple[float, float, float, float, float, float],
        epsg: Union[str, int] = 4326,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
    ) -> "Dataset":
        """create_from_array.

            - Create_from_array method creates a `Dataset` from a given array and geotransform data.

        Parameters
        ----------
        arr: [np.ndarray]
            numpy array.
        geo : [Tuple]
            geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator)
        no_data_value : Any, optional
            no data value to mask the cells out of the domain. The default is -9999.
        driver_type: [str] optional
            driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM"
        path : [str]
            path to save the driver.

        Returns
        -------
        dst: [DataSet].
            Dataset object will be returned.
        """
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
        bands: int = None,
        geo: Tuple[float, float, float, float, float, float] = None,
        epsg: Union[str, int] = None,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
    ) -> "Dataset":
        dtype = numpy_to_gdal_dtype(arr)
        dst_ds = Dataset._create_dataset(
            cols, rows, bands, dtype, driver=driver_type, path=path
        )

        srse = Dataset._create_sr_from_epsg(epsg=epsg)
        dst_ds.SetProjection(srse.ExportToWkt())
        dst_ds.SetGeoTransform(geo)
        if path is None:
            access = "write"
        else:
            access = "read_only"
        dst_obj = Dataset(dst_ds, access=access)
        dst_obj._set_no_data_value(no_data_value=no_data_value)

        if bands == 1:
            dst_obj.raster.GetRasterBand(1).WriteArray(arr)
        else:
            for i in range(bands):
                dst_obj.raster.GetRasterBand(i + 1).WriteArray(arr[i, :, :])

        return dst_obj

    @classmethod
    def dataset_like(
        cls,
        src,
        array: np.ndarray,
        driver: str = "MEM",
        path: str = None,
    ) -> Union["Dataset", None]:
        """dataset_like.

            dataset_like method creates a Dataset from an array like another source dataset. The new dataset
            will have the same `projection`, `coordinates` or the `top left corner` of the original dataset,
            `cell size`, `no_data_velue`, and number of `rows` and `columns`.
            the array and the source dataset should have the same number of columns and rows

        Parameters
        ----------
        src: [gdal.dataset]
            source raster to get the spatial information
        array: [numpy array]
            to store in the new raster
        driver:[str]
            gdal driver type. Default is "GTiff"
        path : [String]
            path to save the new raster including new raster name and extension (.tif)

        Returns
        -------
        None:
            if the driver is "GTiff" the function will save the new raster to the given path.
        Datacube:
            if the driver is "MEM" the function will return the created raster in memory.

        Example
        -------
        >>> src_array = np.load("RAIN_5k.npy")
        >>> src_dataset = Dataset.read_file("DEM.tif")
        >>> name = "rain.tif"
        >>> new_dataset = src_dataset.dataset_like(src, src_array, driver="GTiff")
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array should be of type numpy array")

        if len(array.shape) == 2:
            bands = 1
        else:
            bands = array.shape[0]

        dtype = numpy_to_gdal_dtype(array)

        dst = Dataset._create_dataset(
            src.columns, src.rows, bands, dtype, driver=driver, path=path
        )

        dst.SetGeoTransform(src.geotransform)
        dst.SetProjection(src.crs)
        # setting the NoDataValue does not accept double precision numbers
        if path is None:
            access = "write"
        else:
            access = "read_only"
        dst_obj = cls(dst, access=access)
        dst_obj._set_no_data_value(no_data_value=src.no_data_value[0])

        dst_obj.raster.GetRasterBand(1).WriteArray(array)
        if path is not None:
            dst_obj.raster.FlushCache()
            dst_obj = None

        return dst_obj

    def write_array(self, array: np.array, pivot_cell_indexes: List[Any] = None):
        """Write an array to the dataset at the given xoff, yoff position.

        Parameters
        ----------
        array : np.ndarray
            The array to write
        pivot_cell_indexes : List[float, float]
            indexes [row, column]/[y_offset, x_offset] of the cell to write the array to. If None, the array will be
            written to the top left corner of the dataset.

        Raises
        ------
        Exception
            If the array is not written successfully.

        Hint
        ----
        - The `Dataset` has to be opened in a write mode `read_only=False`.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> from pyramids.dataset import Dataset
        >>> path = "tests.tif"
        >>> dataset = Dataset.read_file(path, read_only=False)
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> dataset.write_array(arr, xoff=5, yoff=3)
        """
        yoff, xoff = pivot_cell_indexes
        try:
            self._raster.WriteArray(array, xoff=xoff, yoff=yoff)
        except Exception as e:
            raise e

    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        return self.raster.GetProjection()

    def set_crs(self, crs: Optional = None, epsg: int = None):
        """set_crs.

            Set the Coordinate Reference System (CRS) of a

        Parameters
        ----------
        crs: [str]
            optional if epsg is specified
            WKT string.
            i.e. 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
            0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        epsg: [int]
            optional if crs is specified
            EPSG code specifying the projection.
        """
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if self.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then reset the crs"
            )
        else:
            if crs is not None:
                self.raster.SetProjection(crs)
                self._epsg = FeatureCollection.get_epsg_from_prj(crs)
            else:
                sr = Dataset._create_sr_from_epsg(epsg)
                self.raster.SetProjection(sr.ExportToWkt())
                self._epsg = epsg

    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbour",
        maintain_alignment: int = False,
        inplace: bool = False,
    ) -> Union["Dataset", None]:
        """to_epsg.

        to_epsg reprojects a raster to any projection
        (default the WGS84 web mercator projection, without resampling)

        Parameters
        ----------
        to_epsg: [integer]
            reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        method: [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "nearest neighbour" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear
        maintain_alignment : [bool]
            True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.
        inplace: [bool]
            True to make changes inplace. Default is False.

        Returns
        -------
        Dataset:
            Dataset object, if inplace is True, the method returns None.

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> dataset = Dataset.read_file("path/raster_name.tif")
        >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)
        """
        if not isinstance(to_epsg, int):
            raise TypeError(
                "please enter correct integer number for to_epsg more information "
                f"https://epsg.io/, given {type(to_epsg)}"
            )
        if not isinstance(method, str):
            raise TypeError(
                "Please enter a correct method, for more information, see documentation "
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method: {method} does not exist, existing methods are {INTERPOLATION_METHODS.keys()}"
            )

        method = INTERPOLATION_METHODS.get(method)

        if maintain_alignment:
            dst_obj = self._reproject_with_ReprojectImage(to_epsg, method)
        else:
            dst = gdal.Warp("", self.raster, dstSRS=f"EPSG:{to_epsg}", format="VRT")
            dst_obj = Dataset(dst)

        if inplace:
            self.__init__(dst_obj.raster)
        else:
            return dst_obj

    def _get_epsg(self) -> int:
        """GetEPSG.

            This function reads the projection of a GEOGCS file or tiff file

        Returns
        -------
        epsg : [integer]
            epsg number
        """
        prj = self._get_crs()
        epsg = FeatureCollection.get_epsg_from_prj(prj)

        return epsg

    def count_domain_cells(self, band: int = 0) -> int:
        """Count cells inside the domain.

        Parameters
        ----------
        band: [int]
            band index. Default is 0.

        Returns
        -------
        int:
            Number of cells
        """
        arr = self.read_array(band=band)
        domain_count = np.size(arr[:, :]) - np.count_nonzero(
            (arr[np.isclose(arr, self.no_data_value[band], rtol=0.001)])
        )
        return domain_count

    @staticmethod
    def _create_sr_from_epsg(epsg: int = None) -> SpatialReference:
        """Create a spatial reference object from epsg number.

        https://gdal.org/tutorials/osr_api_tut.html

        Parameters
        ----------
        epsg: [int]
            epsg number.

        Returns
        -------
        SpatialReference object
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(int(epsg))
        return sr

    def _get_band_names(self) -> List[str]:
        """Get band names from band metadata if exists otherwise will return index [1,2, ...].

        Returns
        -------
        list[str]
            list of band names
        """
        names = []
        for i in range(1, self.band_count + 1):
            band_i = self.raster.GetRasterBand(i)

            if band_i.GetDescription():
                # Use the band_i description.
                names.append(band_i.GetDescription())
            else:
                # Check for metadata.
                band_i_name = "Band_{}".format(band_i.GetBand())
                metadata = band_i.GetDataset().GetMetadata_Dict()

                # If in metadata, return the metadata entry, else Band_N.
                if band_i_name in metadata and metadata[band_i_name]:
                    names.append(metadata[band_i_name])
                else:
                    names.append(band_i_name)

        return names

    def _set_band_names(self, name_list: List):
        """Set band names from a given list of names.

        Returns
        -------
        list[str]
            list of band names
        """
        for i in range(self.band_count):
            # first set the band name in the gdal dataset object
            band_i = self.raster.GetRasterBand(i + 1)
            band_i.SetDescription(name_list[i])
            # second change the band names in the _band_names property.
            self._band_names[i] = name_list[i]

    def _check_no_data_value(self, no_data_value: List):
        """Validate The no_data_value with the dtype of the object.

        Parameters
        ----------
        no_data_value

        Returns
        -------
        no_data_value:
            convert the no_data_value to comply with the dtype
        """
        # convert the no_data_value based on the dtype of each raster band.
        for i, val in enumerate(self.gdal_dtype):
            try:
                val = no_data_value[i]
                # if not None or np.nan
                if val is not None and not np.isnan(val):
                    # if val < np.iinfo(self.dtype[i]).min or val > np.iinfo(self.dtype[i]).max:
                    # if the no_data_value is out of the range of the data type
                    no_data_value[i] = self.numpy_dtype[i](val)
                else:
                    # None and np.nan
                    if self.dtype[i].startswith("u"):
                        # only Unsigned integer data types.
                        # if None or np.nan it will make a problem with the unsigned integer data type
                        # use the max bound of the data type as a no_data_value
                        no_data_value[i] = np.iinfo(self.dtype[i]).max
                    else:
                        # no_data_type is None/np,nan and all other data types that is not Unsigned integer
                        no_data_value[i] = val

            except OverflowError:
                # no_data_value = -3.4028230607370965e+38, numpy_dtype = np.int64
                warnings.warn(
                    f"The no_data_value:{no_data_value[i]} is out of range, Band data type is {self.numpy_dtype[i]}"
                )
                no_data_value[i] = self.numpy_dtype[i](DEFAULT_NO_DATA_VALUE)
        return no_data_value

    def _set_no_data_value(
        self, no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE
    ):
        """setNoDataValue.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - used only when creating an empty driver.

            now the no_data_value is converted to the dtype of the raster bands and updated in the
            dataset attribute, gdal no_data_value attribute, used to fill the raster band.
            from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.

        Parameters
        ----------
        no_data_value: [numeric]
            no data value to fill the masked part of the array.
        """
        if not isinstance(no_data_value, list):
            no_data_value = [no_data_value] * self.band_count

        no_data_value = self._check_no_data_value(no_data_value)

        for band in range(self.band_count):
            try:
                # now the no_data_value is converted to the dtype of the raster bands and updated in the
                # dataset attribute, gdal no_data_value attribute, used to fill the raster band.
                # from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.
                self._set_no_data_value_backend(band, no_data_value[band])
            except Exception as e:
                if str(e).__contains__(
                    "Attempt to write to read only dataset in GDALRasterBand::Fill()."
                ):
                    raise ReadOnlyError(
                        "The Dataset is open with a read only, please read the raster using update "
                        "access mode"
                    )
                elif str(e).__contains__(
                    "in method 'Band_SetNoDataValue', argument 2 of type 'double'"
                ):
                    self._set_no_data_value_backend(
                        band, np.float64(no_data_value[band])
                    )
                else:
                    self._set_no_data_value_backend(band, DEFAULT_NO_DATA_VALUE)
                    logger.warning(
                        "the type of the given no_data_value differs from the dtype of the raster"
                        f"no_data_value now is set to {DEFAULT_NO_DATA_VALUE} in the raster"
                    )

    def _calculate_bbox(self) -> List:
        """Calculate bounding box."""
        xmin, ymax = self.pivot_point
        ymin = ymax - self.rows * self.cell_size
        xmax = xmin + self.columns * self.cell_size
        return [xmin, ymin, xmax, ymax]

    def _calculate_bounds(self) -> GeoDataFrame:
        """Get the bbox as a geodataframe with a polygon geometry."""
        xmin, ymin, xmax, ymax = self._calculate_bbox()
        coords = [(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)]
        poly = FeatureCollection.create_polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly])
        gdf.set_crs(epsg=self.epsg, inplace=True)
        return gdf

    def _set_no_data_value_backend(self, band_i: int, no_data_value: Any):
        """
            - band_i starts from 0 to the number of bands-1.

        Parameters
        ----------
        band_i:
            band index, starts from 0.
        no_data_value:
            Numerical value.
        """
        # check if the dtype of the no_data_value comply with the dtype of the raster itself.
        self._change_no_data_value_attr(band_i, no_data_value)
        # initialize the band with the nodata value instead of 0
        # the no_data_value may have changed inside the _change_no_data_value_attr method to float64, so redefine it.
        no_data_value = self.no_data_value[band_i]
        try:
            self.raster.GetRasterBand(band_i + 1).Fill(no_data_value)
        except Exception as e:
            if str(e).__contains__(" argument 2 of type 'double'"):
                self.raster.GetRasterBand(band_i + 1).Fill(np.float64(no_data_value))
            else:
                raise ValueError(
                    f"Failed to fill the band {band_i} with value: {no_data_value}, because of {e}"
                )
        # update the no_data_value in the Dataset object
        self.no_data_value[band_i] = no_data_value

    def _change_no_data_value_attr(self, band: int, no_data_value):
        """Change the no_data_value attribute.

            - Change only the no_data_value attribute in the gdal Datacube object.
            - Change the no_data_value in the Dataset object for the given band index.
            - The corresponding value in the array will not be changed.

        Parameters
        ----------
        band: [int]
            band index starts from 0.
        no_data_value: [Any]
            no data value.
        """
        try:
            self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)
        except Exception as e:
            if str(e).__contains__(
                "Attempt to write to read only dataset in GDALRasterBand::Fill()."
            ):
                raise ReadOnlyError(
                    "The Dataset is open with a read only, please read the raster using update "
                    "access mode"
                )
            # TypeError
            elif e.args == (
                "in method 'Band_SetNoDataValue', argument 2 of type 'double'",
            ):
                no_data_value = np.float64(no_data_value)
                self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)

        self._no_data_value[band] = no_data_value

    def change_no_data_value(self, new_value: Any, old_value: Any = None):
        """Change No Data Value.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.

        Parameters
        ----------
        new_value: [numeric]
            no data value to set in the raster bands.

        old_value: [numeric]
            old no data value that is already in the raster bands.
        """
        if not isinstance(new_value, list):
            new_value = [new_value] * self.band_count

        if old_value is not None and not isinstance(old_value, list):
            old_value = [old_value] * self.band_count

        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.raster, 0)
        # create a new dataset
        new_dataset = Dataset(dst)
        # the new_value could change inside the _set_no_data_value method before it is used to set the no_data_value
        # attribute in the gdal object/pyramids object and to fill the band.
        new_dataset._set_no_data_value(new_value)
        # now we have to use the no_data_value value in the no_data_value attribute in the Dataset object as it is
        # updated.
        new_value = new_dataset.no_data_value
        for band in range(self.band_count):
            arr = self.read_array(band)
            try:
                if old_value is not None:
                    arr[np.isclose(arr, old_value, rtol=0.001)] = new_value[band]
                else:
                    arr[np.isnan(arr)] = new_value[band]
            except TypeError:
                raise NoDataValueError(
                    f"The dtype of the given no_data_value: {new_value[band]} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self.gdal_dtype[band])}"
                )
            new_dataset.raster.GetRasterBand(band + 1).WriteArray(arr)

        self.__init__(new_dataset.raster)

    def get_cell_coords(
        self, location: str = "center", mask: bool = False
    ) -> np.ndarray:
        """get_cell_coords.

        Returns the coordinates of the cell centers inside the domain (only the cells that
        do not have nodata value)

        Parameters
        ----------
        location: [str]
            location of the coordinates "center" for the center of a cell, corner for the corner of the cell.
            the corner is the top left corner.
        mask: [bool]
            True to exclude the cells out of the domain. Default is False.

        Returns
        -------
        coords : [np.ndarray]
            Array with a list of the coordinates to be interpolated, without the Nan
        mat_range : [np.ndarray]
            Array with all the centers of cells in the domain of the DEM
        """
        # check the location parameter
        location = location.lower()
        if location not in ["center", "corner"]:
            raise ValueError(
                "The location parameter can have one of these values: 'center', 'corner', "
                f"but the value: {location} is given."
            )

        if location == "center":
            # Adding 0.5*cell size to get the center
            add_value = 0.5
        else:
            add_value = 0
        # Getting data for the whole grid
        (
            x_init,
            cell_size_x,
            xy_span,
            y_init,
            yy_span,
            cell_size_y,
        ) = self.geotransform
        if cell_size_x != cell_size_y:
            if np.abs(cell_size_x) != np.abs(cell_size_y):
                logger.warning(
                    f"The given raster does not have a square cells, the cell size is {cell_size_x}*{cell_size_y} "
                )

        # data in the array
        no_val = self.no_data_value[0] if self.no_data_value[0] is not None else np.nan
        arr = self.read_array(band=0)
        if mask is not None and no_val not in arr:
            logger.warning(
                "The no data value does not exist in the band, so all the cells will be considered, and the "
                "mask will not be considered."
            )

        if mask:
            mask = [no_val]
        else:
            mask = None
        indices = get_indices2(arr, mask=mask)

        # exclude the no_data_values cells.
        f1 = [i[0] for i in indices]
        f2 = [i[1] for i in indices]
        x = [x_init + cell_size_x * (i + add_value) for i in f2]
        y = [y_init + cell_size_y * (i + add_value) for i in f1]
        coords = np.array(list(zip(x, y)))

        return coords

    def get_cell_polygons(self, mask=False) -> GeoDataFrame:
        """Get a polygon shapely geometry for the raster cells.

        Parameters
        ----------
        mask: [bool]
            True to get the polygons of the cells inside the domain.

        Returns
        -------
        GeoDataFrame:
            with two columns, geometry, and id
        """
        coords = self.get_cell_coords(location="corner", mask=mask)
        cell_size = self.geotransform[1]
        epsg = self._get_epsg()
        x = np.zeros((coords.shape[0], 4))
        y = np.zeros((coords.shape[0], 4))
        # fill the top left corner point
        x[:, 0] = coords[:, 0]
        y[:, 0] = coords[:, 1]
        # fill the top right
        x[:, 1] = x[:, 0] + cell_size
        y[:, 1] = y[:, 0]
        # fill the bottom right
        x[:, 2] = x[:, 0] + cell_size
        y[:, 2] = y[:, 0] - cell_size

        # fill the bottom left
        x[:, 3] = x[:, 0]
        y[:, 3] = y[:, 0] - cell_size

        coords_tuples = [list(zip(x[:, i], y[:, i])) for i in range(4)]
        polys_coords = [
            (
                coords_tuples[0][i],
                coords_tuples[1][i],
                coords_tuples[2][i],
                coords_tuples[3][i],
            )
            for i in range(len(x))
        ]
        polygons = list(map(FeatureCollection.create_polygon, polys_coords))
        gdf = gpd.GeoDataFrame(geometry=polygons)
        gdf.set_crs(epsg=epsg, inplace=True)
        gdf["id"] = gdf.index
        return gdf

    def get_cell_points(self, location: str = "center", mask=False) -> GeoDataFrame:
        """Get a point shapely geometry for the raster cells center point.

        Parameters
        ----------
        location: [str]
            location of the point, ["corner", "center"]. Default is "center".
        mask: [bool]
            True to get the polygons of the cells inside the domain.

        Returns
        -------
        GeoDataFrame:
            with two columns, geometry, and id
        """
        coords = self.get_cell_coords(location=location, mask=mask)
        epsg = self._get_epsg()

        coords_tuples = list(zip(coords[:, 0], coords[:, 1]))
        points = FeatureCollection.create_point(coords_tuples)
        gdf = gpd.GeoDataFrame(geometry=points)
        gdf.set_crs(epsg=epsg, inplace=True)
        gdf["id"] = gdf.index
        return gdf

    def to_file(self, path: str, band: int = 0, tile_length: int = None) -> None:
        """to_file.

            to_file a raster to a path, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Parameters
        ----------
        path: [string]
            a path including the name of the dataset whti the extention at the end (i.e. "data/cropped.tif").
        band: [int]
            band index, needed only in case of ascii drivers. Default is 0.
        tile_length: int, Optional, Default 256.
            length of the tiles in the driver.

        Examples
        --------
        >>> dataset = Dataset.read_file("path/to/file/***.tif")
        >>> dataset.to_file("save_raster_test.tif")

        Notes
        -----
        The object will still refer to the dataset before saving. if you want to use the new saved dataset you have
        to read the file again.
        """
        if not isinstance(path, str):
            raise TypeError("path input should be string type")

        extension = path.split(".")[-1]
        driver = CATALOG.get_driver_name_by_extension(extension)
        driver_name = CATALOG.get_gdal_name(driver)

        if driver == "ascii":
            arr = self.read_array(band=band)
            no_data_value = self.no_data_value[band]
            xmin, ymin, _, _ = self.bbox
            _io.to_ascii(arr, self.cell_size, xmin, ymin, no_data_value, path)
        else:
            # saving rasters with color table fails with a runtime error
            options = ["COMPRESS=DEFLATE"]
            if tile_length is not None:
                options += [
                    "TILED=YES",
                    f"TILE_LENGTH={tile_length}",
                ]
            if self._block_size is not None and self._block_size != []:
                options += [
                    "BLOCKXSIZE={}".format(self._block_size[0][0]),
                    "BLOCKYSIZE={}".format(self._block_size[0][1]),
                ]

            try:
                dst = gdal.GetDriverByName(driver_name).CreateCopy(
                    path, self.raster, 0, options=options
                )
            except RuntimeError:
                if not os.path.exists(path):
                    raise FailedToSaveError(
                        f"Failed to save the {driver_name} raster to the path: {path}"
                    )
            dst = None  # Flush the dataset to disk
            # print to go around the assigned but never used pre-commit issue
            print(dst)

    def convert_longitude(self, inplace: bool = False):
        """Convert Longitude.

            - convert the longitude from 0 - 360 to -180 - 180.
            - currently the function works correctly if the raster covers the whole world, it means that the columns
            in the rasters covers from longitude 0 to 360.

        Parameters
        ----------
        inplace: [bool]
            True to make the changes in place.

        Returns
        -------
        Dataset.
        """
        # dst = gdal.Warp(
        #     "",
        #     self.raster,
        #     dstSRS="+proj=longlat +ellps=WGS84 +datum=WGS84 +lon_0=0 +over",
        #     format="VRT",
        # )
        lon = self.lon
        src = self.raster
        # create a copy
        drv = gdal.GetDriverByName("MEM")
        dst = drv.CreateCopy("", src, 0)
        # convert the 0 to 360 to -180 to 180
        if lon[-1] <= 180:
            raise ValueError("The raster should cover the whole globe")

        first_to_translated = np.where(lon > 180)[0][0]

        ind = list(range(first_to_translated, len(lon)))
        ind_2 = list(range(0, first_to_translated))

        for band in range(self.band_count):
            arr = self.read_array(band=band)
            arr_rearranged = arr[:, ind + ind_2]
            dst.GetRasterBand(band + 1).WriteArray(arr_rearranged)

        # correct the geotransform
        pivot_point = self.pivot_point
        gt = list(self.geotransform)
        if lon[-1] > 180:
            new_gt = pivot_point[0] - 180
            gt[0] = new_gt

        dst.SetGeoTransform(gt)
        if not inplace:
            return Dataset(dst)
        else:
            self.__init__(dst)

    def _band_to_polygon(self, band: int, col_name: str):
        band = self.raster.GetRasterBand(band + 1)
        srs = osr.SpatialReference(wkt=self.crs)

        dst_ds = FeatureCollection.create_ds("memory")
        dst_layer = dst_ds.CreateLayer(col_name, srs=srs)
        dtype = gdal_to_ogr_dtype(self.raster)
        new_field = ogr.FieldDefn(col_name, dtype)
        dst_layer.CreateField(new_field)
        gdal.Polygonize(band, band, dst_layer, 0, [], callback=None)

        vector = FeatureCollection(dst_ds)
        gdf = vector._ds_to_gdf()
        return gdf

    def to_feature_collection(
        self,
        vector_mask: GeoDataFrame = None,
        add_geometry: str = None,
        tile: bool = False,
        tile_size: int = 1500,
        touch: bool = True,
    ) -> Union[DataFrame, GeoDataFrame]:
        """Convert a raster to a vector.

        The function does the following
            - Flatten the array in each band in the raster then mask the values if a vector_mask
            file is given otherwise it will flatten all values.
            - Put the values for each band in a column in a dataframe under the name of the raster band, but if no meta
            data in the raster band exists, an index number will be used [1, 2, 3, ...]
            - The function has an add_geometry parameter with two possible values ["point", "polygon"], which you can
            specify the type of shapely geometry you want to create from each cell,
                - If point is chosen, the created point will be at the center of each cell
                - If a polygon is chosen, a square polygon will be created that covers the entire cell.

        Parameters
        ----------
        vector_mask : Optional[GeoDataFrame]
            GeoDataFrame for the vector_mask. If given, it will be used to clip the raster
        add_geometry: [str]
            "Polygon", or "Point" if you want to add a polygon geometry of the cells as  column in dataframe.
            Default is None.
        tile: [bool]
            True to use tiles in extracting the values from the raster. Default is False.
        tile_size: [int]
            tile size. Default is 1500.
        touch: [bool]
            to include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
            Default is True.

        Returns
        -------
        DataFrame/GeoDataFrame
            columndL:
                >>> print(gdf.columns)
                >>> Index(['Band_1', 'geometry'], dtype='object')

        the resulted geodataframe will have the band value under the name of the band (if the raster file has a
        metadata, if not, the bands will be indexed from 1 to the number of bands)
        """
        # Get raster band names. open the dataset using gdal.Open
        band_names = self.band_names

        # Create a mask from the pixels touched by the vector_mask.
        if vector_mask is not None:
            src = self.crop(mask=vector_mask, touch=touch)
        else:
            src = self

        if tile:
            df_list = []  # DataFrames of each tile.
            for arr in Dataset.get_tile(src.raster):
                # Assume multi-band
                idx = (1, 2)
                if arr.ndim == 2:
                    # Handle single band rasters
                    idx = (0, 1)

                mask_arr = np.ones((arr.shape[idx[0]], arr.shape[idx[1]]))
                pixels = get_pixels(arr, mask_arr).transpose()
                df_list.append(pd.DataFrame(pixels, columns=band_names))

            # Merge all the tiles.
            df = pd.concat(df_list)
        else:
            arr = src.read_array()

            if self.band_count == 1:
                pixels = arr.flatten()
            else:
                pixels = (
                    arr.flatten()
                    .reshape(src.band_count, src.columns * src.rows)
                    .transpose()
                )
            df = pd.DataFrame(pixels, columns=band_names)
            # mask no data values.
            if src.no_data_value[0] is not None:
                df.replace(src.no_data_value[0], np.nan, inplace=True)
            df.dropna(axis=0, inplace=True, ignore_index=True)

        if add_geometry:
            if add_geometry.lower() == "point":
                coords = src.get_cell_points(mask=True)
            else:
                coords = src.get_cell_polygons(mask=True)

        df = df.drop(columns=["burn_value", "geometry"], errors="ignore")
        if add_geometry:
            df = gpd.GeoDataFrame(df, geometry=coords["geometry"])

        return df

    def apply(self, fun, band: int = 0) -> "Dataset":
        """apply.

            - apply method executes a mathematical operation on raster array and returns
            the result
            - The apply method executes the function only on one cell at a time.

        Parameters
        ----------
        fun: [function]
            defined function that takes one input which is the cell value.
        band: [int]
            band number.

        Returns
        -------
        Datacube
            gdal dataset object

        Examples
        --------
        >>> src_raster = gdal.Open("evap.tif")
        >>> func = np.abs
        >>> new_raster = Dataset.apply(src_raster, func)
        """
        if not callable(fun):
            raise TypeError("The second argument should be a function")

        no_data_value = self.no_data_value[band]
        src_array = self.read_array(band)
        dtype = self.gdal_dtype[band]

        # fill the new array with the nodata value
        new_array = np.ones((self.rows, self.columns)) * no_data_value
        # execute the function on each cell
        # TODO: optimize executing a function over a whole array
        for i in range(self.rows):
            for j in range(self.columns):
                if not np.isclose(src_array[i, j], no_data_value, rtol=0.001):
                    new_array[i, j] = fun(src_array[i, j])

        # create the output raster
        dst = Dataset._create_dataset(self.columns, self.rows, 1, dtype, driver="MEM")
        # set the geotransform
        dst.SetGeoTransform(self.geotransform)
        # set the projection
        dst.SetProjection(self.crs)
        dst_obj = Dataset(dst)
        dst_obj._set_no_data_value(no_data_value=no_data_value)
        dst_obj.raster.GetRasterBand(band + 1).WriteArray(new_array)

        return dst_obj

    def fill(
        self, val: Union[float, int], driver: str = "MEM", path: str = None
    ) -> Union["Dataset", None]:
        """Fill.

            Fill takes a raster and fills it with one value

        Parameters
        ----------
        val: [numeric]
            numeric value
        driver: [str]
            driver type ["GTiff", "MEM"]
        path : [str]
            path including the extension (.tif)

        Returns
        -------
        Dataset:
            the returned value will be a Dataset.
        """
        no_data_value = self.no_data_value[0]
        src_array = self.raster.ReadAsArray()

        if no_data_value is None:
            no_data_value = np.nan

        if not np.isnan(no_data_value):
            src_array[~np.isclose(src_array, no_data_value, rtol=0.000001)] = val
        else:
            src_array[~np.isnan(src_array)] = val
        dst = Dataset.dataset_like(self, src_array, driver=driver, path=path)
        return dst

    def resample(
        self, cell_size: Union[int, float], method: str = "nearest neighbour"
    ) -> "Dataset":
        """resample.

        resample method reproject a raster to any projection
        (default the WGS84 web mercator projection, without resampling)
        The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

        Parameters
        ----------
        cell_size : [integer]
             new cell size to resample the raster.
            (default empty so raster will not be resampled)
        method : [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "nearest neighbour" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear

        Returns
        -------
        Dataset:
             Dataset object.
        """
        if not isinstance(method, str):
            raise TypeError(
                "Please enter correct method, for more information, see documentation"
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method does not exist, existing methods are {INTERPOLATION_METHODS.keys()}"
            )

        method = INTERPOLATION_METHODS.get(method)

        sr_src = osr.SpatialReference(wkt=self.crs)

        ulx = self.geotransform[0]
        uly = self.geotransform[3]
        # transform the right lower corner point
        lrx = self.geotransform[0] + self.geotransform[1] * self.columns
        lry = self.geotransform[3] + self.geotransform[5] * self.rows

        # new geotransform
        new_geo = (
            self.geotransform[0],
            cell_size,
            self.geotransform[2],
            self.geotransform[3],
            self.geotransform[4],
            -1 * cell_size,
        )
        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / cell_size))
        rows = int(np.round(abs(uly - lry) / cell_size))
        dtype = self.gdal_dtype[0]
        bands = self.band_count

        dst = Dataset._create_dataset(cols, rows, bands, dtype)
        # set the geotransform
        dst.SetGeoTransform(new_geo)
        # set the projection
        dst.SetProjection(sr_src.ExportToWkt())
        dst_obj = Dataset(dst)
        # set the no data value
        dst_obj._set_no_data_value(self.no_data_value)
        # perform the projection & resampling
        gdal.ReprojectImage(
            self.raster,
            dst_obj.raster,
            sr_src.ExportToWkt(),
            sr_src.ExportToWkt(),
            method,
        )

        return dst_obj

    def _reproject_with_ReprojectImage(
        self, to_epsg: int, method: str = "nearest neighbour"
    ) -> "Dataset":
        src_gt = self.geotransform
        src_x = self.columns
        src_y = self.rows

        src_sr = osr.SpatialReference(wkt=self.crs)
        src_epsg = self.epsg

        dst_sr = self._create_sr_from_epsg(to_epsg)

        # in case the source crs is GCS and longitude is in the west hemisphere, gdal
        # reads longitude from 0 to 360 and a transformation factor wont work with values
        # greater than 180
        if src_epsg != to_epsg:
            if src_epsg == "4326" and src_gt[0] > 180:
                lng_new = src_gt[0] - 360
                # transformation factors
                tx = osr.CoordinateTransformation(src_sr, dst_sr)
                # transform the right upper corner point
                (ulx, uly, ulz) = tx.TransformPoint(lng_new, src_gt[3])
                # transform the right lower corner point
                (lrx, lry, lrz) = tx.TransformPoint(
                    lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                )
            else:
                xs = [src_gt[0], src_gt[0] + src_gt[1] * src_x]
                ys = [src_gt[3], src_gt[3] + src_gt[5] * src_y]

                [uly, lry], [ulx, lrx] = FeatureCollection.reproject_points(
                    ys, xs, from_epsg=src_epsg, to_epsg=to_epsg
                )
                # old transform
                # # transform the right upper corner point
                # (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
                # # transform the right lower corner point
                # (lrx, lry, lrz) = tx.TransformPoint(
                #     src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                # )

        else:
            ulx = src_gt[0]
            uly = src_gt[3]
            lrx = src_gt[0] + src_gt[1] * src_x
            lry = src_gt[3] + src_gt[5] * src_y

        # get the cell size in the source raster and convert it to the new crs
        # x coordinates or longitudes
        xs = [src_gt[0], src_gt[0] + src_gt[1]]
        # y coordinates or latitudes
        ys = [src_gt[3], src_gt[3]]

        if src_epsg != to_epsg:
            # transform the two-point coordinates to the new crs to calculate the new cell size
            new_ys, new_xs = FeatureCollection.reproject_points(
                ys, xs, from_epsg=src_epsg, to_epsg=to_epsg, precision=6
            )
        else:
            new_xs = xs
            # new_ys = ys

        # TODO: the function does not always maintain alignment, based on the conversion of the cell_size and the
        # pivot point
        pixel_spacing = np.abs(new_xs[0] - new_xs[1])

        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))

        dtype = self.gdal_dtype[0]
        dst = Dataset._create_dataset(cols, rows, self.band_count, dtype)

        # new geotransform
        new_geo = (
            ulx,
            pixel_spacing,
            src_gt[2],
            uly,
            src_gt[4],
            np.sign(src_gt[-1]) * pixel_spacing,
        )
        # set the geotransform
        dst.SetGeoTransform(new_geo)
        # set the projection
        dst.SetProjection(dst_sr.ExportToWkt())
        # set the no data value
        dst_obj = Dataset(dst)
        dst_obj._set_no_data_value(self.no_data_value)
        # perform the projection & resampling
        gdal.ReprojectImage(
            self.raster,
            dst_obj.raster,
            src_sr.ExportToWkt(),
            dst_sr.ExportToWkt(),
            method,
        )
        return dst_obj

    def fill_gaps(self, mask, src_array: np.ndarray) -> np.ndarray:
        """fill_gaps.

        Parameters
        ----------
        mask: [np.ndarray]
        src_array: [np.ndarray]

        Returns
        -------
        np.ndarray
        """
        # align function only equate the no of rows and columns only
        # match nodatavalue inserts nodatavalue in src raster to all places like mask
        # still places that has nodatavalue in the src raster, but it is not nodatavalue in the mask
        # and now has to be filled with values
        # compare no of element that is not nodatavalue in both rasters to make sure they are matched
        # if both inputs are rasters
        mask_array = mask.read_array()
        row = mask.rows
        col = mask.columns
        mask_noval = mask.no_data_value[0]

        if isinstance(mask, Dataset) and isinstance(self, Dataset):
            # there might be cells that are out of domain in the src but not out of domain in the mask
            # so change all the src_noval to mask_noval in the src_array
            # src_array[np.isclose(src_array, self.no_data_value[0], rtol=0.001)] = mask_noval
            # then count them (out of domain cells) in the src_array
            elem_src = src_array.size - np.count_nonzero(
                (src_array[np.isclose(src_array, self.no_data_value[0], rtol=0.001)])
            )
            # count the out of domain cells in the mask
            elem_mask = mask_array.size - np.count_nonzero(
                (mask_array[np.isclose(mask_array, mask_noval, rtol=0.001)])
            )

            # if not equal, then store indices of those cells that don't match
            if elem_mask > elem_src:
                rows = [
                    i
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], self.no_data_value[0], rtol=0.001)
                    and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
                cols = [
                    j
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], self.no_data_value[0], rtol=0.001)
                    and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
            # interpolate those missing cells by the nearest neighbour
            if elem_mask > elem_src:
                src_array = Dataset._nearest_neighbour(
                    src_array, self.no_data_value[0], rows, cols
                )
            return src_array

    def _crop_aligned(
        self,
        mask: Union[gdal.Dataset, np.ndarray],
        mask_noval: Union[int, float] = None,
        fill_gaps: bool = False,
    ) -> "Dataset":
        """_crop_aligned.

        _crop_aligned clip/crop (matches the location of nodata value from mask to src
        raster),
            - Both rasters have to have the same dimensions (no of rows & columns)
            so MatchRasterAlignment should be used prior to this function to align both
            rasters

        Parameters
        ----------
        mask: [Dataset/np.ndarray]
            mask raster to get the location of the NoDataValue and
            where it is in the array
        mask_noval: [numeric]
            in case the mask is np.ndarray, the mask_noval have to be given.
        fill_gaps: [bool]
            Default is False.

        Returns
        -------
        dst: [gdal.dataset]
            the second raster with NoDataValue stored in its cells
            exactly the same like src raster
        """
        if isinstance(mask, Dataset):
            mask_gt = mask.geotransform
            mask_epsg = mask.epsg
            row = mask.rows
            col = mask.columns
            mask_noval = mask.no_data_value[0]
            mask_array = mask.read_array()
        elif isinstance(mask, np.ndarray):
            if mask_noval is None:
                raise ValueError(
                    "You have to enter the value of the no_val parameter when the mask is a numpy array"
                )
            mask_array = mask.copy()
            row, col = mask.shape
        else:
            raise TypeError(
                "The second parameter 'mask' has to be either gdal.Datacube or numpy array"
                f"given - {type(mask)}"
            )

        band_count = self.band_count
        src_sref = osr.SpatialReference(wkt=self.crs)
        src_array = self.read_array()

        if not row == self.rows or not col == self.columns:
            raise ValueError(
                "Two rasters have different number of columns or rows, please resample or match both rasters"
            )

        if isinstance(mask, Dataset):
            if (
                not self.pivot_point == mask.pivot_point
                or not self.cell_size == mask.cell_size
            ):
                raise ValueError(
                    "the location of the upper left corner of both rasters is not the same or cell size is "
                    "different please match both rasters first "
                )

            if not mask_epsg == self.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate systems please reproject one of them to "
                    "the other raster coordinate system"
                )

        if band_count > 1:
            # check if the no data value for the src complies with the dtype of the src as sometimes the band is full
            # of values and the no_data_value is not used at all in the band, and when we try to replace any value in
            # the array with the no_data_value it will raise an error.
            no_data_value = self._check_no_data_value(self.no_data_value)

            for band in range(self.band_count):
                if mask_noval is None:
                    src_array[band, np.isnan(mask_array)] = self.no_data_value[band]
                else:
                    src_array[band, np.isclose(mask_array, mask_noval, rtol=0.001)] = (
                        no_data_value[band]
                    )
        else:
            if mask_noval is None:
                src_array[np.isnan(mask_array)] = self.no_data_value[0]
            else:
                src_array[np.isclose(mask_array, mask_noval, rtol=0.001)] = (
                    self.no_data_value[0]
                )

        if fill_gaps:
            src_array = self.fill_gaps(mask, src_array)

        dst = Dataset._create_dataset(
            col, row, band_count, self.gdal_dtype[0], driver="MEM"
        )
        # but with a lot of computations,
        # if the mask is an array and the mask_gt is not defined, use the src_gt as both the mask and the src
        # are aligned, so they have the sam gt
        try:
            # set the geotransform
            dst.SetGeoTransform(mask_gt)
            # set the projection
            dst.SetProjection(mask.crs)
        except UnboundLocalError:
            dst.SetGeoTransform(self.geotransform)
            dst.SetProjection(src_sref.ExportToWkt())

        dst_obj = Dataset(dst)
        # set the no data value
        dst_obj._set_no_data_value(self.no_data_value)
        if band_count > 1:
            for band in range(band_count):
                dst_obj.raster.GetRasterBand(band + 1).WriteArray(src_array[band, :, :])
        else:
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)
        return dst_obj

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster."""
        if not isinstance(mask, Dataset):
            raise TypeError("The second parameter should be a Dataset")

        return self.rows == mask.rows and self.columns == mask.columns

    def align(
        self,
        alignment_src,
    ) -> "Dataset":
        """align.

        align method copies the following data
            - The coordinate system
            - The number of rows & columns
            - cell size
        from alignment_src to the raster (the source of data values in cells)

        the result will be a raster with the same structure as alignment_src but with
        values from data_src using the Nearest Neighbour interpolation algorithm

        Parameters
        ----------
        alignment_src : [Dataset]
            spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns)
            data values source raster to get the data (values of each cell)

        Returns
        -------
        dst : [Dataset]
            Dataset object

        Examples
        --------
        >>> A = gdal.Open("examples/GIS/data/acc4000.tif")
        >>> B = gdal.Open("examples/GIS/data/soil_raster.tif")
        >>> RasterBMatched = Dataset.align(A,B)
        """
        if isinstance(alignment_src, Dataset):
            src = alignment_src
        else:
            raise TypeError(
                "First parameter should be a Dataset read using Dataset.openRaster or a path to the raster, "
                f"given {type(alignment_src)}"
            )

        # reproject the raster to match the projection of alignment_src
        if not self.epsg == src.epsg:
            reprojected_RasterB = self.to_crs(src.epsg)
        else:
            reprojected_RasterB = self
        # create a new raster
        dst = Dataset._create_dataset(
            src.columns, src.rows, self.band_count, src.gdal_dtype[0], driver="MEM"
        )
        # set the geotransform
        dst.SetGeoTransform(src.geotransform)
        # set the projection
        dst.SetProjection(src.crs)
        # set the no data value
        dst_obj = Dataset(dst)
        dst_obj._set_no_data_value(self.no_data_value)
        # perform the projection & resampling
        method = gdal.GRA_NearestNeighbour
        # resample the reprojected_RasterB
        gdal.ReprojectImage(
            reprojected_RasterB.raster,
            dst_obj.raster,
            src.crs,
            src.crs,
            method,
        )

        return dst_obj

    def _crop_with_raster(
        self,
        mask: Union[gdal.Dataset, str],
    ) -> "Dataset":
        """crop.

            crop method crops a raster using another raster.

        Parameters
        -----------
        string/Dataset:
            The raster you want to use as a mask to crop another raster,
            the mask can be also a path or a gdal object.

        Returns
        -------
        Dataset:
            The cropped raster.
        """
        # get information from the mask raster
        if isinstance(mask, str):
            mask = Dataset.read_file(mask)
        elif isinstance(mask, Dataset):
            mask = mask
        else:
            raise TypeError(
                "The second parameter has to be either path to the mask raster or a gdal.Datacube object"
            )
        if not self._check_alignment(mask):
            # first align the mask with the src raster
            mask = mask.align(self)
        # crop the src raster with the aligned mask
        dst_obj = self._crop_aligned(mask)

        return dst_obj

    def _crop_with_polygon_by_rasterizing(self, poly: GeoDataFrame) -> "Dataset":
        """cropWithPolygon.

            Clip the Raster object using a polygon vector.

        Parameters
        ----------
        poly: [Polygon GeoDataFrame]
            GeodataFrame with a geometry of polygon type.

        Returns
        -------
        Dataset
        """
        if not isinstance(poly, GeoDataFrame):
            raise TypeError(
                "The second parameter: poly should be of type GeoDataFrame "
            )

        poly_epsg = poly.crs.to_epsg()
        src_epsg = self.epsg
        if poly_epsg != src_epsg:
            raise ValueError(
                "Projection Error: the raster and vector polygon have different projection please "
                "unify projection"
            )
        vector = FeatureCollection(poly)
        mask = vector.to_dataset(dataset=self)
        cropped_obj = self._crop_with_raster(mask)

        return cropped_obj

    def _crop_with_polygon_warp(
        self, feature: Union[FeatureCollection, GeoDataFrame], touch: bool = True
    ) -> "Dataset":
        """Crop raster with polygon.

            - do not convert the polygon into a raster but rather use it directly to crop the raster using the
            gdal.warp function.

        Parameters
        ----------
        feature: [FeatureCollection]
                vector mask.
        touch: [bool]
            To include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
            Default is True.

        Returns
        -------
        Dataset Object.
        """
        if isinstance(feature, GeoDataFrame):
            feature = FeatureCollection(feature)
        else:
            if not isinstance(feature, FeatureCollection):
                raise TypeError(
                    f"The function takes only a FeatureCollection or GeoDataFrame, given {type(feature)}"
                )

        feature = feature._gdf_to_ds()
        warp_options = gdal.WarpOptions(
            format="VRT",
            # outputBounds=feature.total_bounds,
            cropToCutline=not touch,
            cutlineDSName=feature.file_name,
            # cutlineLayer=feature.layer_names[0],
            multithread=True,
        )
        dst = gdal.Warp("", self.raster, options=warp_options)
        dst_obj = Dataset(dst)

        if touch:
            dst_obj = Dataset.correct_wrap_cutline_error(dst_obj)

        return dst_obj

    @staticmethod
    def correct_wrap_cutline_error(src: "Dataset"):
        """Correct wrap cutline error.

        https://github.com/Serapieum-of-alex/pyramids/issues/74
        """
        big_array = src.read_array()
        value_to_remove = src.no_data_value[0]
        """Remove rows and columns that are all filled with a certain value from a 2D array."""
        # Find rows and columns to be removed
        if big_array.ndim == 2:
            rows_to_remove = np.all(big_array == value_to_remove, axis=1)
            cols_to_remove = np.all(big_array == value_to_remove, axis=0)
            # Use boolean indexing to remove rows and columns
            small_array = big_array[~rows_to_remove][:, ~cols_to_remove]
        elif big_array.ndim == 3:
            rows_to_remove = np.all(big_array == value_to_remove, axis=(0, 2))
            cols_to_remove = np.all(big_array == value_to_remove, axis=(0, 1))
            # Use boolean indexing to remove rows and columns
            small_array = big_array[:, ~rows_to_remove, ~cols_to_remove]
            n_rows = np.count_nonzero(~rows_to_remove)
            n_cols = np.count_nonzero(~cols_to_remove)
            small_array = small_array.reshape((src.band_count, n_rows, n_cols))
        else:
            raise ValueError("Array must be 2D or 3D")

        x_ind = np.where(~rows_to_remove)[0][0]
        y_ind = np.where(~cols_to_remove)[0][0]
        new_x = src.x[y_ind] - src.cell_size / 2
        new_y = src.y[x_ind] + src.cell_size / 2
        new_gt = (new_x, src.cell_size, 0, new_y, 0, -src.cell_size)
        new_src = src.create_from_array(
            small_array, new_gt, epsg=src.epsg, no_data_value=src.no_data_value
        )
        return new_src

    def crop(
        self,
        mask: Union[GeoDataFrame, FeatureCollection],
        touch: bool = True,
        inplace: bool = False,
    ) -> Union["Dataset", None]:
        """crop.

            Crop/Clip the Dataset object using a polygon/raster.

        Parameters
        ----------
        mask: [Polygon GeoDataFrame/Dataset]
            GeodataFrame with a polygon geometry, or a Dataset object.
        touch: [bool]
            To include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
            Default is True.
        inplace: [bool]
            True to make the changes in place.

        Returns
        -------
        Dataset Object
        """
        if isinstance(mask, GeoDataFrame):
            # dst = self._crop_with_polygon_by_rasterizing(mask)
            dst = self._crop_with_polygon_warp(mask, touch=touch)
        elif isinstance(mask, Dataset):
            dst = self._crop_with_raster(mask)
        else:
            raise TypeError(
                "The second parameter: mask could be either GeoDataFrame or Dataset object"
            )

        if inplace:
            self.__init__(dst.raster)
        else:
            return dst

    @staticmethod
    def _nearest_neighbour(
        array: np.ndarray, nodatavalue: Union[float, int], rows: list, cols: list
    ) -> np.ndarray:
        """_nearest_neighbour.

            - The _nearest_neighbour method fills the cells with a given indices in rows and cols with the value of the
            nearest neighbour.
            - Ss the raster grid is square, so the 4 perpendicular directions are of the same proximity, so the function
            gives priority to the right, left, bottom, and then top and the same for 45 degree inclined direction
            right bottom then left bottom then left Top then right Top.

        Parameters
        ----------
        array: [numpy.array]
            Array to fill some of its cells with the Nearest value.
        nodatavalue: [float32]
            value stored in cells that is out of the domain
        rows: [List]
            list of the rows' index of the cells you want to fill it with the nearest neighbour.
        cols: [List]
            list of the column index of the cells you want to fill it with the nearest neighbour.

        Returns
        -------
        array: [numpy array]
            Cells of given indices will be filled with the value of the Nearest neighbour

        Examples
        --------
        >>> raster = gdal.Open("dem.tif")
        >>> req_rows = [3,12]
        >>> req_cols = [9,2]
        >>> new_array = Dataset._nearest_neighbour(raster, req_rows, req_cols)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "src should be read using gdal (gdal dataset please read it using gdal library) "
            )
        if not isinstance(rows, list):
            raise TypeError("rows input has to be of type list")
        if not isinstance(cols, list):
            raise TypeError("cols input has to be of type list")

        no_rows = np.shape(array)[0]
        no_cols = np.shape(array)[1]

        for i in range(len(rows)):
            # give the cell the value of the cell that is at the right
            if array[rows[i], cols[i] + 1] != nodatavalue and cols[i] + 1 <= no_cols:
                array[rows[i], cols[i]] = array[rows[i], cols[i] + 1]

            elif array[rows[i], cols[i] - 1] != nodatavalue and cols[i] - 1 > 0:
                # give the cell the value of the cell that is at the left
                array[rows[i], cols[i]] = array[rows[i], cols[i] - 1]

            elif array[rows[i] - 1, cols[i]] != nodatavalue and rows[i] - 1 > 0:
                # give the cell the value of the cell that is at the bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i]]

            elif array[rows[i] + 1, cols[i]] != nodatavalue and rows[i] + 1 <= no_rows:
                # give the cell the value of the cell that is at the Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i]]

            elif (
                array[rows[i] - 1, cols[i] + 1] != nodatavalue
                and rows[i] - 1 > 0
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] + 1]

            elif (
                array[rows[i] - 1, cols[i] - 1] != nodatavalue
                and rows[i] - 1 > 0
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] - 1] != nodatavalue
                and rows[i] + 1 <= no_rows
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] + 1] != nodatavalue
                and rows[i] + 1 <= no_rows
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] + 1]
            else:
                print("the cell is isolated (No surrounding cells exist)")
        return array

    def map_to_array_coordinates(
        self,
        points: Union[GeoDataFrame, FeatureCollection, DataFrame],
    ) -> np.ndarray:
        """map_to_array_coordinates.

            - map_to_array_coordinates locates a point with real coordinates (x, y) or (lon, lat) on the array by
            finding the cell indices (rows, col) of the nearest cell in the raster.
            - The point coordinate system of the raster has to be projected to be able to calculate the distance

        Parameters
        ----------
        points: [GeoDataFrame/Dataframe/FeatureCollection]
            - GeoDataFrame:
                GeoDataFrame with POINT geometry.
            - DataFrame:
                DataFrame with x, y columns.

        Returns
        -------
        array:
            array with a shape (any, 2), for the row, column indices in the array.
            array([[ 5,  4],
                   [ 2,  9],
                   [ 5,  9]])
        """
        if isinstance(points, GeoDataFrame):
            points = FeatureCollection(points)
        elif isinstance(points, DataFrame):
            if all(elem not in points.columns for elem in ["x", "y"]):
                raise ValueError(
                    "If the input is a DataFrame, it should have two columns x, and y"
                )
        else:
            if not isinstance(points, FeatureCollection):
                raise TypeError(
                    "please check points input it should be GeoDataFrame/DataFrame/FeatureCollection - given"
                    f" {type(points)}"
                )
        if not isinstance(points, DataFrame):
            # get the x, y coordinates.
            points.xy()
            points = points.feature.loc[:, ["x", "y"]].values
        else:
            points = points.loc[:, ["x", "y"]].values

        # since the first row is x-coords so the first column in the indices is the column index
        indices = locate_values(points, self.x, self.y)
        # rearrange the columns to make the row index first
        indices = indices[:, [1, 0]]
        return indices

    @staticmethod
    def array_to_map_coordinates(
        top_left_x: Number,
        top_left_y: Number,
        cell_size: Number,
        column_index: Union[List[Number], np.ndarray],
        rows_index: Union[List[Number], np.ndarray],
        center: bool = False,
    ) -> Tuple[List[Number], List[Number]]:
        """Array indexes to map coordinates.

            - array_to_map_coordinates converts the array indices (rows, cols) to real coordinates (x, y) or (lon, lat)

        Parameters
        ----------
        top_left_x: [Number]
            the x coordinate of the top left corner of the raster.
        top_left_y: [Number]
            the y coordinate of the top left corner of the raster.
        cell_size: [Number]
            the cell size of the raster.
        column_index: [Union[List[Number], np.ndarray]]
            the column index of the cells in the raster array.
        rows_index: [Union[List[Number], np.ndarray]]
            the row index of the cells in the raster array.
        center: [bool]
            if True, the coordinates will be the center of the cell. Default is False.

        Returns
        -------
        x_coords: [List[Number]]
            the x coordinates of the cells.
        y_coords: [List[Number]]
            the y coordinates of the cells.
        """
        if center:
            # for the top left corner of the cell
            top_left_x = top_left_x + cell_size / 2
            top_left_y = top_left_y - cell_size / 2

        x_coord_fn = lambda x: top_left_x + x * cell_size
        y_coord_fn = lambda y: top_left_y - y * cell_size

        x_coords = list(map(x_coord_fn, column_index))
        y_coords = list(map(y_coord_fn, rows_index))

        return x_coords, y_coords

    def extract(
        self,
        exclude_value: Any = None,
        feature: Union[FeatureCollection, GeoDataFrame] = None,
    ) -> np.ndarray:
        """Extract.

            - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
            - If the feature parameter is given, the raster will be clipped to the extent of the given feature and the
            values within the feature are extracted.

        Parameters
        ----------
        exclude_value: [Numeric]
            values you want to exclude from extracted values
        feature: [FeatureCollection/GeoDataFrame]
            vector file contains geometries you want to extract the values at their location. Default is None.
        """
        # Optimize: make the read_array return only the array for inside the mask feature, and not to read the whole
        #  raster
        arr = self.read_array()
        no_data_value = (
            self.no_data_value[0] if self.no_data_value[0] is not None else np.nan
        )
        if feature is None:
            mask = (
                [no_data_value, exclude_value]
                if exclude_value is not None
                else [no_data_value]
            )
            values = get_pixels2(arr, mask)
        else:
            indices = self.map_to_array_coordinates(feature)
            values = arr[indices[:, 0], indices[:, 1]]
        return values

    def overlay(
        self,
        classes_map,
        band: int = 0,
        exclude_value: Union[float, int] = None,
    ) -> Dict[List[float], List[float]]:
        """Overlay.

            overlay extracts all the values in raster file if you have two maps one with classes, and the other map
            contains any type of values, and you want to know the values in each class.

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object for the raster that have classes you want to overlay with the raster.
        band: [int]
            if the raster is multi-band raster choose the band you want to overlay with the classes map. Default is 0.
        exclude_value: [Numeric]
            values you want to exclude from extracted values.

        Returns
        -------
        Dictionary:
            dictionary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path.
        """
        if not self._check_alignment(classes_map):
            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, please use the method "
                "'align' to align both rasters."
            )
        arr = self.read_array(band=band)
        no_data_value = (
            self.no_data_value[0] if self.no_data_value[0] is not None else np.nan
        )
        mask = (
            [no_data_value, exclude_value]
            if exclude_value is not None
            else [no_data_value]
        )
        ind = get_indices2(arr, mask)
        classes = classes_map.read_array()
        values = dict()

        # extract values
        for i, ind_i in enumerate(ind):
            # first check if the sub-basin has a list in the dict if not create a list
            key = classes[ind_i[0], ind_i[1]]
            if key not in list(values.keys()):
                values[key] = list()

            values[key].append(arr[ind_i[0], ind_i[1]])

        return values

    def get_mask(self, band: int = 0) -> np.ndarray:
        """get_mask.

        Parameters
        ----------
        band: [int]
            band index. Default is 0.

        Returns
        -------
        np.ndarray:
            array of the mask. 0 value for cells out of the domain, and 255 for cells in the domain.
        """
        # TODO: there is a CreateMaskBand method in the gdal.Dataset class, it creates a mask band for the dataset
        #   either internally or externally.
        arr = self._iloc(band).GetMaskBand().ReadAsArray()
        return arr

    def footprint(
        self,
        band: int = 0,
        exclude_values: Optional[List[Any]] = None,
    ) -> Union[GeoDataFrame, None]:
        """footprint.

            extract_extent takes a gdal raster object and returns

        Parameters
        ----------
        band: [int]
            band index. Default is 0.
        exclude_values:
            if you want to exclude_values a certain value in the raster with another value inter the two values as a
            list of tuples a [(value_to_be_exclude_valuesd, new_value)]
            >>> exclude_values = [0]
            - This parameter is introduced particularly for the case of rasters that has the nodatavalue stored in the
            array does not match the value stored in array, so this option can correct this behavior.

        Returns
        -------
        GeoDataFrame:
            - geodataframe containing the polygon representing the extent of the raster. the extent column should
            contain a value of 2 only.
            - if the dataset had separate polygons, each polygon will be in a separate row.
        """
        arr = self.read_array(band=band)
        no_data_val = self.no_data_value[band]

        if no_data_val is None:
            if not (np.isnan(arr)).any():
                logger.warning(
                    "The nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the nodatavalue stored in the raster is"
                    " not correct"
                )
        else:
            if not (np.isclose(arr, no_data_val, rtol=0.00001)).any():
                logger.warning(
                    "the nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the nodatavalue stored in the raster is"
                    " not correct"
                )
        # if you want to exclude_values any value in the raster
        if exclude_values:
            for val in exclude_values:
                try:
                    # in case the val2 is None, and the array is int type, the following line will give error as None
                    # is considered as float
                    arr[np.isclose(arr, val)] = no_data_val
                except TypeError:
                    arr = arr.astype(np.float32)
                    arr[np.isclose(arr, val)] = no_data_val

        # replace all the values with 2
        if no_data_val is None:
            # check if the whole raster is full of no_data_value
            if (np.isnan(arr)).all():
                logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isnan(arr)] = 2
        else:
            # check if the whole raster is full of nodatavalue
            if (np.isclose(arr, no_data_val, rtol=0.00001)).all():
                logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isclose(arr, no_data_val, rtol=0.00001)] = 2
        new_dataset = self.create_from_array(
            arr, self.geotransform, epsg=self.epsg, no_data_value=self.no_data_value
        )
        # then convert the raster into polygon
        gdf = new_dataset.cluster2(band=band)
        gdf.rename(columns={"Band_1": self.band_names[band]}, inplace=True)

        return gdf

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        """Normalize.

        Normalizes numpy arrays into scale 0.0 - 1.0

        Parameters
        ----------
        array : [array]
            numpy array

        Returns
        -------
        array
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return val

    def _window(self, size: int = 256):
        """Dataset square window size/offsets.

        Parameters
        ----------
        size : [int]
            Size of window in pixels. One value required which is used for both the
            x and y size. e.g 256 means a 256x256 window.

        Yields
        ------
        tuple[int]
            4 element tuple containing the x offset, y offset, x size and y size  of the window.
            >>> dataset = Dataset.read_file("examples/GIS/data/acc4000.tif")
            >>> tile_dimensions = list(dataset._window(6))
            >>> print(tile_dimensions)
            >>> [
            >>>     (0, 0, 6, 6), (0, 6, 6, 6), (0, 12, 6, 1),
            >>>     (6, 0, 6, 6), (6, 6, 6, 6), (6, 12, 6, 1),
            >>>     (12, 0, 2, 6), (12, 6, 2, 6), (12, 12, 2, 1)
            >>> ]
        """
        cols = self.columns
        rows = self.rows
        for xoff in range(0, cols, size):
            xsize = size if size + xoff <= cols else cols - xoff
            for yoff in range(0, rows, size):
                ysize = size if size + yoff <= rows else rows - yoff
                yield xoff, yoff, xsize, ysize

    def get_tile(self, size=256) -> Generator[np.ndarray, None, None]:
        """Get tile.

        Parameters
        ----------
        size : int
            Size of the window in pixels. One value required which is used for both the
            x and y size. e.g., 256 means a 256x256 window.

        Yields
        ------
        np.ndarray
            Dataset array in form [band][y][x].
        """
        for xoff, yoff, xsize, ysize in self._window(size=size):
            # read the array at a certain indices
            yield self.raster.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )

    @staticmethod
    def _group_neighbours(
        array, i, j, lowervalue, uppervalue, position, values, count, cluster
    ):
        """Group neighbouring cells with the same values."""
        # bottom cell
        if (
            lowervalue <= array[i + 1, j] <= uppervalue
            and cluster[i + 1, j] == 0
            and i + 1 < array.shape[0]
        ):
            position.append([i + 1, j])
            values.append(array[i + 1, j])
            cluster[i + 1, j] = count
            Dataset._group_neighbours(
                array,
                i + 1,
                j,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # bottom right
        if (
            j + 1 < array.shape[1]
            and i + 1 < array.shape[0]
            and lowervalue <= array[i + 1, j + 1] <= uppervalue
            and cluster[i + 1, j + 1] == 0
        ):
            position.append([i + 1, j + 1])
            values.append(array[i + 1, j + 1])
            cluster[i + 1, j + 1] = count
            Dataset._group_neighbours(
                array,
                i + 1,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # right
        if (
            j + 1 < array.shape[1]
            and lowervalue <= array[i, j + 1] <= uppervalue
            and cluster[i, j + 1] == 0
        ):
            position.append([i, j + 1])
            values.append(array[i, j + 1])
            cluster[i, j + 1] = count
            Dataset._group_neighbours(
                array,
                i,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # upper right
        if (
            j + 1 < array.shape[1]
            and i - 1 >= 0
            and lowervalue <= array[i - 1, j + 1] <= uppervalue
            and cluster[i - 1, j + 1] == 0
        ):
            position.append([i - 1, j + 1])
            values.append(array[i - 1, j + 1])
            cluster[i - 1, j + 1] = count
            Dataset._group_neighbours(
                array,
                i - 1,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # top
        if (
            i - 1 >= 0
            and lowervalue <= array[i - 1, j] <= uppervalue
            and cluster[i - 1, j] == 0
        ):
            position.append([i - 1, j])
            values.append(array[i - 1, j])
            cluster[i - 1, j] = count
            Dataset._group_neighbours(
                array,
                i - 1,
                j,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # top left
        if (
            i - 1 >= 0
            and j - 1 >= 0
            and lowervalue <= array[i - 1, j - 1] <= uppervalue
            and cluster[i - 1, j - 1] == 0
        ):
            position.append([i - 1, j - 1])
            values.append(array[i - 1, j - 1])
            cluster[i - 1, j - 1] = count
            Dataset._group_neighbours(
                array,
                i - 1,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # left
        if (
            j - 1 >= 0
            and lowervalue <= array[i, j - 1] <= uppervalue
            and cluster[i, j - 1] == 0
        ):
            position.append([i, j - 1])
            values.append(array[i, j - 1])
            cluster[i, j - 1] = count
            Dataset._group_neighbours(
                array,
                i,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # bottom left
        if (
            j - 1 >= 0
            and i + 1 < array.shape[0]
            and lowervalue <= array[i + 1, j - 1] <= uppervalue
            and cluster[i + 1, j - 1] == 0
        ):
            position.append([i + 1, j - 1])
            values.append(array[i + 1, j - 1])
            cluster[i + 1, j - 1] = count
            Dataset._group_neighbours(
                array,
                i + 1,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )

    def cluster(
        self, lower_bound: Any, upper_bound: Any
    ) -> Tuple[np.ndarray, int, list, list]:
        """Cluster.

            - group all the connected values between two numbers in a raster in clusters.

        Parameters
        ----------
        lower_bound : [numeric]
            lower bound of the cluster.
        upper_bound : [numeric]
            upper bound of the cluster.

        Returns
        -------
        cluster : [array]
            array contains integer numbers representing the number of the cluster.
        count : [integer]
            number of the clusters in the array.
        position : [list]
            list contains two indices [x,y] for the position of each value.
        values : [numeric]
            the values stored in each cell in the cluster.
        """
        data = self.read_array()
        position = []
        values = []
        count = 1
        cluster = np.zeros(shape=(data.shape[0], data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if lower_bound <= data[i, j] <= upper_bound and cluster[i, j] == 0:
                    self._group_neighbours(
                        data,
                        i,
                        j,
                        lower_bound,
                        upper_bound,
                        position,
                        values,
                        count,
                        cluster,
                    )
                    if cluster[i, j] == 0:
                        position.append([i, j])
                        values.append(data[i, j])
                        cluster[i, j] = count
                    count = count + 1

        return cluster, count, position, values

    def cluster2(
        self,
        band: Union[int, List[int]] = None,
    ) -> GeoDataFrame:
        """to_polygon.

            - to_polygon creates vector polygons for all connected regions of pixels in the raster sharing a common
            pixel value (group neighboring cells with the same value into one polygon).

        Parameters
        ----------
        band: [int]
            raster band index 0, 1, 2, 3,…

        Returns
        -------
        GeoDataFrame
        """
        if band is None:
            band = 0

        name = self.band_names[band]
        gdf = self._band_to_polygon(band, name)

        return gdf

    @property
    def overview_count(self) -> List[int]:
        """Number of the overviews for each band."""
        overview_number = []
        for i in range(self.band_count):
            overview_number.append(self._iloc(i).GetOverviewCount())

        return overview_number

    def create_overviews(
        self, resampling_method: str = "nearest", overview_levels: list = None
    ):
        """Create overviews for the dataset.

        Parameters
        ----------
        resampling_method : str, optional
            The resampling method used to create the overviews, by default "nearest"
            possible values are:
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE_MAGPHASE", "RMS",
                "BILINEAR".
        overview_levels : list, optional
            The overview levels, overview_levels are restricted to the typical power-of-two reduction factors.
            Default [2, 4, 8, 16, 32]

        Returns
        -------
        internal/external overviews:
            The overview (also known as pyramids) could be internal or external depending on the state you read
            the dataset with.
            - External (.ovr file):
                If the dataset is read with a`read_only=True` then the overviews' file will be created as an
                in the same directory of the dataset, with the same name of the dataset and .ovr extension.
            - Internal:
                If the dataset is read with a`read_only=False` then the overviews will be created internally in the
                dataset, and the dataset needs to be saved/flushed to save the new changes to disk.
        overview_count: [list]
            a list property attribute of the overviews for each band.
        """
        if overview_levels is None:
            overview_levels = OVERVIEW_LEVELS
        else:
            if not isinstance(overview_levels, list):
                raise TypeError("overview_levels should be a list")

            # if self.raster.HasArbitraryOverviews():
            if not all(elem in OVERVIEW_LEVELS for elem in overview_levels):
                raise ValueError(
                    "overview_levels are restricted to the typical power-of-two reduction factors "
                    "(like 2, 4, 8, 16, etc.)"
                )

        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Define the overview levels (the reduction factor).
        # e.g., 2 means the overview will be half the resolution of the original dataset.

        # Build overviews using nearest neighbor resampling
        # NEAREST is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        self.raster.BuildOverviews(resampling_method, overview_levels)

    def recreate_overviews(self, resampling_method: str = "nearest"):
        """Recreate overviews for the dataset.

        Parameters
        ----------
        resampling_method : str, optional
            The resampling method used to create the overviews, by default "nearest"
            possible values are:
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE_MAGPHASE", "RMS",
                "BILINEAR".

        Raises
        ------
        ValueError
            resampling_method should be one of {"NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS",
            "MODE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR"}
        ReadOnlyError
            If the overviews are internal and the Dataset is opened with a read only. Please read the dataset using
            read_only=False
        """
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Build overviews using nearest neighbor resampling
        # nearest is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        try:
            for i in range(self.band_count):
                band = self._iloc(i)
                for j in range(self.overview_count[i]):
                    ovr = self.get_overview(i, j)
                    # TODO: if this method takes a long time, we can use the gdal.RegenerateOverviews() method
                    #  which is faster but it does not give the option to choose the resampling method. and the
                    #  overviews has to be given to the function as a list.
                    #  overviews = [band.GetOverview(i) for i in range(band.GetOverviewCount())]
                    #  band.RegenerateOverviews(overviews) or gdal.RegenerateOverviews(overviews)
                    gdal.RegenerateOverview(band, ovr, resampling_method)
        except RuntimeError:
            raise ReadOnlyError(
                "The Dataset is opened with a read only. Please read the dataset using read_only=False"
            )

    def get_overview(self, band: int = 0, overview_index: int = 0) -> gdal.Band:
        """Get an overview of a band.

        Parameters
        ----------
        band : int, optional
            The band index, by default 0
        overview_index: [int]
            index of the overview. Default is 0.

        Returns
        -------
        gdal.Band
            gdal band object
        """
        band = self._iloc(band)
        n_views = band.GetOverviewCount()
        if n_views == 0:
            raise ValueError(
                "The band has no overviews, please use the `create_overviews` method to build the overviews"
            )

        if overview_index >= n_views:
            raise ValueError(f"overview_level should be less than {n_views}")

        # TODO:find away to create a Dataset object from the overview band and to return the Dataset object instead
        #  of the gdal band.
        return band.GetOverview(overview_index)

    def read_overview_array(
        self, band: int = None, overview_index: int = 0
    ) -> np.ndarray:
        """Read Array.

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data, If None, the data of all bands will be read. Default is None
        overview_index: [int]
            index of the overview. Default is 0.

        Returns
        -------
        array : [array]
            array with all the values in the raster.
        """
        if band is None and self.band_count > 1:
            if any(elem == 0 for elem in self.overview_count):
                raise ValueError(
                    "Some bands do not have overviews, please create overviews first"
                )
            # read the array from the first overview to get the size of the array.
            arr = self.get_overview(0, 0).ReadAsArray()
            arr = np.ones(
                (
                    self.band_count,
                    arr.shape[0],
                    arr.shape[1],
                ),
                dtype=self.numpy_dtype[0],
            )
            for i in range(self.band_count):
                arr[i, :, :] = self.get_overview(i, overview_index).ReadAsArray()
        else:
            if band is None:
                band = 0
            else:
                if band > self.band_count - 1:
                    raise ValueError(
                        f"band index should be between 0 and {self.band_count - 1}"
                    )
                if self.overview_count[band] == 0:
                    raise ValueError(
                        f"band {band} has no overviews, please create overviews first"
                    )
            arr = self.get_overview(band, overview_index).ReadAsArray()

        return arr

    @property
    def band_color(self) -> Dict[int, str]:
        """Band colors."""
        color_dict = {}
        for i in range(self.band_count):
            band_color = self._iloc(i).GetColorInterpretation()
            band_color = band_color if band_color is not None else 0
            color_dict[i] = gdal_constant_to_color_name(band_color)
        return color_dict

    @band_color.setter
    def band_color(self, values: Dict[int, str]):
        """band_color.

        Parameters
        ----------
        values: [Dict[int, str]]
            dictionary with band index as key and color name as value.
            e.g. {1: 'Red', 2: 'Green', 3: 'Blue'}
        """
        for key, val in values.items():
            if key > self.band_count:
                raise ValueError(
                    f"band index should be between 0 and {self.band_count}"
                )
            gdal_const = color_name_to_gdal_constant(val)
            self._iloc(key).SetColorInterpretation(gdal_const)

    def get_band_by_color(self, color_name: str) -> int:
        """get_band_by_color.

        Returns
        -------
        [type]
            [description]
        """
        colors = list(self.band_color.values())
        if color_name not in colors:
            band_index = None
        else:
            band_index = colors.index(color_name)
        return band_index

    # TODO: find a better way to handle the color table in accordance with attribute_table
    # and figure out how to take a color ramp and convert it to a color table.
    # use the SetColorInterpretation method to assign the color (R/G/B) to a band.
    @property
    def color_table(self, band: int = None) -> DataFrame:
        """Color table."""
        return self._get_color_table(band)

    @color_table.setter
    def color_table(self, df: DataFrame):
        """color_table.

        Parameters
        ----------
        df: [DataFrame]
            DataFrame with columns: band, values, color
            print(df)
                    band  values    color
                0    1       1  #709959
                1    1       2  #F2EEA2
                2    1       3  #F2CE85
                3    2       1  #C28C7C
                4    2       2  #D6C19C
                5    2       3  #D6C19C
        """
        if not isinstance(df, DataFrame):
            raise TypeError(f"df should be a DataFrame not {type(df)}")

        if not {"band", "values", "color"}.issubset(df.columns):
            raise ValueError(  # noqa
                "df should have the following columns: band, values, color"
            )

        self._set_color_table(df, overwrite=True)

    def _set_color_table(self, color_df: DataFrame, overwrite: bool = False):
        """_set_color_table.

        Parameters
        ----------
        color_df: [DataFrame]
            DataFrame with columns: band, values, color
            print(df)
                    band  values    color
                0    1       1  #709959
                1    1       2  #F2EEA2
                2    1       3  #F2CE85
                3    2       1  #C28C7C
                4    2       2  #D6C19C
                5    2       3  #D6C19C
        overwrite: [bool]
            True if you want to overwrite the existing color table. Default is False.
        """
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info"
            " check https://github.com/Serapieum-of-alex/cleopatra"
        )
        from cleopatra.colors import Colors

        color = Colors(color_df["color"].tolist())
        color_rgb = color.get_rgb(normalized=False)
        color_df.loc[:, ["red", "green", "blue"]] = color_rgb

        for band, df_band in color_df.groupby("band"):
            band = self.raster.GetRasterBand(band)

            if overwrite:
                color_table = gdal.ColorTable()
            else:
                color_table = band.GetColorTable()

            for i, row in df_band.iterrows():
                color_table.SetColorEntry(
                    row["values"], (row["red"], row["green"], row["blue"])
                )

            band.SetColorTable(color_table)
            # band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    def _get_color_table(self, band: int = None) -> DataFrame:
        """get_color_table.

        Parameters
        ----------
        band: [int], optional
            band index, Default is None.

        Returns
        -------
        [type]
            [description]
        """
        df = pd.DataFrame(columns=["band", "values", "red", "green", "blue", "alpha"])
        bands = range(self.band_count) if band is None else band
        for band in bands:
            color_table = self.raster.GetRasterBand(band + 1).GetRasterColorTable()
            for i in range(color_table.GetCount()):
                df.loc[i, ["red", "green", "blue", "alpha"]] = (
                    color_table.GetColorEntry(i)
                )
                df.loc[i, ["band", "values"]] = band + 1, i

        return df

    def get_histogram(
        self,
        band: int = 0,
        bins: int = 6,
        min_value: float = None,
        max_value: float = None,
        include_out_of_range: bool = False,
        approx_ok: bool = False,
    ) -> Tuple[Any, np.ndarray]:
        """get_histogram.

        Parameters
        ----------
        band: [int], optional
            band index, Default is 1.
        bins: [int], optional
            number of bins, Default is 6.
        min_value: [float], optional
            minimum value, Default is None.
        max_value: [float], optional
            maximum value, Default is None.
        include_out_of_range : bool, default=False
            if ``True``, add out-of-range values into the first and last buckets
        approx_ok : bool, default=True
            if ``True``, compute an approximate histogram by using subsampling or overviews

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            histogram values and bin edges.

        Hint
        ----
        - The value of the histogram will be stored in an xml file by the name of the raster file with the extension of
        .aux.xml, the content of the file will be like the following:

        <PAMDataset>
          <PAMRasterBand band="1">
            <Description>Band_1</Description>
            <Histograms>
              <HistItem>
                <HistMin>0</HistMin>
                <HistMax>88</HistMax>
                <BucketCount>6</BucketCount>
                <IncludeOutOfRange>0</IncludeOutOfRange>
                <Approximate>0</Approximate>
                <HistCounts>75|6|0|4|2|1</HistCounts>
              </HistItem>
            </Histograms>
          </PAMRasterBand>
        </PAMDataset>
        """
        band = self._iloc(band)
        min_val, max_val = band.ComputeRasterMinMax()
        if min_value is None:
            min_value = min_val
        if max_value is None:
            max_value = max_val

        bin_width = (max_value - min_value) / bins
        ranges = [
            (min_val + i * bin_width, min_val + (i + 1) * bin_width)
            for i in range(bins)
        ]

        hist = band.GetHistogram(
            min=min_value,
            max=max_value,
            buckets=bins,
            include_out_of_range=include_out_of_range,
            approx_ok=approx_ok,
        )
        return hist, ranges

    # def get_coverage(self, band: int = 0, polygon = GeoDataFrame) -> float:
    #     """get_coverage.
    #
    #     Parameters
    #     ----------
    #     band: [int], optional
    #         band index, Default is 1.
    #
    #     Returns
    #     -------
    #     [float]
    #         percentage of non-zero values.
    #     """
    #     # convert the polygon vertices to array indices using the map_to_array_coordinates method
    #     # then use the array indices in the GetDataCoverageStatus
    #     #1- get the extent of the polygon
    #     #2- convert the extent to array indices
    #     arr = Dataset.map_to_array_coordinates()
    #     # 3- use the array indices to get the GetDataCoverageStatus method
    #     band = self._iloc(band)
    #     sampling = 1
    #     x_off, y_off =  # the top left corner point of the polygon.
    #     flags, percent = band.GetDataCoverageStatus(x_off, y_off, x_size, y_size, sampling)
    #     return percent
