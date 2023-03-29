"""
raster contains python functions to handle raster data align them together based on a source raster, perform any
algebric operation on cell's values. gdal class: https://gdal.org/java/org/gdal/gdal/package-summary.html.
"""
import datetime as dt
from pathlib import Path

import os
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from loguru import logger
import shutil
import tempfile
import uuid
import geopandas as gpd
import numpy as np
import pandas as pd

from geopandas.geodataframe import GeoDataFrame, DataFrame
from osgeo import gdal, osr, ogr  # gdalconst,
from osgeo.osr import SpatialReference
from pyramids.utils import (
    gdal_to_ogr_dtype,
    INTERPOLATION_METHODS,
    NUMPY_GDAL_DATA_TYPES,
    gdal_to_numpy_dtype,
    numpy_to_gdal_dtype,
    create_time_conversion_func,
    Catalog,
)
from pyramids.errors import (
    ReadOnlyError,
    DatasetNoFoundError,
    NoDataValueError,
    AlignmentError,
    DriverNotExistError,
)

try:
    from osgeo_utils import gdal_merge
except ModuleNotFoundError:
    logger.warning(
        "osgeo_utils module does not exist try install pip install osgeo-utils "
    )

from pyramids.array import get_pixels, _get_indices2, _get_pixels2, locate_values
from pyramids.featurecollection import FeatureCollection
from pyramids import io

DEFAULT_NO_DATA_VALUE = -9999
CATALOG = Catalog(raster_driver=True)

# By default, the GDAL and OGR Python bindings do not raise exceptions when errors occur. Instead they return an error
# value such as None and write an error message to sys.stdout, to report errors by raising exceptions. You can enable
# this behavior in GDAL and OGR by calling the UseExceptions()
gdal.UseExceptions()
# gdal.ErrorReset()


class Dataset:
    """Dataset class contains methods to deal with rasters and netcdf files, change projection and coordinate systems."""

    default_no_data_value = DEFAULT_NO_DATA_VALUE

    def __init__(self, src: gdal.Dataset):
        if not isinstance(src, gdal.Dataset):
            raise TypeError(
                "src should be read using gdal (gdal dataset please read it using gdal"
                f" library) given {type(src)}"
            )
        self._raster = src
        self._geotransform = src.GetGeoTransform()
        self._cell_size = self.geotransform[1]
        self._meta_data = src.GetMetadata()
        self._file_name = src.GetDescription()
        # projection data
        # the epsg property returns the value of the _epsg attribute so if the projection changes in any function the
        # function should also change the value of the _epsg attribute.
        self._epsg = self._get_epsg()
        # variables and subsets
        self.subsets = src.GetSubDatasets()
        self._variables = self.get_variables()
        # array and dimensions
        self._rows = src.RasterYSize
        self._columns = src.RasterXSize
        self._band_count = src.RasterCount
        if len(self.subsets) > 0:
            self._time_stamp = self._get_time_variable()
            self._lat, self._lon = self._get_lat_lon()

        self._no_data_value = [
            src.GetRasterBand(i).GetNoDataValue() for i in range(1, self.band_count + 1)
        ]
        self._dtype = [
            src.GetRasterBand(i).DataType for i in range(1, self.band_count + 1)
        ]

        self._band_names = self.get_band_names()

    def __str__(self):
        message = f"""
            File: {self.file_name}
            Cell size: {self.cell_size}
            EPSG: {self.epsg}
            Variables: {self.variables}
            Number of Bands: {self.band_count}
            Band names: {self.band_names}
            Dimension: {self.rows * self.columns}
            Mask: {self._no_data_value[0]}
            Data type: {self.dtype[0]}
        """
        return message

    def __repr__(self):
        message = """
            File: {0}
            Cell size: {1}
            EPSG: {2}
            Variables: {3}
            Number of Bands: {4}
            Band names: {5}
            Dimension: {6} * {7}
            Mask: {8}
            Data type: {9}
            projection: {10}
            Metadata: {11}
        """.format(
            self.file_name,
            self.cell_size,
            self.epsg,
            self.variables,
            self.band_count,
            self.band_names,
            self.rows,
            self.columns,
            self._no_data_value[0],
            self.dtype[0],
            self.crs,
            self.meta_data,
        )
        return message

    @property
    def raster(self):
        return self._raster

    @raster.setter
    def raster(self, value: gdal.Dataset):
        self._raster = value

    @property
    def rows(self):
        """Number of rows in the raster array."""
        return self._rows

    @property
    def columns(self):
        """Number of columns in the raster array."""
        return self._columns

    @property
    def geotransform(self):
        """WKT projection."""
        return self._geotransform

    @property
    def pivot_point(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    def epsg(self):
        """WKT projection."""
        return self._epsg

    @property
    def lon(self):
        """Longitude coordinates"""
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
        """Latitude-coordinate"""
        if not hasattr(self, "_lat"):
            pivot_y = self.pivot_point[1]
            cell_size = self.cell_size
            y_coords = [
                pivot_y - i * cell_size - cell_size / 2 for i in range(self.columns)
            ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @property
    def x(self):
        """x-coordinate"""
        # X_coordinate = upperleft corner x + index * cell size + celsize/2
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
    def y(self):
        """y-coordinate"""
        # X_coordinate = upperleft corner x + index * cell size + celsize/2
        if not hasattr(self, "_lat"):
            pivot_y = self.pivot_point[1]
            cell_size = self.cell_size
            y_coords = [
                pivot_y - i * cell_size - cell_size / 2 for i in range(self.columns)
            ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @property
    def crs(self):
        """Coordinate reference system."""
        return self._get_crs()

    @crs.setter
    def crs(self, value):
        """Coordinate reference system."""
        self.set_crs(value)

    @property
    def cell_size(self):
        """Cell size."""
        return self._cell_size

    @property
    def band_count(self):
        """Number of bands in the raster."""
        return self._band_count

    @property
    def band_names(self):
        """Band names."""
        return self._band_names

    @property
    def variables(self):
        """Variables in the raster (resambles the variables in netcdf files.)"""
        return self._variables

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain"""
        return self._no_data_value

    @property
    def meta_data(self):
        """Meta data"""
        return self._meta_data

    @property
    def dtype(self):
        """Data Type"""
        return self._dtype

    @property
    def file_name(self):
        """file name"""
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    def time_stamp(self):
        """Time stamp"""
        return self._time_stamp

    @property
    def driver_type(self):
        """Driver Type"""
        driver_type = self.raster.GetDriver().GetDescription()
        return CATALOG.get_driver_name(driver_type)

    @classmethod
    def read_file(cls, path: str, read_only=True):
        """read file.

        Parameters
        ----------
        path : [str]
            Path of file to open(works for ascii, geotiff).
        read_only : [bool]
            File mode, set to False to open in "update" mode.

        Returns
        -------
        GDAL dataset
        """
        src = io.read_file(path, read_only)
        return cls(src)

    @classmethod
    def _create_empty_driver(
        cls, src: gdal.Dataset, path: str = None, bands: int = 1, no_data_value=None
    ):
        """Create a new empty driver from another dataset.

        Parameters
        ----------
        src : [gdal.Datacube]
            gdal dataset
        path : str
        bands : int or None
            Number of bands to create in the output raster.
        no_data_value : float or None
            No data value, if None uses the same as ``src``.

        Returns
        -------
        gdal.DataSet
        """
        bands = int(bands) if bands is not None else src.RasterCount
        # create the obhect
        src_obj = cls(src)
        # Create the driver.
        dst = src_obj._create_dataset(
            src_obj.columns, src_obj.rows, bands, src_obj.dtype[0], path=path
        )

        # Set the projection.
        dst.SetGeoTransform(src_obj.geotransform)
        dst.SetProjection(src_obj.raster.GetProjectionRef())
        dst = cls(dst)
        if no_data_value is not None:
            dst._set_no_data_value(no_data_value=float(no_data_value))

        return dst

    @classmethod
    def _create_driver_from_scratch(
        cls,
        cell_size: int,
        rows: int,
        columns: int,
        dtype: int,
        bands: int,
        top_left_coords: Tuple,
        epsg: int,
        no_data_value: Any = None,
        path: str = None,
    ):
        """Create a new empty driver from another dataset.

        Parameters
        ----------
        cell_size: [Any]
            cell size.
        rows: [int]
            number of rows.
        columns: [int]
            number of columns.
        dtype: [int]
            data type.
        bands : int or None
            Number of bands to create in the output raster.
        top_left_coords: [Tuple]
            coordinates of the top left corner point.
        epsg: [int]
            epsg number to identify the projection of the coordinates in the created raster.
        no_data_value : float or None
            No data value, if None uses the same as ``src``.
        path : [str]
            path on disk.

        Returns
        -------
        gdal.DataSet
        """
        # Create the driver.
        dst = Dataset._create_dataset(columns, rows, bands, dtype, path=path)
        geotransform = (
            top_left_coords[0],
            cell_size,
            0,
            top_left_coords[1],
            0,
            -1 * cell_size,
        )
        sr = Dataset._create_sr_from_epsg(epsg)

        # Set the projection.
        dst.SetGeoTransform(geotransform)
        dst.SetProjection(sr.ExportToWkt())
        dst = cls(dst)
        if no_data_value is not None:
            dst._set_no_data_value(no_data_value=no_data_value)

        return dst

    def read_array(self, band: int = None) -> np.ndarray:
        """Read Array

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0

        Returns
        -------
        array : [array]
            array with all the values in the raster.
        """
        if band is None and self.band_count > 1:
            arr = np.ones(
                (
                    self.band_count,
                    self.rows,
                    self.columns,
                )
            )
            for i in range(self.band_count):
                arr[i, :, :] = self._raster.GetRasterBand(i + 1).ReadAsArray()
        else:
            if band is None:
                band = 0
            else:
                if band > self.band_count - 1:
                    raise ValueError(
                        f"band index should be between 0 and {self.band_count - 1}"
                    )
            arr = self.raster.GetRasterBand(band + 1).ReadAsArray()

        return arr

    def _get_time_variable(self):
        """

        Returns
        -------

        """
        # time_vars = [(i, self.meta_data.get(i)) for i in self.meta_data.keys() if i.startswith("time")]
        # time_var_name = time_vars[0][0].split("#")[0]
        extra_dim = self.meta_data.get("NETCDF_DIM_EXTRA")
        if extra_dim is not None:
            time_var_name = extra_dim.replace("{", "").replace("}", "")
            units = self.meta_data.get(f"{time_var_name}#units")
            func = create_time_conversion_func(units)
            time_vals = self._read_variable(time_var_name)
            time_stamp = list(map(func, time_vals[0]))
        else:
            time_stamp = None
        return time_stamp

    def _get_lat_lon(self):
        lon = self._read_variable("lon")
        lat = self._read_variable("lat")
        return lat, lon

    def _read_variable(self, var: str):
        return gdal.Open(f"NETCDF:{self.file_name}:{var}").ReadAsArray()

    @staticmethod
    def _create_dataset(
        cols: int,
        rows: int,
        bands: int,
        dtype: int,
        driver: str = "MEM",
        path: str = None,
    ) -> gdal.Dataset:
        """Create GDAL driver.

            creates a driver and save it to disk and in memory if path is not given.

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
            # LZW is a lossless compression method achieve the highst compression but with lot of computation
            src = gdal.GetDriverByName(driver).Create(
                path, cols, rows, bands, dtype, ["COMPRESS=LZW"]
            )
        else:
            # for memory drivers
            driver = "MEM"
            src = gdal.GetDriverByName(driver).Create("", cols, rows, bands, dtype)
        return src

    @classmethod
    def create_dataset(
        cls,
        path: str = None,
        arr: Union[str, gdal.Dataset, np.ndarray] = "",
        geo: Union[str, tuple] = "",
        epsg: Union[str, int] = "",
        nodatavalue: Any = DEFAULT_NO_DATA_VALUE,
        bands: int = 1,
    ) -> Union[gdal.Dataset, None]:
        """create_raster.

            - create_raster method creates a raster from a given array and geotransform data
            and save the tif file if a Path is given or it will return the gdal.Datacube

        Parameters
        ----------
        path : [str], optional
            Path to save the Dataset, if '' is given a memory raster will be returned. The default is ''.
        arr : [array], optional
            numpy array. The default is ''.
        geo : [list], optional
            geotransform list [minimum lon, pixelsize, rotation, maximum lat, rotation,
                pixelsize]. The default is ''.
        nodatavalue : TYPE, optional
            DESCRIPTION. The default is -9999.
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator )
        bands: [int]
            band number.

        Returns
        -------
        dst : [gdal.Datacube/save raster to drive].
            if a path is given the created raster will be saved to drive, if not
            a gdal.Datacube will be returned.
        """
        try:
            if np.isnan(nodatavalue):
                nodatavalue = DEFAULT_NO_DATA_VALUE
        except TypeError:
            # np.isnan fails sometimes with the following error
            # TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
            if pd.isnull(nodatavalue):
                nodatavalue = DEFAULT_NO_DATA_VALUE

        if path is None:
            driver_type = "MEM"
        else:
            if not isinstance(path, str):
                raise TypeError("first parameter Path should be string")

            driver_type = "GTiff"

        cols = int(arr.shape[1])
        rows = int(arr.shape[0])
        dtype = numpy_to_gdal_dtype(arr)
        dst_ds = Dataset._create_dataset(
            cols, rows, bands, dtype, driver=driver_type, path=path
        )

        srse = Dataset._create_sr_from_epsg(epsg=epsg)
        dst_ds.SetProjection(srse.ExportToWkt())
        dst_obj = cls(dst_ds)
        dst_obj._set_no_data_value(no_data_value=nodatavalue)

        dst_obj.raster.SetGeoTransform(geo)
        dst_obj.raster.GetRasterBand(1).WriteArray(arr)

        if path is None:
            return dst_obj
        else:
            dst_ds = None
            return

    @classmethod
    def raster_like(
        cls,
        src,
        array: np.ndarray,
        driver: str = "MEM",
        path: str = None,
    ) -> Union[gdal.Dataset, None]:
        """rasterLike.

        rasterLike method creates a Geotiff raster like another input raster, new raster
        will have the same projection, coordinates or the top left corner of the original
        raster, cell size, nodata velue, and number of rows and columns
        the raster and the dem should have the same number of columns and rows

        Parameters
        ----------
        src : [gdal.dataset]
            source raster to get the spatial information
        array : [numpy array]
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
        >>> array = np.load("RAIN_5k.npy")
        >>> src = gdal.Open("DEM.tif")
        >>> name = "rain.tif"
        >>> Dataset.raster_like(src, array, driver="GTiff", path=name)
        - or create a raster in memory
        >>> array = np.load("RAIN_5k.npy")
        >>> src = gdal.Open("DEM.tif")
        >>> dst = Dataset.raster_like(src, array, driver="MEM")
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
        dst_obj = cls(dst)
        dst_obj._set_no_data_value(no_data_value=src.no_data_value[0])

        dst_obj.raster.GetRasterBand(1).WriteArray(array)
        if path is not None:
            dst_obj.raster.FlushCache()
            dst_obj = None

        return dst_obj

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
                self._epsg = FeatureCollection.get_epsg_from_Prj(crs)
            else:
                sr = Dataset._create_sr_from_epsg(epsg)
                self.raster.SetProjection(sr.ExportToWkt())
                self._epsg = epsg

    def _get_epsg(self) -> int:
        """GetEPSG.

            This function reads the projection of a GEOGCS file or tiff file

        Returns
        -------
        epsg : [integer]
            epsg number
        """
        prj = self._get_crs()
        epsg = FeatureCollection.get_epsg_from_Prj(prj)

        return epsg

    def get_variables(self):
        """

        Returns
        -------

        """
        variables = {}
        for i, var in enumerate(self.subsets):
            name = var[1].split(" ")[1]
            src = gdal.Open(self.subsets[i][0])
            variables[name] = Dataset(src)

        return variables

    def count_domain_cells(self):
        """Count cells inside the domain

        Returns
        -------
        int:
            Number of cells
        """
        # count cells inside the domain
        arr = self.raster.ReadAsArray()
        domain_count = np.size(arr[:, :]) - np.count_nonzero(
            (arr[np.isclose(arr, self.no_data_value[0], rtol=0.001)])
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

        # if epsg is None:
        #     sr.SetWellKnownGeogCS("WGS84")
        # else:
        #     try:
        #         if not sr.SetWellKnownGeogCS(epsg) == 6:
        #             sr.SetWellKnownGeogCS(epsg)
        #         else:
        #             try:
        #                 sr.ImportFromEPSG(int(epsg))
        #             except:
        #                 sr.ImportFromWkt(epsg)
        #     except:
        #         try:
        #             sr.ImportFromEPSG(int(epsg))
        #         except:
        #             sr.ImportFromWkt(epsg)
        return sr

    def get_band_names(self) -> List[str]:
        """Get band names from band meta data if exists otherwise will return idex [1,2, ...]

        Parameters
        ----------

        Returns
        -------
        list[str]
        """
        names = []
        for i in range(1, self.raster.RasterCount + 1):
            band_i = self.raster.GetRasterBand(i)

            if band_i.GetDescription():
                # Use the band_i description.
                names.append(band_i.GetDescription())
            else:
                # Check for metedata.
                band_i_name = "Band_{}".format(band_i.GetBand())
                metadata = band_i.GetDataset().GetMetadata_Dict()

                # If in metadata, return the metadata entry, else Band_N.
                if band_i_name in metadata and metadata[band_i_name]:
                    names.append(metadata[band_i_name])
                else:
                    names.append(band_i_name)

        return names

    def _set_no_data_value(
        self, no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE
    ):
        """setNoDataValue.

            - Set the no data value in a all raster bands.
            - Fills the whole raster with the no_data_value.
            - used only when creating an empty driver.

        Parameters
        ----------
        no_data_value: [numeric]
            no data value to fill the masked part of the array
        """
        if not isinstance(no_data_value, list):
            no_data_value = [no_data_value] * self.band_count
        for i, val in enumerate(self.dtype):
            if gdal_to_numpy_dtype(val).__contains__("float"):
                no_data_value[i] = float(no_data_value[i])
            elif gdal_to_numpy_dtype(val).__contains__("int"):
                no_data_value[i] = int(no_data_value[i])
            else:
                raise TypeError("NoDataValue has a complex data type")

        for band in range(self.band_count):
            try:
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
        no_dtype = str(type(no_data_value)).split("'")[1]
        potential_dtypes = [
            i for i in list(NUMPY_GDAL_DATA_TYPES.keys()) if i.__contains__(no_dtype)
        ]
        potential_dtypes = [NUMPY_GDAL_DATA_TYPES.get(i) for i in potential_dtypes]

        if not self.dtype[band_i] in potential_dtypes:
            raise NoDataValueError(
                f"The dtype of the given no_data_value{no_data_value}: {no_dtype} differs from the dtype of the "
                f"band: {gdal_to_numpy_dtype(self.dtype[band_i])}"
            )

        self.change_no_data_value_attr(band_i, no_data_value)
        # initialize the band with the nodata value instead of 0
        self.raster.GetRasterBand(band_i + 1).Fill(no_data_value)
        # update the no_data_value in the Dataset object
        self.no_data_value[band_i] = no_data_value

    def change_no_data_value_attr(self, band: int, no_data_value):
        """Change the no_data_value attribute.

            - Change only the no_data_value attribute in the gdal Datacube object.
            - Change the no_data_value in the Dataset object for the given band index.
            - The corresponding value in the array will not be changed.
            - Band index starts from 0

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
            elif str(e).__contains__(
                "in method 'Band_SetNoDataValue', argument 2 of type 'double'"
            ):
                self.raster.GetRasterBand(band + 1).SetNoDataValue(
                    np.float64(no_data_value)
                )
        self.no_data_value[band] = no_data_value

    def change_no_data_value(self, new_value: Any, old_value: Any = None):
        """change_no_data_value.

            - Set the no data value in a all raster bands.
            - Fills the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.

        Parameters
        ----------
        new_value: [numeric]
            no data value to set in the raster bands.

        old_value: [numeric]
            old no data value that are already in the raster bands.
        """
        if not isinstance(new_value, list):
            new_value = [new_value] * self.band_count

        if old_value is not None and not isinstance(old_value, list):
            old_value = [old_value] * self.band_count

        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.raster, 0)
        for band in range(self.band_count):
            arr = self.read_array(band)
            arr[np.isclose(arr, old_value, rtol=0.001)] = new_value[band]
            dst.GetRasterBand(band + 1).WriteArray(arr)

        self._raster = dst

        for band in range(self.band_count):
            self.change_no_data_value_attr(band, new_value[band])

    def get_cell_coords(
        self, location: str = "center", mask: bool = False
    ) -> np.ndarray:
        """GetCoords.

        Returns the coordinates of the cell centres inside the domain (only the cells that
        does not have nodata value)

        Parameters
        ----------
        location: [str]
            location of the coordinates "center" for the center of a cell, corner for the corner of the cell.
            the corner is the top left corner.
        mask: [bool]
            True to Execlude the cells out of the doain. Default is False.

        Returns
        -------
        coords : [np.ndarray]
            Array with a list of the coordinates to be interpolated, without the Nan
        mat_range : [np.ndarray]
            Array with all the centres of cells in the domain of the DEM
        """
        # check the location parameter
        location = location.lower()
        if location not in ["center", "corner"]:
            raise ValueError(
                "The location parameter can have one of these values: 'center', 'corner', "
                f"but the value: {location} is given."
            )

        if location == "center":
            # Adding 0.5*cell size to get the centre
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
        no_val = self.no_data_value[0]
        arr = self.read_array(band=0)
        if mask is not None and no_val not in arr:
            logger.warning(
                "The no data value does not exist in the band, so all the cells will be cosidered and the "
                "mask will not be cosidered."
            )

        if mask:
            mask = [no_val]
        else:
            mask = None
        indices = _get_indices2(arr, mask=mask)

        # execlude the no_data_values cells.
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

    def to_file(self, path: str, driver: str = "geotiff", band: int = 1) -> None:
        """Save to geotiff format.

            saveRaster saves a raster to a path

        Parameters
        ----------
        path: [string]
            a path includng the name of the raster and extention.
            >>> path = "data/cropped.tif"
        driver: [str]
            driver = "geotiff".
        band: [int]
            band index, needed only in case of ascii drivers. Default is 1.

        Examples
        --------
        >>> raster_obj = Dataset.read_file("path/to/file/***.tif")
        >>> output_path = "examples/GIS/data/save_raster_test.tif"
        >>> raster_obj.to_file(output_path)
        """
        if not isinstance(path, str):
            raise TypeError("Raster_path input should be string type")

        if not CATALOG.exists(driver):
            raise DriverNotExistError(f"The given driver: {driver} does not exist")

        driver_name = CATALOG.get_gdal_name(driver)

        if driver == "ascii":
            self._to_ascii(path, band=band)
        else:
            dst = gdal.GetDriverByName(driver_name).CreateCopy(path, self.raster, 0)
            dst = None  # Flush the dataset to disk
            # print to go around the assigned but never used pre-commit issue
            print(dst)

    def _to_ascii(self, path: str, band: int = 1) -> None:
        """write raster into ascii file.

            to_ascii reads writes the raster to disk into an ascii format.

        Parameters
        ----------
        path: [str]
            name of the ASCII file you want to convert and the name
            should include the extension ".asc"
        band: [int]
            band index.
        """
        if not isinstance(path, str):
            raise TypeError("path input should be string type")

        if os.path.exists(path):
            raise FileExistsError(
                f"There is a file with the same path you have provided: {path}"
            )

        y_lower_side = self.geotransform[3] - self.rows * self.cell_size
        # write the the ASCII file details
        File = open(path, "w")
        File.write("ncols         " + str(self.columns) + "\n")
        File.write("nrows         " + str(self.rows) + "\n")
        File.write("xllcorner     " + str(self.geotransform[0]) + "\n")
        File.write("yllcorner     " + str(y_lower_side) + "\n")
        File.write("cellsize      " + str(self.cell_size) + "\n")
        File.write("NODATA_value  " + str(self.no_data_value[band - 1]) + "\n")
        arr = self.raster.ReadAsArray()
        # write the array
        for i in range(np.shape(arr)[0]):
            File.writelines(list(map(Dataset.stringSpace, arr[i, :])))
            File.write("\n")

        File.close()

    @staticmethod
    def stringSpace(inp):
        return str(inp) + "  "

    def to_polygon(
        self,
        band: int = 1,
        col_name: Any = "id",
        path: str = None,
        driver: str = "memory",
    ) -> Union[GeoDataFrame, None]:
        """polygonize.

            RasterToPolygon takes a gdal Datacube object and group neighboring cells with the same value into one
            polygon, the resulted vector will be saved to disk as a geojson file

        Parameters
        ----------
        band: [int]
            raster band index [1,2,3,..]
        path:[str]
            path where you want to save the polygon, the path should include the extension at the end
            (i.e. path/vector_name.geojson)
        col_name:
            name of the column where the raster data will be stored.
        driver: [str]
            vector driver, for all possible drivers check https://gdal.org/drivers/vector/index.html .
            Default is "GeoJSON".

        Returns
        -------
        None
        """
        band = self.raster.GetRasterBand(band)
        srs = osr.SpatialReference(wkt=self.crs)
        if path is None:
            dst_layername = "id"
        else:
            dst_layername = path.split(".")[0].split("/")[-1]

        dst_ds = FeatureCollection.create_ds(driver, path)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
        dtype = gdal_to_ogr_dtype(self.raster)
        newField = ogr.FieldDefn(col_name, dtype)
        dst_layer.CreateField(newField)
        gdal.Polygonize(band, band, dst_layer, 0, [], callback=None)
        if path:
            dst_layer = None
            dst_ds = None
        else:
            vector = FeatureCollection(dst_ds)
            gdf = vector._ds_to_gdf()
            return gdf

    def to_geodataframe(
        self,
        vector_mask: Union[str, GeoDataFrame] = None,
        add_geometry: str = None,
        tile: bool = False,
        tile_size: int = 1500,
    ) -> Union[DataFrame, GeoDataFrame]:
        """Convert a raster to a GeoDataFrame.

            The function do the following
            - Flatten the array in each band in the raster then mask the values if a vector_mask
            file is given otherwise it will flatten all values.
            - Put the values for each band in a column in a dataframe under the name of the raster band, but if no meta
            data in the raster band exists, an index number will be used [1, 2, 3, ...]
            - The function has a add_geometry parameter with two possible values ["point", "polygon"], which you can
            specify the type of shapely geometry you want to create from each cell,
                - If point is chosen, the created point will be at the center of each cell
                - If a polygon is chosen, a square polygon will be created that covers the entire cell.

        Parameters
        ----------
        vector_mask : Optional[GeoDataFrame/str]
            GeoDataFrame for the vector_mask file path to vector_mask file. If given, it will be used to clip the raster
        add_geometry: [str]
            "Polygon", or "Point" if you want to add a polygon geometry of the cells as  column in dataframe.
            Default is None.
        tile: [bool]
            True to use tiles in extracting the values from the raster. Default is False.
        tile_size: [int]
            tile size. Default is 1500.

        Returns
        -------
        DataFrame/GeoDataFrame
            columndL:
                >>> print(gdf.columns)
                >>> Index(['Band_1', 'geometry'], dtype='object')

        the resulted geodataframe will have the band value under the name of the band (if the raster file has a metadata,
        if not, the bands will be indexed from 1 to the number of bands)
        """
        temp_dir = None

        # Get raster band names. open the dataset using gdal.Open
        band_names = self.get_band_names()

        # Create a mask from the pixels touched by the vector_mask.
        if vector_mask is not None:
            # Create a temporary directory for files.
            temp_dir = tempfile.mkdtemp()
            new_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}")

            # read the vector with geopandas
            if isinstance(vector_mask, GeoDataFrame):
                vector = FeatureCollection(vector_mask)
            elif isinstance(vector_mask, FeatureCollection):
                vector = vector_mask
            else:
                raise TypeError(
                    f"The vector_mask should be of type: [GeoDataFrame/FeatureCollection], given: "
                    f"{type(vector_mask)}"
                )

            # add a unique value for each rows to use it to rasterize the vector
            vector.feature["burn_value"] = list(range(1, len(vector.feature) + 1))
            # save the new vector to disk to read it with ogr later
            vector.feature.to_file(new_vector_path, driver="GeoJSON")

            # rasterize the vector by burning the unique values as cell values.
            # rasterized_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.tif")
            rasterized_vector = vector.to_dataset(
                src=self, vector_field="burn_value"
            )  # rasterized_vector_path,
            if add_geometry:
                if add_geometry.lower() == "point":
                    coords = rasterized_vector.get_cell_points(mask=True)
                else:
                    coords = rasterized_vector.get_cell_polygons(mask=True)

            # Loop over mask values to extract pixels.
            # DataFrames of each tile.
            df_list = []
            mask_arr = rasterized_vector.raster.GetRasterBand(1).ReadAsArray()

            for arr in self.get_tile(tile_size):

                mask_dfs = []
                for mask_val in vector.feature["burn_value"].values:
                    # Extract only masked pixels.
                    flatten_masked_values = get_pixels(
                        arr, mask_arr, mask_val=mask_val
                    ).transpose()
                    fid_px = np.ones(flatten_masked_values.shape[0]) * mask_val

                    # Create a DataFrame of masked flatten_masked_values and their FID.
                    mask_df = pd.DataFrame(flatten_masked_values, columns=band_names)
                    mask_df["burn_value"] = fid_px
                    mask_dfs.append(mask_df)

                # Concat the mask DataFrames.
                mask_df = pd.concat(mask_dfs)

                # Join with pixels with vector attributes using the FID.
                df_list.append(
                    mask_df.merge(vector.feature, how="left", on="burn_value")
                )

            # Merge all the tiles.
            out_df = pd.concat(df_list)
        else:
            if tile:
                df_list = []  # DataFrames of each tile.
                for arr in Dataset.get_tile(self.raster):
                    # Assume multiband
                    idx = (1, 2)
                    if arr.ndim == 2:
                        # Handle single band rasters
                        idx = (0, 1)

                    mask_arr = np.ones((arr.shape[idx[0]], arr.shape[idx[1]]))
                    pixels = get_pixels(arr, mask_arr).transpose()
                    df_list.append(pd.DataFrame(pixels, columns=band_names))

                # Merge all the tiles.
                out_df = pd.concat(df_list)
            else:
                # Warning: not checked yet for multi bands
                arr = self.raster.ReadAsArray()
                pixels = arr.flatten()
                out_df = pd.DataFrame(pixels, columns=band_names)

            if add_geometry:
                if add_geometry.lower() == "point":
                    coords = self.get_cell_points(mask=True)
                else:
                    coords = self.get_cell_polygons(mask=True)

        out_df = out_df.drop(columns=["burn_value", "geometry"], errors="ignore")
        if add_geometry:
            out_df = gpd.GeoDataFrame(out_df, geometry=coords["geometry"])

        # TODO mask no data values.

        # Remove temporary files.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Return dropping any extra cols.
        return out_df

    def apply(self, fun, band: int = 0):
        """mapAlgebra.

        - mapAlgebra executes a mathematical operation on raster array and returns
        the result
        - The mapAlgebra executes the function only on one cell at a time.

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
            raise TypeError("second argument should be a function")

        no_data_value = self.no_data_value[band]
        src_array = self.read_array(band)
        dtype = self.dtype[band]

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
    ) -> Union[None, gdal.Dataset]:
        """Fill.

            Fill takes a raster and fill it with one value

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
        raster : [None/gdal.Datacube]
            if the raster is saved directly to the path you provided the returned value will be None, otherwise the
            returned value will be the gdal.Datacube itself.
        """
        no_data_value = self.no_data_value[0]
        src_array = self.raster.ReadAsArray()

        if no_data_value is None:
            no_data_value = np.nan

        if not np.isnan(no_data_value):
            src_array[~np.isclose(src_array, no_data_value, rtol=0.000001)] = val
        else:
            src_array[~np.isnan(src_array)] = val
        dst = Dataset.raster_like(self, src_array, driver=driver, path=path)
        return dst

    def resample(self, cell_size: Union[int, float], method: str = "nearest neibour"):
        """resampleRaster.

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
            "nearest neibour" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear

        Returns
        -------
        raster : [gdal.Datacube]
             gdal object (you can read it by ReadAsArray)
        """
        if not isinstance(method, str):
            raise TypeError(
                " please enter correct method more information see docmentation "
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

        pixel_spacing = cell_size
        # new geotransform
        new_geo = (
            self.geotransform[0],
            pixel_spacing,
            self.geotransform[2],
            self.geotransform[3],
            self.geotransform[4],
            -1 * pixel_spacing,
        )
        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))
        dtype = self.raster.GetRasterBand(1).DataType

        dst = Dataset._create_dataset(cols, rows, 1, dtype)
        # set the geotransform
        dst.SetGeoTransform(new_geo)
        # set the projection
        dst.SetProjection(sr_src.ExportToWkt())
        dst_obj = Dataset(dst)
        # set the no data value
        no_data_value = self.raster.GetRasterBand(1).GetNoDataValue()
        dst_obj._set_no_data_value(no_data_value)
        # perform the projection & resampling
        gdal.ReprojectImage(
            self.raster,
            dst_obj.raster,
            sr_src.ExportToWkt(),
            sr_src.ExportToWkt(),
            method,
        )

        return dst_obj

    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neibour",
        maintain_alighment: int = False,
    ):
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
            "nearest neibour" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear
        maintain_alighment : [bool]
            True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.

        Returns
        -------
        raster:
            gdal dataset (you can read it by ReadAsArray)

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> src = Dataset.read_file("path/raster_name.tif")
        >>> projected_raster = src.to_crs(to_epsg=3857)
        """
        if not isinstance(to_epsg, int):
            raise TypeError(
                "please enter correct integer number for to_epsg more information "
                f"https://epsg.io/, given {type(to_epsg)}"
            )
        if not isinstance(method, str):
            raise TypeError(
                "please enter correct method more information see " "docmentation "
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method does not exist, existing methods are {INTERPOLATION_METHODS.keys()}"
            )

        method = INTERPOLATION_METHODS.get(method)

        if maintain_alighment:
            dst_obj = self._reproject_with_ReprojectImage(to_epsg, method)
        else:
            dst = gdal.Warp("", self.raster, dstSRS=f"EPSG:{to_epsg}", format="VRT")
            dst_obj = Dataset(dst)

        return dst_obj

    def _reproject_with_ReprojectImage(
        self, to_epsg: int, method: str = "nearest neibour"
    ) -> object:
        src_proj = self.crs
        src_gt = self.geotransform
        src_x = self.columns
        src_y = self.rows

        src_sr = osr.SpatialReference(wkt=src_proj)
        src_epsg = self._get_epsg()

        ### distination raster
        # spatial ref
        dst_sr = self._create_sr_from_epsg(to_epsg)

        # in case the source crs is GCS and longitude is in the west hemisphere gdal
        # reads longitude from 0 to 360 and transformation factor wont work with values
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
            # transform the two points coordinates to the new crs to calculate the new cell size
            new_ys, new_xs = FeatureCollection.reproject_points(
                ys, xs, from_epsg=src_epsg, to_epsg=to_epsg, precision=6
            )
        else:
            new_xs = xs
            # new_ys = ys

        pixel_spacing = np.abs(new_xs[0] - new_xs[1])

        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))

        dtype = self.dtype[0]
        dst = Dataset._create_dataset(cols, rows, 1, dtype)

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
        no_data_value = self.raster.GetRasterBand(1).GetNoDataValue()
        dst_obj = Dataset(dst)
        dst_obj._set_no_data_value(no_data_value)
        # perform the projection & resampling
        gdal.ReprojectImage(
            self.raster,
            dst_obj.raster,
            src_sr.ExportToWkt(),
            dst_sr.ExportToWkt(),
            method,
        )
        return dst_obj

    def _crop_alligned(
        self,
        mask: Union[gdal.Dataset, np.ndarray],
        mask_noval: Union[int, float] = None,
        band: int = 1,
    ) -> Union[np.ndarray, gdal.Dataset]:
        """cropAlligned.

        cropAlligned clip/crop (matches the location of nodata value from mask to src
        raster),
            - Both rasters have to have the same dimensions (no of rows & columns)
            so MatchRasterAlignment should be used prior to this function to align both
            rasters

        Parameters
        ----------
        TODO: the oriiginal function had the ability to crop an array with a dataset object and the opposite
        src: [gdal.dataset/np.ndarray]
            raster you want to clip/store NoDataValue in its cells
            exactly the same like mask raster
        mask: [gdal.dataset/np.ndarray]
            mask raster to get the location of the NoDataValue and
            where it is in the array
        mask_noval: [numeric]
            in case the mask is np.ndarray, the mask_noval have to be given.
        band: [int]
            band index, starts from 1 to the number of bands.

        Returns
        -------
        dst: [gdal.dataset]
            the second raster with NoDataValue stored in its cells
            exactly the same like src raster
        """
        # check the mask
        if isinstance(mask, Dataset):
            mask_gt = mask.geotransform
            # mask_proj = mask.proj
            # mask_sref = osr.SpatialReference(wkt=mask_proj)
            mask_epsg = mask.epsg
            row = mask.rows
            col = mask.columns
            mask_noval = mask.no_data_value[band - 1]
            mask_array = mask.raster.ReadAsArray()
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

        # if the to be clipped object is raster
        src_noval = self.no_data_value[band - 1]
        dtype = self.dtype[band - 1]

        src_sref = osr.SpatialReference(wkt=self.crs)
        src_array = self.raster.ReadAsArray()
        # Warning: delete later the self.raster will never be an array
        if isinstance(self.raster, np.ndarray):
            # if the object to be cropped is array
            src_array = self.raster.copy()
            dtype = self.raster.dtype

        # check proj
        if not mask_array.shape == src_array.shape:
            raise ValueError(
                "Two rasters has different number of columns or rows please resample or match both rasters"
            )

        # if both inputs are rasters
        if isinstance(mask, Dataset):
            if not self.geotransform == mask_gt:
                raise ValueError(
                    "location of upper left corner of both rasters are not the same or cell size is "
                    "different please match both rasters first "
                )

            if not mask_epsg == self.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate system please reproject one of them to "
                    "the other raster coordinate system"
                )

        src_array[np.isclose(mask_array, mask_noval, rtol=0.001)] = mask_noval

        # align function only equate the no of rows and columns only
        # match nodatavalue inserts nodatavalue in src raster to all places like mask
        # still places that has nodatavalue in the src raster but it is not nodatavalue in the mask
        # and now has to be filled with values
        # compare no of element that is not nodatavalue in both rasters to make sure they are matched
        # if both inputs are rasters
        if isinstance(mask, gdal.Dataset) and isinstance(self.raster, gdal.Dataset):
            # there might be cells that are out of domain in the src but not out of domain in the mask
            # so change all the src_noval to mask_noval in the src_array
            src_array[np.isclose(src_array, src_noval, rtol=0.001)] = mask_noval
            # then count them (out of domain cells) in the src_array
            elem_src = np.size(src_array[:, :]) - np.count_nonzero(
                (src_array[np.isclose(src_array, mask_noval, rtol=0.001)])
            )
            # count the out of domian cells in the mask
            elem_mask = np.size(mask_array[:, :]) - np.count_nonzero(
                (mask_array[np.isclose(mask_array, mask_noval, rtol=0.001)])
            )

            # if not equal then store indices of those cells that doesn't matchs
            if elem_mask > elem_src:
                rows = [
                    i
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], mask_noval, rtol=0.001)
                    and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
                cols = [
                    j
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], mask_noval, rtol=0.001)
                    and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
            # interpolate those missing cells by nearest neighbour
            if elem_mask > elem_src:
                src_array = Dataset._nearest_neighbour(
                    src_array, mask_noval, rows, cols
                )

        # if the dst is a raster
        if isinstance(self.raster, gdal.Dataset):
            dst = Dataset._create_dataset(col, row, 1, dtype, driver="MEM")
            # but with lot of computation
            # if the mask is an array and the mask_gt is not defined use the src_gt as both the mask and the src
            # are aligned so they have the sam gt
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
            dst_obj._set_no_data_value(mask_noval)
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)
            return dst_obj
        else:
            return src_array

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster"""
        if not isinstance(mask, Dataset):
            raise TypeError("The second parameter should be a Dataset")

        return self.rows == mask.rows and self.columns == mask.columns

    def align(
        self,
        alignment_src,
    ) -> gdal.Dataset:
        """matchRasterAlignment.

        matchRasterAlignment method copies the following data
            - The coordinate system
            - The number of of rows & columns
            - cell size
        from alignment_src to a data_src raster (the source of data values in cells)

        the result will be a raster with the same structure like alignment_src but with
        values from data_src using Nearest Neighbour interpolation algorithm

        Parameters
        ----------
        alignment_src : [gdal.dataset/string]
            spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns)
            data values source raster to get the data (values of each cell)

        Returns
        -------
        dst : [Raster]
            Raster object

        Examples
        --------
        >>> A = gdal.Open("examples/GIS/data/acc4000.tif")
        >>> B = gdal.Open("examples/GIS/data/soil_raster.tif")
        >>> RasterBMatched = Dataset.align(A,B)
        """
        if isinstance(alignment_src, Dataset):
            src = alignment_src
        elif isinstance(alignment_src, str):
            src = Dataset.read_file(alignment_src)
        else:
            raise TypeError(
                "First parameter should be a Dataset read using Dataset.openRaster or a path to the raster, "
                f"given {type(alignment_src)}"
            )

        # reproject the raster to match the projection of alignment_src
        reprojected_RasterB = self.to_crs(src.epsg)
        # create a new raster
        dst = Dataset._create_dataset(
            src.columns, src.rows, 1, src.dtype[0], driver="MEM"
        )
        # set the geotransform
        dst.SetGeoTransform(src.geotransform)
        # set the projection
        dst.SetProjection(src.crs)
        # set the no data value
        dst_obj = Dataset(dst)
        dst_obj._set_no_data_value(src.no_data_value[0])
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
    ) -> gdal.Dataset:
        """crop.

            crop method crops a raster using another raster (both rasters does not have to be aligned).

        Parameters
        -----------
        mask : [string/Dataset]
            the raster you want to use as a mask to crop other raster,
            the mask can be also a path or a gdal object.

        Returns
        -------
        dst : [Dataset]
            the cropped raster will be returned, if the save parameter was True,
            the cropped raster will also be saved to disk in the OutputPath
            directory.
        """
        # get information from the mask raster
        if isinstance(mask, str):
            mask = Dataset.read_file(mask)
        elif isinstance(mask, Dataset):
            mask = mask
        else:
            raise TypeError(
                "Second parameter has to be either path to the mask raster or a gdal.Datacube object"
            )
        if not self._check_alignment(mask):
            # first align the mask with the src raster
            mask = mask.align(self)
        # crop the src raster with the aligned mask
        dst_obj = self._crop_alligned(mask)

        return dst_obj

    def _crop_with_polygon(self, poly: GeoDataFrame):
        """cropWithPolygon.

            clip the Raster object using a polygon vector.

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
        mask = vector.to_dataset(src=self)
        cropped_obj = self._crop_with_raster(mask)

        # xmin, ymin, xmax, ymax = poly.bounds.values.tolist()[0]
        # window = (xmin, ymax, xmax, ymin)
        # # gdal.TranslateOptions(dst, ss, projWin=window)
        # # copy the src raster
        # drv = gdal.GetDriverByName("MEM")
        # dst = drv.CreateCopy("", self.raster, 0)
        # try:
        #     gdal.Translate(dst, self.raster, projWin=window)
        # except RuntimeError:
        #     pass

        # cropped_obj = Dataset(dst)

        return cropped_obj

    def crop(self, mask: Union[GeoDataFrame]):
        """

            clip the Dataset object using a polygon/another raster (both rasters does not have to be aligned).

        Parameters
        ----------
        mask: [Polygon GeoDataFrame/Dataset object]
            GeodataFrame with a geometry of polygon type

        Returns
        -------
        Dataset Object
        """
        if isinstance(mask, GeoDataFrame):
            dst = self._crop_with_polygon(mask)
        elif isinstance(mask, Dataset):
            dst = self._crop_with_raster(mask)
        else:
            raise TypeError(
                "The second parameter: mask could be either GeoDataFrame or Dataset object"
            )

        return dst

    # def clipRasterWithPolygon(
    #     self,
    #     vector_psth: str,
    #     save: bool = False,
    #     output_path: str = None,
    # ) -> gdal.Datacube:
    #     """ClipRasterWithPolygon.
    #
    #         ClipRasterWithPolygon method clip a raster using polygon shapefile
    #
    #     Parameters
    #     ----------
    #     raster_path : [String]
    #         path to the input raster including the raster extension (.tif)
    #     vector_psth : [String]
    #         path to the input shapefile including the shapefile extension (.shp)
    #     save : [Boolen]
    #         True or False to decide whether to save the clipped raster or not
    #         default is False
    #     output_path : [String]
    #         path to the place in your drive you want to save the clipped raster
    #         including the raster name & extension (.tif), default is None
    #
    #     Returns
    #     -------
    #     projected_raster:
    #         [gdal object] clipped raster
    #     if save is True function is going to save the clipped raster to the output_path
    #
    #     Examples
    #     --------
    #     >>> src_path = r"data/Evaporation_ERA-Interim_2009.01.01.tif"
    #     >>> shp_path = "data/"+"Outline.shp"
    #     >>> clipped_raster = Dataset.clipRasterWithPolygon(raster_path,vector_psth)
    #     or
    #     >>> dst_path = r"data/cropped.tif"
    #     >>> clipped_raster = Dataset.clipRasterWithPolygon(src_path, shp_path, True, dst_path)
    #     """
    #     if isinstance(vector_psth, str):
    #         poly = gpd.read_file(vector_psth)
    #     elif isinstance(vector_psth, gpd.geodataframe.GeoDataFrame):
    #         poly = vector_psth
    #     else:
    #         raise TypeError("vector_psth input should be string type")
    #
    #     if not isinstance(save, bool):
    #         raise TypeError("save input should be bool type (True or False)")
    #
    #     if save:
    #         if not isinstance(output_path, str):
    #             raise ValueError("Pleaase enter a path to save the clipped raster")
    #     # inputs value
    #     if save:
    #         ext = output_path[-4:]
    #         if not ext == ".tif":
    #             raise TypeError(
    #                 "please add the extention at the end of the output_path input"
    #             )
    #
    #     src_epsg = self.epsg
    #     gt = self.geotransform
    #
    #     # first check if the crs is GCS if yes check whether the long is greater than 180
    #     # geopandas read -ve longitude values if location is west of the prime meridian
    #     # while rasterio and gdal not
    #     if src_epsg == "4326" and gt[0] > 180:
    #         # reproject the raster to web mercator crs
    #         raster = self.reproject()
    #         out_transformed = os.environ["Temp"] + "/transformed.tif"
    #         # save the raster with the new crs
    #         raster.to_geotiff(out_transformed)
    #         raster = rasterio.open(out_transformed)
    #         # delete the transformed raster
    #         os.remove(out_transformed)
    #     else:
    #         # crs of the raster was not GCS or longitudes less than 180
    #         if isinstance(raster_path, str):
    #             raster = rasterio.open(raster_path)
    #         else:
    #             raster = rasterio.open(raster_path.GetDescription())
    #
    #     ### Cropping the raster with the shapefile
    #     # Re-project into the same coordinate system as the raster data
    #     shpfile = poly.to_crs(crs=raster.crs.data)
    #
    #     # Get the geometry coordinates by using the function.
    #     coords = Feature.getFeatures(shpfile)
    #
    #     out_img, out_transform = rio_mask(dataset=raster, shapes=coords, crop=True)
    #
    #     # copy the metadata from the original data file.
    #     out_meta = raster.meta.copy()
    #
    #     # Next we need to parse the EPSG value from the CRS so that we can create
    #     # a Proj4 string using PyCRS library (to ensure that the projection information is saved correctly).
    #     epsg_code = int(raster.crs.data["init"][5:])
    #
    #     # close the transformed raster
    #     raster.close()
    #
    #     # Now we need to update the metadata with new dimensions, transform (affine) and CRS (as Proj4 text)
    #     out_meta.update(
    #         {
    #             "driver": "GTiff",
    #             "height": out_img.shape[1],
    #             "width": out_img.shape[2],
    #             "transform": out_transform,
    #             "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
    #         }
    #     )
    #
    #     # save the clipped raster.
    #     temp_path = os.environ["Temp"] + "/cropped.tif"
    #     with rasterio.open(temp_path, "w", **out_meta) as dest:
    #         dest.write(out_img)
    #         dest.close()
    #         dest = None
    #
    #     # read the clipped raster
    #     raster = gdal.Open(temp_path, gdal.GA_ReadOnly)
    #     # reproject the clipped raster back to its original crs
    #     projected_raster = Dataset.projectRaster(
    #         raster, int(src_epsg.GetAttrValue("AUTHORITY", 1))
    #     )
    #     raster = None
    #     # delete the clipped raster
    #     # try:
    #     # TODO: fix ClipRasterWithPolygon as it does not delete the the cropped.tif raster from the temp_path
    #     # the following line through an error
    #     os.remove(temp_path)
    #     # except:
    #     #     print(temp_path + " - could not be deleted")
    #
    #     # write the raster to the file
    #     if save:
    #         Dataset.saveRaster(projected_raster, output_path)
    #
    #     return projected_raster

    # @staticmethod
    # def clip2(
    #     src: Union[rasterio.io.DatasetReader, str],
    #     poly: Union[GeoDataFrame, str],
    #     save: bool = False,
    #     output_path: str = "masked.tif",
    # ) -> gdal.Datacube:
    #     """Clip2.
    #
    #         Clip function takes a rasterio object and clip it with a given geodataframe
    #         containing a polygon shapely object
    #
    #     Parameters
    #     ----------
    #     src : [rasterio.io.DatasetReader]
    #         the raster read by rasterio .
    #     poly : [geodataframe]
    #         geodataframe containing the polygon you want clip the raster based on.
    #     save : [Bool], optional
    #         to save the clipped raster to your drive. The default is False.
    #     output_path : [String], optional
    #         path iincluding the extention (.tif). The default is 'masked.tif'.
    #
    #     Returns
    #     -------
    #     out_img : [rasterio object]
    #         the clipped raster.
    #     metadata : [dictionay]
    #             dictionary containing number of bands, coordinate reference system crs
    #             dtype, geotransform, height and width of the raster
    #     """
    #     ### 1- Re-project the polygon into the same coordinate system as the raster data.
    #     # We can access the crs of the raster using attribute .crs.data:
    #     if isinstance(poly, str):
    #         # read the shapefile
    #         poly = gpd.read_file(poly)
    #     elif isinstance(poly, gpd.geodataframe.GeoDataFrame):
    #         poly = poly
    #     else:
    #         raise TypeError("Polygongdf input should be string type")
    #
    #     if isinstance(src, str):
    #         src = rasterio.open(src)
    #     elif isinstance(src, rasterio.io.DatasetReader):
    #         src = src
    #     else:
    #         raise TypeError("Rasterobj input should be string type")
    #
    #     # Project the Polygon into same CRS as the grid
    #     poly = poly.to_crs(crs=src.crs.data)
    #
    #     # Print crs
    #     # geo.crs
    #     ### 2- Convert the polygon into GeoJSON format for rasterio.
    #
    #     # Get the geometry coordinates by using the function.
    #     coords = [json.loads(poly.to_json())["features"][0]["geometry"]]
    #
    #     # print(coords)
    #
    #     ### 3-Clip the raster with Polygon
    #     out_img, out_transform = rasterio.mask.mask(
    #         dataset=src, shapes=coords, crop=True
    #     )
    #
    #     ### 4- update the metadata
    #     # Copy the old metadata
    #     out_meta = src.meta.copy()
    #     # print(out_meta)
    #
    #     # Next we need to parse the EPSG value from the CRS so that we can create
    #     # a Proj4 -string using PyCRS library (to ensure that the projection
    #     # information is saved correctly).
    #
    #     # Parse EPSG code
    #     epsg_code = int(src.crs.data["init"][5:])
    #     # print(epsg_code)
    #
    #     out_meta.update(
    #         {
    #             "driver": "GTiff",
    #             "height": out_img.shape[1],
    #             "width": out_img.shape[2],
    #             "transform": out_transform,
    #             "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
    #         }
    #     )
    #     if save:
    #         with rasterio.open(output_path, "w", **out_meta) as dest:
    #             dest.write(out_img)
    #
    #     return out_img, out_meta

    @staticmethod
    def _nearest_neighbour(
        array: np.ndarray, nodatavalue: Union[float, int], rows: list, cols: list
    ) -> np.ndarray:
        """nearestNeighbour.

            - fills the cells of a given indices in rows and cols with the value of the nearest
            neighbour.
            - Ss the raster grid is square so the 4 perpendicular direction are of the same proximity so the function
            gives priority to the right then left then bottom then top and the same for 45 degree inclined direction
            right bottom then left bottom then left Top then right Top.

        Parameters
        ----------
        array: [numpy.array]
            Array to fill some of its cells with Nearest value.
        nodatavalue: [float32]
            value stored in cells that is out of the domain
        rows: [List]
            list of the rows index of the cells you want to fill it with
            nearest neighbour.
        cols: [List]
            list of the column index of the cells you want to fill it with
            nearest neighbour.

        Returns
        -------
        array: [numpy array]
            Cells of given indices will be filled with value of the Nearest neighbour

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

        #    array=raster.ReadAsArray()
        #    nodatavalue=np.float32(raster.GetRasterBand(1).GetNoDataValue())

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

    def locate_points(
        self,
        points: Union[GeoDataFrame, FeatureCollection],
    ) -> np.ndarray:
        """nearestCell.

            nearestCell calculates the the indices (rows, col) of nearest cell in a given
            raster to a station
            coordinate system of the raster has to be projected to be able to calculate
            the distance

        Parameters
        ----------
        points: [Dataframe]
            dataframe with POINT geometry.

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
        else:
            if not isinstance(points, FeatureCollection):
                raise TypeError(
                    f"please check points input it should be GeoDataFrame/FeatureCollection - given {type(points)}"
                )
        # get the x, y coordinates.
        points.xy()
        points = points.feature.loc[:, ["x", "y"]].values
        grid = np.array([self.x, self.y]).transpose()
        # since first row is x-coords so the first column in the indices is the column index
        indices = locate_values(points, grid)
        # rearrange the columns to make the row index first
        indices = indices[:, [1, 0]]
        return indices

    def extract(
        self,
        exclude_value: Any = None,
        feature: Union[FeatureCollection, GeoDataFrame] = None,
    ) -> List:
        """Extract Values.

            - this function is written to extract and return a list of all the values in a map.

        Parameters
        ----------
        exclude_value: [Numeric]
            values you want to exclude from exteacted values
        feature: [FeatureCollection/GeoDataFrame]
            vectpr file contains geometries you want to extract the values at their location. Default is None.
        """
        arr = self.read_array()

        if feature is None:
            mask = (
                [self.no_data_value[0], exclude_value]
                if exclude_value is not None
                else [self.no_data_value[0]]
            )
            values = _get_pixels2(arr, mask)
        else:
            indices = self.locate_points(feature)
            values = arr[indices[:, 0], indices[:, 1]]
        return values

    def overlay(
        self,
        classes_map,
        exclude_value: Union[float, int] = None,
    ) -> Dict[List[float], List[float]]:
        """OverlayMap.

            OverlayMap extracts and return a list of all the values in an ASCII file,
            if you have two maps one with classes, and the other map contains any type of values,
            and you want to know the values in each class

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object fpr the raster that have classes you want to overlay with the raster.
        exclude_value: [Numeric]
            values you want to exclude from extracted values.

        Returns
        -------
        Dictionary:
            dictonary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path.
        """
        if not self._check_alignment(classes_map):
            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, plase use the method "
                "'align' to align both rasters."
            )
        arr = self.read_array()
        mask = (
            [self.no_data_value[0], exclude_value]
            if exclude_value is not None
            else [self.no_data_value[0]]
        )
        ind = _get_indices2(arr, mask)
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

    @staticmethod
    def normalize(array: np.ndarray):
        """
        Normalizes numpy arrays into scale 0.0 - 1.0

        Parameters
        ----------
        array : [array]
            numpy array

        Returns
        -------
        array
            DESCRIPTION.
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return val

    @staticmethod
    def _window(src: gdal.Dataset, size: int = 256):
        """Dataset square window size/offsets.

        Parameters
        ----------
        src : [gdal.Datacube]
            gdal Datacube object.
        size : [int]
            Size of window in pixels. One value required which is used for both the
            x and y size. E.g 256 means a 256x256 window.

        Yields
        ------
        tuple[int]
            4 element tuple containing the x size, y size, x offset and y offset
            of the window.
        """
        cols = src.RasterXSize
        rows = src.RasterYSize
        for xoff in range(0, cols, size):
            xsize = size if size + xoff <= cols else cols - xoff
            for yoff in range(0, rows, size):
                ysize = size if size + yoff <= rows else rows - yoff
                yield xsize, ysize, xoff, yoff

    def get_tile(self, size=256):
        """gets a raster array in tiles.

        Parameters
        ----------
        size : int
            Size of window in pixels. One value required which is used for both the
            x and y size. E.g 256 means a 256x256 window.

        Yields
        ------
        np.ndarray
            Dataset array in form [band][y][x].
        """
        for xsize, ysize, xoff, yoff in self._window(self.raster, size=size):
            # read the array at a certain indeces
            yield self.raster.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )

    def listAttributes(self):
        """Print Attributes List."""

        print("\n")
        print(
            f"Attributes List of: { repr(self.__dict__['name'])} - {self.__class__.__name__}  Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")


class Datacube:

    files: List[str]
    data: np.ndarray

    """
    files:
        list of geotiff files' names
    """

    def __init__(
        self,
        src: Dataset,
        time_length: int,
        files: List[str] = None,
    ):
        self._base = src
        self.files = files
        self._time_length = time_length

        pass

    @property
    def base(self) -> Dataset:
        """base.

        Base Dataset
        """
        return self._base

    @property
    def time_length(self) -> int:
        """length of the dataset."""
        return self._time_length

    @classmethod
    def create_cube(cls, src: Dataset, dataset_length: int):
        """Create Datacube.

            - Create Datacube from a sample raster and

        Parameters
        ----------
        src: [Dataset]
            RAster Object
        dataset_length: [int]
            length of the dataset.

        Returns
        -------
        Datacube object.
        """
        return cls(src, dataset_length)

    def update_cube(self, array: np.ndarray):
        """Update dataset data.

            - This function creates a Geotiff raster like another input raster, new raster will have the same
            projection, coordinates or the top left corner of the original raster, cell size, nodata velue, and number
            of rows and columns
            - the raster and the given array should have the same number of columns and rows.

        Parameters
        ----------
        array: [numpy array]
            3D array to be stores as a rasters, the dimensions should be
            [rows, columns, timeseries length]

        Returns
        -------
        save the new raster to the given path

        Examples
        --------
        >>> src_raster = gdal.Open("DEM.tif")
        >>> name = ["Q_2012_01_01_01.tif","Q_2012_01_01_02.tif","Q_2012_01_01_03.tif","Q_2012_01_01_04.tif"]
        >>> Datacube.rastersLike(src_raster, data, name)
        """
        self.data = array

    @classmethod
    def read_separate_files(
        cls,
        path: Union[str, List[str]],
        with_order: bool = True,
        start: str = None,
        end: str = None,
        fmt: str = None,
        freq: str = "daily",
        separator: str = "_",
        extension: str = ".tif",
    ):
        """read_separate_files.

            - reads rasters from a folder and creates a 3d array with the same 2d dimensions of the first raster in
            the folder and len as the number of files

        inside the folder.
            - All rasters should have the same dimensions
            - Folder should only contain raster files
            - raster file name should have the date at the end of the file name before the extension directly
              with the YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD
              >>> "50_MSWEP_1979.01.01.tif"

        Parameters
        ----------
        path:[String/list]
            path of the folder that contains all the rasters or
            a list contains the paths of the rasters to read.
        with_order: [bool]
            True if the rasters follows a certain order, then the rasters names should have a
            number at the beginning of the file name indicating the order.
            >>> "01_MSWEP_1979.01.01.tif"
            >>> "02_MSWEP_1979.01.02.tif"
            >>> ...
            >>> "20_MSWEP_1979.01.20.tif"
            - currently the function depends mainly on the separator "_" that separate the order number from the rest of
            file name.
            - the separator between the date parts YYYY.MM.DD ir YYYY_MM_DD or any other separator does not matter,
            however the separator has to be only one letter.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input raster for a specific period only and not all rasters,
            if not given all rasters in the given path will be read.
            Hint:
                The date in the raster file name should be the last string befor the file extension
                >>> "20_MSWEP_YYYY.MM.DD.tif"
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
            Hint:
                The date in the raster file name should be the last string befor the file extension
                >>> "20_MSWEP_YYYY.MM.DD.tif"
        freq: [str]
            frequency of the rasters "daily", Hourly, monthly
        separator: [str]
            separator between the order in the beginning of the raster file name and the rest of the file
            name. Default is "_".
        extension: [str]
            the extension of the files you want to read from the given path. Default is ".tif".

        Returns
        -------
        arr_3d: [numpy.ndarray]
            3d array contains arrays read from all rasters in the folder.

        Example
        -------
        >>> from pyramids.dataset import Datacube
        >>> raster_folder = "examples/GIS/data/raster-folder"
        >>> prec = Datacube.read_separate_files(raster_folder)

        >>> import glob
        >>> search_criteria = "*.tif"
        >>> file_list = glob.glob(os.path.join(raster_folder, search_criteria))
        >>> prec = Datacube.read_separate_files(file_list, with_order=False)
        """
        if not isinstance(path, str) and not isinstance(path, list):
            raise TypeError(f"path input should be string/list type, given{type(path)}")

        if isinstance(path, str):
            # check wether the path exist or not
            if not os.path.exists(path):
                raise FileNotFoundError("The path you have provided does not exist")
            # get list of all files
            files = os.listdir(path)
            files = [i for i in files if i.endswith(extension)]
            # files = glob.glob(os.path.join(path, "*.tif"))
            # check whether there are files or not inside the folder
            if len(files) < 1:
                raise FileNotFoundError("The path you have provided is empty")
        else:
            files = path[:]
        # to sort the files in the same order as the first number in the name
        if with_order:
            try:
                filesNo = [int(i.split(separator)[0]) for i in files]
            except ValueError:
                raise ValueError(
                    "please include a number at the beginning of the"
                    "rasters name to indicate the order of the rasters. to do so please"
                    "use the Inputs.RenameFiles method to solve this issue and don't "
                    "include any other files in the folder with the rasters"
                )

            file_tuple = sorted(zip(filesNo, files))
            files = [x for _, x in file_tuple]

        if start is not None or end is not None:
            start = dt.datetime.strptime(start, fmt)
            end = dt.datetime.strptime(end, fmt)

            # get the dates for each file
            dates = list()
            for i, file_i in enumerate(files):
                if freq == "daily":
                    l = len(file_i) - 4
                    day = int(file_i[l - 2 : l])
                    month = int(file_i[l - 5 : l - 3])
                    year = int(file_i[l - 10 : l - 6])
                    dates.append(dt.datetime(year, month, day))
                elif freq == "hourly":
                    year = int(file_i.split("_")[-4])
                    month = int(file_i.split("_")[-3])
                    day = int(file_i.split("_")[-2])
                    hour = int(file_i.split("_")[-1].split(".")[0])
                    dates.append(dt.datetime(year, month, day, hour))

            starti = dates.index(start)
            endi = dates.index(end) + 1
            assert all(
                file_i.endswith(".tif") for file_i in files[starti:endi]
            ), "all files in the given folder should have .tif extension"
        else:
            starti = 0
            endi = len(files)
            # check that folder only contains rasters
            assert all(
                file_i.endswith(extension) for file_i in files
            ), "all files in the given folder should have .tif extension"

        # files to be read
        files = files[starti:endi]
        if not isinstance(path, list):
            # add the path to all the files
            files = [f"{path}/{i}" for i in files]
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        sample = Dataset.read_file(files[0])

        return cls(sample, len(files), files)

    def read_dataset(self, band: int = 1):
        """Read array.

            Read values form the given bands as Arrays for all files

        Parameters
        ----------
        band: [int]
            number of the band you want to read default is 1.

        Returns
        -------
        Array
        """
        # check the given band number
        if not hasattr(self, "base"):
            raise ValueError(
                "please use the read_separate_files method to get the files (tiff/ascii) in the"
                "dataset directory"
            )
        if band > self.base.band_count:
            raise ValueError(
                f"the raster has only {self.base.band_count} check the given band number"
            )
        # fill the array with no_data_value data
        self._data = np.ones(
            (
                self.time_length,
                self.base.rows,
                self.base.columns,
            )
        )
        self._data[:, :, :] = np.nan

        for i, file_i in enumerate(self.files):
            # read the tif file
            raster_i = gdal.Open(f"{file_i}")
            self._data[i, :, :] = raster_i.GetRasterBand(band).ReadAsArray()

    @property
    def data(self) -> np.ndarray:
        """data attribute.

        - The attribute where the dataset array is stored.
        - the 3D numpy array, [dataset length, rows, cols], [dataset length, lons, lats]
        """
        return self._data

    @data.setter
    def data(self, val):
        """Data attribute.

        - setting the data (array) does not allow different dimension from the dimension that have been
        defined increating the dataset.
        """
        # if the attribute is defined before check the dimension
        if hasattr(self, "data"):
            if self._data.shape != val.shape:
                raise ValueError(
                    f"The dimension of the new data: {val.shape}, differs from the dimension of the "
                    f"original dataset: {self._data.shape}, please redefine the base Dataset and "
                    f"dataset_length first"
                )

        self._data = val

    @data.deleter
    def data(self):
        self._data = None

    def iloc(self, i):
        """iloc.

            - Access dataset array using index.

        Parameters
        ----------
        i: [int]
            index

        Returns
        -------
        Dataset:
            Dataset object.
        """
        if not hasattr(self, "data"):
            raise DatasetNoFoundError("please read the dataset first")
        arr = self._data[i, :, :]
        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.base.raster, 0)
        dst.GetRasterBand(1).WriteArray(arr)
        return Dataset(dst)

    def to_file(self, path: str, driver: str = "geotiff", band: int = 1):
        """Save to geotiff format.

            saveRaster saves a raster to a path

        Parameters
        ----------
        path: [string]
            a path includng the name of the raster and extention.
            >>> path = "data/cropped.tif"
        driver: [str]
            driver = "geotiff".
        band: [int]
            band index, needed only in case of ascii drivers. Default is 1.

        Examples
        --------
        >>> raster_obj = Dataset.read_file("path/to/file/***.tif")
        >>> output_path = "examples/GIS/data/save_raster_test.tif"
        >>> raster_obj.to_file(output_path)
        """
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)
        ext = CATALOG.get_extension(driver)

        for i in range(self.time_length):
            src = self.iloc(i)
            src.to_file(f"{path}/{i}.{ext}", driver=driver, band=band)

    def to_crs(
        self,
        to_epsg: int = 3857,
        method: str = "nearest neibour",
        maintain_alighment: int = False,
    ):
        """to_epsg.

            - to_epsg reprojects a raster to any projection (default the WGS84 web mercator projection,
            without resampling) The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

        Parameters
        ----------
        to_epsg: [integer]
            reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        method: [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "Nearest" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear
        maintain_alighment : [bool]
            True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.

        Returns
        -------
        raster:
            gdal dataset (you can read it by ReadAsArray)

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> src = Dataset.read_file("path/raster_name.tif")
        >>> projected_raster = src.to_crs(to_epsg=3857)
        """
        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.to_crs(
                to_epsg, method=method, maintain_alighment=maintain_alighment
            )
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (
                            self.time_length,
                            arr.shape[0],
                            arr.shape[1],
                        )
                    )
                    * np.nan
                )
            array[i, :, :] = arr

        self._data = array
        # use the last src as
        self._base = dst

    def crop(
        self,
        mask: Union[Dataset, str],
    ) -> None:
        """cropAlignedFolder.

            cropAlignedFolder matches the location of nodata value from src raster to dst
            raster, Mask is where the NoDatavalue will be taken and the location of
            this value src_dir is path to the folder where rasters exist where we
            need to put the NoDataValue of the mask in RasterB at the same locations

        Parameters
        ----------
        mask : [str/Dataset]
            path/Dataset object of the mask raster to crop the rasters (to get the NoData value
            and it location in the array) Mask should include the name of the raster and the
            extension like "data/dem.tif", or you can read the mask raster using gdal and use
            is the first parameter to the function.

        Returns
        -------
        new rasters have the values from rasters in B_input_path with the NoDataValue in the same
        locations like raster A.

        Examples
        --------
        >>> dem_path = "examples/GIS/data/acc4000.tif"
        >>> src_path = "examples/GIS/data/aligned_rasters/"
        >>> out_path = "examples/GIS/data/crop_aligned_folder/"
        >>> Datacube.crop(dem_path, src_path, out_path)
        """
        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.crop(mask)
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (
                            self.time_length,
                            arr.shape[0],
                            arr.shape[1],
                        )
                    )
                    * np.nan
                )
            array[i, :, :] = arr

        self._data = array
        # use the last src as
        self._base = dst

    # # TODO: merge ReprojectDataset and ProjectRaster they are almost the same
    # # TODO: still needs to be tested
    # @staticmethod
    # def to_epsg(
    #         src: gdal.Datacube,
    #         to_epsg: int = 3857,
    #         cell_size: int = [],
    #         method: str = "Nearest",
    #
    # ) -> gdal.Datacube:
    #     """to_epsg.
    #
    #         - to_epsg reprojects and resamples a folder of rasters to any projection
    #         (default the WGS84 web mercator projection, without resampling)
    #
    #     Parameters
    #     ----------
    #     src: [gdal dataset]
    #         gdal dataset object (src=gdal.Open("dem.tif"))
    #     to_epsg: [integer]
    #          reference number to the new projection (https://epsg.io/)
    #         (default 3857 the reference no of WGS84 web mercator )
    #     cell_size: [integer]
    #          number to resample the raster cell size to a new cell size
    #         (default empty so raster will not be resampled)
    #     method: [String]
    #         resampling technique default is "Nearest"
    #         https://gisgeography.com/raster-resampling/
    #         "Nearest" for nearest neighbour,"cubic" for cubic convolution,
    #         "bilinear" for bilinear
    #
    #     Returns
    #     -------
    #     raster: [gdal Datacube]
    #          a GDAL in-memory file object, where you can ReadAsArray etc.
    #     """
    #     if not isinstance(src, gdal.Datacube):
    #         raise TypeError(
    #             "src should be read using gdal (gdal dataset please read it using gdal"
    #             f" library) given {type(src)}"
    #         )
    #     if not isinstance(to_epsg, int):
    #         raise TypeError(
    #             "please enter correct integer number for to_epsg more information "
    #             f"https://epsg.io/, given {type(to_epsg)}"
    #         )
    #     if not isinstance(method, str):
    #         raise TypeError(
    #             "please enter correct method more information see " "docmentation "
    #         )
    #
    #     if cell_size:
    #         assert isinstance(cell_size, int) or isinstance(
    #             cell_size, float
    #         ), "please enter an integer or float cell size"
    #
    #     if method == "Nearest":
    #         method = gdal.GRA_NearestNeighbour
    #     elif method == "cubic":
    #         method = gdal.GRA_Cubic
    #     elif method == "bilinear":
    #         method = gdal.GRA_Bilinear
    #
    #     src_proj = src.GetProjection()
    #     src_gt = src.GetGeoTransform()
    #     src_x = src.RasterXSize
    #     src_y = src.RasterYSize
    #     dtype = src.GetRasterBand(1).DataType
    #     # spatial ref
    #     src_sr = osr.SpatialReference(wkt=src_proj)
    #     src_epsg = src_sr.GetAttrValue("AUTHORITY", 1)
    #
    #     # distination
    #     # spatial ref
    #     dst_epsg = osr.SpatialReference()
    #     dst_epsg.ImportFromEPSG(to_epsg)
    #     # transformation factors
    #     tx = osr.CoordinateTransformation(src_sr, dst_epsg)
    #
    #     # incase the source crs is GCS and longitude is in the west hemisphere gdal
    #     # reads longitude fron 0 to 360 and transformation factor wont work with valeus
    #     # greater than 180
    #     if src_epsg == "4326" and src_gt[0] > 180:
    #         lng_new = src_gt[0] - 360
    #         # transform the right upper corner point
    #         (ulx, uly, ulz) = tx.TransformPoint(lng_new, src_gt[3])
    #         # transform the right lower corner point
    #         (lrx, lry, lrz) = tx.TransformPoint(
    #             lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
    #         )
    #     else:
    #         # transform the right upper corner point
    #         (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
    #         # transform the right lower corner point
    #         (lrx, lry, lrz) = tx.TransformPoint(
    #             src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
    #         )
    #
    #     if not cell_size:
    #         # the result raster has the same pixcel size as the source
    #         # check if the coordinate system is GCS convert the distance from angular to metric
    #         if src_epsg == "4326":
    #             coords_1 = (src_gt[3], src_gt[0])
    #             coords_2 = (src_gt[3], src_gt[0] + src_gt[1])
    #             #            pixel_spacing=geopy.distance.vincenty(coords_1, coords_2).m
    #             pixel_spacing = FeatureCollection.GCSDistance(coords_1, coords_2)
    #         else:
    #             pixel_spacing = src_gt[1]
    #     else:
    #         # if src_epsg.GetAttrValue('AUTHORITY', 1) != "4326":
    #         #     assert (cell_size > 1), "please enter cell size greater than 1"
    #         # if the user input a cell size resample the raster
    #         pixel_spacing = cell_size
    #
    #     # create a new raster
    #     cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
    #     rows = int(np.round(abs(uly - lry) / pixel_spacing))
    #     dst = Dataset._create_dataset(cols, rows, 1, dtype, driver="MEM")
    #
    #     # new geotransform
    #     new_geo = (ulx, pixel_spacing, src_gt[2], uly, src_gt[4], -pixel_spacing)
    #     # set the geotransform
    #     dst.SetGeoTransform(new_geo)
    #     # set the projection
    #     dst.SetProjection(dst_epsg.ExportToWkt())
    #     # set the no data value
    #     no_data_value = src.GetRasterBand(1).GetNoDataValue()
    #     dst = Dataset._set_no_data_value(dst, no_data_value)
    #     # perform the projection & resampling
    #     gdal.ReprojectImage(
    #         src, dst, src_sr.ExportToWkt(), dst_epsg.ExportToWkt(), method
    #     )
    #
    #     return dst

    def align(self, alignment_src: Dataset):
        """matchDataAlignment.

        this function matches the coordinate system and the number of of rows & columns
        between two rasters
        Raster A is the source of the coordinate system, no of rows and no of columns & cell size
        rasters_dir is path to the folder where Raster B exist where  Raster B is
        the source of data values in cells
        the result will be a raster with the same structure like RasterA but with
        values from RasterB using Nearest Neighbour interpolation algorithm

        Parameters
        ----------
        alignment_src: [String]
            path to the spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns) alignment_src should include the name of the raster
            and the extension like "data/dem.tif"

        Returns
        -------
        new rasters:
            ٌRasters have the values from rasters in rasters_dir with the same
            cell size, no of rows & columns, coordinate system and alignment like raster A

        Examples
        --------
        >>> dem_path = "01GIS/inputs/4000/acc4000.tif"
        >>> prec_in_path = "02Precipitation/CHIRPS/Daily/"
        >>> prec_out_path = "02Precipitation/4km/"
        >>> Dataset.align(dem_path,prec_in_path,prec_out_path)
        """
        if not isinstance(alignment_src, Dataset):
            raise TypeError("alignment_src input should be a Dataset object")

        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.align(alignment_src)
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (
                            self.time_length,
                            arr.shape[0],
                            arr.shape[1],
                        )
                    )
                    * np.nan
                )
            array[i, :, :] = arr

        self._data = array
        # use the last src as
        self._base = dst

    @staticmethod
    def merge(
        src: List[str],
        dst: str,
        no_data_value: Union[float, int, str] = "0",
        init: Union[float, int, str] = "nan",
        n: Union[float, int, str] = "nan",
    ):
        """merge.

            merges group of rasters into one raster

        Parameters
        ----------
        src: List[str]
            list of the path to all input raster
        dst: [str]
            path to the output raster
        no_data_value: [float/int]
            Assign a specified nodata value to output bands.
        init: [float/int]
            Pre-initialize the output image bands with these values. However, it is not marked as the nodata value
            in the output file. If only one value is given, the same value is used in all the bands.
        n: [float/int]
            Ignore pixels from files being merged in with this pixel value.

        Returns
        -------
        None
        """
        # run the command
        # cmd = "gdal_merge.py -o merged_image_1.tif"
        # subprocess.call(cmd.split() + file_list)
        # vrt = gdal.BuildVRT("merged.vrt", file_list)
        # src = gdal.Translate("merged_image.tif", vrt)

        parameters = (
            ["", "-o", dst]
            + src
            + [
                "-co",
                "COMPRESS=LZW",
                "-init",
                str(init),
                "-a_nodata",
                str(no_data_value),
                "-n",
                str(n),
            ]
        )  # '-separate'
        gdal_merge.main(parameters)

    # @staticmethod
    # def rasterio_merge(
    #     raster_list: list, save: bool = False, path: str = "MosaicedRaster.tif"
    # ):
    #     """mosaic.
    #
    #     Parameters
    #     ----------
    #     raster_list : [list]
    #         list of the raster files to mosaic.
    #     save : [Bool], optional
    #         to save the clipped raster to your drive. The default is False.
    #     path : [String], optional
    #         Path iincluding the extention (.tif). The default is 'MosaicedRaster.tif'.
    #
    #     Returns
    #     -------
    #     Mosaiced raster: [Rasterio object]
    #         the whole mosaiced raster
    #     metadata : [dictionay]
    #         dictionary containing number of bands, coordinate reference system crs
    #         dtype, geotransform, height and width of the raster
    #     """
    #     # List for the source files
    #     RasterioObjects = []
    #
    #     # Iterate over raster files and add them to source -list in 'read mode'
    #     for file in raster_list:
    #         src = rasterio.open(file)
    #         RasterioObjects.append(src)
    #
    #     # Merge function returns a single mosaic array and the transformation info
    #     dst, dst_trans = rasterio.merge.merge(RasterioObjects)
    #
    #     # Copy the metadata
    #     dst_meta = src.meta.copy()
    #     epsg_code = int(src.crs.data["init"][5:])
    #     # Update the metadata
    #     dst_meta.update(
    #         {
    #             "driver": "GTiff",
    #             "height": dst.shape[1],
    #             "width": dst.shape[2],
    #             "transform": dst_trans,
    #             "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
    #         }
    #     )
    #
    #     if save:
    #         # Write the mosaic raster to disk
    #         with rasterio.open(path, "w", **dst_meta) as dest:
    #             dest.write(dst)
    #
    #     return dst, dst_meta

    def apply(self, ufunc: Callable):
        """folderCalculator.

        this function matches the location of nodata value from src raster to dst
        raster
        Dataset A is where the NoDatavalue will be taken and the location of this value
        B_input_path is path to the folder where Dataset B exist where  we need to put
        the NoDataValue of RasterA in RasterB at the same locations

        Parameters
        ----------
        ufunc: [function]
            callable universal function ufunc (builtin or user defined)
            https://numpy.org/doc/stable/reference/ufuncs.html
            - to create a ufunc from a normal function (https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html)

        Returns
        -------
        new rasters will be saved to the save_to

        Examples
        --------
        >>> def func(val):
        >>>    return val%2
        >>> ufunc = np.frompyfunc(func, 1, 1)
        >>> dataset.apply(ufunc)
        """
        if not callable(ufunc):
            raise TypeError("second argument should be a function")
        arr = self.data
        no_data_value = self.base.no_data_value[0]
        # execute the function on each raster
        arr[~np.isclose(arr, no_data_value, rtol=0.001)] = ufunc(
            arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        )

    def overlay(
        self,
        classes_map,
        exclude_value: Union[float, int] = None,
    ) -> Dict[List[float], List[float]]:
        """this function is written to extract and return a list of all the values in an ASCII file.

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object fpr the raster that have classes you want to overlay with the raster.
        exclude_value: [Numeric]
            values you want to exclude from extracted values.

        Returns
        -------
        Dictionary:
            dictonary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path.
        """
        values = {}
        for i in range(self.time_length):
            src = self.iloc(i)
            dict_i = src.overlay(classes_map, exclude_value)

            # these are the destinct values from the BaseMap which are keys in the
            # ExtractedValuesi dict with each one having a list of values
            classes = list(dict_i.keys())

            for class_i in classes:
                if class_i not in values.keys():
                    values[class_i] = list()

                values[class_i] = values[class_i] + dict_i[class_i]

        return values

    # @staticmethod
    # def readNC(
    #         path,
    #         save_to: str,
    #         separator: str = "_",
    #         time_var_name: str = None,
    #         prefix: str = None,
    # ):
    #
    #     if isinstance(path, str):
    #         nc = netCDF4.Datacube(path)
    #     elif isinstance(path, list):
    #         nc = netCDF4.MFDataset(path)
    #     else:
    #         raise TypeError(
    #             "First parameter to the nctoTiff function should be either str or list"
    #         )
    #
    #     # get the variable
    #     Var = list(nc.subsets.keys())[-1]
    #     # extract the data
    #     dataset = nc[Var]
    #     # get the details of the file
    #     geo, epsg, _, _, time_len, time_var, no_data_value, datatype = NC.getNCDetails(
    #         nc, time_var_name=time_var_name
    #     )
    #     print("sss")
