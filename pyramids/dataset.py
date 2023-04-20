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
    import_cleopatra,
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

from hpc.filter import get_pixels, get_indices2, get_pixels2, locate_values
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
    def raster(self) -> gdal.Dataset:
        """GDAL Dataset"""
        return self._raster

    @raster.setter
    def raster(self, value: gdal.Dataset):
        self._raster = value

    @property
    def values(self) -> np.ndarray:
        """array values."""
        return self.read_array(band=None)

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
        """Dataset shape"""
        return self.band_count, self.rows, self.columns

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
    def bounds(self) -> GeoDataFrame:
        """bounds"""
        return self._calculate_bounds()

    @property
    def bbox(self) -> List:
        """[xmin, ymin, xmax, ymax]"""
        return self._calculate_bbox()

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
                pivot_y - i * cell_size - cell_size / 2 for i in range(self.rows)
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
                pivot_y - i * cell_size - cell_size / 2 for i in range(self.rows)
            ]
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @property
    def crs(self) -> str:
        """Coordinate reference system."""
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
        return self._band_names

    @property
    def variables(self):
        """Variables in the raster (resambles the variables in netcdf files.)"""
        return self._variables

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain"""
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, value: object):
        """No data value that marks the cells out of the domain"""
        if isinstance(value, list):
            for i, val in enumerate(value):
                self._change_no_data_value_attr(i, val)
        else:
            self._change_no_data_value_attr(0, value)

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
        if hasattr(self, "_time_stamp"):
            val = self._time_stamp
        else:
            val = None

        return val

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
        dst = src_obj._create_gdal_dataset(
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
        dst = Dataset._create_gdal_dataset(columns, rows, bands, dtype, path=path)
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

    def plot(self, band: int = 0, exclude_value: Any = None, **kwargs):
        """Read Array

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0
        exclude_value: [Any]
            value to execlude from the plot. Default is None.
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
                the color of the anotation of the point. Default is blue.
            pid_size: [Any]
                size of the point annotation.
            figsize : [tuple], optional
                figure size. The default is (8,8).
            title : [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size : [integer], optional
                title size. The default is 15.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            rotation : [number], optional
                rotation of the colorbar label. The default is -90.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            cbar_length : [float], optional
                ratio to control the height of the colorbar. The default is 0.75.
            ticks_spacing : [integer], optional
                Spacing in the colorbar ticks. The default is 2.
            cbar_label_size : integer, optional
                size of the color bar label. The default is 12.
            cbar_label : str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale : integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- color_scale 1 is the normal scale
                2- color_scale 2 is the power scale
                3- color_scale 3 is the SymLogNorm scale
                4- color_scale 4 is the PowerNorm scale
                5- color_scale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma : [float], optional
                    value needed for option 2 . The default is 1./2..
                line_threshold : [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale : [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint : [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value : [bool]
                True if you want to display the values of the cells as a text
            num_size : integer, optional
                size of the numbers plotted intop of each cells. The default is 8.
            background_color_threshold : [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        import_cleopatra(
            "The current funcrion uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/Serapieum-of-alex/cleopatra"
        )
        from cleopatra.array import Array

        exclude_value = (
            [self.no_data_value[band], exclude_value]
            if exclude_value is not None
            else [self.no_data_value[band]]
        )

        arr = self.read_array(band=band)
        cleo = Array(arr, exclude_value=exclude_value)
        fig, ax = cleo.plot(**kwargs)
        return fig, ax

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
    def _create_gdal_dataset(
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
    def create_from_array(
        cls,
        arr: Union[str, gdal.Dataset, np.ndarray],
        geo: Union[str, tuple],
        epsg: Union[str, int],
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
    ):
        """create_raster.

            - create_raster method creates a raster from a given array and geotransform data
            and save the tif file if a Path is given or it will return the gdal.Datacube

        Parameters
        ----------
        arr : [array], optional
            numpy array. The default is ''.
        geo : [list], optional
            geotransform list [minimum lon, pixelsize, rotation, maximum lat, rotation,
                pixelsize]. The default is ''.
        no_data_value : TYPE, optional
            DESCRIPTION. The default is -9999.
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator )

        Returns
        -------
        dst : [DataSet].
            Dataset object will be returned.
        """
        if len(arr.shape) == 2:
            bands = 1
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            bands = arr.shape[0]
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        dtype = numpy_to_gdal_dtype(arr)
        dst_ds = Dataset._create_gdal_dataset(
            cols, rows, bands, dtype, driver="MEM", path=None
        )

        srse = Dataset._create_sr_from_epsg(epsg=epsg)
        dst_ds.SetProjection(srse.ExportToWkt())
        dst_obj = cls(dst_ds)
        dst_obj._set_no_data_value(no_data_value=no_data_value)

        dst_obj.raster.SetGeoTransform(geo)
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
        >>> Dataset.dataset_like(src, array, driver="GTiff", path=name)
        - or create a raster in memory
        >>> array = np.load("RAIN_5k.npy")
        >>> src = gdal.Open("DEM.tif")
        >>> dst = Dataset.dataset_like(src, array, driver="MEM")
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array should be of type numpy array")

        if len(array.shape) == 2:
            bands = 1
        else:
            bands = array.shape[0]

        dtype = numpy_to_gdal_dtype(array)

        dst = Dataset._create_gdal_dataset(
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

    def _check_no_data_value(self, no_data_value: List):
        """Validate The no_data_value with the dtype of the object.

        Parameters
        ----------
        no_data_value

        Returns
        -------
        no_data_value:
            convert the no_data_value to comly with the dtype
        """
        for i, val in enumerate(self.dtype):
            if gdal_to_numpy_dtype(val).__contains__("float"):
                no_data_value[i] = (
                    float(no_data_value[i]) if no_data_value[i] is not None else None
                )
            elif gdal_to_numpy_dtype(val).__contains__("int"):
                no_data_value[i] = (
                    int(no_data_value[i]) if no_data_value[i] is not None else None
                )
            else:
                raise TypeError("NoDataValue has a complex data type")

        return no_data_value

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

        self._check_no_data_value(no_data_value)

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

    def _calculate_bbox(self) -> List:
        """calculate bounding box"""
        xmin, ymax = self.pivot_point
        ymin = ymax - self.rows * self.cell_size
        xmax = xmin + self.columns * self.cell_size
        return [xmin, ymin, xmax, ymax]

    def _calculate_bounds(self) -> GeoDataFrame:
        """get the bbox as a geodataframe with a polygon geometry"""
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
        no_dtype = str(type(no_data_value)).split("'")[1]
        potential_dtypes = [
            i for i in list(NUMPY_GDAL_DATA_TYPES.keys()) if i.__contains__(no_dtype)
        ]
        potential_dtypes = [NUMPY_GDAL_DATA_TYPES.get(i) for i in potential_dtypes]

        if no_data_value is not None:
            if not self.dtype[band_i] in potential_dtypes:
                raise NoDataValueError(
                    f"The dtype of the given no_data_value{no_data_value}: {no_dtype} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self.dtype[band_i])}"
                )

        self._change_no_data_value_attr(band_i, no_data_value)
        # initialize the band with the nodata value instead of 0
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
        self._no_data_value[band] = no_data_value

    def change_no_data_value(self, new_value: Any, old_value: Any = None):
        """Change No Data Value.

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
        # create a new dataset
        new_dataset = Dataset(dst)
        new_dataset._set_no_data_value(new_value)

        for band in range(self.band_count):
            arr = self.read_array(band)
            try:
                arr[np.isclose(arr, old_value, rtol=0.001)] = new_value[band]
            except TypeError:
                raise NoDataValueError(
                    f"The dtype of the given no_data_value: {new_value[band]} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self.dtype[band])}"
                )
            new_dataset.raster.GetRasterBand(band + 1).WriteArray(arr)

        self.__init__(new_dataset.raster)

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
        indices = get_indices2(arr, mask=mask)

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

    def to_file(self, path: str, driver: str = "geotiff", band: int = 0) -> None:
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
            band index, needed only in case of ascii drivers. Default is 0.

        Examples
        --------
        >>> raster_obj = Dataset.read_file("path/to/file/***.tif")
        >>> output_path = "examples/GIS/data/save_raster_test.tif"
        >>> raster_obj.to_file(output_path)
        """
        if not isinstance(path, str):
            raise TypeError("path input should be string type")

        if not CATALOG.exists(driver):
            raise DriverNotExistError(f"The given driver: {driver} does not exist")

        driver_name = CATALOG.get_gdal_name(driver)

        if driver == "ascii":
            arr = self.read_array(band=band)
            no_data_value = self.no_data_value[band]
            xmin, ymin, _, _ = self.bbox
            io.to_ascii(arr, self.cell_size, xmin, ymin, no_data_value, path)
        else:
            dst = gdal.GetDriverByName(driver_name).CreateCopy(path, self.raster, 0)
            dst = None  # Flush the dataset to disk
            # print to go around the assigned but never used pre-commit issue
            print(dst)

    def to_polygon(
        self,
        band: int = 0,
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
        band = self.raster.GetRasterBand(band + 1)
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
                dataset=self, column_name="burn_value"
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
        dst = Dataset._create_gdal_dataset(
            self.columns, self.rows, 1, dtype, driver="MEM"
        )
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
        dst = Dataset.dataset_like(self, src_array, driver=driver, path=path)
        return dst

    def resample(self, cell_size: Union[int, float], method: str = "nearest neibour"):
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
            "nearest neibour" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear

        Returns
        -------
        raster : [Dataset]
             Dataset object.
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
        dtype = self.dtype[0]
        bands = self.band_count

        dst = Dataset._create_gdal_dataset(cols, rows, bands, dtype)
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
        dst = Dataset._create_gdal_dataset(cols, rows, 1, dtype)

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

    def fill_gaps(self, mask, src_array):
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
        # still places that has nodatavalue in the src raster but it is not nodatavalue in the mask
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
            # count the out of domian cells in the mask
            elem_mask = mask_array.size - np.count_nonzero(
                (mask_array[np.isclose(mask_array, mask_noval, rtol=0.001)])
            )

            # if not equal then store indices of those cells that doesn't matchs
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
            # interpolate those missing cells by nearest neighbour
            if elem_mask > elem_src:
                src_array = Dataset._nearest_neighbour(
                    src_array, self.no_data_value[0], rows, cols
                )
            return src_array

    def _crop_alligned(
        self,
        mask: Union[gdal.Dataset, np.ndarray],
        mask_noval: Union[int, float] = None,
        fill_gaps: bool = False,
    ) -> Union[np.ndarray, gdal.Dataset]:
        """cropAlligned.

        cropAlligned clip/crop (matches the location of nodata value from mask to src
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
                "Two rasters has different number of columns or rows please resample or match both rasters"
            )

        if isinstance(mask, Dataset):
            if (
                not self.pivot_point == mask.pivot_point
                or not self.cell_size == mask.cell_size
            ):
                raise ValueError(
                    "location of upper left corner of both rasters are not the same or cell size is "
                    "different please match both rasters first "
                )

            if not mask_epsg == self.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate system please reproject one of them to "
                    "the other raster coordinate system"
                )

        if band_count > 1:
            for band in range(self.band_count):
                if mask_noval is None:
                    src_array[band, np.isnan(mask_array)] = self.no_data_value[band]
                else:
                    src_array[
                        band, np.isclose(mask_array, mask_noval, rtol=0.001)
                    ] = self.no_data_value[band]
        else:
            if mask_noval is None:
                src_array[np.isnan(mask_array)] = self.no_data_value[0]
            else:
                src_array[
                    np.isclose(mask_array, mask_noval, rtol=0.001)
                ] = self.no_data_value[0]

        if fill_gaps:
            src_array = self.fill_gaps(mask, src_array)

        dst = Dataset._create_gdal_dataset(
            col, row, band_count, self.dtype[0], driver="MEM"
        )
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
        dst_obj._set_no_data_value(self.no_data_value)
        if band_count > 1:
            for band in range(band_count):
                dst_obj.raster.GetRasterBand(band + 1).WriteArray(src_array[band, :, :])
        else:
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)
        return dst_obj

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster"""
        if not isinstance(mask, Dataset):
            raise TypeError("The second parameter should be a Dataset")

        return self.rows == mask.rows and self.columns == mask.columns

    def align(
        self,
        alignment_src,
    ) -> gdal.Dataset:
        """align.

        align method copies the following data
            - The coordinate system
            - The number of of rows & columns
            - cell size
        from alignment_src to a data_src raster (the source of data values in cells)

        the result will be a raster with the same structure like alignment_src but with
        values from data_src using Nearest Neighbour interpolation algorithm

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
        reprojected_RasterB = self.to_crs(src.epsg)
        # create a new raster
        dst = Dataset._create_gdal_dataset(
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
        mask = vector.to_dataset(dataset=self)
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
        points: Union[GeoDataFrame, FeatureCollection, DataFrame],
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
        elif isinstance(points, DataFrame):
            if all(elem not in points.columns for elem in ["x", "y"]):
                raise ValueError(
                    "If the input in a DataFrame, it should have two columns x, and y"
                )
        else:
            if not isinstance(points, FeatureCollection):
                raise TypeError(
                    f"please check points input it should be GeoDataFrame/FeatureCollection - given {type(points)}"
                )
        if not isinstance(points, DataFrame):
            # get the x, y coordinates.
            points.xy()
            points = points.feature.loc[:, ["x", "y"]].values
        else:
            points = points.loc[:, ["x", "y"]].values

        # since first row is x-coords so the first column in the indices is the column index
        indices = locate_values(points, self.x, self.y)
        # rearrange the columns to make the row index first
        indices = indices[:, [1, 0]]
        return indices

    def extract(
        self,
        exclude_value: Any = None,
        feature: Union[FeatureCollection, GeoDataFrame] = None,
    ) -> List:
        """Extract.

            - Extract method get all the values in a raster, and exclude the values in the exclude_value parameter.
            - If the feature parameter is given the raster will be cliped to the extent of the given feature and the
            values within the feature are extracted.

        Parameters
        ----------
        exclude_value: [Numeric]
            values you want to exclude from exteacted values
        feature: [FeatureCollection/GeoDataFrame]
            vector file contains geometries you want to extract the values at their location. Default is None.
        """
        arr = self.read_array()

        if feature is None:
            mask = (
                [self.no_data_value[0], exclude_value]
                if exclude_value is not None
                else [self.no_data_value[0]]
            )
            values = get_pixels2(arr, mask)
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

            overlay extracts all the values in rasater file, if you have two maps one with classes, and the other map
            contains any type of values, and you want to know the values in each class.

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object for the raster that have classes you want to overlay with the raster.
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
            if you want to exclude_values a certain value in the raster with another value inter the two values as a list of tuples a
            [(value_to_be_exclude_valuesd, new_value)]
            >>> exclude_values = [0]
            - This parameter is introduced particularly for the case of rasters that has the nodatavalue stored in the
            array does not match the value stored in array, so this option can correct this behavior.

        Returns
        -------
        GeoDataFrame:
            - geodataframe containing the polygon representing the extent of the raster. the extent column should
            contains a value of  2 only.
            - if the dataset had separate polygons each polygon will be in a separate row.
        """
        arr = self.read_array(band=band)
        no_data_val = self.no_data_value[band]

        if no_data_val is None:
            if not (np.isnan(arr)).any():
                logger.warning(
                    "the nodata values stored in the raster does not exist in the raster "
                    "so either the raster extent is all full with data or the nodatavalue stored in the raster is"
                    " not correct"
                )
        else:
            if not (np.isclose(arr, no_data_val, rtol=0.00001)).any():
                logger.warning(
                    "the nodata values stored in the raster does not exist in the raster "
                    "so either the raster extent is all full with data or the nodatavalue stored in the raster is"
                    " not correct"
                )
        # if you want to exclude_values any value in the raster
        if exclude_values:
            for val in exclude_values:
                try:
                    # in case the val2 is None and the array is int type the following line will give error as None
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
            arr, self.geotransform, self.epsg, self.no_data_value
        )
        # then convert the rasater into polygon
        gdf = new_dataset.to_polygon(band=band)

        return gdf

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

    @staticmethod
    def _groupNeighbours(
        array, i, j, lowervalue, uppervalue, position, values, count, cluster
    ):
        """group neibiuring cells with the same values"""

        # bottom cell
        if (
            lowervalue <= array[i + 1, j] <= uppervalue
            and cluster[i + 1, j] == 0
            and i + 1 < array.shape[0]
        ):
            position.append([i + 1, j])
            values.append(array[i + 1, j])
            cluster[i + 1, j] = count
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
            Dataset._groupNeighbours(
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
        """Cluster

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
            list contains two indeces [x,y] for the position of each value .
        values : [numeric]
            the values stored in each cell in the cluster .
        """
        data = self.read_array()
        position = []
        values = []
        count = 1
        cluster = np.zeros(shape=(data.shape[0], data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                if lower_bound <= data[i, j] <= upper_bound and cluster[i, j] == 0:
                    self._groupNeighbours(
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
    """DataCube."""

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

    def __str__(self):
        message = f"""
            Files: {len(self.files)}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Variables: {self._base.variables}
            Dimension: {self.rows * self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

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

    @property
    def rows(self):
        """Number of rows"""
        return self._base.rows

    @property
    def shape(self):
        """Number of rows"""
        return self.time_length, self.rows, self.columns

    @property
    def columns(self):
        """Number of columns"""
        return self._base.columns

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
        with_order: bool = False,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
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

    def read_dataset(self, band: int = 0):
        """Read array.

            Read values form the given bands as Arrays for all files

        Parameters
        ----------
        band: [int]
            index of the band you want to read default is 0.

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
        if band > self.base.band_count - 1:
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
            self._data[i, :, :] = raster_i.GetRasterBand(band + 1).ReadAsArray()

    @property
    def data(self) -> np.ndarray:
        """data attribute.

        - The attribute where the dataset array is stored.
        - the 3D numpy array, [dataset length, rows, cols], [dataset length, lons, lats]
        """
        if not hasattr(self, "_data"):
            raise ValueError("please use the read_dataset to read the data first.")

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

    def __getitem__(self, key):
        if not hasattr(self, "data"):
            raise AttributeError("Please use the read_dataset method t read the data")
        return self._data[key, :, :]

    def __setitem__(self, key, value: np.ndarray):
        if not hasattr(self, "data"):
            raise AttributeError("Please use the read_dataset method t read the data")
        self._data[key, :, :] = value

    def __len__(self):
        return self._data.shape[0]

    def __iter__(self):
        return iter(self._data[:])

    def head(self, n: int = 5):
        """First 5 Datasets."""
        return self._data[:n, :, :]

    def tail(self, n: int = -5):
        """Last 5 Datasets."""
        return self._data[n:, :, :]

    def first(self):
        """First Dataset."""
        return self._data[0, :, :]

    def last(self):
        """Last Dataset."""
        return self._data[-1, :, :]

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

    def plot(self, band: int = 0, exclude_value: Any = None, **kwargs):
        """Read Array

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0
        exclude_value: [Any]
            value to execlude from the plot. Default is None.
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
                the color of the anotation of the point. Default is blue.
            pid_size: [Any]
                size of the point annotation.
            figsize : [tuple], optional
                figure size. The default is (8,8).
            title : [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size : [integer], optional
                title size. The default is 15.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            rotation : [number], optional
                rotation of the colorbar label. The default is -90.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            cbar_length : [float], optional
                ratio to control the height of the colorbar. The default is 0.75.
            ticks_spacing : [integer], optional
                Spacing in the colorbar ticks. The default is 2.
            cbar_label_size : integer, optional
                size of the color bar label. The default is 12.
            cbar_label : str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale : integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- color_scale 1 is the normal scale
                2- color_scale 2 is the power scale
                3- color_scale 3 is the SymLogNorm scale
                4- color_scale 4 is the PowerNorm scale
                5- color_scale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma : [float], optional
                    value needed for option 2 . The default is 1./2..
                line_threshold : [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale : [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint : [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value : [bool]
                True if you want to display the values of the cells as a text
            num_size : integer, optional
                size of the numbers plotted intop of each cells. The default is 8.
            background_color_threshold : [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        import_cleopatra(
            "The current funcrion uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/Serapieum-of-alex/cleopatra"
        )
        from cleopatra.array import Array

        data = self.data

        exclude_value = (
            [self.base.no_data_value[band], exclude_value]
            if exclude_value is not None
            else [self.base.no_data_value[band]]
        )

        cleo = Array(data, exclude_value=exclude_value)
        time = list(range(self.time_length))
        anim = cleo.animate(time, **kwargs)
        return anim

    def to_file(self, path: str, driver: str = "geotiff", band: int = 0):
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
