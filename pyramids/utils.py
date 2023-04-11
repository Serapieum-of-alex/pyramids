"""Utility module"""
import yaml
import datetime as dt
import numpy as np
from osgeo import gdal, gdal_array, ogr
from osgeo.gdal import Dataset
from pyramids.errors import OptionalPackageDoesNontExist
from pyramids import __path__

# from urllib.parse import urlparse as parse_url

# mapping between gdal type and ogr field type
GDAL_OGR_DATA_TYPES = {
    gdal.GDT_Unknown: ogr.OFTInteger,
    gdal.GDT_Byte: ogr.OFTInteger,
    gdal.GDT_UInt16: ogr.OFTInteger,
    gdal.GDT_Int16: ogr.OFTInteger,
    gdal.GDT_UInt32: ogr.OFTInteger,
    gdal.GDT_Int32: ogr.OFTInteger,
    gdal.GDT_Float32: ogr.OFTReal,
    gdal.GDT_Float64: ogr.OFTReal,
    gdal.GDT_CInt16: ogr.OFTInteger,
    gdal.GDT_CInt32: ogr.OFTInteger,
    gdal.GDT_CFloat32: ogr.OFTReal,
    gdal.GDT_CFloat64: ogr.OFTReal,
}

NUMPY_GDAL_DATA_TYPES = {
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "float64": 7,
    "complex64": 10,
    "complex128": 11,
}
INTERPOLATION_METHODS = {
    "nearest neibour": gdal.GRA_NearestNeighbour,
    "cubic": gdal.GRA_Cubic,
    "bilinear": gdal.GRA_Bilinear,
}


# TODO: check the gdal.GRA_Lanczos, gdal.GRA_Average resampling method


def numpy_to_gdal_dtype(arr: np.ndarray):
    """mapping functiuon between numpy and gdal data types.

    Parameters
    ----------
    arr: [np.ndarray]
        numpy array

    Returns
    -------
    gdal data type
    """
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
    loc = list(NUMPY_GDAL_DATA_TYPES.values()).index(type_code)
    gdal_type = list(GDAL_OGR_DATA_TYPES.keys())[loc]
    return gdal_type

    # return GDAL_NUMPY_DATA_TYPES[list(NUMPY_GDAL_DATA_TYPES.keys())[loc]]


def gdal_to_numpy_dtype(dtype: int):
    """converts gdal dtype into numpy dtype

    Parameters
    ----------
    dtype: [int]

    Returns
    -------
    str
    """
    ind = list(NUMPY_GDAL_DATA_TYPES.values()).index(dtype)
    return list(NUMPY_GDAL_DATA_TYPES.keys())[ind]


def gdal_to_ogr_dtype(src: Dataset, band: int = 1):
    """return the coresponding data type grom ogr to each gdal data type.

    Parameters
    ----------
    src: [DataSet]
        gdal Datacube
    band: [gda Band]
        gdal band

    Returns
    -------
    gdal data type
    """
    band = src.GetRasterBand(band)
    loc = list(GDAL_OGR_DATA_TYPES.keys()).index(band.DataType) + 1
    key = list(GDAL_OGR_DATA_TYPES.keys())[loc]
    return GDAL_OGR_DATA_TYPES[key]


def create_time_conversion_func(time: str) -> callable:
    """Create a function to convert the ordinal time to gregorian date

    Parameters
    ----------
    time: [str]
        time unit string extracted from netcdf file
        >>> 'seconds since 1970-01-01'

    Returns
    -------
    callacle
    """
    time_unit, start = time.split(" since ")
    datum = dt.datetime.strptime(start, "%Y-%m-%d")

    def ordinal_to_date(time_step: int):
        if time_unit == "microseconds":
            gregorian = datum + dt.timedelta(microseconds=time_step)
        elif time_unit == "milliseconds":
            gregorian = datum + dt.timedelta(milliseconds=time_step)
        elif time_unit == "seconds":
            gregorian = datum + dt.timedelta(seconds=time_step)
        elif time_unit == "hours":
            gregorian = datum + dt.timedelta(hours=time_step)
        elif time_unit == "minutes":
            gregorian = datum + dt.timedelta(minutes=time_step)
        elif time_unit == "hours":
            gregorian = datum + dt.timedelta(hours=time_step)
        elif time_unit == "days":
            gregorian = datum + dt.timedelta(days=time_step)
        else:
            raise ValueError(f"The given time unit is not available: {time_unit}")
        return gregorian

    return ordinal_to_date


class Catalog:
    """Data Catalog."""

    def __init__(self, raster_driver=True):
        if raster_driver:
            path = "gdal_drivers.yaml"
        else:
            path = "ogr_drivers.yaml"
        self.catalog = self._get_gdal_catalog(path)

    def _get_gdal_catalog(self, path: str):
        with open(f"{__path__[0]}/{path}", "r") as stream:
            gdal_catalog = yaml.safe_load(stream)

        return gdal_catalog

    def get_driver(self, driver: str):
        """get Driver data from the catalog"""
        return self.catalog.get(driver)

    def get_gdal_name(self, driver: str):
        """Get GDAL name"""
        driver = self.get_driver(driver)
        return driver.get("GDAL Name")

    def exists(self, driver: str):
        """check if the driver exist in the catalog"""
        return driver in self.catalog.keys()

    def get_extension(self, driver: str):
        """Get driver extension."""
        driver = self.get_driver(driver)
        return driver.get("extension")

    def get_driver_name(self, gdal_name) -> str:
        """Get drivern name"""
        for key, value in self.catalog.items():
            name = value.get("GDAL Name")
            if gdal_name == name:
                break
        return key


def import_geopy(message: str):
    """try to import geopy."""
    try:
        import geopy  # noqa
    except ImportError:
        raise OptionalPackageDoesNontExist(message)


def import_cleopatra(message: str):
    """try to import cleopatra."""
    try:
        import cleopatra  # noqa
    except ImportError:
        raise OptionalPackageDoesNontExist(message)
