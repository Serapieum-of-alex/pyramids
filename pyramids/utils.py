"""Utility module"""
import gzip
import os
from loguru import logger
import datetime as dt
import numpy as np
from osgeo import gdal, gdal_array, ogr
from osgeo.gdal import Dataset

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
        gdal Dataset
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


def extractFromGZ(input_file: str, output_file: str, delete=False):
    """ExtractFromGZ method extract data from the zip/.gz files, save the data.

    Parameters
    ----------
    input_file : [str]
        zipped file name .
    output_file : [str]
        directory where the unzipped data must be
                            stored.
    delete : [bool]
        True if you want to delete the zipped file after the extracting the data
    Returns
    -------
    None.
    """
    with gzip.GzipFile(input_file, "rb") as zf:
        content = zf.read()
        save_file_content = open(output_file, "wb")
        save_file_content.write(content)

    save_file_content.close()
    zf.close()

    if delete:
        os.remove(input_file)


class ReadOnlyError(Exception):
    """ReadOnlyError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class DatasetNoFoundError(Exception):
    """DatasetNoFoundError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class NoDataValueError(Exception):
    """NoDataValueError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class AlignmentError(Exception):
    """Alignment Error"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


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
        if time_unit == "seconds":
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
