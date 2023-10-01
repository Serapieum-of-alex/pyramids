"""Utility module"""
from typing import Union
import yaml
import datetime as dt
import numpy as np
from pandas import DataFrame
from osgeo import gdal, ogr, gdalconst  # gdal_array,
from osgeo.gdal import Dataset
from pyramids._errors import OptionalPackageDoesNontExist, DriverNotExistError
from pyramids import __path__

# from urllib.parse import urlparse as parse_url
# OGR_DATA_TYPES = {
#     ogr.OFTInteger: 0,
#     ogr.OFTIntegerList: 1,
#     ogr.OFTReal: 2,
#     ogr.OFTRealList: 3,
#     ogr.OFTString: 4,
#     ogr.OFTStringList: 5,
#     ogr.OFTWideString: 6,
#     ogr.OFTWideStringList: 7,
#     ogr.OFTBinary: 8,
#     ogr.OFTDate: 9,
#     ogr.OFTTime: 10,
#     ogr.OFTDateTime: 11,
#     ogr.OFTInteger64: 12,
#     ogr.OFTInteger64List: 13,
# }

# OGR_NUMPY_DATA_TYPES = {
#     0: np.int64,  # ogr.OFTInteger is actually int32 but to unify it with how geopandas read it, we will use int64.
#     12: np.int64,  # ogr.OFTInteger64
#     2: np.float64,  # ogr.OFTReal
#     4: np.object_,  # ogr.OFTString
#     11: np.datetime64,  # ogr.OFTDateTime
#     9: np.datetime64,  # ogr.OFTDate
#     10: np.datetime64,  # ogr.OFTTime
# }

DTYPE_NAMES = [
    None,
    "byte",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "float32",
    "float64",
    "complex-int16",
    "complex-int32",
    "complex-float32",
    "complex-float64",
    "uint64",
    "int64",
    "int8",
    "count",
]

GDAL_DTYPE = [
    gdalconst.GDT_Unknown,
    gdalconst.GDT_Byte,
    gdalconst.GDT_UInt16,
    gdalconst.GDT_Int16,
    gdalconst.GDT_UInt32,
    gdalconst.GDT_Int32,
    gdalconst.GDT_Float32,
    gdalconst.GDT_Float64,
    gdalconst.GDT_CInt16,
    gdalconst.GDT_CInt32,
    gdalconst.GDT_CFloat32,
    gdalconst.GDT_CFloat64,
    gdalconst.GDT_UInt64,
    gdalconst.GDT_Int64,
    gdalconst.GDT_Int8,
    gdalconst.GDT_TypeCount,
]

GDAL_DTYPE_CODE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

OGR_DTYPE = [
    None,
    ogr.OFTInteger,
    ogr.OFTInteger,
    ogr.OFTInteger,
    ogr.OFTInteger64,
    ogr.OFTInteger64,
    ogr.OFTReal,
    ogr.OFTReal,
    None,
    None,
    None,
    None,
    ogr.OFTInteger64,
    ogr.OFTInteger64,
    ogr.OFTInteger,
    None,
]

NUMPY_DTYPE = [
    None,
    np.uint8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.float32,
    np.float64,
    np.complex64,
    np.complex64,
    np.complex64,
    np.complex128,
    np.uint64,
    np.int64,
    np.int8,
    None,
]

DTYPE_CONVERSION_DF = DataFrame(
    columns=["id", "name", "numpy", "gdal", "ogr"],
    data=list(zip(GDAL_DTYPE_CODE, DTYPE_NAMES, NUMPY_DTYPE, GDAL_DTYPE, OGR_DTYPE)),
)

INTERPOLATION_METHODS = {
    "nearest neibour": gdal.GRA_NearestNeighbour,
    "cubic": gdal.GRA_Cubic,
    "bilinear": gdal.GRA_Bilinear,
}


# TODO: check the gdal.GRA_Lanczos, gdal.GRA_Average resampling method


def numpy_to_gdal_dtype(arr: Union[np.ndarray, np.dtype]) -> int:
    """mapping functiuon between numpy and gdal data types.

    Parameters
    ----------
    arr: [np.ndarray/np.dtype]
        numpy array or numpy data type

    Returns
    -------
    gdal data type
    """
    if isinstance(arr, np.ndarray):
        np_dtype = arr.dtype
    else:
        np_dtype = arr
    # integer as gdal does not accept the dtype if it is int64
    gdal_type = int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["numpy"] == np_dtype, "gdal"
        ].values[0]
    )
    return gdal_type


def ogr_to_numpy_dtype(dtype_code: int):
    """converts OGR dtype into numpy dtype

    Parameters
    ----------
    dtype_code: [int]
        OGR data type code
        ogr.OFTInteger: 0,
        ogr.OFTIntegerList: 1,
        ogr.OFTReal: 2,
        ogr.OFTRealList: 3,
        ogr.OFTString: 4,
        ogr.OFTStringList: 5,
        ogr.OFTWideString: 6,
        ogr.OFTWideStringList: 7,
        ogr.OFTBinary:8,
        ogr.OFTDate:9,
        ogr.OFTTime:10,
        ogr.OFTDateTime:11,
        ogr.OFTInteger64: 12,
        ogr.OFTInteger64List: 13,

    Returns
    -------
    Numpy data type
    """
    # since there are more than one numpy dtype for the ogr.OFTInteger (0), and the ogr.OFTInteger64 (12),
    # we will return int32 for 0 and int64 for 12.
    if dtype_code == 0:
        numpy_dtype = np.int32
    elif dtype_code == 12:
        numpy_dtype = np.int64
    elif dtype_code == 2:
        numpy_dtype = np.float64
    else:
        numpy_dtype = DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["ogr"] == dtype_code, "numpy"
        ]

        if len(numpy_dtype) == 0:
            raise ValueError(
                f"The given OGR data type is not supported: {dtype_code}, available types are: "
                f"{DTYPE_CONVERSION_DF['ogr'].unique().tolist()}"
            )
        else:
            numpy_dtype = numpy_dtype.values[0]

    return numpy_dtype


def gdal_to_numpy_dtype(dtype: int) -> str:
    """converts gdal dtype into numpy dtype

    Parameters
    ----------
    dtype: [int]

    Returns
    -------
    str
    """
    gdal_dtypes = DTYPE_CONVERSION_DF.loc[DTYPE_CONVERSION_DF["gdal"] == dtype, "numpy"]
    if len(gdal_dtypes) == 0:
        raise ValueError(
            f"The given GDAL data type is not supported: {dtype}, available types are: "
            f"{DTYPE_CONVERSION_DF['gdal'].unique().tolist()}"
        )
    else:
        gdal_dtypes = gdal_dtypes.values[0].__name__

    return gdal_dtypes


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
    gdal_dtype = band.DataType
    return int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["gdal"] == gdal_dtype, "ogr"
        ].values[0]
    )


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

    def get_driver_name_by_extension(self, extension: str):
        """Get driver by extension.

        Parameters
        ----------
        extension: [str]
            extenstion of the file.

        Returns
        -------
        str:
            Driver name.
        """
        try:
            key = next(
                key
                for key, value in self.catalog.items()
                if value.get("extension") is not None
                and value.get("extension") == extension
            )
        except StopIteration:
            raise DriverNotExistError(
                f"The given extension: {extension} is not associated with any driver in the "
                "driver catalog, if this driver is supported by gdal please open and issue to "
                "asking for youe extension to be added to the catalog"
                "https://github.com/Serapieum-of-alex/pyramids/issues/new?assignees=&labels=&template=feature_request.md&title=add%20extension"
            )

        return key

    def get_driver_by_extension(self, extension):
        """Get driver by extension.

        Parameters
        ----------
        extension: [str]
            extenstion of the file.

        Returns
        -------
        Dict:
            Driver dictionary
        """
        diver_name = self.get_driver_name_by_extension(extension)
        return self.get_driver(diver_name)

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


def ogr_ds_togdal_dataset(ogr_ds: ogr.DataSource) -> gdal.Dataset:
    """Convert ogr.Datasource object to a gdal.Dataset

    Parameters
    ----------
    ogr_ds: [Datasource]
        ogr.Datasource object

    Returns
    -------
    gdal.Dataset
    """
    gdal_ds = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, gdal.GDT_Unknown)

    for i in range(ogr_ds.GetLayerCount()):
        layer = ogr_ds.GetLayerByIndex(i)
        gdal_layer = gdal_ds.CreateLayer(
            layer.GetName(), layer.GetSpatialRef(), layer.GetLayerDefn().GetGeomType()
        )
        for field in layer.schema:
            gdal_layer.CreateField(ogr.FieldDefn(field.name, field.type))
        for feature in layer:
            gdal_feature = ogr.Feature(feature.GetDefnRef())
            gdal_feature.SetGeometry(feature.GetGeometryRef())
            for field in layer.schema:
                field_value = feature.GetField(field.name)
                gdal_feature.SetField(field.name, field_value)
            gdal_layer.CreateFeature(gdal_feature)

    return gdal_ds
