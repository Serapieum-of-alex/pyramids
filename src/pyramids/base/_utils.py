"""Utility module."""

from typing import Union
from pathlib import Path
import yaml
import datetime as dt
import numpy as np
from pandas import DataFrame
from osgeo import gdal, ogr, gdalconst  # gdal_array,
from osgeo.gdal import Dataset
from pyramids.base._errors import OptionalPackageDoesNotExist, DriverNotExistError
from pyramids import __path__


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

COLOR_INTERPRETATIONS = [
    gdal.GCI_Undefined,  # 0
    gdal.GCI_GrayIndex,  # 1
    gdal.GCI_PaletteIndex,  # 2
    gdal.GCI_RedBand,  # 3
    gdal.GCI_GreenBand,  # 4
    gdal.GCI_BlueBand,  # 5
    gdal.GCI_AlphaBand,  # 6
    gdal.GCI_HueBand,  # 7
    gdal.GCI_SaturationBand,  # 8
    gdal.GCI_LightnessBand,  # 9
    gdal.GCI_CyanBand,  # 10
    gdal.GCI_MagentaBand,  # 11
    gdal.GCI_YellowBand,  # 12
    gdal.GCI_BlackBand,  # 13
    gdal.GCI_YCbCr_YBand,  # 14
    gdal.GCI_YCbCr_CbBand,  # 15
    gdal.GCI_YCbCr_CrBand,  # 16
]

COLOR_NAMES = [
    "undefined",
    "gray_index",
    "palette_index",
    "red",
    "green",
    "blue",
    "alpha",
    "hue",
    "saturation",
    "lightness",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "YCbCr_YBand",
    "YCbCr_CbBand",
    "YCbCr_CrBand",
]

COLOR_TABLE = DataFrame(
    columns=["id", "gdal_constant", "name"],
    data=list(zip(range(len(COLOR_NAMES)), COLOR_INTERPRETATIONS, COLOR_NAMES)),
)
INTERPOLATION_METHODS = {
    "nearest neighbor": gdal.GRA_NearestNeighbour,
    "cubic": gdal.GRA_Cubic,
    "bilinear": gdal.GRA_Bilinear,
}


def color_name_to_gdal_constant(color_name: str) -> int:
    """Convert color name to GDAL constant.

    Args:
        color_name (str): Color name.

    Returns:
        int: GDAL constant corresponding to the color name.
    """
    if color_name not in COLOR_NAMES:
        raise ValueError(
            f"{color_name} is not a valid color name, possible names are: {COLOR_NAMES}"
        )

    gdal_constant = int(
        COLOR_TABLE.loc[COLOR_TABLE["name"] == color_name, "gdal_constant"].values[0]
    )
    return gdal_constant


def gdal_constant_to_color_name(gdal_constant: int) -> str:
    """Convert GDAL constant to color name.

    Args:
        gdal_constant (int): GDAL constant.

    Returns:
        str: Color name corresponding to the GDAL constant.
    """
    if gdal_constant not in COLOR_INTERPRETATIONS:
        raise ValueError(
            f"{gdal_constant} is not a valid gdal constant, possible constants are: {COLOR_INTERPRETATIONS}"
        )
    color_name = COLOR_TABLE.loc[
        COLOR_TABLE["gdal_constant"] == gdal_constant, "name"
    ].values[0]
    return color_name


def numpy_to_gdal_dtype(arr: Union[np.ndarray, np.dtype, str]) -> int:
    """Map function between numpy and GDAL data types.

    Args:
        arr (np.ndarray | np.dtype | str): Numpy array or numpy data type.

    Returns:
        int: GDAL data type code.
    """
    if isinstance(arr, np.ndarray):
        np_dtype = arr.dtype
    elif isinstance(arr, np.dtype):
        np_dtype = arr
    elif isinstance(arr, str):
        np_dtype = np.dtype(arr)
    else:
        raise ValueError(
            "The given input is not a numpy array or a numpy data type, please provide a valid input"
        )
    # integer as gdal does not accept the dtype if it is int64
    gdal_type = int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["numpy"] == np_dtype, "gdal"
        ].values[0]
    )
    return gdal_type


def ogr_to_numpy_dtype(dtype_code: int):
    """Convert OGR dtype into numpy dtype.

    Args:
        dtype_code (int): OGR data type code
            - ogr.OFTInteger: 0
            - ogr.OFTIntegerList: 1
            - ogr.OFTReal: 2
            - ogr.OFTRealList: 3
            - ogr.OFTString: 4
            - ogr.OFTStringList: 5
            - ogr.OFTWideString: 6
            - ogr.OFTWideStringList: 7
            - ogr.OFTBinary: 8
            - ogr.OFTDate: 9
            - ogr.OFTTime: 10
            - ogr.OFTDateTime: 11
            - ogr.OFTInteger64: 12
            - ogr.OFTInteger64List: 13

    Returns:
        numpy.dtype: Numpy data type corresponding to the OGR code.
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
    """Convert GDAL dtype into numpy dtype.

    Args:
        dtype (int): GDAL data type code.

    Returns:
        str: Name of the corresponding numpy dtype.
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
    """Get the corresponding OGR data type for a given GDAL band.

    Args:
        src (gdal.Dataset): GDAL dataset.
        band (int): Band index (1-based). Default is 1.

    Returns:
        int: OGR data type code corresponding to the band GDAL dtype.
    """
    band = src.GetRasterBand(band)
    gdal_dtype = band.DataType
    return int(
        DTYPE_CONVERSION_DF.loc[
            DTYPE_CONVERSION_DF["gdal"] == gdal_dtype, "ogr"
        ].values[0]
    )


def create_time_conversion_func(time: str) -> callable:
    """Create a function to convert the ordinal time to Gregorian date.

    Args:
        time (str): Time unit string extracted from netcdf file, e.g., 'seconds since 1970-01-01'.

    Returns:
        callable: A function that converts an integer time step to a datetime.
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
        """Initialize the catalog."""
        if raster_driver:
            path = "gdal_drivers.yaml"
        else:
            path = "ogr_drivers.yaml"
        self.catalog = self._get_gdal_catalog(path)

    @staticmethod
    def _get_gdal_catalog(path: str):
        path = Path(__path__[0]) / f"/base/{path}"
        with open(path, "r") as stream:
            gdal_catalog = yaml.safe_load(stream)

        return gdal_catalog

    def get_driver(self, driver: str):
        """Get Driver data from the catalog."""
        return self.catalog.get(driver)

    def get_gdal_name(self, driver: str):
        """Get GDAL name."""
        driver = self.get_driver(driver)
        return driver.get("GDAL Name")

    def get_driver_name_by_extension(self, extension: str):
        """Get driver by extension.

        Args:
            extension (str): Extension of the file.

        Returns:
            str: Driver name.
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

        Args:
            extension (str): Extension of the file.

        Returns:
            dict: Driver dictionary.
        """
        driver_name = self.get_driver_name_by_extension(extension)
        return self.get_driver(driver_name)

    def exists(self, driver: str):
        """Check if the driver exist in the catalog."""
        return driver in self.catalog.keys()

    def get_extension(self, driver: str):
        """Get driver extension."""
        driver = self.get_driver(driver)
        return driver.get("extension")

    def get_driver_name(self, gdal_name) -> str:
        """Get drivern name."""
        for key, value in self.catalog.items():
            name = value.get("GDAL Name")
            if gdal_name == name:
                break
        return key


def import_geopy(message: str):
    """Import geopy."""
    try:
        import geopy  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def import_cleopatra(message: str):
    """Import cleopatra."""
    try:
        import cleopatra  # noqa
    except ImportError:
        raise OptionalPackageDoesNotExist(message)


def ogr_ds_to_gdal_dataset(ogr_ds: ogr.DataSource) -> gdal.Dataset:
    """Convert ogr.DataSource object to a gdal.Dataset.

    Args:
        ogr_ds (ogr.DataSource): OGR data source object.

    Returns:
        gdal.Dataset: An in-memory GDAL dataset converted from the OGR source.
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
