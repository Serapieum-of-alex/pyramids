import geopandas as gpd
import numpy as np
import pytest

# import datetime as dt
from osgeo import gdal
from osgeo.gdal import Dataset


@pytest.fixture(scope="module")
def src() -> Dataset:
    return gdal.Open("examples/data/acc4000.tif")


@pytest.fixture(scope="module")
def src_arr(src: Dataset) -> np.ndarray:
    return src.ReadAsArray()


@pytest.fixture(scope="module")
def src_shape(src: Dataset) -> tuple:
    return src.ReadAsArray().shape


@pytest.fixture(scope="module")
def src_no_data_value(src: Dataset) -> float:
    return src.GetRasterBand(1).GetNoDataValue()


@pytest.fixture(scope="module")
def src_epsg() -> int:
    return 32618


@pytest.fixture(scope="module")
def src_geotransform(src: Dataset) -> tuple:
    return src.GetGeoTransform()


@pytest.fixture(scope="module")
def src_arr_first_4_rows() -> np.ndarray:
    return np.array(
        [
            [434968.12061706, 520007.78799918],
            [434968.12061706, 520007.78799918],
            [434968.12061706, 520007.78799918],
            [434968.12061706, 520007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def src_arr_last_4_rows() -> np.ndarray:
    return np.array(
        [
            [478968.12061706, 520007.78799918],
            [478968.12061706, 520007.78799918],
            [478968.12061706, 520007.78799918],
            [478968.12061706, 520007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def cells_centerscoords() -> np.ndarray:
    return np.array(
        [
            [434968.12061706, 520007.78799918],
            [438968.12061706, 520007.78799918],
            [442968.12061706, 520007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def soil_raster() -> Dataset:
    return gdal.Open("examples/data/soil_raster.tif")


@pytest.fixture(scope="module")
def save_raster_path() -> str:
    return "examples/data/save_raster_test.tif"


@pytest.fixture(scope="module")
def raster_like_path() -> str:
    return "examples/data/raster_like_saved.tif"


def func1(val):
    if val < 20:
        val = 1
    elif val < 40:
        val = 2
    elif val < 60:
        val = 3
    elif val < 80:
        val = 4
    elif val < 100:
        val = 5
    else:
        val = 0
    return val


@pytest.fixture(scope="module")
def mapalgebra_function():
    return func1


@pytest.fixture(scope="module")
def fill_raster_path() -> str:
    return "examples/data/fill_raster_saved.tif"


@pytest.fixture(scope="module")
def fill_raster_value() -> int:
    return 20


@pytest.fixture(scope="module")
def resample_raster_cell_size() -> int:
    return 100


@pytest.fixture(scope="module")
def resample_raster_resample_technique() -> str:
    return "bilinear"


@pytest.fixture(scope="module")
def resample_raster_result_dims() -> tuple:
    return 520, 560


@pytest.fixture(scope="module")
def project_raster_to_epsg() -> int:
    return 4326


@pytest.fixture(scope="module")
def aligned_raster() -> Dataset:
    return gdal.Open(
        "examples/data/Evaporation_ECMWF_ERA-Interim_mm_daily_2009.01.01.tif"
    )


@pytest.fixture(scope="module")
def aligned_raster_arr(aligned_raster) -> np.ndarray:
    return aligned_raster.ReadAsArray()


@pytest.fixture(scope="module")
def crop_aligned_folder_path() -> str:
    return "examples/data/aligned_rasters/"


@pytest.fixture(scope="module")
def crop_aligned_folder_saveto() -> str:
    return "examples/data/crop_aligned_folder/"


@pytest.fixture(scope="module")
def crop_saveto() -> str:
    return "examples/data/crop_using_crop.tif"


@pytest.fixture(scope="module")
def rasters_folder_path() -> str:
    return "examples/data/raster-folder"


@pytest.fixture(scope="module")
def rasters_folder_rasters_number() -> int:
    return 6


@pytest.fixture(scope="module")
def rasters_folder_dim() -> tuple:
    return 125, 93


@pytest.fixture(scope="module")
def rasters_folder_start_date() -> str:
    return "1979-01-02"


@pytest.fixture(scope="module")
def rasters_folder_end_date() -> str:
    return "1979-01-05"


@pytest.fixture(scope="module")
def rasters_folder_date_fmt() -> str:
    return "%Y-%m-%d"


@pytest.fixture(scope="module")
def rasters_folder_between_dates_raster_number() -> int:
    return 4


@pytest.fixture(scope="module")
def basin_polygon() -> gpd.GeoDataFrame:
    return gpd.read_file("tests/data/basin.geojson")


@pytest.fixture(scope="module")
def ascii_file_path() -> str:
    return "tests/data/asci_example.asc"


@pytest.fixture(scope="module")
def ascii_file_save_to() -> str:
    return "tests/data/asci_write_test.asc"


@pytest.fixture(scope="module")
def ascii_shape() -> tuple:
    return 13, 14


@pytest.fixture(scope="module")
def ascii_geotransform() -> tuple:
    return 13, 14, 432968.1206170588, 468007.787999178, 4000.0, -3.4028230607370965e38
