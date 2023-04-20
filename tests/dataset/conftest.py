import glob
import os
from typing import List

import geopandas as gpd
import pandas as pd
from geopandas.geodataframe import GeoDataFrame, DataFrame
import numpy as np
import pytest
from osgeo import gdal
from osgeo.gdal import Dataset
from shapely import wkt


@pytest.fixture(scope="module")
def src_path() -> str:
    return "tests/data/acc4000.tif"


@pytest.fixture(scope="module")
def src(src_path: str) -> Dataset:
    return gdal.OpenShared(src_path, gdal.GA_ReadOnly)


@pytest.fixture(scope="module")
def chang_no_data_dataset(src_path: str) -> Dataset:
    return gdal.OpenShared("tests/data/acc4000-change-no-data.tif", gdal.GA_ReadOnly)


@pytest.fixture(scope="module")
def lon_coords() -> list:
    return [
        434968.1206170588,
        438968.1206170588,
        442968.1206170588,
        446968.1206170588,
        450968.1206170588,
        454968.1206170588,
        458968.1206170588,
        462968.1206170588,
        466968.1206170588,
        470968.1206170588,
        474968.1206170588,
        478968.1206170588,
        482968.1206170588,
        486968.1206170588,
    ]


@pytest.fixture(scope="module")
def lat_coords() -> list:
    return [
        518007.787999178,
        514007.787999178,
        510007.787999178,
        506007.787999178,
        502007.787999178,
        498007.787999178,
        494007.787999178,
        490007.787999178,
        486007.787999178,
        482007.787999178,
        478007.787999178,
        474007.787999178,
        470007.787999178,
    ]


@pytest.fixture(scope="module")
def src_reset_crs() -> Dataset:
    return gdal.Open("tests/data/reset_crs.tif")


@pytest.fixture(scope="module")
def src_set_no_data_value() -> Dataset:
    return gdal.Open("tests/data/src-set_no_data_value.tif")


@pytest.fixture(scope="module")
def src_update() -> Dataset:
    return gdal.Open("tests/data/acc4000-update.tif", gdal.GA_Update)


@pytest.fixture(scope="module")
def multi_band() -> Dataset:
    return gdal.Open("tests/data/geotiff/multi_bands.tif")


@pytest.fixture(scope="module")
def nc_path() -> str:
    return "examples/data/MSWEP_1979010100.nc"


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
def src_masked_values(src_arr: np.ndarray, src_no_data_value: float) -> np.ndarray:
    return src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)]


@pytest.fixture(scope="module")
def src_masked_cells_center_coords_last4() -> np.ndarray:
    return np.array(
        [
            [474968.12061706, 474007.78799918],
            [478968.12061706, 474007.78799918],
            [438968.12061706, 470007.78799918],
            [442968.12061706, 470007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def src_masked_values_len(src_masked_values: np.ndarray) -> int:
    return src_masked_values.shape[0]


@pytest.fixture(scope="module")
def src_epsg() -> int:
    return 32618


@pytest.fixture(scope="module")
def src_geotransform(src: Dataset) -> tuple:
    return src.GetGeoTransform()


@pytest.fixture(scope="module")
def src_cell_center_coords_first_4_rows() -> np.ndarray:
    return np.array(
        [
            [434968.12061706, 518007.78799918],
            [438968.12061706, 518007.78799918],
            [442968.12061706, 518007.78799918],
            [446968.12061706, 518007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def src_cell_center_coords_last_4_rows() -> np.ndarray:
    return np.array(
        [
            [474968.12061706, 470007.78799918],
            [478968.12061706, 470007.78799918],
            [482968.12061706, 470007.78799918],
            [486968.12061706, 470007.78799918],
        ]
    )


@pytest.fixture(scope="module")
def src_cells_corner_coords_last4() -> np.ndarray:
    return np.array(
        [
            [472968.12061706, 472007.78799918],
            [476968.12061706, 472007.78799918],
            [480968.12061706, 472007.78799918],
            [484968.12061706, 472007.78799918],
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
    return "tests/data/crop_aligned_folder/"


@pytest.fixture(scope="module")
def crop_saveto() -> str:
    return "examples/data/crop_using_crop.tif"


@pytest.fixture(scope="module")
def rasters_folder_path() -> str:
    return "tests/data/raster-folder"


@pytest.fixture(scope="module")
def rhine_raster(rasters_folder_path: str) -> Dataset:
    return gdal.Open(f"{rasters_folder_path}/1_MSWEP_1979.01.02.tif")


@pytest.fixture(scope="module")
def ascii_folder_path() -> str:
    return "tests/data/ascii-folder"


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
def polygon_mask() -> gpd.GeoDataFrame:
    return gpd.read_file("tests/data/polygon_germany.geojson")


@pytest.fixture(scope="module")
def raster_mask() -> Dataset:
    return gdal.Open("tests/data/raster_mask.tif")


@pytest.fixture(scope="module")
def basin_polygon() -> gpd.GeoDataFrame:
    return gpd.read_file("tests/data/basin.geojson")


@pytest.fixture(scope="module")
def ascii_file_path() -> str:
    return "tests/data/asci_example.asc"


@pytest.fixture(scope="module")
def ascii_without_projection() -> str:
    return "tests/data/asci_without_projection.asc"


@pytest.fixture(scope="module")
def ascii_file_save_to() -> str:
    return "tests/data/asci_write_test.asc"


@pytest.fixture(scope="module")
def ascii_shape() -> tuple:
    return 13, 14


@pytest.fixture(scope="module")
def ascii_geotransform() -> tuple:
    return 13, 14, 432968.1206170588, 468007.787999178, 4000.0, -3.4028230607370965e38


@pytest.fixture(scope="module")
def merge_input_raster() -> List[str]:
    search_criteria = "splitted-raster*.tif"
    path = "tests/data/merge"
    return glob.glob(os.path.join(path, search_criteria))


@pytest.fixture(scope="module")
def merge_output() -> str:
    return r"tests/data/merge/merged_raster.tif"


@pytest.fixture(scope="module")
def match_alignment_dataset() -> str:
    return "tests/data/match-align-dataset"


@pytest.fixture(scope="module")
def germany_classes() -> str:
    return "tests/data/germany-classes.tif"


@pytest.fixture(scope="module")
def coello_gauges() -> GeoDataFrame:
    return gpd.read_file("tests/data/coello-gauges.geojson")


@pytest.fixture(scope="module")
def points_location_in_array() -> np.ndarray:
    return np.array([[4, 9, 9, 4, 8, 10], [5, 2, 5, 7, 7, 13]]).transpose()


@pytest.fixture(scope="module")
def bounds_gdf() -> GeoDataFrame:
    poly = wkt.loads(
        "POLYGON ((432968.1206170588 520007.787999178, 432968.1206170588 468007.787999178, 488968.1206170588 "
        "468007.787999178, 488968.1206170588 520007.787999178, 432968.1206170588 520007.787999178))"
    )
    gdf = gpd.GeoDataFrame(geometry=[poly])
    gdf.set_crs(epsg=32618, inplace=True)
    return gdf


@pytest.fixture(scope="module")
def footprint_test() -> Dataset:
    return gdal.Open("tests/data/footprint_test.tif")


@pytest.fixture(scope="module")
def gauges_df() -> DataFrame:
    x = [454795.6728, 443847.5736, 454044.6935, 464533.7067, 463231.1242, 487292.5152]
    y = [503143.3264, 481850.7151, 481189.4256, 502683.6482, 486656.3455, 478045.5720]
    df = pd.DataFrame(columns=["x", "y"])
    df["x"] = x
    df["y"] = y
    return df


@pytest.fixture(scope="module")
def rhine_dem() -> gdal.Dataset:
    return gdal.Open("tests/data/dem/DEM5km_Rhine_burned_acc.tif")


@pytest.fixture(scope="module")
def clusters() -> np.ndarray:
    return np.load("tests/data/dem/cluster.npy")


@pytest.fixture(scope="module")
def sentinel_raster() -> gdal.Dataset:
    return gdal.Open(
        "tests/data/geotiff/S2A_MSIL2A_20200215T082021_N0214_R121_T36SXA_20200215T110825_image_0_0.tif"
    )


@pytest.fixture(scope="module")
def sentinel_crop() -> gdal.Dataset:
    return gdal.Open("tests\data\geotiff\sentinel_crop.tif")


@pytest.fixture(scope="module")
def int_none_nodatavalue_attr_0_stored() -> gdal.Dataset:
    return gdal.Open("tests/data/geotiff/int_none_nodatavalue_attr_0_stored.tif")
