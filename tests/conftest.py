from pathlib import Path
from typing import Tuple, List

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from osgeo.gdal import Dataset
from shapely import wkt
from shapely.geometry import Polygon

from tests.catchment.conftest import *
from tests.dataset.conftest import *
from tests.feature.conftest import *


@pytest.fixture(scope="session")
def test_image_path() -> Path:
    return Path("tests/data/test_image.tif").absolute()


@pytest.fixture(scope="session")
def test_image(test_image_path: Path) -> Dataset:
    return gdal.Open(str(test_image_path))


@pytest.fixture(scope="session")
def polygonized_raster_path() -> str:
    return "tests/data/polygonized.geojson"


@pytest.fixture(scope="session")
def vector_mask_path() -> str:
    return "tests/data/mask.geojson"


@pytest.fixture(scope="session")
def vector_mask_gdf(vector_mask_path: str) -> GeoDataFrame:
    return gpd.read_file(vector_mask_path)


@pytest.fixture(scope="session")
def rasterized_mask_path() -> str:
    return "tests/data/rasterized_mask.tif"


@pytest.fixture(scope="session")
def rasterized_mask_array() -> np.ndarray:
    return np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])


@pytest.fixture(scope="session")
def rasterized_mask_values() -> np.ndarray:
    return np.array([1, 2, 3, 4, 15, 16, 17, 18, 29, 30, 31, 32, 43, 44, 45, 46])


@pytest.fixture(scope="session")
def raster_to_df_path() -> str:
    return "tests/data/raster_to_df.tif"


@pytest.fixture(scope="session")
def raster_to_df_dataset(raster_to_df_path: str) -> Dataset:
    return gdal.Open(raster_to_df_path)


@pytest.fixture(scope="session")
def raster_to_df_arr(raster_to_df_dataset: Dataset) -> np.ndarray:
    return raster_to_df_dataset.ReadAsArray()


@pytest.fixture(scope="session")
def arr() -> np.ndarray:
    return np.array([[1, 2], [3, 4]], dtype=np.float32)


@pytest.fixture(scope="session")
def lat_lon() -> Tuple[float, float]:
    return 89.83, -157.30


@pytest.fixture(scope="session")
def h3_resolution() -> int:
    return 5


@pytest.fixture(scope="session")
def hex_index() -> str:
    return "8503262bfffffff"


@pytest.fixture(scope="session")
def hex_8503262bfffffff_res5_wkt() -> str:
    return "POLYGON ((-172.2805493519069 89.93768444958565, 174.22059417805994 89.84926816814777, -166.73742982292657 89.77732549241466, -144.1902385971999 89.75925746884819, -123.07348396782385 89.80388603154616, -112.33490395174248 89.8939283971607, -172.2805493519069 89.93768444958565))"


@pytest.fixture(scope="session")
def hex_8503262bfffffff_res5_polygon(hex_8503262bfffffff_res5_wkt: str) -> Polygon:
    return wkt.loads(hex_8503262bfffffff_res5_wkt)


@pytest.fixture(scope="session")
def index_column() -> str:
    return "hex"


@pytest.fixture(scope="session")
def point_gdf() -> GeoDataFrame:
    geom_list = [
        "POINT (-75.13333 21.93333)",
        "POINT (108.00000 3.00000)",
        "POINT (90.00000 -1.00000)",
        "POINT (42.00000 14.00000)",
    ]
    hex_index = [
        "854c91cffffffff",
        "854c91cffffffff",
        "854c91cffffffff",
        "854c91cffffffff",
    ]
    geoms = list(map(wkt.loads, geom_list))
    gdf = GeoDataFrame(geometry=geoms)
    gdf["hex"] = hex_index
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


@pytest.fixture(scope="module")
def one_compressed_file_gzip() -> str:
    return "tests/data/one_compressed_file.gz"


@pytest.fixture(scope="module")
def unzip_gzip_file_name() -> str:
    return "tests/data/chirps-v2.0.2009.01.01.tif"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip() -> str:
    return "tests/data/multiple_compressed_files.gz"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def one_compressed_file_zip() -> str:
    return "tests/data/one_compressed_file.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip() -> str:
    return "tests/data/multiple_compressed_files.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def multiple_compressed_file_7z() -> str:
    return "tests/data/multiple_compressed_files.7z"


@pytest.fixture(scope="module")
def multiple_compressed_file_tar() -> str:
    return "tests/data/multiple_compressed_files.tar"


@pytest.fixture(scope="module")
def one_compressed_file_7z() -> str:
    return "tests/data/one_compressed_file.7z"


@pytest.fixture(scope="module")
def one_compressed_file_tar() -> str:
    return "tests/data/one_compressed_file.tar"
