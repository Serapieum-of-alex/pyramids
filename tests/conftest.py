from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from osgeo.gdal import Dataset

from tests.catchment.conftest import *
from tests.raster.conftest import *
from tests.utils.conftest import *
from tests.vector.conftest import *


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
