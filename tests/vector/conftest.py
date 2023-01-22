# import numpy as np
from typing import List, Tuple
import geopandas as gpd
import pytest
from geopandas.geodataframe import GeoDataFrame

# import datetime as dt
from osgeo import ogr
from osgeo.ogr import DataSource

@pytest.fixture(scope="module")
def coordinates() -> List[Tuple[int, int]]:
    return [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]

@pytest.fixture(scope="module")
def coordinates_wkt() -> str:
    return 'POLYGON ((-106.64 24.0 0,-106.49 24.05 0,-106.49 24.01 0,-106.49 23.98 0))'

@pytest.fixture(scope="module")
def create_vector_path() -> str:
    return "tests/data/create_geojson_datasource.geojson"


@pytest.fixture(scope="module")
def test_vector_path() -> str:
    return "tests/data/test_vector.geojson"


@pytest.fixture(scope="module")
def data_source(test_vector_path: str) -> DataSource:
    return ogr.Open(test_vector_path)


@pytest.fixture(scope="module")
def ds_geodataframe(test_vector_path: str) -> GeoDataFrame:
    return gpd.read_file(test_vector_path)


@pytest.fixture(scope="module")
def gdf(test_vector_path: str) -> GeoDataFrame:
    return gpd.read_file(test_vector_path)


@pytest.fixture(scope="module")
def test_save_vector_path() -> str:
    return "tests/data/test_save_vector.geojson"
