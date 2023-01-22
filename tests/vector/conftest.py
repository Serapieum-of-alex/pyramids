# import numpy as np
import geopandas as gpd
import pytest
from geopandas.geodataframe import GeoDataFrame

# import datetime as dt
from osgeo import ogr
from osgeo.ogr import DataSource


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
