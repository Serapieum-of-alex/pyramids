# import numpy as np
import pytest

# import datetime as dt
from osgeo import ogr
from osgeo.ogr import DataSource
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame


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
def gdf(test_vector_path: str) -> GeoDataFrame:
    return gpd.read_file(test_vector_path)