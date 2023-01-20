# import geopandas as gpd
# import numpy as np
import pytest

# import datetime as dt
# from osgeo import ogr
# from osgeo.ogr import DataSource
# from osgeo.gdal import Dataset

@pytest.fixture(scope="module")
def create_vector_path() -> str:
    return "tests/data/create_geojson_datasource.geojson"

@pytest.fixture(scope="module")
def test_vector_path() -> str:
    return "tests/data/test_vector.geojson"
