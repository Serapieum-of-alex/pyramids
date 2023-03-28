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
    return "POLYGON ((-106.64 24, -106.49 24.05, -106.49 24.01, -106.49 23.98, -106.64 24))"


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


@pytest.fixture(scope="session")
def gdf_bound() -> list:
    return [
        -92.03658482384117,
        41.27256171142609,
        -92.03450381936825,
        41.27464271589901,
    ]


@pytest.fixture(scope="module")
def test_save_vector_path() -> str:
    return "tests/data/test_save_vector.geojson"


@pytest.fixture(scope="module")
def points_gdf() -> GeoDataFrame:
    return gpd.read_file("tests/data/geometries/points.geojson")


@pytest.fixture(scope="module")
def points_gdf_x() -> list:
    return [
        457856.6918398,
        448255.85541664,
        440500.96721295,
        452283.93249295,
        459949.38026518,
        469463.5364471,
        480065.41504773,
        471982.66728089,
        486715.13473295,
        478295.75897145,
    ]


@pytest.fixture(scope="module")
def points_gdf_y() -> list:
    return [
        510972.07404114,
        499571.95362685,
        483558.69331881,
        479106.24112879,
        488578.97325903,
        494864.72736348,
        481107.79467665,
        477336.23141282,
        476997.17491253,
        473979.83130823,
    ]
