# import geopandas as gpd
# import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

# import datetime as dt
# from osgeo import gdal
# from osgeo.gdal import Dataset


@pytest.fixture(scope="module")
def points() -> DataFrame:
    return pd.read_csv("tests/data/points.csv")


@pytest.fixture(scope="module")
def points_location_in_array() -> DataFrame:
    data = dict(rows=[4, 9, 9, 4, 8, 10], cols=[5, 2, 5, 7, 7, 13])
    return pd.DataFrame(data)
