import pytest
from osgeo import gdal


@pytest.fixture(scope="module")
def coello_df_4000() -> gdal.Dataset:
    return gdal.Open("tests/data/dem/fd4000.tif")
