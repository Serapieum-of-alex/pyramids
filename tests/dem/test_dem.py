import numpy as np
from osgeo import gdal
from pyramids.dem import DEM


def test_create_dem_instance(rhine_raster: gdal.Dataset):
    dem = DEM(rhine_raster)
    assert isinstance(dem, DEM)
    assert hasattr(dem, "crs")
    assert hasattr(dem, "epsg")
    assert hasattr(dem, "band_count")


def test_d8(coello_df_4000: gdal.Dataset):
    dem = DEM(coello_df_4000)
    fd_cell, elev_sinkless = dem.D8()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)
    assert elev_sinkless.shape == (dem.rows, dem.columns)


def test_flowDirectionIndex(coello_df_4000: gdal.Dataset):
    dem = DEM(coello_df_4000)
    fd_cell = dem.flowDirectionIndex()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)


def test_flowDirectionTable(coello_df_4000: gdal.Dataset):
    dem = DEM(coello_df_4000)
    fd_table = dem.flowDirectionTable()
    assert isinstance(fd_table, dict)


# def test_cluster(rhine_raster: gdal.Dataset):
#     dem = DEM(rhine_raster)
#     arr = dem.read_array()
#     f = DEM.cluster(arr, 1, 5)
#     print("ccc")
