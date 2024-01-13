import numpy as np
from osgeo import gdal
from pyramids.dem import DEM


def test_create_dem_instance(rhine_raster: gdal.Dataset):
    dem = DEM(rhine_raster)
    assert isinstance(dem, DEM)
    assert hasattr(dem, "crs")
    assert hasattr(dem, "epsg")
    assert hasattr(dem, "band_count")


def test_fill_sinks(coello_df_4000: gdal.Dataset, elev_sinkless_valid: np.ndarray):
    dem = DEM(coello_df_4000)
    elev = dem.read_array(band=0).astype(np.float32)
    # get the value stores in no data value cells
    dem_no_val = dem.no_data_value[0]
    elev[np.isclose(elev, dem_no_val, rtol=0.00001)] = np.nan

    dem_filled = dem.fill_sinks(elev)
    assert dem_filled.shape == (dem.rows, dem.columns)
    assert np.array_equal(dem_filled, elev_sinkless_valid, equal_nan=True)


def test_d8(
    coello_df_4000: gdal.Dataset,
    flow_direction_array: np.ndarray,
):
    dem = DEM(coello_df_4000)
    # fd_cell, elev = dem.D8_new()
    fd_cell = dem.D8()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)
    assert np.array_equal(fd_cell, flow_direction_array, equal_nan=True)


def test_flowDirectionIndex(coello_df_4000: gdal.Dataset):
    dem = DEM(coello_df_4000)
    fd_cell = dem.flow_direction_index()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)


def test_flowDirectionTable(coello_df_4000: gdal.Dataset):
    dem = DEM(coello_df_4000)
    fd_cell = dem.flow_direction_table()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)


def test_flowDirectionTable(coello_df_4000: gdal.Dataset, coello_fdt):
    dem = DEM(coello_df_4000)
    fd_table = dem.flow_direction_table()
    assert fd_table == coello_fdt


# def test_cluster(rhine_raster: gdal.Dataset):
#     dem = DEM(rhine_raster)
#     arr = dem.read_array()
#     f = DEM.cluster(arr, 1, 5)
#     print("ccc")
