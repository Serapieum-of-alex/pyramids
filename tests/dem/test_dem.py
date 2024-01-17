import numpy as np
from osgeo import gdal
from pyramids.dem import DEM


def test_create_dem_instance(rhine_raster: gdal.Dataset):
    dem = DEM(rhine_raster)
    assert isinstance(dem, DEM)
    assert hasattr(dem, "crs")
    assert hasattr(dem, "epsg")
    assert hasattr(dem, "band_count")


class TestProperties:
    def test_values(self, coello_dem_4000: gdal.Dataset):
        """Test if the 'values' property actually replaces the no data values with np.nan"""
        dem = DEM(coello_dem_4000)
        arr = dem.values
        assert isinstance(arr, np.ndarray)
        assert np.isnan(arr[0, 0])


def test_fill_sinks(coello_dem_4000: gdal.Dataset, elev_sink_free: np.ndarray):
    dem = DEM(coello_dem_4000)
    dem_filled = dem.fill_sinks()
    assert isinstance(dem_filled, DEM)
    assert dem_filled.shape == dem.shape
    assert np.array_equal(dem_filled.values, elev_sink_free, equal_nan=True)
    # test if the changes are made inplace
    dem_filled = dem.fill_sinks(inplace=True)
    assert dem_filled is None


class TestSlope:
    def test_get_8_direction_slopes(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_slope: np.ndarray,
    ):
        dem = DEM(coello_dem_4000)
        slope = dem._get_8_direction_slopes()
        assert isinstance(slope, np.ndarray)
        assert np.array_equal(slope, coello_slope, equal_nan=True)

    def test_slope(
        self,
        coello_dem_4000: gdal.Dataset,
        coello_max_slope: np.ndarray,
    ):
        dem = DEM(coello_dem_4000)
        slope = dem.slope()
        assert isinstance(slope, DEM)
        assert slope.shape == dem.shape
        assert np.array_equal(slope.values, coello_max_slope, equal_nan=True)


def test_flow_direction(
    coello_dem_4000: gdal.Dataset,
    coello_flow_direction_cell_index: np.ndarray,
):
    dem = DEM(coello_dem_4000)
    fd = dem.flow_direction()
    assert isinstance(fd, np.ndarray)
    assert np.array_equal(fd, coello_flow_direction_cell_index, equal_nan=True)


def test_flow_direction_array_cells_indices(
    coello_dem_4000: gdal.Dataset,
    flow_direction_array_cells_indices: np.ndarray,
):
    dem = DEM(coello_dem_4000)
    fd_cell = dem.convert_flow_direction_to_cell_indices()
    assert isinstance(fd_cell, np.ndarray)
    assert fd_cell.shape == (dem.rows, dem.columns, 2)
    assert np.array_equal(fd_cell, flow_direction_array_cells_indices, equal_nan=True)


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
