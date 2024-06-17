import os
import pytest
from osgeo import gdal
import numpy as np
from pyramids.dataset import Dataset
from pyramids.datacube import DataCube


class TestProperties:
    def test__str__(self, noah_nc_path: str):
        src = DataCube.read_file(noah_nc_path)
        assert isinstance(src.__str__(), str)

    def test__repr__(self, noah_nc_path: str):
        src = DataCube.read_file(noah_nc_path)
        assert isinstance(src.__repr__(), str)

    def test_x_lon_y_lat(self, noah_nc_path: str):
        cube = DataCube.read_file(noah_nc_path)
        np.testing.assert_array_equal(cube.x, cube.lon)
        np.testing.assert_array_equal(cube.y, cube.lat)

    def test_geotransform(self, noah_nc_path: str):
        cube = DataCube.read_file(noah_nc_path)
        assert cube.geotransform == (-0.25, 1.0, 0, -89.25, 0, -1.0)


@pytest.fixture(scope="module")
def test_netcdf_create_from_array(
    src_arr: np.ndarray,
    src_geotransform: tuple,
    src_epsg: int,
    src_no_data_value: float,
) -> DataCube:
    src_arr = np.array([src_arr, src_arr, src_arr])
    variable_name = "values"
    cube = DataCube.create_from_array(
        arr=src_arr,
        geo=src_geotransform,
        epsg=src_epsg,
        no_data_value=src_no_data_value,
        driver_type="netcdf",
        path=None,
        variable_name=variable_name,
    )
    assert isinstance(cube.raster, gdal.Dataset)
    assert cube.variable_names == [variable_name]
    var = cube.get_variable(variable_name)
    assert var.shape == (3, 13, 14)
    np.testing.assert_array_equal(var.read_array(), src_arr)
    assert var.cell_size == 4000
    assert var.read_array(0, [3, 3, 1, 1]) == 0
    np.testing.assert_allclose(
        var.no_data_value, [-3.402823e38, -3.402823e38, -3.402823e38]
    )
    # np.testing.assert_allclose(var.geotransform, src_geotransform)
    path = "save_created_netcdf_file.nc"
    assert cube.to_file(path) is None
    os.remove(path)
    return cube


class TestReadNetCDF:
    def test_standard_netcdf(self, noah_nc_path):
        dataset = DataCube.read_file(noah_nc_path)
        assert dataset.raster is not None
        assert dataset.shape == (0, 512, 512)
        assert dataset.variable_names == ["Band1", "Band2", "Band3", "Band4"]
        # assert list(dataset.get_variable().keys()) == ["Band1", "Band2", "Band3", "Band4"]
        assert not dataset.is_subset
        assert not dataset.is_md_array
        var = dataset.get_variable("Band1")
        assert var.is_subset
        assert not var.is_md_array
        assert var.shape == (1, 360, 720)
        assert var.dtype == ["float32"]
        assert np.isclose(var.no_data_value[0], -9.96920996e36)
        assert var.block_size == [[720, 1]]
        assert dataset.block_size == []
        assert var.cell_size == 0.5

    def test_read_netcdf_file_created_by_pyramids(self, pyramids_created_nc_3d: str):
        dataset = DataCube.read_file(pyramids_created_nc_3d)
        # arr = dataset.read_array()
        assert dataset.variable_names == []
        dataset = DataCube.read_file(
            pyramids_created_nc_3d, open_as_multi_dimensional=True
        )
        assert dataset.variable_names == ["values"]
        var = dataset.get_variable("values")
        assert var.shape == (3, 13, 14)
        assert var.cell_size == 4000
        assert np.isclose(
            var.no_data_value, [-3.402823e38, -3.402823e38, -3.402823e38]
        ).all()
        assert var.block_size == [[14, 1], [14, 1], [14, 1]]
        assert var.read_array().shape == (3, 13, 14)
        assert np.equal(var.read_array(0, [3, 3, 1, 1]), 0)


class TestCreateNetCDF:
    @pytest.fixture(scope="module")
    def test_create_netcdf_from_array_2d(self, src: gdal.Dataset):
        dataset = Dataset(src)
        rows = dataset.rows
        cols = dataset.columns
        arr = dataset.read_array()
        epsg = dataset.epsg
        geo = dataset.geotransform
        no_data_value = dataset.no_data_value[0]
        band_values = [1]
        variable_name = "values"
        src = DataCube._create_netcdf_from_array(
            arr,
            variable_name,
            cols,
            rows,
            band_values,
            geo,
            epsg,
            no_data_value,
            driver_type="netcdf",
        )
        rg = src.GetRootGroup()
        assert rg.GetMDArrayNames() == ["values", "x", "y"]
        dims = rg.GetDimensions()
        assert [dim.GetName() for dim in dims] == ["x", "y"]
        dim_x = rg.OpenMDArray("x")
        np.testing.assert_allclose(
            dim_x.ReadAsArray(),
            [
                434968.12,
                438968.12,
                442968.12,
                446968.12,
                450968.12,
                454968.12,
                458968.12,
                462968.12,
                466968.12,
                470968.12,
                474968.12,
                478968.12,
                482968.12,
                486968.12,
            ],
        )
        return src

    def test_instantiate_dataset_from_2d_mdarray(
        self, test_create_netcdf_from_array_2d: gdal.Dataset
    ):
        """
        mainly to test the self.get_variables() method
        """
        dataset = DataCube(test_create_netcdf_from_array_2d)
        assert dataset.variable_names == ["values"]
        assert dataset.get_variable("values").shape == (1, 13, 14)


class TestAddVariable:
    def test_add_variable(self, test_netcdf_create_from_array: DataCube):
        dataset = test_netcdf_create_from_array
        dataset.add_variable(test_netcdf_create_from_array)

        assert all(
            [item in dataset.variable_names for item in ["values", "values-new"]]
        )

        var = dataset.get_variable("values-new")
        assert var.shape == (3, 13, 14)

    def test_remove_variable_in_memory_driver(self, test_netcdf_create_from_array):
        dataset = test_netcdf_create_from_array
        variable_name = "values"
        dataset.remove_variable(variable_name)
        assert variable_name not in dataset.variable_names


class TestMultiVariablesNC:
    def test_x_lon_y_lat(self, two_variable_nc: str):
        cube = DataCube.read_file(two_variable_nc)
        np.testing.assert_array_equal(cube.x, np.array(range(-10, 11), dtype=float))
        np.testing.assert_array_equal(cube.lon, np.array(range(-10, 11), dtype=float))
        np.testing.assert_array_equal(cube.y, np.array(range(-10, 11), dtype=float))
        np.testing.assert_array_equal(cube.lat, np.array(range(-10, 11), dtype=float))

    def test_geotransform(self, two_variable_nc: str):
        cube = DataCube.read_file(two_variable_nc)
        assert cube.geotransform == (-10.5, 1.0, 0, -9.5, 0, -1.0)

    def test_variables(self, two_variable_nc: str):
        cube = DataCube.read_file(two_variable_nc)
        assert cube.variable_names == ["z", "q"]
        assert isinstance(cube.variables["q"], DataCube)
        assert isinstance(cube.variables["z"], DataCube)
        var = cube.variables["q"]
        assert var.shape == (1, 21, 21)

    def test_variables_x_lon_y_lat(self, two_variable_nc: str):
        cube = DataCube.read_file(two_variable_nc)
        var = cube.variables["q"]
        print(var.x)
        print(var.geotransform)
