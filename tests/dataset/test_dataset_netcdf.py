import os
import pytest
from osgeo import gdal
import numpy as np
from pyramids.dataset import Dataset


@pytest.fixture(scope="module")
def test_netcdf_create_from_array(
    src_arr: np.ndarray,
    src_geotransform: tuple,
    src_epsg: int,
    src_no_data_value: float,
):
    src_arr = np.array([src_arr, src_arr, src_arr])
    variable_name = "values"
    src = Dataset.create_from_array(
        arr=src_arr,
        geo=src_geotransform,
        epsg=src_epsg,
        no_data_value=src_no_data_value,
        driver_type="netcdf",
        path=None,
        variable_name=variable_name,
    )
    assert isinstance(src.raster, gdal.Dataset)
    assert src.variable_names == [variable_name]
    var = src.variables[variable_name]
    assert var.shape == (3, 13, 14)
    assert np.isclose(var.read_array(), src_arr, rtol=0.00001).all()
    assert var.cell_size == 4000
    assert var.read_array(0, [3, 3, 1, 1]) == 0
    assert np.isclose(
        var.no_data_value, [-3.402823e38, -3.402823e38, -3.402823e38]
    ).all()

    assert np.isclose(var.geotransform, src_geotransform).all()
    path = "save_created_netcdf_file.nc"
    assert src.to_file(path) is None
    os.remove(path)
    return src


class TestReadNetCDF:
    def test_standard_netcdf(self, noah_nc_path):
        dataset = Dataset.read_file(noah_nc_path)
        assert dataset.raster is not None
        assert dataset.shape == (0, 512, 512)
        assert dataset.variable_names == ["Band1", "Band2", "Band3", "Band4"]
        assert list(dataset.variables.keys()) == ["Band1", "Band2", "Band3", "Band4"]
        assert not dataset.is_subset
        assert not dataset.is_md_array
        var = dataset.variables["Band1"]
        assert var.is_subset
        assert not var.is_md_array
        assert var.shape == (1, 360, 720)
        assert var.dtype == ["float32"]
        assert np.isclose(var.no_data_value[0], -9.96920996e36)
        assert var.block_size == [[720, 1]]
        assert dataset.block_size == []
        assert var.cell_size == 0.5

    def test_read_netcdf_file_created_by_pyramids(self, pyramids_created_nc_3d: str):
        dataset = Dataset.read_file(pyramids_created_nc_3d)
        # arr = dataset.read_array()
        assert dataset.variable_names == []
        dataset = Dataset.read_file(
            pyramids_created_nc_3d, open_as_multi_dimensional=True
        )
        assert dataset.variable_names == ["values"]
        var = dataset.variables["values"]
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
        src = Dataset._create_netcdf_from_array(
            arr,
            variable_name,
            cols,
            rows,
            1,
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
        assert np.isclose(
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
        ).all()
        return src

    def test_instantiate_dataset_from_2d_mdarray(
        self, test_create_netcdf_from_array_2d: gdal.Dataset
    ):
        """
        mainly to test the self.get_variables() method
        """
        dataset = Dataset(test_create_netcdf_from_array_2d)
        assert dataset.variable_names == ["values"]
        assert dataset.variables["values"].shape == (1, 13, 14)


class TestAddVariable:
    def test_add_variable(self, test_netcdf_create_from_array):
        dataset = test_netcdf_create_from_array
        dataset.add_variable("new_variable", test_netcdf_create_from_array)

        assert all(
            [item in dataset.variable_names for item in ["values", "values-new"]]
        )

        var = dataset.variables["values-new"]
        assert var.shape == (3, 13, 14)

    def test_remove_variable_in_memory_driver(self, test_netcdf_create_from_array):
        dataset = test_netcdf_create_from_array
        variable_name = "values"
        dataset.remove_variable(variable_name)
        assert variable_name not in dataset.variable_names
