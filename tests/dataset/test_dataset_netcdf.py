import os
from osgeo import gdal
import numpy as np
from pyramids.dataset import Dataset


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
        arr = dataset.read_array()
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
    def test_netcdf_create_from_array(
        self,
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
    ):
        src_arr = np.array([src_arr, src_arr, src_arr])
        src = Dataset.create_from_array(
            arr=src_arr,
            geo=src_geotransform,
            epsg=src_epsg,
            no_data_value=src_no_data_value,
            driver_type="netcdf",
            path=None,
        )
        assert isinstance(src.raster, gdal.Dataset)
        assert src.variable_names == ["values"]
        var = src.variables["values"]
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
