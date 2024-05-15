import numpy as np
from pyramids.dataset import Dataset


class TestReadNetCDF:
    def test_standard_netcdf(self, noah_nc_path):
        dataset = Dataset.read_file(noah_nc_path)
        assert dataset.raster is not None
        assert dataset.shape == (0, 512, 512)
        assert dataset.variable_names == ["Band1", "Band2", "Band3", "Band4"]
        assert list(dataset.variables.keys()) == ["Band1", "Band2", "Band3", "Band4"]
        assert dataset.variables["Band1"].shape == (1, 360, 720)
        assert dataset.variables["Band1"].dtype == ["float32"]
        assert np.isclose(dataset.variables["Band1"].no_data_value[0], -9.96920996e36)
        assert dataset.variables["Band1"].block_size == [[720, 1]]
        assert dataset.block_size == []
        assert dataset.variables["Band1"].cell_size == 0.5

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

    # def test_netcdf_create_from_array(
    #         self,
    #         src_arr: np.ndarray,
    #         src_geotransform: tuple,
    #         src_epsg: int,
    #         src_no_data_value: float,
    # ):
    #     src_arr = np.array([src_arr, src_arr, src_arr])
    #     src = Dataset.create_from_array(
    #         arr=src_arr,
    #         geo=src_geotransform,
    #         epsg=src_epsg,
    #         no_data_value=src_no_data_value,
    #         driver_type="netcdf"
    #     )
    #     assert isinstance(src.raster, gdal.Dataset)
    #     assert np.isclose(src.raster.ReadAsArray(), src_arr, rtol=0.00001).all()
    #     assert np.isclose(
    #         src.raster.GetRasterBand(1).GetNoDataValue(),
    #         src_no_data_value,
    #         rtol=0.00001,
    #     )
    #     assert src.raster.GetGeoTransform() == src_geotransform
