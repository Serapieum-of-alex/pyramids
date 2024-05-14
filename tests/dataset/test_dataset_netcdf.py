import numpy as np
from pyramids.dataset import Dataset


class TestReadNetCDF:
    def test(self, noah_nc_path):
        dataset = Dataset.read_file(noah_nc_path)
        assert dataset.raster is not None
        assert dataset.shape == (0, 512, 512)
        assert dataset.variable_names == [
            (
                'NETCDF:"tests/data/netcdf/noah-precipitation-1979.nc":Band1',
                "[360x720] Band1 (32-bit floating-point)",
            ),
            (
                'NETCDF:"tests/data/netcdf/noah-precipitation-1979.nc":Band2',
                "[360x720] Band2 (32-bit floating-point)",
            ),
            (
                'NETCDF:"tests/data/netcdf/noah-precipitation-1979.nc":Band3',
                "[360x720] Band3 (32-bit floating-point)",
            ),
            (
                'NETCDF:"tests/data/netcdf/noah-precipitation-1979.nc":Band4',
                "[360x720] Band4 (32-bit floating-point)",
            ),
        ]
        assert list(dataset.variables.keys()) == ["Band1", "Band2", "Band3", "Band4"]
        assert dataset.variables["Band1"].shape == (1, 360, 720)
        assert dataset.variables["Band1"].dtype == ["float32"]
        assert np.isclose(dataset.variables["Band1"].no_data_value[0], -9.96920996e36)
        assert dataset.variables["Band1"].block_size == [[720, 1]]
        assert dataset.block_size == []
        assert dataset.variables["Band1"].cell_size == 0.5
