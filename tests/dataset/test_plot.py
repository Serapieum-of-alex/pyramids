import pytest
import numpy as np
from osgeo import gdal
from pyramids.dataset import Dataset, Datacube


import matplotlib

matplotlib.use("TkAgg")


class TestPlotDataSet:
    @pytest.mark.plot
    def test_single_band(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        from matplotlib.figure import Figure

        dataset = Dataset(src)
        fig, ax = dataset.plot(band=0)
        assert isinstance(fig, Figure)

    @pytest.mark.plot
    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        from matplotlib.figure import Figure

        dataset = Dataset(sentinel_raster)
        fig, ax = dataset.plot(rgb=[3, 2, 1])
        assert isinstance(fig, Figure)


class TestPlotDataCube:
    @pytest.mark.plot
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        from matplotlib.animation import FuncAnimation

        cube = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        cube.read_dataset()
        anim = cube.plot()
        assert isinstance(anim, FuncAnimation)
