import pytest
import numpy as np
from pyramids.dataset import Dataset, Datacube


# import matplotlib
# matplotlib.use("TkAgg")


class TestPlotDataSet:
    @pytest.mark.plot
    def test_plot_1(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        from matplotlib.figure import Figure

        dataset = Dataset(src)
        fig, ax = dataset.plot(band=0)
        assert isinstance(fig, Figure)


class TestReadDataset:
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
