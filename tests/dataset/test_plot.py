import numpy as np
from pyramids.dataset import Dataset

# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib.figure import Figure


class TestPlotDataSet:
    def test_plot_1(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        dataset = Dataset(src)
        fig, ax = dataset.plot(band=0)
        assert isinstance(fig, Figure)
