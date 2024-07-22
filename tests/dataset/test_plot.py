import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame
from osgeo import gdal
from pyramids.dataset import Dataset
from pyramids.datacube import Datacube


class TestPlotDataSet:
    import matplotlib

    matplotlib.use("agg")

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

    @pytest.mark.plot
    def test_multi_band_overviews(
        self,
        era5_image_internal_overviews_read_only_true: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        from matplotlib.figure import Figure

        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        fig, ax = dataset.plot(band=0, overview=True, overview_index=0)

        assert isinstance(fig, Figure)


class TestPlotDataCube:
    @pytest.mark.plot
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        from cleopatra.array_glyph import ArrayGlyph
        from matplotlib.animation import FuncAnimation

        cube = Datacube.read_multiple_files(rasters_folder_path, with_order=False)
        cube.open_datacube()
        cleo = cube.plot()
        assert isinstance(cleo, ArrayGlyph)


class TestColorTable:

    @pytest.mark.plot
    def test_generated_data(self):
        arr = np.random.randint(1, 3, size=(2, 5, 5))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )

        # without alpha
        color_table = pd.DataFrame(
            {
                "band": [1, 1, 1, 2, 2, 2],
                "values": [1, 2, 3, 1, 2, 3],
                "color": [
                    "#709959",
                    "#F2EEA2",
                    "#F2CE85",
                    "#C28C7C",
                    "#D6C19C",
                    "#D6C19C",
                ],
            }
        )
        dataset.color_table = color_table
        retrieved_color_table = dataset.color_table
        assert all(
            ["band", "values", "red", "green", "blue", "alpha"]
            == retrieved_color_table.columns
        )

    @pytest.mark.plot
    def test_get_color_table(self, src_with_color_table: Dataset):
        dataset = Dataset(src_with_color_table)
        df = dataset._get_color_table()
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)
        # test the color_table property
        df = dataset.color_table
        assert isinstance(df, DataFrame)
        assert all(df.columns == ["band", "values", "red", "green", "blue", "alpha"])
        assert all(df.band == 1)

    @pytest.mark.plot
    def test_set_color_table(self, src_without_color_table: Dataset):
        color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
        values = [1, 3, 5, 7, 9]
        df = pd.DataFrame(columns=["band", "values", "color"])
        df.loc[:, "values"] = values
        df.loc[:, "band"] = 1
        df.loc[:, "color"] = color_hex

        dataset = Dataset(src_without_color_table)
        dataset._set_color_table(df, overwrite=True)

        color_table = dataset.raster.GetRasterBand(1).GetColorTable()
        assert color_table is not None, "the color table should not be None"
        assert color_table.GetCount() == 10, "the color table should have 5 colors"
        colors = [color_table.GetColorEntry(i) for i in range(color_table.GetCount())]
        assert colors == [
            (0, 0, 0, 0),
            (112, 153, 89, 255),
            (0, 0, 0, 0),
            (242, 238, 162, 255),
            (0, 0, 0, 0),
            (242, 206, 133, 255),
            (0, 0, 0, 0),
            (194, 140, 124, 255),
            (0, 0, 0, 0),
            (214, 193, 156, 255),
        ]
        # test the color_table property
        dataset.color_table = df


class TestColorRelief:
    color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
    values = [1, 3, 5, 7, 9]
    df = pd.DataFrame(columns=["values", "color"])
    df.loc[:, "values"] = values
    df.loc[:, "color"] = color_hex

    @pytest.mark.plot
    def test_process_color_table(self):

        color_table = Dataset._process_color_table(self.df)
        assert isinstance(color_table, DataFrame)
        assert all(color_table.columns == ["values", "red", "green", "blue", "alpha"])

    def test_create_color_relief(self):
        arr = np.random.randint(0, 15, size=(10, 10))
        dataset = Dataset.create_from_array(
            arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
        )
        color_relief = dataset.create_color_relief(band=0, color_table=self.df)
        assert color_relief.band_count == 4
        assert color_relief.band_color == {0: "red", 1: "green", 2: "blue", 3: "alpha"}
        df = color_relief.stats()
        assert all((0 < df["min"]) & (df["min"] <= 255))
        assert all((0 < df["max"]) & (df["max"] <= 255))
