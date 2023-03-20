from typing import List
import os
import numpy as np
import shutil
from osgeo import gdal
import geopandas as gpd
from pyramids.raster import Raster, Dataset


class TestReadDataset:
    def test_read_all_without_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        assert isinstance(dataset.sample, Raster)
        assert dataset.sample.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_lenth == rasters_folder_rasters_number
        assert dataset.sample.rows == rasters_folder_dim[0]
        assert dataset.sample.columns == rasters_folder_dim[1]

    def test_read_all_with_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=True)
        assert isinstance(dataset.sample, Raster)
        assert dataset.sample.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_lenth == rasters_folder_rasters_number
        assert dataset.sample.rows == rasters_folder_dim[0]
        assert dataset.sample.columns == rasters_folder_dim[1]

    def test_read_between_dates(
        self,
        rasters_folder_path: str,
        rasters_folder_start_date: str,
        rasters_folder_end_date: str,
        rasters_folder_date_fmt: str,
        rasters_folder_dim: tuple,
        rasters_folder_between_dates_raster_number: int,
    ):
        dataset = Dataset.read_separate_files(
            rasters_folder_path,
            with_order=True,
            start=rasters_folder_start_date,
            end=rasters_folder_end_date,
            fmt=rasters_folder_date_fmt,
        )
        assert isinstance(dataset.sample, Raster)
        assert dataset.sample.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_lenth == rasters_folder_between_dates_raster_number
        assert dataset.sample.rows == rasters_folder_dim[0]
        assert dataset.sample.columns == rasters_folder_dim[1]

    # def test_from_netcdf(nc_path: str):
    #     Dataset.readNC(nc_path, "", separator="_")


class TestAscii:
    def test_read_all_without_order(
        self,
        ascii_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        assert isinstance(dataset.sample, Raster)
        assert dataset.sample.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_lenth == rasters_folder_rasters_number
        assert dataset.sample.rows == rasters_folder_dim[0]
        assert dataset.sample.columns == rasters_folder_dim[1]


class TestReadArray:
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        assert dataset.array.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )

    def test_ascii(
        self,
        ascii_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        dataset.read_dataset()
        assert dataset.array.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )


class TestAccessDataset:
    def test_iloc(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        src = dataset.iloc(2)
        assert isinstance(src, Raster)
        arr = src.read_array()
        assert isinstance(arr, np.ndarray)


class TestReproject:
    def test_to_epsg(
        self,
        rasters_folder_path: str,
    ):
        to_epsg = 4326
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_epsg(to_epsg)
        assert dataset.sample.epsg == to_epsg
        arr = dataset.array
        assert dataset.sample.rows == arr.shape[1]
        assert dataset.sample.columns == arr.shape[2]
        assert dataset.time_lenth == arr.shape[0]
        assert dataset.sample.epsg == to_epsg


class TestSaveDataset:
    def test_to_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        path = "tests/data/dataset/save_geotiff"
        if os.path.exists(path):
            shutil.rmtree(path)

        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_geotiff(path)
        files = os.listdir(path)
        assert len(files) == 6
        shutil.rmtree(path)

    def test_to_ascii(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        path = "tests/data/dataset/save_ascii"
        if os.path.exists(path):
            shutil.rmtree(path)

        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_ascii(path)
        files = os.listdir(path)
        assert len(files) == 6
        shutil.rmtree(path)


class TestCrop:
    def test_crop_with_raster(
        self,
        raster_mask: Dataset,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        mask = Raster(raster_mask)
        dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.crop(mask)
        # dataset.to_geotiff(crop_aligned_folder_saveto)
        arr = dataset.array[0, :, :]
        no_data_value = dataset.sample.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 720
        # shutil.rmtree(crop_aligned_folder_saveto)

    # def test_crop_with_polygon(
    #         self,
    #         polygon_mask: gpd.GeoDataFrame,
    #         rasters_folder_path: str,
    #         crop_aligned_folder_saveto: str,
    # ):
    #     # if os.path.exists(crop_aligned_folder_saveto):
    #     #     shutil.rmtree(crop_aligned_folder_saveto)
    #     #     os.mkdir(crop_aligned_folder_saveto)
    #     # else:
    #     #     os.mkdir(crop_aligned_folder_saveto)
    #
    #     dataset = Dataset.read_separate_files(rasters_folder_path, with_order=False)
    #     dataset.read_dataset()
    #     dataset.crop(polygon_mask)
    #     dataset.to_geotiff(crop_aligned_folder_saveto)
    #     arr = dataset.array[0, :, :]
    #     no_data_value = dataset.sample.no_data_value[0]
    #     arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
    #     assert arr1.shape[0] == 720
    #     # shutil.rmtree(crop_aligned_folder_saveto)


def test_merge(
    merge_input_raster: List[str],
    merge_output: str,
):
    Dataset.gdal_merge(merge_input_raster, merge_output)
    assert os.path.exists(merge_output)
    src = gdal.Open(merge_output)
    assert src.GetRasterBand(1).GetNoDataValue() == 0
