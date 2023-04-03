from typing import List
import os
import numpy as np
import shutil
from osgeo import gdal
import geopandas as gpd
from pyramids.dataset import Dataset, Datacube


class TestCreateDataCube:
    def test_read_all_without_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_all_with_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=True)
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_between_dates(
        self,
        rasters_folder_path: str,
        rasters_folder_start_date: str,
        rasters_folder_end_date: str,
        rasters_folder_date_fmt: str,
        rasters_folder_dim: tuple,
        rasters_folder_between_dates_raster_number: int,
    ):
        dataset = Datacube.read_separate_files(
            rasters_folder_path,
            with_order=True,
            start=rasters_folder_start_date,
            end=rasters_folder_end_date,
            fmt=rasters_folder_date_fmt,
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_between_dates_raster_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    # def test_from_netcdf(nc_path: str):
    #     Datacube.readNC(nc_path, "", separator="_")


class TestAscii:
    def test_read_all_without_order(
        self,
        ascii_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]


class TestReadDataset:
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        assert dataset.data.shape == (
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
        dataset = Datacube.read_separate_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        dataset.read_dataset()
        assert dataset.data.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )


class TestUpdateDataset:
    def test_different_dimensions(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        # access the data attribute
        arr = dataset.data
        # modify the array
        arr = arr[0:4, :, :] * np.nan
        try:
            dataset.update_cube(arr)
        except ValueError:
            pass

    def test_same_dimensions(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        # access the data attribute
        arr = dataset.data
        # modify the array
        arr = arr * np.nan
        dataset.update_cube(arr)


class TestAccessDataset:
    def test_iloc(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        src = dataset.iloc(2)
        assert isinstance(src, Dataset)
        arr = src.read_array()
        assert isinstance(arr, np.ndarray)


class TestReproject:
    def test_to_epsg(
        self,
        rasters_folder_path: str,
    ):
        to_epsg = 4326
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_crs(to_epsg)
        assert dataset.base.epsg == to_epsg
        arr = dataset.data
        assert dataset.base.rows == arr.shape[1]
        assert dataset.base.columns == arr.shape[2]
        assert dataset.time_length == arr.shape[0]
        assert dataset.base.epsg == to_epsg


def test_match_alignment(
    match_alignment_dataset: str,
    src: Datacube,
):
    dataset = Datacube.read_separate_files(match_alignment_dataset, with_order=False)
    dataset.read_dataset()
    mask_obj = Dataset(src)
    dataset.align(mask_obj)
    assert dataset.base.rows == mask_obj.rows
    assert dataset.base.columns == mask_obj.columns


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

        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_file(path)
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

        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.to_file(path, driver="ascii", band=0)
        files = os.listdir(path)
        assert len(files) == 6
        shutil.rmtree(path)


class TestCrop:
    def test_crop_with_raster(
        self,
        raster_mask: Datacube,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        mask = Dataset(raster_mask)
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.crop(mask)
        # dataset.to_geotiff(crop_aligned_folder_saveto)_crop_with_polygon
        arr = dataset.data[0, :, :]
        no_data_value = dataset.base.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 720
        # shutil.rmtree(crop_aligned_folder_saveto)

    def test_crop_with_polygon(
        self,
        polygon_mask: gpd.GeoDataFrame,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        dataset.crop(polygon_mask)
        # dataset.to_geotiff(crop_aligned_folder_saveto)
        arr = dataset.data[0, :, :]
        no_data_value = dataset.base.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 806
        # shutil.rmtree(crop_aligned_folder_saveto)


def test_merge(
    merge_input_raster: List[str],
    merge_output: str,
):
    Datacube.merge(merge_input_raster, merge_output)
    assert os.path.exists(merge_output)
    src = gdal.Open(merge_output)
    assert src.GetRasterBand(1).GetNoDataValue() == 0


class TestApply:
    def test_1(
        self,
        rasters_folder_path: str,
    ):
        dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
        dataset.read_dataset()
        func = np.abs
        dataset.apply(func)


def test_overlay(rasters_folder_path: str, germany_classes: str):
    dataset = Datacube.read_separate_files(rasters_folder_path, with_order=False)
    dataset.read_dataset()

    classes_src = Dataset.read_file(germany_classes)
    class_dict = dataset.overlay(classes_src)
    arr = classes_src.read_array()
    class_values = np.unique(arr)
    assert len(class_dict.keys()) == len(class_values) - 1
    extracted_classes = list(class_dict.keys())
    real_classes = class_values.tolist()[:-1]
    assert all(i in real_classes for i in extracted_classes)
