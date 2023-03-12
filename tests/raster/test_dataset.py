import os
import numpy as np
# from osgeo.gdal import Dataset
from pyramids.raster import Raster, Dataset

class TestReadDataset:
    def test_read_all_inside_folder_without_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = Dataset.readDataset(rasters_folder_path, with_order=False)
        assert np.shape(dataset.array) == (
            rasters_folder_dim[0],
            rasters_folder_dim[1],
            rasters_folder_rasters_number,
        )

    def test_read_all_inside_folder(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        arr = Raster.readRastersFolder(rasters_folder_path, with_order=True)
        assert np.shape(arr) == (
            rasters_folder_dim[0],
            rasters_folder_dim[1],
            rasters_folder_rasters_number,
        )

    def test_read_between_dates(
        self,
        rasters_folder_path: str,
        rasters_folder_start_date: str,
        rasters_folder_end_date: str,
        rasters_folder_date_fmt: str,
        rasters_folder_dim: tuple,
        rasters_folder_between_dates_raster_number: int,
    ):
        arr = Raster.readRastersFolder(
            rasters_folder_path,
            with_order=True,
            start=rasters_folder_start_date,
            end=rasters_folder_end_date,
            fmt=rasters_folder_date_fmt,
        )
        assert np.shape(arr) == (
            rasters_folder_dim[0],
            rasters_folder_dim[1],
            rasters_folder_between_dates_raster_number,
        )

def test_from_netcdf(nc_path: str):
    Dataset.readNC(nc_path, "", separator="_")

def test_crop_folder(
    src: Dataset,
    crop_aligned_folder_path: str,
    crop_aligned_folder_saveto: str,
):
    Raster.cropAlignedFolder(
        crop_aligned_folder_path, src, crop_aligned_folder_saveto
    )
    assert len(os.listdir(crop_aligned_folder_saveto)) == 3