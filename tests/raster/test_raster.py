import os
from typing import List, Tuple

# import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from osgeo.gdal import Dataset

from pyramids.raster import Raster

# @pytest.fixture(scope="module")
class TestCreateRasterObject:
    def test_from_gdal_dataset(
        self,
        src: str,
        src_no_data_value: float,
    ):
        src = Raster(src)
        assert isinstance(src, Raster)

    def test_from_open_raster(
        self,
        src_path: str,
        src_no_data_value: float,
    ):
        src = Raster.openRaster(src_path)
        assert isinstance(src, Raster)

    def test_from_create_empty_driver(
        self,
        src: Dataset,
        src_no_data_value: float,
    ):
        src = Raster.createEmptyDriver(src)
        assert isinstance(src, Raster)

    def test_create_raster(
        self,
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
    ):
        src = Raster.createRaster(
            arr=src_arr,
            geo=src_geotransform,
            epsg=src_epsg,
            nodatavalue=src_no_data_value,
        )
        assert isinstance(src.raster, Dataset)
        assert np.isclose(src.raster.ReadAsArray(), src_arr, rtol=0.00001).all()
        assert np.isclose(
            src.raster.GetRasterBand(1).GetNoDataValue(),
            src_no_data_value,
            rtol=0.00001,
        )
        assert src.raster.GetGeoTransform() == src_geotransform


    class TestRasterLike:
        def test_create_raster_like_to_disk(
            self,
            src: Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
            raster_like_path: str,
        ):
            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

            Raster.rasterLike(src, arr2, driver="GTiff", path=raster_like_path)
            assert os.path.exists(raster_like_path)
            dst = gdal.Open(raster_like_path)
            arr = dst.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src.GetGeoTransform() == dst.GetGeoTransform()

        def test_create_raster_like_to_mem(
            self,
            src: Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
        ):
            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

            dst = Raster.rasterLike(src, arr2, driver="MEM")

            arr = dst.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src.GetGeoTransform() == dst.raster.GetGeoTransform()

class TestSpatialProperties:
    def test_GetRasterData(
            self,
            src: Dataset,
            src_no_data_value: float,
    ):
        src = Raster(src)
        arr, nodataval = src.getRasterData()
        assert np.isclose(src_no_data_value, nodataval, rtol=0.001)
        assert isinstance(arr, np.ndarray)


    def test_get_raster_details(self, src: Dataset, src_shape: tuple):
        src = Raster(src)
        cols, rows, prj, bands, gt, no_data_value, dtypes = src.getRasterDetails()
        assert cols == src_shape[1]
        assert rows == src_shape[0]
        assert isinstance(no_data_value, list)
        assert isinstance(dtypes, list)
        assert isinstance(gt, tuple)


    def test_GetProjectionData(
            self,
            src: Dataset,
            src_epsg: int,
            src_geotransform: tuple,
    ):
        src = Raster(src)
        epsg, geo = src.getProjectionData()
        assert epsg == src_epsg
        assert geo == src_geotransform


    def test_get_band_names(self, src: Dataset):
        src = Raster(src)
        names = src.getBandNames()
        assert isinstance(names, list)
        assert names == ["Band_1"]

    class TestGetCellCoords:
        def test_cell_center_all_cells(
            self,
            src: Dataset,
            src_shape: tuple,
            src_cell_center_coords_first_4_rows,
            src_cell_center_coords_last_4_rows,
            cells_centerscoords: np.ndarray,
        ):
            """get center coordinates of all cells."""
            src = Raster(src)
            coords = src.getCellCoords(location="center", mask=False)
            assert len(coords) == src_shape[0] * src_shape[1]
            assert np.isclose(
                coords[:4, :], src_cell_center_coords_first_4_rows, rtol=0.000001
            ).all(), (
                "the coordinates of the first 4 rows differs from the validation coords"
            )
            assert np.isclose(
                coords[-4:, :], src_cell_center_coords_last_4_rows, rtol=0.000001
            ).all(), "the coordinates of the last 4 rows differs from the validation coords"

        def test_cell_corner_all_cells(
            self,
            src: Dataset,
            src_cells_corner_coords_last4,
        ):
            src = Raster(src)
            coords = src.getCellCoords(location="corner")
            assert np.isclose(
                coords[-4:, :], src_cells_corner_coords_last4, rtol=0.000001
            ).all()

        def test_cell_center_masked_cells(
            self,
            src: Dataset,
            src_masked_values_len: int,
            src_masked_cells_center_coords_last4,
        ):
            """get cell coordinates from cells inside the domain only."""
            src = Raster(src)
            coords = src.getCellCoords(location="center", mask=True)
            assert coords.shape[0] == src_masked_values_len
            assert np.isclose(
                coords[-4:, :], src_masked_cells_center_coords_last4, rtol=0.000001
            ).all()


class TestCreateCellGeometry:
    def test_create_cell_polygon(self, src: Dataset, src_shape: Tuple, src_epsg: int):
        src = Raster(src)
        gdf = src.getCellPolygons()
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points(self, src: Dataset, src_shape: Tuple, src_epsg: int):
        src = Raster(src)
        gdf = src.getCellPoints()
        # check the size
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg


class TestSave:
    def test_save_rasters(
            self,
            src: Dataset,
            save_raster_path: str,
    ):
        src = Raster(src)
        src.to_geotiff(save_raster_path)
        assert os.path.exists(save_raster_path)
        os.remove(save_raster_path)

class TestMathOperations:
    def test_map_algebra(
            self,
            src: Dataset,
            mapalgebra_function,
    ):
        src = Raster(src)
        dst = src.mapAlgebra(mapalgebra_function)
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals == [1, 2, 3, 4, 5]


class TestFillRaster:

    def test_memory_raster(self, src: Dataset, fill_raster_path: str, fill_raster_value: int):
        src = Raster(src)
        dst = src.fill(fill_raster_value, driver="MEM", path=fill_raster_path)
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value

    def test_disk_raster(self, src: Dataset, fill_raster_path: str, fill_raster_value: int):
        src = Raster(src)
        src.fill(fill_raster_value, driver="GTiff", path=fill_raster_path)
        "now the resulted raster is saved to disk"
        dst = gdal.Open(fill_raster_path)
        arr = dst.ReadAsArray()
        nodataval = dst.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value


def test_resample(
    src: Dataset,
    resample_raster_cell_size: int,
    resample_raster_resample_technique: str,
    resample_raster_result_dims: tuple,
):
    src = Raster(src)
    dst = src.resample(
        resample_raster_cell_size,
        resample_technique=resample_raster_resample_technique,
    )

    dst_arr = dst.raster.ReadAsArray()
    assert dst_arr.shape == resample_raster_result_dims
    assert (
        dst.raster.GetGeoTransform()[1] == resample_raster_cell_size
        and dst.raster.GetGeoTransform()[-1] == -1 * resample_raster_cell_size
    )
    assert np.isclose(
        dst.raster.GetRasterBand(1).GetNoDataValue(),
        src.raster.GetRasterBand(1).GetNoDataValue(),
        rtol=0.00001,
    )
    assert dst.raster.GetProjection() == src.raster.GetProjection()


class TestProjectRaster:
    def test_option1(
        self,
        src: Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Raster(src)
        dst = src.reproject(to_epsg=project_raster_to_epsg, option=1)

        proj = dst.raster.GetProjection()
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option2(
        self,
        src: Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Raster(src)
        dst = src.reproject(to_epsg=project_raster_to_epsg, option=2)

        proj = dst.proj
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape


# TODO: test ReprojectDataset
class TestCropAlligned:
    # TODO: still create a test for the case that the src and the mask does not have the same alignments
    def test_crop_arr_with_gdal_obj(
        self,
        src: Dataset,
        aligned_raster_arr,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        dst_arr_cropped = Raster.cropAlligned(aligned_raster_arr, src)
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_gdal_obj_with_gdal_obj(
        self,
        src: Dataset,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        dst_cropped = Raster.cropAlligned(aligned_raster, src)
        dst_arr_cropped = dst_cropped.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_gdal_obj_with_array(
        self,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        dst_cropped = Raster.cropAlligned(
            aligned_raster, src_arr, mask_noval=src_no_data_value
        )
        dst_arr_cropped = dst_cropped.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_folder(
        self,
        src: Dataset,
        crop_aligned_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        Raster.cropAlignedFolder(
            crop_aligned_folder_path, src, crop_aligned_folder_saveto
        )
        assert len(os.listdir(crop_aligned_folder_saveto)) == 3


def test_crop(
    soil_raster: Dataset,
    aligned_raster: Dataset,
    crop_saveto: str,
):
    # the soil raster has epsg=2116 and
    # Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)
    # the aligned_raster has a epsg = 32618 and
    # Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)
    Raster.crop(aligned_raster, soil_raster, save=True, output_path=crop_saveto)
    assert os.path.exists(crop_saveto)


# def test_ClipRasterWithPolygon():


class TestASCII:
    def test_read_ascii(
        self,
        ascii_file_path: str,
        ascii_shape: tuple,
        ascii_geotransform: tuple,
    ):
        arr, details = Raster.readASCII(ascii_file_path, pixel_type=1)
        assert arr.shape == ascii_shape
        assert details == ascii_geotransform

    def test_write_ascii(self, ascii_geotransform: tuple, ascii_file_save_to: str):
        arr = np.ones(shape=(13, 14)) * 0.03
        Raster.writeASCII(ascii_file_save_to, ascii_geotransform, arr)
        assert os.path.exists(ascii_file_save_to)


def test_match_raster_alignment(
    src: Dataset,
    src_shape: tuple,
    src_no_data_value: float,
    src_geotransform: tuple,
    soil_raster: Dataset,
):
    soil_aligned = Raster.matchRasterAlignment(src, soil_raster)
    assert soil_aligned.ReadAsArray().shape == src_shape
    nodataval = soil_aligned.GetRasterBand(1).GetNoDataValue()
    assert np.isclose(nodataval, src_no_data_value, rtol=0.000001)
    geotransform = soil_aligned.GetGeoTransform()
    assert src_geotransform == geotransform


class TestReadRastersFolder:
    def test_read_all_inside_folder_without_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        arr = Raster.readRastersFolder(rasters_folder_path, with_order=False)
        assert np.shape(arr) == (
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


def test_merge(
    merge_input_raster: List[str],
    merge_output: str,
):
    Raster.gdal_merge(merge_input_raster, merge_output)
    assert os.path.exists(merge_output)
    src = gdal.Open(merge_output)
    assert src.GetRasterBand(1).GetNoDataValue() == 0
