import os
from typing import List, Tuple

import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
import numpy as np
import pytest
from osgeo import gdal, osr
from osgeo.gdal import Dataset
from pyramids.raster import Raster
from pyramids.raster import ReadOnlyError


class TestCreateRasterObject:
    def test_from_gdal_dataset(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Raster(src)
        assert hasattr(src, "subsets")
        assert hasattr(src, "meta_data")
        assert hasattr(src, "variables")
        assert isinstance(src, Raster)

    def test_from_gdal_dataset_multi_band(
        self,
        multi_band: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Raster(multi_band)
        assert hasattr(src, "subsets")
        assert hasattr(src, "meta_data")
        assert hasattr(src, "variables")
        assert src.band_count == 13
        assert isinstance(src, Raster)

    def test_from_open_ascii_file(
        self,
        ascii_file_path: str,
        ascii_shape: tuple,
        ascii_geotransform: tuple,
    ):
        # src_obj = Raster.readASCII(ascii_file_path, dtype=1)
        src_obj = Raster.read(ascii_file_path)
        assert src_obj.band_count == 1
        assert src_obj.epsg == 6326
        assert isinstance(src_obj.raster, Dataset)
        assert src_obj.geotransform == (
            432968.1206170588,
            4000.0,
            0.0,
            520007.787999178,
            0.0,
            -4000.0,
        )

    def test_from_create_empty_driver(
        self,
        src: Dataset,
        src_no_data_value: float,
    ):
        src = Raster.create_empty_driver(src)
        assert isinstance(src, Raster)

    def test_create_raster(
        self,
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
    ):
        src = Raster.create_raster(
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
            # remove the file if it exists
            if os.path.exists(raster_like_path):
                os.remove(raster_like_path)

            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
            src_obj = Raster(src)
            Raster.raster_like(src_obj, arr2, driver="GTiff", path=raster_like_path)
            assert os.path.exists(raster_like_path)
            dst_obj = Raster.read(raster_like_path)
            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform

        def test_create_raster_like_to_mem(
            self,
            src: Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
        ):
            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

            src_obj = Raster(src)
            dst_obj = Raster.raster_like(src_obj, arr2, driver="MEM")

            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform


class TestSpatialProperties:
    def test_read_array(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        src = Raster(src)
        arr = src.read_array(band=0)
        assert np.array_equal(src_arr, arr)

    def test_read_array_multi_bands(
        self,
        multi_band: Dataset,
    ):
        src = Raster(multi_band)
        arr = src.read_array()
        assert np.array_equal(multi_band.ReadAsArray(), arr)

    def test_get_band_names(self, src: Dataset):
        src = Raster(src)
        names = src.get_band_names()
        assert isinstance(names, list)
        assert names == ["Band_1"]

    def test_set_no_data_value_error_read_only(
        self,
        src_set_no_data_value: Dataset,
        src_no_data_value: float,
    ):
        src = Raster(src_set_no_data_value)
        try:
            src._set_no_data_value(-99999.0)
        except ReadOnlyError:
            pass

    def test_set_no_data_value(
        self,
        src_update: Dataset,
        src_no_data_value: float,
    ):
        src = Raster(src_update)
        src._set_no_data_value(5.0)
        # check if the no_data_value in the Dataset object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == 5
        # check if the no_data_value of the Raster object is set
        assert src.no_data_value[0] == 5

    def test_change_no_data_value(
        self,
        src: Dataset,
        src_no_data_value: float,
    ):
        src = Raster(src)
        arr = src.read_array()
        old_value = arr[0, 0]
        new_val = -6666
        src.change_no_data_value(new_val, old_value)
        # check if the no_data_value in the Dataset object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Raster object is set
        assert src.no_data_value[0] == new_val
        # check if the new_val for the no_data_value is set in the bands
        arr = src.read_array(0)
        val = arr[0, 0]
        assert val == new_val


class TestGetCellCoordsAndCreateCellGeometry:
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
        coords = src.get_cell_coords(location="center", mask=False)
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
        coords = src.get_cell_coords(location="corner")
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
        coords = src.get_cell_coords(location="center", mask=True)
        assert coords.shape[0] == src_masked_values_len
        assert np.isclose(
            coords[-4:, :], src_masked_cells_center_coords_last4, rtol=0.000001
        ).all()

    def test_create_cell_polygon(self, src: Dataset, src_shape: Tuple, src_epsg: int):
        src = Raster(src)
        gdf = src.get_cell_polygons()
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points(self, src: Dataset, src_shape: Tuple, src_epsg: int):
        src = Raster(src)
        gdf = src.get_cell_points()
        # check the size
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    # TODO: create a tesk using a mask


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

    def test_save_ascii(
        self,
        src: Dataset,
        ascii_file_save_to: str,
    ):
        if os.path.exists(ascii_file_save_to):
            os.remove(ascii_file_save_to)

        src = Raster(src)
        src.to_ascii(ascii_file_save_to)
        assert os.path.exists(ascii_file_save_to)
        os.remove(ascii_file_save_to)


class TestMathOperations:
    def test_apply(
        self,
        src: Dataset,
        mapalgebra_function,
    ):
        src = Raster(src)
        dst = src.apply(mapalgebra_function)
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestFillRaster:
    def test_memory_raster(
        self, src: Dataset, fill_raster_path: str, fill_raster_value: int
    ):
        src = Raster(src)
        dst = src.fill(fill_raster_value, driver="MEM")
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value

    def test_disk_raster(
        self, src: Dataset, fill_raster_path: str, fill_raster_value: int
    ):
        if os.path.exists(fill_raster_path):
            os.remove(fill_raster_path)
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
        method=resample_raster_resample_technique,
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


class TestReproject:
    def test_option_maintain_alighment(
        self,
        src: Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Raster(src)
        dst = src.to_epsg(to_epsg=project_raster_to_epsg, maintain_alighment=True)

        proj = dst.raster.GetProjection()
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_donot_maintain_alighment(
        self,
        src: Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Raster(src)
        dst = src.to_epsg(to_epsg=project_raster_to_epsg, maintain_alighment=False)

        proj = dst.proj
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape


def test_match_raster_alignment(
    src: Dataset,
    src_shape: tuple,
    src_no_data_value: float,
    src_geotransform: tuple,
    soil_raster: Dataset,
):
    mask_obj = Raster(src)
    soil_raster_obj = Raster(soil_raster)
    soil_aligned = soil_raster_obj.match_alignment(mask_obj)
    assert soil_aligned.raster.ReadAsArray().shape == src_shape
    nodataval = soil_aligned.raster.GetRasterBand(1).GetNoDataValue()
    assert np.isclose(nodataval, src_no_data_value, rtol=0.000001)
    geotransform = soil_aligned.raster.GetGeoTransform()
    assert src_geotransform == geotransform


class TestCrop:
    def test_crop_gdal_obj_with_gdal_obj(
        self,
        src: Dataset,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        mask_obj = Raster(src)
        aligned_raster = Raster(aligned_raster)
        croped = aligned_raster.crop_alligned(mask_obj)
        dst_arr_cropped = croped.raster.ReadAsArray()
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
        aligned_raster = Raster(aligned_raster)
        croped = aligned_raster.crop_alligned(src_arr, mask_noval=src_no_data_value)
        dst_arr_cropped = croped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    # def test_crop_arr_with_gdal_obj(
    #     self,
    #     src: Dataset,
    #     aligned_raster_arr,
    #     src_arr: np.ndarray,
    #     src_no_data_value: float,
    # ):
    #     dst_arr_cropped = src.cropAlligned(aligned_raster_arr, src)
    #     # check that all the places of the nodatavalue are the same in both arrays
    #     src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
    #     dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
    #     assert (dst_arr_cropped == src_arr).all()

    def test_crop_un_aligned(
        self,
        soil_raster: Dataset,
        aligned_raster: Dataset,
        crop_saveto: str,
    ):
        # the soil raster has epsg=2116 and
        # Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)
        # the aligned_raster has a epsg = 32618 and
        # Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)
        mask_obj = Raster(soil_raster)
        aligned_raster = Raster(aligned_raster)
        aligned_raster._crop_un_aligned(mask_obj)


class TestCropWithPolygon:
    def test_crop_with_polygon(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: gpd.GeoDataFrame,
    ):
        src_obj = Raster(rhine_raster)
        cropped_raster = src_obj._crop_with_polygon(polygon_mask)
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.geotransform == src_obj.geotransform
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]


class TestToPolygon:
    """Tect converting raster to polygon."""

    def test_save_polygon_to_disk(
        self, test_image: Dataset, polygonized_raster_path: str
    ):
        im_obj = Raster(test_image)
        im_obj.to_polygon(path=polygonized_raster_path, driver="GeoJSON")
        assert os.path.exists(polygonized_raster_path)
        gdf = gpd.read_file(polygonized_raster_path)
        assert len(gdf) == 4
        assert all(gdf.geometry.geom_type == "Polygon")
        os.remove(polygonized_raster_path)

    def test_save_polygon_to_memory(
        self, test_image: Dataset, polygonized_raster_path: str
    ):
        im_obj = Raster(test_image)
        gdf = im_obj.to_polygon()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.geometry.geom_type == "Polygon")


class TestToDataFrame:
    def test_dataframe_without_mask(
        self, raster_to_df_dataset: gdal.Dataset, raster_to_df_arr: np.ndarray
    ):
        """the input raster is given as a string path on disk.

        Parameters
        ----------
        raster_to_df_dataset: Raster
        raster_to_df_arr: array for comparison
        """
        src = Raster(raster_to_df_dataset)
        gdf = src.to_geodataframe(add_geometry="Point")
        assert isinstance(gdf, GeoDataFrame)
        rows, cols = raster_to_df_arr.shape
        # get values and reshape arrays for comparison
        arr_flatten = raster_to_df_arr.reshape((rows * cols, 1))
        extracted_values = gdf.loc[:, gdf.columns[0]].values
        extracted_values = extracted_values.reshape(arr_flatten.shape)
        assert np.array_equal(extracted_values, arr_flatten), (
            "the extracted values in the dataframe does not equa the real "
            "values in the array"
        )

    def test_to_dataframe_with_mask_as_path_input(
        self,
        raster_to_df_dataset: gdal.Dataset,
        vector_mask_path: str,
        rasterized_mask_values: np.ndarray,
    ):
        """the input mask vector is given as a string path on disk.

        Parameters
        ----------
        raster_to_df_dataset: path on disk
        vector_mask_path: path on disk
        rasterized_mask_values: for camparioson
        """
        src = Raster(raster_to_df_dataset)
        gdf = src.to_geodataframe(vector_mask_path, add_geometry="Point")
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == len(rasterized_mask_values)
        assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
            "the extracted values in the dataframe "
            "does not "
            "equa the real "
            "values in the array"
        )

    def test_to_dataframe_with_gdf_mask(
        self,
        raster_to_df_dataset: gdal.Dataset,
        vector_mask_gdf: GeoDataFrame,
        rasterized_mask_values: np.ndarray,
    ):
        """the input mask vector is given as geodataframe.

        Parameters
        ----------
        raster_to_df_dataset: path on disk
        vector_mask_gdf: geodataframe for the vector mask
        rasterized_mask_values: array for comparison
        """
        src = Raster(raster_to_df_dataset)
        gdf = src.to_geodataframe(vector_mask_gdf, add_geometry="Point")
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == len(rasterized_mask_values)
        assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
            "the extracted values in the dataframe "
            "does not "
            "equa the real "
            "values in the array"
        )
