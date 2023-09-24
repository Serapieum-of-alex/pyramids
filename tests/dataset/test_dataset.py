import os
from typing import List, Tuple

import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame, DataFrame
import numpy as np
import pytest
from osgeo import gdal, osr
from pyramids.dataset import Dataset
from pyramids._errors import ReadOnlyError, NoDataValueError


class TestCreateRasterObject:
    def test_from_gdal_dataset(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        assert hasattr(src, "subsets")
        assert hasattr(src, "meta_data")
        assert hasattr(src, "variables")
        assert isinstance(src, Dataset)

    def test_from_gdal_dataset_multi_band(
        self,
        multi_band: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(multi_band)
        assert hasattr(src, "subsets")
        assert hasattr(src, "meta_data")
        assert hasattr(src, "variables")
        assert src.band_count == 13
        assert isinstance(src, Dataset)

    def test_from_open_ascii_file(
        self,
        ascii_file_path: str,
        ascii_shape: tuple,
        ascii_geotransform: tuple,
    ):
        src_obj = Dataset.read_file(ascii_file_path)
        assert src_obj.band_count == 1
        assert src_obj.epsg == 6326
        assert isinstance(src_obj.raster, gdal.Dataset)
        assert src_obj.geotransform == (
            432968.1206170588,
            4000.0,
            0.0,
            520007.787999178,
            0.0,
            -4000.0,
        )

    def test_from_read_file_zip_file(
        self,
        ascii_file_path: str,
        ascii_shape: tuple,
        ascii_geotransform: tuple,
    ):
        src_obj = Dataset.read_file(ascii_file_path)
        assert src_obj.band_count == 1
        assert src_obj.epsg == 6326
        assert isinstance(src_obj.raster, gdal.Dataset)
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
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset._create_empty_driver(src)
        assert isinstance(src, Dataset)

    def test_create_from_array(
        self,
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
    ):
        src = Dataset.create_from_array(
            arr=src_arr,
            geo=src_geotransform,
            epsg=src_epsg,
            no_data_value=src_no_data_value,
        )
        assert isinstance(src.raster, gdal.Dataset)
        assert np.isclose(src.raster.ReadAsArray(), src_arr, rtol=0.00001).all()
        assert np.isclose(
            src.raster.GetRasterBand(1).GetNoDataValue(),
            src_no_data_value,
            rtol=0.00001,
        )
        assert src.raster.GetGeoTransform() == src_geotransform

    def test_create_driver_from_scratch(self):
        cell_size = 4000
        rows = 13
        columns = 14
        dtype = 5  # np.int32
        bands_count = 1
        top_left_coords = (432968.1206170588, 520007.787999178)
        ds_epsg = 32618
        no_data_value = -3.4028230607370965e38
        dataset_n = Dataset.create_driver_from_scratch(
            cell_size,
            rows,
            columns,
            dtype,
            bands_count,
            top_left_coords,
            ds_epsg,
            no_data_value,
        )
        assert dataset_n.rows == rows
        assert dataset_n.columns == columns
        assert dataset_n.epsg == ds_epsg
        assert dataset_n.cell_size == cell_size
        assert dataset_n.pivot_point == top_left_coords
        assert dataset_n.band_count == bands_count
        assert dataset_n.dtype == ["int32"]
        # the dtype is np.int32, and the no_data_value is -3.4028230607370965e+38
        # Dataset_check_no_data_value()
        # trying to convert the no_data_value to int32 will give the following error
        # "OverflowError: Python int too large to convert to C long"
        # then the default_no_data_value (-9999) will be converted to int32 and used as the no_data_value

        # Dataset._change_no_data_value_attr(band=0, no_data_value=-9999.0)
        # the _change_no_data_value_attr method will try to change the no_data_value to -9999.0 (int32)
        # but the self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value) will raise an error
        # "TypeError: in method 'Band_SetNoDataValue', argument 2 of type 'double'" , so the no_data_value will be
        # changed to float64
        new_no_data_value = np.float64(dataset_n.default_no_data_value)
        assert dataset_n.no_data_value[0] == [new_no_data_value]
        assert isinstance(dataset_n.no_data_value[0], np.float64)
        arr = dataset_n.read_array()
        assert arr[0, 0] == new_no_data_value

    class TestRasterLike:
        def test_to_disk(
            self,
            src: gdal.Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
            raster_like_path: str,
        ):
            # remove the file if it exists
            if os.path.exists(raster_like_path):
                os.remove(raster_like_path)

            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
            src_obj = Dataset(src)
            Dataset.dataset_like(src_obj, arr2, driver="GTiff", path=raster_like_path)
            assert os.path.exists(raster_like_path)
            dst_obj = Dataset.read_file(raster_like_path)
            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform

        def test_to_mem(
            self,
            src: gdal.Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
        ):
            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

            src_obj = Dataset(src)
            dst_obj = Dataset.dataset_like(src_obj, arr2, driver="MEM")

            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform


class TestProperties:
    def test_pivot_point(self, src: gdal.Dataset):
        dataset = Dataset(src)
        xy = dataset.pivot_point
        assert xy[0] == 432968.1206170588
        assert xy[1] == 520007.787999178

    def test_lon_lat(self, src: gdal.Dataset, lon_coords: list, lat_coords: list):
        dataset = Dataset(src)
        assert all(np.isclose(dataset.lon, lon_coords, rtol=0.00001))
        assert all(np.isclose(dataset.x, lon_coords, rtol=0.00001))
        assert all(np.isclose(dataset.lat, lat_coords, rtol=0.00001))
        assert all(np.isclose(dataset.y, lat_coords, rtol=0.00001))

    def test_create_bounds(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        poly = dataset._calculate_bounds()
        assert isinstance(poly, GeoDataFrame)
        assert all(bounds_gdf == poly)

    def test_create_bbox(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        bbox = dataset._calculate_bbox()
        assert isinstance(bbox, list)
        assert bbox == [
            432968.1206170588,
            468007.787999178,
            488968.1206170588,
            520007.787999178,
        ]
        bbox = dataset.bbox
        assert bbox == [
            432968.1206170588,
            468007.787999178,
            488968.1206170588,
            520007.787999178,
        ]

    def test_bounds_property(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        assert all(dataset.bounds == bounds_gdf)

    def test_shape(self, src: gdal.Dataset):
        dataset = Dataset(src)
        assert dataset.shape == (1, 13, 14)

    def test_values(self, src: gdal.Dataset):
        dataset = Dataset(src)
        assert isinstance(dataset.values, np.ndarray)

    def test_get_band_names(self, src: gdal.Dataset):
        src = Dataset(src)
        names = src._get_band_names()
        assert isinstance(names, list)
        assert names == ["Band_1"]

    def test_set_band_names(self, src: gdal.Dataset):
        src = Dataset(src)
        name_list = ["new_name"]
        src._set_band_names(name_list)
        # chack that the name is chaged in the dataset object
        assert src.band_names == name_list
        assert src.raster.GetRasterBand(1).GetDescription() == name_list[0]
        # return back the old name so that the test_get_band_names pass the test.
        src._set_band_names(["Band_1"])

    def test_band_names(self, src: gdal.Dataset):
        name_list = ["new_name"]
        src = Dataset(src)
        assert src.band_names == ["Band_1"]
        src.band_names = name_list
        assert src.band_names == name_list
        src.band_names = ["Band_1"]

    def test_numpy_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.numpy_dtype == [np.float32]

    def test_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.dtype == ["float32"]

    def test_gdal_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.gdal_dtype == [6]


class TestSpatialProperties:
    def test_read_array(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        src = Dataset(src)
        arr = src.read_array(band=0)
        assert np.array_equal(src_arr, arr)

    def test_read_array_multi_bands(
        self,
        multi_band: gdal.Dataset,
    ):
        src = Dataset(multi_band)
        arr = src.read_array()
        assert np.array_equal(multi_band.ReadAsArray(), arr)

    def test_create_sr_from_epsg(self):
        sr = Dataset._create_sr_from_epsg(4326)
        assert sr.GetAuthorityCode(None) == f"{4326}"


class TestNoDataValue:
    def test_set_no_data_value_error_read_only(
        self,
        src_set_no_data_value: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_set_no_data_value)
        try:
            src._set_no_data_value(-99999.0)
        except ReadOnlyError:
            pass

    def test_set_no_data_value(
        self,
        src_update: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_update)
        src._set_no_data_value(5.0)
        # check if the no_data_value in the Datacube object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == 5
        # check if the no_data_value of the Dataset object is set5
        assert src.no_data_value[0] == 5

    def test_change_no_data_value(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        arr = src.read_array()
        old_value = arr[0, 0]
        new_val = -6666
        src.change_no_data_value(new_val, old_value)
        # check if the no_data_value in the Datacube object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert src.no_data_value[0] == new_val
        # check that the no_data_value type has changed to float like the band dtype
        assert isinstance(src.no_data_value[0], float)
        # check if the new_val for the no_data_value is set in the bands
        arr = src.read_array(0)
        val = arr[0, 0]
        assert val == new_val

    def test_change_no_data_value_setter(
        self,
        chang_no_data_dataset: gdal.Dataset,
        src_no_data_value: float,
    ):
        """
        check setting the gdal attribute only but not the value of the nodata cells
        """
        dataset = Dataset(chang_no_data_dataset)
        new_val = -6666
        dataset.no_data_value = new_val
        # check if the no_data_value in the Datacube object is set
        assert dataset.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert dataset.no_data_value == [new_val]

    def test_change_no_data_error_different_data_type(
        self, int_none_nodatavalue_attr_0_stored: gdal.Dataset
    ):
        # try to store None in the array (int)
        dataset = Dataset(int_none_nodatavalue_attr_0_stored)
        try:
            dataset.change_no_data_value(None, 0)
        except NoDataValueError:
            pass


class TestSetCRS:
    def test_geotiff_using_epsg(
        self,
        src_reset_crs: gdal.Dataset,
    ):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        proj_epsg = 4326
        dataset = Dataset(src_reset_crs)
        dataset.set_crs(epsg=proj_epsg)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_geotiff_using_wkt(
        self,
        src_reset_crs: gdal.Dataset,
    ):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        proj_epsg = 4326
        dataset = Dataset(src_reset_crs)
        dataset.set_crs(crs=proj)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_ascii(
        self,
        ascii_without_projection: str,
    ):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        dataset = Dataset.read_file(ascii_without_projection)
        try:
            dataset.set_crs(crs=proj)
        except TypeError:
            pass


class TestGetCellCoordsAndCreateCellGeometry:
    def test_cell_center_masked_cells(
        self,
        src: gdal.Dataset,
        src_masked_values_len: int,
        src_masked_cells_center_coords_last4,
    ):
        """get cell coordinates from cells inside the domain only."""
        src = Dataset(src)
        coords = src.get_cell_coords(location="center", mask=True)
        assert coords.shape[0] == src_masked_values_len
        assert np.isclose(
            coords[-4:, :], src_masked_cells_center_coords_last4, rtol=0.000001
        ).all()

    def test_cell_center_all_cells(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
        src_cell_center_coords_first_4_rows,
        src_cell_center_coords_last_4_rows,
        cells_centerscoords: np.ndarray,
    ):
        """get center coordinates of all cells."""
        src = Dataset(src)
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
        src: gdal.Dataset,
        src_cells_corner_coords_last4,
    ):
        src = Dataset(src)
        coords = src.get_cell_coords(location="corner")
        assert np.isclose(
            coords[-4:, :], src_cells_corner_coords_last4, rtol=0.000001
        ).all()

    def test_create_cell_polygon(
        self, src: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(src)
        gdf = src.get_cell_polygons()
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points(
        self, src: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(src)
        gdf = src.get_cell_points()
        # check the size
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points_no_data_value_is_None(
        self, era5_image: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(era5_image)
        gdf = src.get_cell_points(mask=True)
        # check the size
        assert len(gdf) == 5
        assert gdf.crs.to_epsg() == 4326

    # TODO: create a tesk using a mask


class TestSave:
    def test_save_rasters(
        self,
        src: gdal.Dataset,
        save_raster_path: str,
    ):
        if os.path.exists(save_raster_path):
            os.remove(save_raster_path)
        src = Dataset(src)
        src.to_file(save_raster_path)
        assert os.path.exists(save_raster_path)
        os.remove(save_raster_path)

    def test_save_ascii(
        self,
        src: gdal.Dataset,
        ascii_file_save_to: str,
    ):
        if os.path.exists(ascii_file_save_to):
            os.remove(ascii_file_save_to)

        src = Dataset(src)
        src.to_file(ascii_file_save_to)
        assert os.path.exists(ascii_file_save_to)
        os.remove(ascii_file_save_to)


class TestMathOperations:
    def test_apply(
        self,
        src: gdal.Dataset,
        mapalgebra_function,
    ):
        src = Dataset(src)
        dst = src.apply(mapalgebra_function)
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestFillRaster:
    def test_memory_raster(
        self, src: gdal.Dataset, fill_raster_path: str, fill_raster_value: int
    ):
        src = Dataset(src)
        dst = src.fill(fill_raster_value, driver="MEM")
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value

    def test_disk_raster(
        self, src: gdal.Dataset, fill_raster_path: str, fill_raster_value: int
    ):
        if os.path.exists(fill_raster_path):
            os.remove(fill_raster_path)
        src = Dataset(src)
        src.fill(fill_raster_value, driver="GTiff", path=fill_raster_path)
        "now the resulted raster is saved to disk"
        dst = gdal.Open(fill_raster_path)
        arr = dst.ReadAsArray()
        nodataval = dst.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value


class TestResample:
    def test_single_band(
        self,
        src: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resample_raster_result_dims: tuple,
    ):
        src = Dataset(src)
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

    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resampled_multi_band_dims: tuple,
        sentinel_resample_arr: np.ndarray,
    ):
        resample_raster_cell_size = 0.00015
        src = Dataset(sentinel_raster)
        dst = src.resample(
            resample_raster_cell_size,
            method=resample_raster_resample_technique,
        )

        dst_arr = dst.raster.ReadAsArray()
        assert dst.rows == resampled_multi_band_dims[0]
        assert dst.columns == resampled_multi_band_dims[1]
        assert (
            dst.raster.GetGeoTransform()[1] == resample_raster_cell_size
            and dst.raster.GetGeoTransform()[-1] == -1 * resample_raster_cell_size
        )

        assert np.array_equal(sentinel_resample_arr, dst_arr)
        assert dst.raster.GetProjection() == src.raster.GetProjection()


class TestReproject:
    def test_option_maintain_alighment_single_band(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alighment=True)

        proj = dst.raster.GetProjection()
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_maintain_alighment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alighment=True)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg
        # assert dst.shape == src.shape

    def test_option_donot_maintain_alighment(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alighment=False)

        proj = dst.crs
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_donot_maintain_alighment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alighment=False)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg
        # assert dst.shape == src.shape


class TestAlign:
    def test_align_single_band(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
        # src_no_data_value: float,
        src_geotransform: tuple,
        soil_raster: gdal.Dataset,
    ):
        mask_obj = Dataset(src)
        dataset = Dataset(soil_raster)
        dataset_aligned = dataset.align(mask_obj)
        assert dataset_aligned.raster.ReadAsArray().shape == src_shape
        nodataval = dataset_aligned.raster.GetRasterBand(1).GetNoDataValue()
        src_no_data_value = dataset.no_data_value[0]
        assert np.isclose(nodataval, src_no_data_value, rtol=0.000001)
        geotransform = dataset_aligned.raster.GetGeoTransform()
        assert src_geotransform == geotransform

    def test_align_multi_band(
        self,
        resampled_multiband: gdal.Dataset,
        sentinel_raster: gdal.Dataset,
        resampled_multi_band_dims: tuple,
        src_geotransform: tuple,
    ):
        alignment_src = Dataset(resampled_multiband)
        dataset = Dataset(sentinel_raster)
        dataset_aligned = dataset.align(alignment_src)
        assert dataset_aligned.rows == resampled_multi_band_dims[0]
        assert dataset_aligned.columns == resampled_multi_band_dims[1]
        # assert dataset_aligned.no_data_value == dataset.no_data_value
        assert dataset.pivot_point == dataset_aligned.pivot_point


class TestCrop:
    def test_crop_dataset_with_another_dataset_single_band(
        self,
        src: gdal.Dataset,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        mask_obj = Dataset(src)
        aligned_raster = Dataset(aligned_raster)
        croped = aligned_raster._crop_alligned(mask_obj)
        dst_arr_cropped = croped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_dataset_with_another_dataset_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        sentinel_crop,
        sentinel_crop_arr_without_no_data_value: np.ndarray,
    ):
        mask_obj = Dataset(sentinel_crop)
        aligned_raster = Dataset(sentinel_raster)

        croped = aligned_raster._crop_alligned(mask_obj)
        dst_arr_cropped = croped.raster.ReadAsArray()
        # filter the no_data_value out of the array
        arr = dst_arr_cropped[
            ~np.isclose(dst_arr_cropped, croped.no_data_value[0], rtol=0.001)
        ]
        assert np.array_equal(sentinel_crop_arr_without_no_data_value, arr)

    def test_crop_dataset_with_array(
        self,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        aligned_raster = Dataset(aligned_raster)
        croped = aligned_raster._crop_alligned(src_arr, mask_noval=src_no_data_value)
        dst_arr_cropped = croped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_un_aligned(
        self,
        soil_raster: gdal.Dataset,
        aligned_raster: gdal.Dataset,
        crop_saveto: str,
    ):
        # the soil raster has epsg=2116 and
        # Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)
        # the aligned_raster has a epsg = 32618 and
        # Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)
        mask_obj = Dataset(soil_raster)
        aligned_raster = Dataset(aligned_raster)
        aligned_raster._crop_with_raster(mask_obj)


class TestCropWithPolygon:
    def test_by_rasterizing(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: gpd.GeoDataFrame,
    ):
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj._crop_with_polygon_by_rasterizing(polygon_mask)
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.geotransform == src_obj.geotransform
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_by_warp(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: gpd.GeoDataFrame,
    ):
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj._crop_with_polygon_warp(polygon_mask)
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_with_irrigular_polygon(
        self,
        raster_1band_coello_gdal_dataset: Dataset,
        rasterized_mask_values: np.ndarray,
        coello_irregular_polygon_gdf: GeoDataFrame,
    ):
        """the input mask vector is given as geodataframe.

        Parameters
        ----------
        rasterized_mask_values: array for comparison
        """
        dataset = Dataset(raster_1band_coello_gdal_dataset)
        # test with irrigular mask polygon
        cropped = dataset._crop_with_polygon_warp(coello_irregular_polygon_gdf)
        assert isinstance(cropped, Dataset)
        arr = cropped.raster.ReadAsArray()
        values = arr[~np.isclose(arr, dataset.no_data_value[0], rtol=0.0001)]
        assert np.array_equal(
            values, rasterized_mask_values
        ), "the extracted values in the dataframe does not equa the real values in the array"


class TestCluster2:
    """Tect converting raster to polygon."""

    def test_single_band(
        self,
        test_image: gdal.Dataset,
    ):
        dataset = Dataset(test_image)
        gdf = dataset.cluster2()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.columns == ["GPP", "geometry"])
        assert all(gdf.geometry.geom_type == "Polygon")

    def test_multi_band_all_bands(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        dataset = Dataset(sentinel_raster)
        gdf = dataset.cluster2()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 1767
        assert all(
            elem in gdf.columns for elem in [dataset.band_names[0]] + ["geometry"]
        )
        assert all(gdf.geometry.geom_type == "Polygon")


class TestToFeatureCollection:
    class TestWithoutMask:
        def test_1band(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            raster_to_df_arr: np.ndarray,
        ):
            """the input raster is given as a string path on disk.

            Parameters
            ----------
            raster_to_df_arr: array for comparison
            """
            src = Dataset(raster_1band_coello_gdal_dataset)
            gdf = src.to_feature_collection(add_geometry="Point")
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

        def test_multi_band(
            self, era5_image: gdal.Dataset, era5_image_gdf: GeoDataFrame
        ):
            """the input raster is given as a string path on disk."""
            dataset = Dataset(era5_image)
            gdf = dataset.to_feature_collection(add_geometry="Point")
            assert isinstance(gdf, GeoDataFrame)
            assert gdf.equals(era5_image_gdf), (
                "the extracted values in the dataframe does not equa the real "
                "values in the array"
            )

        def test_cropped_raster(
            self,
            raster_to_df_dataset_with_cropped_cell: gdal.Dataset,
            raster_to_df_arr: np.ndarray,
        ):
            """the input raster is given as a string path on disk.

            Parameters
            ----------
            raster_to_df_arr: array for comparison
            """
            dataset = Dataset(raster_to_df_dataset_with_cropped_cell)
            gdf = dataset.to_feature_collection(add_geometry="Point")
            assert isinstance(gdf, GeoDataFrame)
            # rows, cols = raster_to_df_arr.shape
            # get values and reshape arrays for comparison
            arr_flatten = (
                list(range(47, 54))
                + list(range(60, 68))
                + list(range(74, 82))
                + list(range(87, 96))
                + list(range(101, 110))
                + list(range(115, 124))
                + list(range(129, 138))
            )
            arr_flatten = np.array(arr_flatten)
            extracted_values = gdf.loc[:, gdf.columns[0]].values
            # extracted_values = extracted_values.reshape(arr_flatten.shape)
            assert np.array_equal(extracted_values, arr_flatten), (
                "the extracted values in the dataframe does not equa the real "
                "values in the array"
            )

    # def test_with_mask_multi_band(
    #     self, era5_image: gdal.Dataset, era5_image_gdf: GeoDataFrame, era5_mask: GeoDataFrame
    # ):
    #     """the input raster is given as a string path on disk."""
    #     dataset = Dataset(era5_image)
    #     gdf = dataset.to_feature_collection(add_geometry="Point", vector_mask=era5_mask)
    #     assert isinstance(gdf, GeoDataFrame)
    #     assert gdf.equals(era5_image_gdf), (
    #         "the extracted values in the dataframe does not equa the real "
    #         "values in the array"
    #     )

    class TestWithMask:
        def test_polygon_entirly_inside_raster(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            polygon_corner_coello_gdf: GeoDataFrame,
            rasterized_mask_values: np.ndarray,
        ):
            """the input mask vector is given as geodataframe.

            Parameters
            ----------
            rasterized_mask_values: array for comparison
            """
            dataset = Dataset(raster_1band_coello_gdal_dataset)
            gdf = dataset.to_feature_collection(
                polygon_corner_coello_gdf, add_geometry="Point"
            )

            poly_gdf = dataset.to_feature_collection(
                polygon_corner_coello_gdf, add_geometry="Polygon"
            )
            assert isinstance(gdf, GeoDataFrame)
            assert isinstance(poly_gdf, GeoDataFrame)
            assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(gdf["geometry"].geom_type == "Point")
            assert np.array_equal(poly_gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(poly_gdf["geometry"].geom_type == "Polygon")

        def test_polygon_partly_outside_raster(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            polygon_corner_coello_gdf: GeoDataFrame,
            rasterized_mask_values: np.ndarray,
            coello_irregular_polygon_gdf,
        ):
            """the input mask vector is given as geodataframe.

            Parameters
            ----------
            rasterized_mask_values: array for comparison
            """
            dataset = Dataset(raster_1band_coello_gdal_dataset)
            gdf = dataset.to_feature_collection(
                coello_irregular_polygon_gdf, add_geometry="Point"
            )
            poly_gdf = dataset.to_feature_collection(
                coello_irregular_polygon_gdf, add_geometry="Polygon"
            )
            assert isinstance(gdf, GeoDataFrame)
            assert isinstance(poly_gdf, GeoDataFrame)
            assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(gdf["geometry"].geom_type == "Point")
            assert np.array_equal(poly_gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(poly_gdf["geometry"].geom_type == "Polygon")


class TestExtract:
    def test_single_band(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        values = src.extract(exclude_value=0)
        assert len(values) == 46

    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(sentinel_raster)
        values = src.extract()
        arr = sentinel_raster.ReadAsArray()
        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        assert np.array_equal(arr, values)

    def test_locate_points_using_gdf(
        self,
        coello_gauges: DataFrame,
        src: Dataset,
        points_location_in_array: GeoDataFrame,
    ):
        dataset = Dataset(src)
        loc = dataset.locate_points(coello_gauges)
        assert isinstance(loc, np.ndarray)
        assert np.array_equal(points_location_in_array, loc)

    def test_locate_points_using_df(
        self,
        gauges_df: DataFrame,
        src: Dataset,
        points_location_in_array: GeoDataFrame,
    ):
        dataset = Dataset(src)
        loc = dataset.locate_points(gauges_df)
        assert isinstance(loc, np.ndarray)
        assert np.array_equal(points_location_in_array, loc)

    def test_extract_with_point_geometry_input(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
        coello_gauges: GeoDataFrame,
    ):
        src = Dataset(src)
        values = src.extract(exclude_value=0, feature=coello_gauges)
        assert len(values) == len(coello_gauges)
        assert np.array_equal(values, [4, 6, 1, 5, 49, 88])


class TestOverlay:
    def test_single_band(self, rhine_raster: gdal.Dataset, germany_classes: str):
        src_obj = Dataset(rhine_raster)
        classes_src = Dataset.read_file(germany_classes)
        class_dict = src_obj.overlay(classes_src)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values) - 1
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()[:-1]
        assert all(i in real_classes for i in extracted_classes)

    def test_multi_band(
        self, sentinel_raster: gdal.Dataset, sentinel_classes: gdal.Dataset
    ):
        dataset = Dataset(sentinel_raster)
        classes_src = Dataset(sentinel_classes)
        class_dict = dataset.overlay(classes_src, band=1)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values)
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()
        assert all(i in real_classes for i in extracted_classes)


class TestFootPrint:
    @pytest.mark.fast
    def test_raster_full_of_data(self, test_image: Dataset):
        dataset = Dataset(test_image)
        extent = dataset.footprint()
        # extent.to_file("tests/data/extent1.geojson")
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

    @pytest.mark.fast
    def test_max_depth_raster(self, footprint_test: Dataset, replace_values: List):
        dataset = Dataset(footprint_test)
        extent = dataset.footprint(exclude_values=replace_values)

        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

    @pytest.mark.fast
    def test_raster_full_of_no_data_value(
        self, test_image: gdal.Dataset, nan_raster: str
    ):
        dataset = Dataset(nan_raster)
        extent = dataset.footprint()
        assert extent is None

    @pytest.mark.fast
    def test_modis_with_replace_parameter_several_bands(
        self, modis_surf_temp: gdal.Dataset, replace_values: List
    ):
        dataset = Dataset(modis_surf_temp)
        # modis nodatavalue is gdal object is different than the array
        extent = dataset.footprint(exclude_values=replace_values)
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

    @pytest.mark.fast
    def test_era5_one_band_no_no_data_value_in_raster(
        self, era5_image: gdal.Dataset, replace_values: List
    ):
        dataset = Dataset(era5_image)
        extent = dataset.footprint(exclude_values=replace_values)
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2


def test_cluster(rhine_dem: gdal.Dataset, clusters: np.ndarray):
    dataset = Dataset(rhine_dem)
    lower_value = 0.1
    upper_value = 20
    cluster_array, count, position, values = dataset.cluster(lower_value, upper_value)
    assert count == 155
    assert np.array_equal(cluster_array, clusters)
    assert len(position) == 2364
    assert len(values) == 2364


class TestNCtoGeoTIFF:
    def test_convert_0_360_to_180_180_longitude_new_dataset(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        new_dataset = dataset.convert_longitude()
        lon = new_dataset.lon
        assert lon.max() < 180
        assert new_dataset.pivot_point == (-180, 90)

    def test_convert_0_360_to_180_180_longitude_inplace(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        dataset.convert_longitude(inplace=True)
        lon = dataset.lon
        assert lon.max() < 180
        assert dataset.pivot_point == (-180, 90)
