import os

import geopandas as gpd
import numpy as np
from geopandas.geodataframe import DataFrame, GeoDataFrame
from osgeo import gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource

from pyramids.convert import Convert
from pyramids.raster import Raster


class TestPolygonize:
    def test_save_polygon_to_disk(
        self, test_image: Dataset, polygonized_raster_path: str
    ):
        Convert.rasterToPolygon(test_image, polygonized_raster_path, driver="GeoJSON")
        assert os.path.exists(polygonized_raster_path)
        gdf = gpd.read_file(polygonized_raster_path)
        assert len(gdf) == 4
        assert all(gdf.geometry.geom_type == "Polygon")
        os.remove(polygonized_raster_path)

    def test_save_polygon_to_memory(
        self, test_image: Dataset, polygonized_raster_path: str
    ):
        gdf = Convert.rasterToPolygon(test_image)
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.geometry.geom_type == "Polygon")


class TestPolygonToRaster:
    def test_disk_inputs_and_outputs(
        self,
        vector_mask_path,
        raster_to_df_path,
        raster_to_df_dataset: Dataset,
        rasterized_mask_path: str,
        rasterized_mask_array: np.ndarray,
    ):
        """All inputs are in disk.

            - The inputs to the function are in disk.
            - The output will be written to disk.

        Parameters
        ----------
        vector_mask_path
        raster_to_df_path
        raster_to_df_dataset
        rasterized_mask_path
        rasterized_mask_array
        """
        src = Convert.polygonToRaster(
            vector_mask_path, raster_to_df_path, rasterized_mask_path
        )
        assert os.path.exists(rasterized_mask_path), (
            "The output raster should have been saved to disk at the "
            f"following path: {raster_to_df_path}"
        )
        src = gdal.Open(rasterized_mask_path)
        assert Raster.getEPSG(src) == 32618
        geo = src.GetGeoTransform()
        geo_source = raster_to_df_dataset.GetGeoTransform()
        assert geo == geo_source
        arr, no_vata_val = Raster.getRasterData(src)
        assert no_vata_val == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 16

    def test_gdf_input(
        self,
        vector_mask_gdf: GeoDataFrame,
        raster_to_df_path: str,
        raster_to_df_dataset: Dataset,
        rasterized_mask_path: str,
        rasterized_mask_array: np.ndarray,
    ):
        """Geodataframe input polygon.

            - The inputs to the function are in disk.
            - The output will be written to disk.

        Parameters
        ----------
        vector_mask_gdf
        raster_to_df_path
        raster_to_df_dataset
        rasterized_mask_path
        rasterized_mask_array
        """
        src = Convert.polygonToRaster(
            vector_mask_gdf, raster_to_df_path, rasterized_mask_path
        )
        assert os.path.exists(rasterized_mask_path), (
            "The output raster should have been saved to disk at the "
            f"following path: {raster_to_df_path}"
        )
        src = gdal.Open(rasterized_mask_path)
        assert Raster.getEPSG(src) == 32618
        geo = src.GetGeoTransform()
        geo_source = raster_to_df_dataset.GetGeoTransform()
        assert geo == geo_source
        arr, no_vata_val = Raster.getRasterData(src)
        assert no_vata_val == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 16

    def test_return_output(
        self,
        vector_mask_path: str,
        raster_to_df_path: str,
        raster_to_df_dataset: Dataset,
        rasterized_mask_array: np.ndarray,
    ):
        """Geodataframe input polygon.

            - The inputs to the function are in disk,
            - The output will be returned as gdal.Dataset.

        Parameters
        ----------
        vector_mask_path
        raster_to_df_path
        raster_to_df_dataset
        rasterized_mask_array
        """
        src = Convert.polygonToRaster(vector_mask_path, raster_to_df_path)
        assert Raster.getEPSG(src) == 32618
        geo = src.GetGeoTransform()
        geo_source = raster_to_df_dataset.GetGeoTransform()
        assert geo == geo_source
        arr, no_vata_val = Raster.getRasterData(src)
        assert no_vata_val == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 16


class TestRasterToDataFrame:
    def test_raster_to_dataframe_without_mask(
        self, raster_to_df_path: str, raster_to_df_arr: np.ndarray
    ):
        df = Convert.rasterToDataframe(raster_to_df_path)
        assert isinstance(df, DataFrame)
        rows, cols = raster_to_df_arr.shape
        arr_flatten = raster_to_df_arr.reshape((rows * cols, 1))
        assert np.array_equal(df.values, arr_flatten), (
            "the extracted values in the dataframe does not equa the real "
            "values in the array"
        )

    def test_raster_to_dataframe_with_mask(
        self, raster_to_df_path, vector_mask_path, rasterized_mask_values: np.ndarray
    ):
        df = Convert.rasterToDataframe(raster_to_df_path, vector_mask_path)
        assert isinstance(df, DataFrame)
        assert len(df) == len(rasterized_mask_values)
        assert np.array_equal(df["Band_1"].values, rasterized_mask_values), (
            "the extracted values in the dataframe "
            "does not "
            "equa the real "
            "values in the array"
        )


class TestConvertDataSourceAndGDF:
    def test_ds_to_gdf(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        gdf = Convert._ogrDataSourceToGeoDF(data_source)
        assert all(gdf == ds_geodataframe)

    def test_gdf_to_ds(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        ds = Convert._gdfToOgrDataSource(ds_geodataframe)
        assert isinstance(ds, DataSource)
        assert ds.name == "memory"
