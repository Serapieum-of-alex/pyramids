import os
from typing import List, Tuple
import shutil
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr, gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource
from shapely.geometry.polygon import Polygon

from pyramids.feature import Feature
from pyramids.dataset import Dataset


class TestReadFile:
    def test_open_vector(self, test_vector_path: str):
        ds = Feature.read_file(test_vector_path, engine="ogr")
        assert isinstance(ds.feature, DataSource)
        assert ds.feature.name == test_vector_path

    def test_open_geodataframe(self, test_vector_path: str):
        vector = Feature.read_file(test_vector_path, engine="geopandas")
        assert isinstance(vector.feature, GeoDataFrame)


class TestToFile:
    def test_save_ds(self, data_source: DataSource, test_save_vector_path: str):
        vector = Feature(data_source)
        vector.to_file(test_save_vector_path)
        assert os.path.exists(test_save_vector_path), "The vector file does not exist"
        # read the vector to check it
        assert ogr.GetDriverByName("GeoJSON").Open(test_save_vector_path)
        # clean
        os.remove(test_save_vector_path)

    def test_save_gdf(self, gdf: GeoDataFrame, test_save_vector_path: str):
        vector = Feature(gdf)
        vector.to_file(test_save_vector_path)
        assert os.path.exists(test_save_vector_path), "The vector file does not exist"
        # clean
        os.remove(test_save_vector_path)


class TestCreateDataSource:
    def test_create_geojson_data_source(self, create_vector_path: str):
        if os.path.exists(create_vector_path):
            os.remove(create_vector_path)
        ds = Feature.create_ds(driver="geojson", path=create_vector_path)
        assert isinstance(ds, DataSource)
        ds = None
        assert os.path.exists(
            create_vector_path
        ), "the geojson vector driver was not created in the given path"
        # clean created files
        os.remove(create_vector_path)

    def test_create_memory_data_source(
        self,
    ):
        ds = Feature.create_ds(driver="memory")
        assert isinstance(
            ds, DataSource
        ), "the in memory ogr data source object was not created correctly"
        assert ds.name == "memData"


def test_copy_driver_to_memory(data_source: DataSource):
    name = "test_copy_datasource"
    ds = Feature._copy_driver_to_memory(data_source, name)
    assert isinstance(ds, DataSource)
    assert ds.name == name


class TestConvert:
    def test_ds_to_gdf(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        vector = Feature(data_source)
        gdf = vector._ds_to_gdf()
        assert isinstance(gdf, GeoDataFrame)
        assert all(gdf == ds_geodataframe)

    def test_gdf_to_ds(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        vector = Feature(ds_geodataframe)
        ds = vector._gdf_to_ds()
        assert isinstance(ds, DataSource)
        assert ds.name == "memory"


# def test_geodataframe_to_datasource(gdf: GeoDataFrame):
#     ds = Feature.GeoDataFrameToOgr(gdf)
#     ds.name
#     print("sss")


class TestCreatePolygon:
    def test_create_wkt_str(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        """Test create the wkt from coordinates."""
        coords = Feature.create_polygon(coordinates, wkt=True)
        assert isinstance(coords, str)
        assert coords == coordinates_wkt

    def test_create_polygon_object(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        """Test create the wkt from coordinates."""
        coords = Feature.create_polygon(coordinates)
        assert isinstance(coords, Polygon)


class TestCreatePoint:
    def test_create_point_geometries(self, coordinates: List[Tuple[int, int]]):
        point_list = Feature.create_point(coordinates)
        assert isinstance(point_list, list)
        assert len(point_list) == len(coordinates)


class TestPolygonToRaster:
    def test_with_raster_obj(
        self,
        vector_mask_gdf: GeoDataFrame,
        raster_to_df_path: str,
    ):
        """Geodataframe input polygon.

            - The inputs to the function are in disk,
            - The output will be returned as gdal.Datacube.

        Parameters
        ----------
        raster_to_df_path
        """
        dataset = Dataset.read_file(raster_to_df_path)
        vector = Feature(vector_mask_gdf)
        src = vector.to_raster(src=dataset)
        assert src.epsg == vector_mask_gdf.crs.to_epsg()

        xmin, _, _, ymax, _, _ = dataset.geotransform
        assert src.geotransform[0] == xmin
        assert src.geotransform[3] == ymax
        assert src.cell_size == 4000
        arr = src.read_array()
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 16

    def test_with_cell_size_inout(
        self,
        vector_mask_gdf: GeoDataFrame,
    ):
        """Geodataframe input polygon.

            - The inputs to the function are in disk,
            - The output will be returned as gdal.Datacube.

        Parameters
        ----------
        vector_mask_gdf
        """
        vector = Feature(vector_mask_gdf)
        src = vector.to_raster(cell_size=200)
        assert src.epsg == vector_mask_gdf.crs.to_epsg()
        xmin, _, _, ymax = vector_mask_gdf.bounds.values[0]
        assert src.geotransform[0] == xmin
        assert src.geotransform[3] == ymax
        assert src.cell_size == 200
        arr = src.read_array()
        # assert src.no_data_value[0] == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 6400
