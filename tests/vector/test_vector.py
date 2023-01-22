import os
from typing import List, Tuple
from shapely.geometry.polygon import Polygon
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr
from osgeo.ogr import DataSource

from pyramids.vector import Vector


class TestOpenVector:
    def test_open_vector(self, test_vector_path: str):
        ds = Vector.openVector(test_vector_path)
        assert isinstance(ds, DataSource)
        assert ds.name == test_vector_path

    def test_open_geodataframe(self, test_vector_path: str):
        gdf = Vector.openVector(test_vector_path, geodataframe=True)
        assert isinstance(gdf, GeoDataFrame)


class TestCreateDataSource:
    def test_create_geojson_data_source(self, create_vector_path: str):
        Vector.createDataSource(driver="GeoJSON", path=create_vector_path)
        assert os.path.exists(
            create_vector_path
        ), "the geojson vector driver was not created in the given path"
        # clean created files
        os.remove(create_vector_path)

    def test_create_memory_data_source(
        self,
    ):
        ds = Vector.createDataSource(driver="MEMORY")
        assert isinstance(
            ds, DataSource
        ), "the in memory ogr data source object was not created correctly"
        assert ds.name == "memData"


# def test_geodataframe_to_datasource(gdf: GeoDataFrame):
#     ds = Vector.GeoDataFrameToOgr(gdf)
#     ds.name
#     print("sss")


def test_copy_driver_to_memory(data_source: DataSource):
    name = "test_copy_datasource"
    ds = Vector.copyDriverToMemory(data_source, name)
    assert isinstance(ds, DataSource)
    assert ds.name == name


def test_save_vector(data_source: DataSource, test_save_vector_path: str):
    Vector.saveVector(data_source, test_save_vector_path)
    assert os.path.exists(test_save_vector_path), "The vector file does not exist"
    # read the vector to check it
    assert ogr.GetDriverByName("GeoJSON").Open(test_save_vector_path)
    # clean
    os.remove(test_save_vector_path)


class TestCreatePolygon:
    def test_create_wkt_str(
            self,
            coordinates: List[Tuple[int, int]],
            coordinates_wkt: str,
    ):
        """
        Test create the wkt from coordinates.
        """
        coords = Vector.createPolygon(coordinates, wkt=True)
        assert isinstance(coords, str)
        assert coords == coordinates_wkt


    def test_create_polygon_object(
            self,
            coordinates: List[Tuple[int, int]],
            coordinates_wkt: str,
    ):
        """
        Test create the wkt from coordinates.
        """
        coords = Vector.createPolygon(coordinates)
        assert isinstance(coords, Polygon)

class TestCreatePoint:
    def test_create_point_geometries(self, coordinates: List[Tuple[int, int]]):
        point_list = Vector.createPoint(coordinates)
        assert isinstance(point_list, list)
        assert len(point_list) == len(coordinates)
