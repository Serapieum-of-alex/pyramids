import os
from typing import List, Tuple
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr, gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource
from shapely.geometry.polygon import Polygon

from pyramids.vector import Vector
from pyramids.dataset import Dataset


class TestOpenVector:
    def test_open_vector(self, test_vector_path: str):
        ds = Vector.read(test_vector_path)
        assert isinstance(ds, DataSource)
        assert ds.name == test_vector_path

    def test_open_geodataframe(self, test_vector_path: str):
        gdf = Vector.read(test_vector_path, geodataframe=True)
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
        """Test create the wkt from coordinates."""
        coords = Vector.createPolygon(coordinates, wkt=True)
        assert isinstance(coords, str)
        assert coords == coordinates_wkt

    def test_create_polygon_object(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        """Test create the wkt from coordinates."""
        coords = Vector.createPolygon(coordinates)
        assert isinstance(coords, Polygon)


class TestCreatePoint:
    def test_create_point_geometries(self, coordinates: List[Tuple[int, int]]):
        point_list = Vector.createPoint(coordinates)
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
        src_obj = Dataset.read_file(raster_to_df_path)
        src = Vector.to_raster(vector_mask_gdf, src=src_obj)
        assert src.epsg == vector_mask_gdf.crs.to_epsg()

        xmin, _, _, ymax, _, _ = src_obj.geotransform
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
        src = Vector.to_raster(vector_mask_gdf, cell_size=200)
        assert src.epsg == vector_mask_gdf.crs.to_epsg()
        xmin, _, _, ymax = vector_mask_gdf.bounds.values[0]
        assert src.geotransform[0] == xmin
        assert src.geotransform[3] == ymax
        assert src.cell_size == 200
        arr = src.read_array()
        # assert src.no_data_value[0] == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 6400


class TestConvertDataSourceAndGDF:
    def test_ds_to_gdf(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        gdf = Vector._ogrDataSourceToGeoDF(data_source)
        assert all(gdf == ds_geodataframe)

    def test_gdf_to_ds(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        ds = Vector._gdfToOgrDataSource(ds_geodataframe)
        assert isinstance(ds, DataSource)
        assert ds.name == "memory"
