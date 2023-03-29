import os
from typing import List, Tuple

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

from pyramids.featurecollection import FeatureCollection
from pyramids.dataset import Dataset


class TestAttributes:
    def test_bound(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert feature.bounds == gdf_bound


class TestReadFile:
    def test_open_geodataframe(self, test_vector_path: str):
        vector = FeatureCollection.read_file(test_vector_path)
        assert isinstance(vector.feature, GeoDataFrame)


class TestToFile:
    def test_save_ds(self, data_source: DataSource, test_save_vector_path: str):
        vector = FeatureCollection(data_source)
        vector.to_file(test_save_vector_path)
        assert os.path.exists(test_save_vector_path), "The vector file does not exist"
        # read the vector to check it
        assert ogr.GetDriverByName("GeoJSON").Open(test_save_vector_path)
        # clean
        os.remove(test_save_vector_path)

    def test_save_gdf(self, gdf: GeoDataFrame, test_save_vector_path: str):
        vector = FeatureCollection(gdf)
        vector.to_file(test_save_vector_path)
        assert os.path.exists(test_save_vector_path), "The vector file does not exist"
        # clean
        os.remove(test_save_vector_path)


class TestCreateDataSource:
    def test_create_geojson_data_source(self, create_vector_path: str):
        if os.path.exists(create_vector_path):
            os.remove(create_vector_path)
        ds = FeatureCollection.create_ds(driver="geojson", path=create_vector_path)
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
        ds = FeatureCollection.create_ds(driver="memory")
        assert isinstance(
            ds, DataSource
        ), "the in memory ogr data source object was not created correctly"
        assert ds.name == "memData"


def test_copy_driver_to_memory(data_source: DataSource):
    name = "test_copy_datasource"
    ds = FeatureCollection._copy_driver_to_memory(data_source, name)
    assert isinstance(ds, DataSource)
    assert ds.name == name


class TestConvert:
    def test_ds_to_gdf(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        vector = FeatureCollection(data_source)
        gdf = vector._ds_to_gdf()
        assert isinstance(gdf, GeoDataFrame)
        assert all(gdf == ds_geodataframe)

    def test_ds_to_gdf_inplace(
        self, data_source: DataSource, ds_geodataframe: GeoDataFrame
    ):
        vector = FeatureCollection(data_source)
        gdf = vector._ds_to_gdf(inplace=True)
        assert gdf is None
        assert isinstance(vector.feature, GeoDataFrame)
        assert all(vector.feature == ds_geodataframe)

    def test_gdf_to_ds(self, ds_geodataframe: GeoDataFrame):
        vector = FeatureCollection(ds_geodataframe)
        ds = vector._gdf_to_ds()
        assert isinstance(ds, DataSource)
        # assert ds.name == "memory"

    def test_gdf_to_ds_inplace(self, ds_geodataframe: GeoDataFrame):
        vector = FeatureCollection(ds_geodataframe)
        ds = vector._gdf_to_ds(inplace=True)
        assert ds is None
        assert isinstance(vector.feature, DataSource)
        # assert vector.feature.name == "memory"

    # def test_gdf_to_ds2(
    #         self, ds_geodataframe: GeoDataFrame
    # ):
    #     feature = FeatureCollection(ds_geodataframe)
    #     vector = feature.gdf_to_ds()
    #     layer = vector.GetLayer(0)
    #     assert len(list(layer)) == len(ds_geodataframe)
    #
    #     assert isinstance(vector.feature, DataSource)
    #     # assert vector.feature.name == "memory"


# def test_geodataframe_to_datasource(gdf: GeoDataFrame):
#     ds = FeatureCollection.GeoDataFrameToOgr(gdf)
#     ds.name
#     print("sss")


class TestCreatePolygon:
    def test_create_wkt_str(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        """Test create the wkt from coordinates."""
        coords = FeatureCollection.create_polygon(coordinates, wkt=True)
        assert isinstance(coords, str)
        assert coords == coordinates_wkt

    def test_create_polygon_object(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        """Test create the wkt from coordinates."""
        coords = FeatureCollection.create_polygon(coordinates)
        assert isinstance(coords, Polygon)


class TestCreatePoint:
    def test_create_point_geometries(self, coordinates: List[Tuple[int, int]]):
        point_list = FeatureCollection.create_point(coordinates)
        assert isinstance(point_list, list)
        assert len(point_list) == len(coordinates)


class TestToRaster:
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
        vector = FeatureCollection(vector_mask_gdf)
        src = vector.to_dataset(src=dataset)
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
        vector = FeatureCollection(vector_mask_gdf)
        src = vector.to_dataset(cell_size=200)
        assert src.epsg == vector_mask_gdf.crs.to_epsg()
        xmin, _, _, ymax = vector_mask_gdf.bounds.values[0]
        assert src.geotransform[0] == xmin
        assert src.geotransform[3] == ymax
        assert src.cell_size == 200
        arr = src.read_array()
        # assert src.no_data_value[0] == 0.0
        values = arr[arr[:, :] == 1.0]
        assert values.shape[0] == 6400


class Test_multi_geom_handler:
    def test_multi_points_with_multiple_point(
        self, multi_point_geom, point_coords: list
    ):
        coord_type = "x"
        gtype = "MultiPoint"
        res = FeatureCollection._multi_geom_handler(multi_point_geom, coord_type, gtype)
        assert all(np.isclose(res, [point_coords[0], point_coords[0]], rtol=0.00001))

    def test_multi_points_with_one_point(
        self, multi_point_one_point_geom, point_coords: list
    ):
        coord_type = "x"
        gtype = "MultiPoint"
        res = FeatureCollection._multi_geom_handler(
            multi_point_one_point_geom, coord_type, gtype
        )
        assert np.isclose(res[0], point_coords[0], rtol=0.00001)

    def test_multi_polygons(self, multi_polygon_geom, multi_polygon_coords_x: list):
        coord_type = "x"
        gtype = "MultiPolygon"
        res = FeatureCollection._multi_geom_handler(
            multi_polygon_geom, coord_type, gtype
        )
        assert res == multi_polygon_coords_x

    def test_multi_linestring(self, multi_line_geom, multi_linestring_coords_x: list):
        coord_type = "x"
        gtype = "MULTILINESTRING"
        res = FeatureCollection._multi_geom_handler(multi_line_geom, coord_type, gtype)
        assert res == multi_linestring_coords_x


class TestXY:
    def test_points(self, points_gdf: GeoDataFrame, points_gdf_x, points_gdf_y):
        feature = FeatureCollection(points_gdf)
        feature.xy()
        assert all(np.isclose(feature.feature["y"].values, points_gdf_y, rtol=0.000001))
        assert all(np.isclose(feature.feature["x"].values, points_gdf_x, rtol=0.000001))

    def test_multi_points(
        self, multi_points_gdf: GeoDataFrame, multi_points_gdf_x, multi_points_gdf_y
    ):
        feature = FeatureCollection(multi_points_gdf)
        feature.xy()
        assert all(
            np.isclose(
                feature.feature.loc[0, "y"], multi_points_gdf_y[0][0], rtol=0.000001
            )
        )
        assert all(
            np.isclose(
                feature.feature.loc[0, "x"], multi_points_gdf_x[0][0], rtol=0.000001
            )
        )

    def test_multi_points_2(self, multi_points_gdf_2: GeoDataFrame, point_coords_2):
        feature = FeatureCollection(multi_points_gdf_2)
        feature.xy()
        assert feature.feature.loc[0, "x"] == [
            point_coords_2[0][0],
            point_coords_2[1][0],
        ]
        assert feature.feature.loc[0, "y"] == [
            point_coords_2[0][1],
            point_coords_2[1][1],
        ]

    def test_polygons(
        self, polygons_gdf: GeoDataFrame, polygon_gdf_x: list, polygon_gdf_y: list
    ):
        feature = FeatureCollection(polygons_gdf)
        feature.xy()
        assert isinstance(feature.feature.loc[0, "y"], list)
        assert all(
            np.isclose(feature.feature.loc[0, "y"], polygon_gdf_y, rtol=0.000001)
        )
        assert all(
            np.isclose(feature.feature.loc[0, "x"], polygon_gdf_x, rtol=0.000001)
        )

    def test_multi_polygons(
        self,
        multi_polygon_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(multi_polygon_gdf)
        feature.xy()
        assert isinstance(feature.feature.loc[0, "x"], list)
        assert all(
            np.isclose(
                feature.feature.loc[0, "x"], multi_polygon_gdf_coords_x, rtol=0.000001
            )
        )

    def test_geometry_collection(
        self,
        geometry_collection_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(geometry_collection_gdf)
        feature.xy()
        assert feature.feature.loc[0, "x"] == 100.0
        assert feature.feature.loc[1, "x"] == [101.0, 102.0]
        assert feature.feature.loc[2, "x"] == [
            460717.3717217822,
            456004.5874004898,
            456929.2331169145,
            459285.1699671757,
            462651.74958306097,
            460717.3717217822,
        ]


class TestConcate:
    def test_return_new_gdf(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        gdf = feature.concate(geometry_collection_gdf)
        assert len(gdf) == 2
        assert gdf.loc[0, "geometry"].geom_type == "GeometryCollection"

    def test_inplace(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        gdf = feature.concate(geometry_collection_gdf, inplace=True)
        assert gdf is None
        assert len(feature.feature) == 2
        assert feature.feature.loc[0, "geometry"].geom_type == "GeometryCollection"


def test_center_point(polygons_gdf: GeoDataFrame):
    feature = FeatureCollection(polygons_gdf)
    gdf = feature.center_point()
    assert "center_point" in gdf.columns
    assert isinstance(gdf.loc[0, "center_point"], Point)
