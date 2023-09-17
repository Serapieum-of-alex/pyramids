import os
from typing import List, Tuple

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr, gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

from pyramids.featurecollection import FeatureCollection
from pyramids.dataset import Dataset


class TestAttributes:
    def test_total_bound_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert all(np.isclose(feature.total_bounds, gdf_bound, rtol=0.0001))

    def test_total_bound_ds(self, data_source: DataSource, gdf_bound: List):
        feature = FeatureCollection(data_source)
        assert all(np.isclose(feature.total_bounds, gdf_bound, rtol=0.0001))

    def test_pivot_point_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        point = feature.pivot_point
        assert point == [gdf_bound[0], gdf_bound[-1]]

    def test_pivot_point_ds(self, data_source: DataSource, gdf_bound: List):
        feature = FeatureCollection(data_source)
        point = feature.pivot_point
        assert point == [gdf_bound[0], gdf_bound[-1]]

    def test_layer_count_gdf(self, gdf: GeoDataFrame):
        feature = FeatureCollection(gdf)
        layer_count = feature.layers_count
        assert layer_count is None

    def test_layer_count_ogr_ds(self, data_source: DataSource):
        feature = FeatureCollection(data_source)
        layer_count = feature.layers_count
        assert layer_count == 1

    def test_layer_names_ogr_ds(self, data_source: DataSource):
        feature = FeatureCollection(data_source)
        names = feature.layer_names
        assert names == ["poligonized"]

    def test_columns_ds(self, coello_gauges_ds: DataSource):
        feature = FeatureCollection(coello_gauges_ds)
        columns = feature.column
        assert columns == ["id", "x", "y", "geometry"]

    def test_columns_gdf(self, coello_gauges_gdf: GeoDataFrame):
        feature = FeatureCollection(coello_gauges_gdf)
        columns = feature.column
        assert columns == ["id", "x", "y", "geometry"]

    def test_dtypes_gdf(self, coello_gauges_gdf: GeoDataFrame):
        feature = FeatureCollection(coello_gauges_gdf)
        dtypes = feature.dtypes
        assert isinstance(dtypes, dict)
        assert dtypes == {
            "id": "int64",
            "x": "float64",
            "y": "float64",
            "geometry": "geometry",
        }

    def test_dtypes_ds(self, coello_gauges_ds: DataSource):
        feature = FeatureCollection(coello_gauges_ds)
        dtypes = feature.dtypes
        assert isinstance(dtypes, dict)
        assert dtypes == {
            "id": "int64",
            "x": "float64",
            "y": "float64",
        }


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
        assert isinstance(ds, FeatureCollection)
        assert isinstance(ds.feature, DataSource)
        # assert ds.name == "memory"

    def test_gdf_to_ds_inplace(self, ds_geodataframe: GeoDataFrame):
        vector = FeatureCollection(ds_geodataframe)
        ds = vector._gdf_to_ds(inplace=True)
        assert ds is None
        assert isinstance(vector.feature, DataSource)
        # assert vector.feature.name == "memory"

    def test_gdf_to_ds_if_feature_is_already_ds(
        self, data_source: DataSource, ds_geodataframe: GeoDataFrame
    ):
        vector = FeatureCollection(data_source)
        ds = vector._gdf_to_ds()
        assert isinstance(ds, FeatureCollection)
        assert isinstance(ds.feature, DataSource)

    def test_gdf_to_gdal_ex(self, ds_geodataframe: GeoDataFrame):
        vector = FeatureCollection(ds_geodataframe)
        ds = vector._gdf_to_ds(gdal_dataset=True)
        assert isinstance(ds, FeatureCollection)
        assert isinstance(ds.feature, gdal.Dataset)

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


class TestToDataset:
    """Test Convert feature collection into dataset"""

    class TestSingleColumnVector:
        """
        Descriptions
        ------------
            - Test using the dataset parameter
            - The vector we want to convert into raster has one column (which means the returned raster should have
            one band)

        Tests
        -----
         - test_single_band_dataset_parameter:
            The dataset parameter has one band
        """

        def test_single_band_dataset_parameter(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            """
            Description
            -----------
                - The dataset parameter has one band.
                - The vector itself has one column

            Parameters
            ----------
                - use the dataset parameter is used in the conversion not the cell_size.
                - the column_name parameter is None (the default value) so the function should convert all the
                columns (one column) into bands

                - both the vector and the raster shares the same pivot point, the vector locates in the top left
                corner of the raster, the vector covers only the top left corner, and not the whole raster.

            Returns
            -------
                - The resulted raster should have 3 bands full of 1, 2, 3 respectively.
                - the array of each band in the new dataset will have 4 cells.
                - the cell size of the new dataset is 8000.
            """
            dataset = Dataset.read_file(raster_1band_coello_path)
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            values = arr[arr[:, :] == 1.0]
            assert values.shape[0] == 16

    # def test_dataset_parameter_error_case(
    #         self, era5_image: gdal.Dataset, era5_mask: GeoDataFrame
    # ):
    #     """
    #     This case does not give the right result when we use the dataset parameter
    #     """
    #     dataset = Dataset(era5_image)
    #     vector = FeatureCollection(era5_mask)
    #     src = vector.to_dataset(cell_size=0.080)  #dataset=dataset
    #     # src.to_file("ssss.tif")

    class TestMultiColumnVector:
        def test_single_band_dataset_parameter_one_column(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            """Geodataframe input polygon.

                test convert a vector to a raster using a dataset as a parameter
                - both the vector and the raster shares the same pivot point, the vector locates in the top left
                corner of the raster, the vector covers only the top left corner, and not the whole raster.
                - the raster is a single band.
                - use columne_name parameter to tell the method which column you want to take the value from.

            Parameters
            ----------
            raster_to_df_path
            """
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset, column_name="column_2")
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            values = arr[arr[:, :] == 2]
            assert values.shape[0] == 16

        def test_single_band_dataset_parameter_multiple_columns(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            """Geodataframe input polygon.

                test convert a vector to a raster using a dataset as a parameter
                - both the vector and the raster shares the same pivot point, the vector locates in the top left
                corner of the raster, the vector covers only the top left corner, and not the whole raster.
                - the raster is a single band.
                - use columne_name parameter to tell the method which column you want to take the value from.

            Parameters
            ----------
            raster_to_df_path
            """
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(
                dataset=dataset, column_name=["column_2", "column_3"]
            )
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            assert arr.shape == (2, 13, 14)
            band_1_values = arr[0, arr[0, :, :] == 2]
            band_2_values = arr[1, arr[1, :, :] == 3]
            assert band_1_values.shape[0] == 16
            assert band_2_values.shape[0] == 16

        def test_single_band_dataset_parameter_none_column_name(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            """Geodataframe input polygon.

                test convert a vector to a raster using a dataset as a parameter
                - both the vector and the raster shares the same pivot point, the vector locates in the top left
                corner of the raster, the vector covers only the top left corner, and not the whole raster.
                - the raster is a single band.
                - use columne_name parameter to tell the method which column you want to take the value from.
                - The top left corner of both the raster and the polygon are very close to each other but not identical.

            Parameters
            ----------
            raster_to_df_path
            """
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset, column_name=None)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            assert arr.shape == (3, 13, 14)
            band_1_values = arr[0, arr[0, :, :] == 1]
            band_2_values = arr[1, arr[1, :, :] == 2]
            band_3_values = arr[2, arr[2, :, :] == 3]
            assert band_1_values.shape[0] == 16
            assert band_2_values.shape[0] == 16
            assert band_3_values.shape[0] == 16

    class TestCellSize:
        """
        Test converting feature collection into dataset using cell size parameter.

        Tests
        -----
        test_none_column_name:
            multi columns, very course resolution (8000)
        """

        def test_none_column_name(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
        ):
            """
            Description
            -----------
                - Add more columns to the the one geometry vector to create multi-band raster.

            Parameters
            ----------
                - use the cell_size parameter in the conversion not the dataset.
                - the column_name parameter is None (the default value) so the function should convert all the columns in

            Returns
            -------
                - The resulted raster should have 3 bands full of 1, 2, 3 respectively.
                - the array of each band in the new dataset will have 4 cells.
                - the cell size of the new dataset is 8000.
            """
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            required_cell_size = 8000

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(cell_size=required_cell_size, column_name=None)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, ymax = vector.pivot_point
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == required_cell_size
            arr = src.read_array()
            assert arr.shape == (3, 2, 2)
            band_1_values = arr[0, arr[0, :, :] == 1]
            band_2_values = arr[1, arr[1, :, :] == 2]
            band_3_values = arr[2, arr[2, :, :] == 3]
            assert band_1_values.shape[0] == 4
            assert band_2_values.shape[0] == 4
            assert band_3_values.shape[0] == 4

        def test_one_column(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
        ):
            """
            Description
            -----------
                - The vector has one column

            Parameters
            ----------
                - use the cell_size parameter in the conversion not the dataset.
                - the column_name parameter is None (the default value) so the function should convert all the columns(
                one column)

            Returns
            -------
                - The resulted raster should have 1 bands full of 1.
                - the array of each band in the new dataset will have 6400 cells.
                - the cell size of the new dataset is 200.
            """
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(cell_size=200)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()
            xmin, _, _, ymax = polygon_corner_coello_gdf.bounds.values[0]
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 200
            arr = src.read_array()
            # assert src.no_data_value[0] == 0.0
            values = arr[arr[:, :] == 1.0]
            assert values.shape[0] == 6400

        def test_multi_polygon(
            self,
            polygons_coello_gdf: GeoDataFrame,
        ):
            """
            Description
            -----------
                - Add more columns to the the one geometry vector to create multi-band raster.

            Parameters
            ----------
                - use the cell_size parameter in the conversion not the dataset.
                - the column_name parameter is None (the default value) so the function should convert all the columns in

            Returns
            -------
                - The resulted raster should have 3 bands full of 1, 2, 3 respectively.
                - the array of each band in the new dataset will have 4 cells.
                - the cell size of the new dataset is 4000.
            """
            required_cell_size = 4000

            vector = FeatureCollection(polygons_coello_gdf)
            src = vector.to_dataset(cell_size=required_cell_size, column_name=None)
            assert src.epsg == polygons_coello_gdf.crs.to_epsg()

            xmin, ymax = vector.pivot_point
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == required_cell_size
            arr = src.read_array()
            assert arr.shape == (3, 13, 13)
            band_0_values_1 = arr[0, arr[0, :, :] == 1]
            band_0_values_2 = arr[0, arr[0, :, :] == 2]
            band_0_values_3 = arr[0, arr[0, :, :] == 3]
            band_0_values_4 = arr[0, arr[0, :, :] == 4]
            assert band_0_values_1.shape[0] == 20
            assert band_0_values_2.shape[0] == 27
            assert band_0_values_3.shape[0] == 21
            assert band_0_values_4.shape[0] == 24


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
