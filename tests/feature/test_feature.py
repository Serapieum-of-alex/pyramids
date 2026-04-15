"""Integration tests for FeatureCollection (GeoDataFrame-subclass design).

These tests focus on behavior that survives the ARC-1a / ARC-1b refactor:
GeoDataFrame-only construction, the pyramids-specific properties
(``epsg``, ``total_bounds``, ``top_left_corner``, ``column``, ``dtypes``),
I/O (``read_file``, ``to_file``), geometry factories, rasterization
(``to_dataset``), xy / center_point, concate.

Tests that exercised the old OGR-accepting public surface
(``create_ds``, ``_copy_driver_to_memory``, ``_ds_to_gdf``,
``_gdf_to_ds``, the ``.feature`` property, ``layers_count``,
``layer_names``, ``file_name``) were removed — those surfaces no
longer exist.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry.polygon import Polygon

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection


class TestAttributes:
    def test_total_bound_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert all(np.isclose(feature.total_bounds, gdf_bound, rtol=0.0001))

    def test_top_left_corner_gdf(self, gdf: GeoDataFrame, gdf_bound: List):
        feature = FeatureCollection(gdf)
        assert feature.top_left_corner == [gdf_bound[0], gdf_bound[-1]]

    def test_columns_gdf(self, coello_gauges_gdf: GeoDataFrame):
        feature = FeatureCollection(coello_gauges_gdf)
        assert feature.column == ["id", "x", "y", "geometry"]


class TestReadFile:
    def test_read_returns_featurecollection(self, test_vector_path: str):
        fc = FeatureCollection.read_file(test_vector_path)
        # FeatureCollection IS a GeoDataFrame after ARC-1a, so this must hold.
        assert isinstance(fc, FeatureCollection)
        assert isinstance(fc, GeoDataFrame)


class TestToFile:
    def test_save_gdf(self, gdf: GeoDataFrame, test_save_vector_path: Path):
        vector = FeatureCollection(gdf)
        vector.to_file(test_save_vector_path)
        assert test_save_vector_path.exists()
        test_save_vector_path.unlink()


class TestCreatePolygon:
    def test_create_wkt_str(
        self,
        coordinates: List[Tuple[int, int]],
        coordinates_wkt: str,
    ):
        coords = FeatureCollection.create_polygon(coordinates, wkt=True)
        assert isinstance(coords, str)
        assert coords == coordinates_wkt

    def test_create_polygon_object(self, coordinates: List[Tuple[int, int]]):
        coords = FeatureCollection.create_polygon(coordinates)
        assert isinstance(coords, Polygon)


class TestCreatePoint:
    def test_return_shapely_object(self, coordinates: List[Tuple[int, int]]):
        point_list = FeatureCollection.create_point(coordinates)
        assert isinstance(point_list, list)
        assert len(point_list) == len(coordinates)

    def test_return_featurecollection(self, coordinates: List[Tuple[int, int]]):
        point_fc = FeatureCollection.create_point(coordinates, epsg=4326)
        assert isinstance(point_fc, FeatureCollection)
        assert len(point_fc["geometry"]) == len(coordinates)
        assert point_fc.epsg == 4326


class TestToDataset:
    """Test rasterization via :meth:`FeatureCollection.to_dataset`."""

    class TestSingleColumnVector:
        def test_single_band_dataset_parameter(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset)
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            xmin, _, _, ymax, _, _ = dataset.geotransform
            assert src.geotransform[0] == xmin
            assert src.geotransform[3] == ymax
            assert src.cell_size == 4000
            arr = src.read_array()
            values = arr[~np.isclose(arr, src.no_data_value[0])]
            assert values.shape[0] == 16
            assert values.mean() == vector["fid"].values[0]

    class TestMultiColumnVector:
        def test_single_band_dataset_parameter_one_column(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset, column_name="column_2")
            assert src.epsg == polygon_corner_coello_gdf.crs.to_epsg()

            arr = src.read_array()
            values = arr[~np.isclose(arr, src.no_data_value[0])]
            assert values.shape[0] == 16
            assert values.mean() == 2

        def test_single_band_dataset_parameter_multiple_columns(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(
                dataset=dataset, column_name=["column_2", "column_3"]
            )
            arr = src.read_array()
            assert arr.shape == (2, 13, 14)
            band_1_values = arr[0, ~np.isclose(arr[0, :, :], src.no_data_value[0])]
            band_2_values = arr[1, ~np.isclose(arr[1, :, :], src.no_data_value[1])]
            assert band_1_values.mean() == 2
            assert band_2_values.mean() == 3

        def test_single_band_dataset_parameter_none_column_name(
            self,
            polygon_corner_coello_gdf: GeoDataFrame,
            raster_1band_coello_path: str,
        ):
            dataset = Dataset.read_file(raster_1band_coello_path)
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(dataset=dataset, column_name=None)

            arr = src.read_array()
            assert arr.shape == (3, 13, 14)

    class TestCellSize:
        def test_none_column_name(self, polygon_corner_coello_gdf: GeoDataFrame):
            polygon_corner_coello_gdf["column_2"] = 2
            polygon_corner_coello_gdf["column_3"] = 3
            required_cell_size = 8000

            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(cell_size=required_cell_size, column_name=None)
            assert src.cell_size == required_cell_size
            arr = src.read_array()
            assert arr.shape == (3, 2, 2)

        def test_one_column(self, polygon_corner_coello_gdf: GeoDataFrame):
            vector = FeatureCollection(polygon_corner_coello_gdf)
            src = vector.to_dataset(cell_size=200)
            assert src.cell_size == 200
            arr = src.read_array()
            values = arr[arr[:, :] == 1.0]
            assert values.shape[0] == 6400

        def test_multi_polygon(self, polygons_coello_gdf: GeoDataFrame):
            required_cell_size = 4000
            vector = FeatureCollection(polygons_coello_gdf)
            src = vector.to_dataset(cell_size=required_cell_size, column_name=None)
            arr = src.read_array()
            assert arr.shape == (3, 13, 13)


class TestMultiGeomHandler:
    def test_multi_points_with_multiple_point(
        self, multi_point_geom, point_coords: list
    ):
        res = FeatureCollection._multi_geom_handler(
            multi_point_geom, "x", "MultiPoint"
        )
        assert all(np.isclose(res, [point_coords[0], point_coords[0]], rtol=0.00001))

    def test_multi_points_with_one_point(
        self, multi_point_one_point_geom, point_coords: list
    ):
        res = FeatureCollection._multi_geom_handler(
            multi_point_one_point_geom, "x", "MultiPoint"
        )
        assert np.isclose(res[0], point_coords[0], rtol=0.00001)

    def test_multi_polygons(self, multi_polygon_geom, multi_polygon_coords_x: list):
        res = FeatureCollection._multi_geom_handler(
            multi_polygon_geom, "x", "MultiPolygon"
        )
        assert res == multi_polygon_coords_x

    def test_multi_linestring(self, multi_line_geom, multi_linestring_coords_x: list):
        res = FeatureCollection._multi_geom_handler(
            multi_line_geom, "x", "MULTILINESTRING"
        )
        assert res == multi_linestring_coords_x


class TestXY:
    def test_points(self, points_gdf: GeoDataFrame, points_gdf_x, points_gdf_y):
        feature = FeatureCollection(points_gdf)
        feature.xy()
        assert all(np.isclose(feature["y"].values, points_gdf_y, rtol=0.000001))
        assert all(np.isclose(feature["x"].values, points_gdf_x, rtol=0.000001))

    def test_multi_points(
        self, multi_points_gdf: GeoDataFrame, multi_points_gdf_x, multi_points_gdf_y
    ):
        feature = FeatureCollection(multi_points_gdf)
        feature.xy()
        assert all(
            np.isclose(
                feature.loc[0, "y"], multi_points_gdf_y[0][0], rtol=0.000001
            )
        )
        assert all(
            np.isclose(
                feature.loc[0, "x"], multi_points_gdf_x[0][0], rtol=0.000001
            )
        )

    def test_multi_points_2(self, multi_points_gdf_2: GeoDataFrame, point_coords_2):
        feature = FeatureCollection(multi_points_gdf_2)
        feature.xy()
        assert feature.loc[0, "x"] == [point_coords_2[0][0], point_coords_2[1][0]]
        assert feature.loc[0, "y"] == [point_coords_2[0][1], point_coords_2[1][1]]

    def test_polygons(
        self, polygons_gdf: GeoDataFrame, polygon_gdf_x: list, polygon_gdf_y: list
    ):
        feature = FeatureCollection(polygons_gdf)
        feature.xy()
        assert isinstance(feature.loc[0, "y"], list)
        assert all(np.isclose(feature.loc[0, "y"], polygon_gdf_y, rtol=0.000001))
        assert all(np.isclose(feature.loc[0, "x"], polygon_gdf_x, rtol=0.000001))

    def test_multi_polygons(
        self,
        multi_polygon_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(multi_polygon_gdf)
        feature.xy()
        assert isinstance(feature.loc[0, "x"], list)
        assert all(
            np.isclose(feature.loc[0, "x"], multi_polygon_gdf_coords_x, rtol=0.000001)
        )

    def test_geometry_collection(
        self,
        geometry_collection_gdf: GeoDataFrame,
        multi_polygon_gdf_coords_x: list,
    ):
        feature = FeatureCollection(geometry_collection_gdf)
        feature.xy()
        assert feature.loc[0, "x"] == 100.0
        assert feature.loc[1, "x"] == [101.0, 102.0]
        assert feature.loc[2, "x"] == [
            460717.3717217822,
            456004.5874004898,
            456929.2331169145,
            459285.1699671757,
            462651.74958306097,
            460717.3717217822,
        ]


class TestConcatenate:
    """ARC-11: concatenate() is the canonical spelling; concate is a deprecated alias."""

    def test_return_new_gdf(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        gdf = feature.concatenate(geometry_collection_gdf)
        assert len(gdf) == 2
        assert gdf.loc[0, "geometry"].geom_type == "GeometryCollection"

    def test_inplace(self, geometry_collection_gdf: GeoDataFrame):
        feature = FeatureCollection(geometry_collection_gdf)
        result = feature.concatenate(geometry_collection_gdf, inplace=True)
        assert result is None
        assert len(feature) == 2
        assert feature.loc[0, "geometry"].geom_type == "GeometryCollection"

    def test_concate_is_deprecated_alias(self, geometry_collection_gdf: GeoDataFrame):
        """``concate`` (the old misspelling) still works but warns."""
        import warnings as _w

        feature = FeatureCollection(geometry_collection_gdf)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            gdf = feature.concate(geometry_collection_gdf)
        deprecated = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecated, "concate should emit a DeprecationWarning"
        assert "concatenate" in str(deprecated[0].message)
        assert len(gdf) == 2  # behavior still correct


def test_center_point(polygons_gdf: GeoDataFrame):
    feature = FeatureCollection(polygons_gdf)
    gdf = feature.center_point()
    assert "center_point" in gdf.columns
