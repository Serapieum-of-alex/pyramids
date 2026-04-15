"""Unit tests for FeatureCollection methods (GeoDataFrame-subclass design).

These tests cover the pyramids-specific surface of ``FeatureCollection``
after the ARC-1a/ARC-1b refactor:

- Static helpers: ``_geometry_collection``, ``_explode_gdf``,
  ``_multi_geom_handler``, ``_get_coords``, ``_get_xy_coords``,
  ``_get_point_coords``, ``_create_sr_from_proj``.
- Instance methods: ``center_point``, ``dtypes``, ``__str__``.
- Classmethods / statics: ``create_point``, ``create_polygon``,
  ``reproject_points2``, ``get_epsg_from_prj``.
- The ARC-1b guard: constructing ``FeatureCollection(ogr.DataSource)``
  raises ``TypeError`` and the constructor accepts only what
  ``GeoDataFrame`` accepts.

Tests that exercised the old OGR-accepting public surface
(``create_ds``, ``_copy_driver_to_memory``, ``_ds_to_gdf``,
``_gdf_to_ds``, ``layers_count``, ``layer_names``, ``file_name``,
``_explode_multi_geometry``, the ``.feature`` property) have been
removed with ARC-1a/ARC-1b; those surfaces no longer exist.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas import GeoDataFrame
from osgeo import ogr, osr
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.collection import GeometryCollection

from pyramids.feature import FeatureCollection


@pytest.fixture()
def simple_polygon_gdf() -> GeoDataFrame:
    """GeoDataFrame with a single polygon in WGS 84."""
    poly = box(30.0, 30.0, 31.0, 31.0)
    gdf = gpd.GeoDataFrame({"value": [42]}, geometry=[poly], crs="EPSG:4326")
    return gdf


@pytest.fixture()
def multipolygon_gdf() -> GeoDataFrame:
    """GeoDataFrame whose geometry column contains a MultiPolygon."""
    poly1 = box(0.0, 0.0, 1.0, 1.0)
    poly2 = box(2.0, 2.0, 3.0, 3.0)
    mpoly = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[mpoly], crs="EPSG:4326")
    return gdf


@pytest.fixture()
def multipoint_geom() -> MultiPoint:
    """A MultiPoint geometry with three points."""
    return MultiPoint([Point(1, 2), Point(3, 4), Point(5, 6)])


@pytest.fixture()
def multilinestring_geom() -> MultiLineString:
    """A MultiLineString geometry."""
    return MultiLineString([LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])])


@pytest.fixture()
def multipolygon_geom() -> MultiPolygon:
    """A MultiPolygon geometry."""
    return MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])


@pytest.fixture()
def geometry_collection_geom() -> GeometryCollection:
    """A GeometryCollection with a Point, LineString, and Polygon."""
    return GeometryCollection(
        [
            Point(10, 20),
            LineString([(0, 0), (1, 1), (2, 0)]),
            box(5, 5, 6, 6),
        ]
    )


@pytest.fixture()
def wgs84_wkt() -> str:
    """WKT string for WGS 84."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()


@pytest.fixture()
def utm_proj4() -> str:
    """Proj4 string for UTM zone 36N."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32636)
    return srs.ExportToProj4()


# ── ARC-1a / ARC-1b : subclass contract + OGR guard ─────────────────


class TestSubclassContract:
    """FeatureCollection subclasses GeoDataFrame and preserves identity."""

    def test_is_geodataframe(self, simple_polygon_gdf: GeoDataFrame):
        """FeatureCollection IS a GeoDataFrame."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert isinstance(fc, GeoDataFrame)
        assert isinstance(fc, FeatureCollection)

    def test_slice_returns_featurecollection(self, simple_polygon_gdf):
        """Slicing preserves the FeatureCollection subclass identity."""
        gdf = gpd.GeoDataFrame(
            {"v": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(gdf)
        subset = fc[fc["v"] > 1]
        assert isinstance(subset, FeatureCollection)

    def test_copy_returns_featurecollection(self, simple_polygon_gdf):
        """copy() preserves the FeatureCollection subclass identity."""
        fc = FeatureCollection(simple_polygon_gdf)
        dup = fc.copy()
        assert isinstance(dup, FeatureCollection)

    def test_inherited_to_crs_works(self, simple_polygon_gdf):
        """Inherited ``to_crs`` works directly (no pyramids implementation)."""
        fc = FeatureCollection(simple_polygon_gdf)
        out = fc.to_crs(3857)
        assert isinstance(out, FeatureCollection)
        assert out.crs.to_epsg() == 3857

    def test_constructor_rejects_ogr_datasource(self, tmp_path):
        """ARC-1b: FeatureCollection(ogr.DataSource) raises TypeError."""
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly], crs="EPSG:4326")
        path = tmp_path / "t.geojson"
        gdf.to_file(path, driver="GeoJSON")
        ds = ogr.Open(str(path))
        with pytest.raises(TypeError, match="no longer accepts"):
            FeatureCollection(ds)


# ── static / helper coverage ────────────────────────────────────────


class TestGeometryCollection:
    """Tests for ``_geometry_collection`` static method."""

    def test_extracts_point_x(self, geometry_collection_geom):
        """A Point(10, 20) should appear as 10.0 in the x-coord list."""
        result = FeatureCollection._geometry_collection(geometry_collection_geom, "x")
        assert 10.0 in result

    def test_extracts_point_y(self, geometry_collection_geom):
        """A Point(10, 20) should appear as 20.0 in the y-coord list."""
        result = FeatureCollection._geometry_collection(geometry_collection_geom, "y")
        assert 20.0 in result

    def test_extracts_linestring_coords(self, geometry_collection_geom):
        """LineString [(0,0),(1,1),(2,0)] x-coords should include 0, 1, 2."""
        result = FeatureCollection._geometry_collection(geometry_collection_geom, "x")
        assert 0.0 in result and 1.0 in result and 2.0 in result

    def test_extracts_polygon_coords(self, geometry_collection_geom):
        """Polygon box(5,5,6,6) x-coords should include 5 and 6."""
        result = FeatureCollection._geometry_collection(geometry_collection_geom, "x")
        assert 5.0 in result and 6.0 in result

    def test_empty_geometry_collection(self):
        """An empty GeometryCollection should produce an empty list."""
        gc = GeometryCollection()
        result = FeatureCollection._geometry_collection(gc, "x")
        assert result == []


class TestExplodeGdf:
    """Tests for ``_explode_gdf`` via ``xy()``."""

    def test_multipolygon_exploded_via_xy(self):
        """``xy()`` explodes MultiPolygons into per-row Polygons."""
        poly1 = box(0.0, 0.0, 1.0, 1.0)
        poly2 = box(2.0, 2.0, 3.0, 3.0)
        mpoly = MultiPolygon([poly1, poly2])
        single = box(10.0, 10.0, 11.0, 11.0)
        gdf = gpd.GeoDataFrame(geometry=[mpoly, single], crs="EPSG:4326")
        fc = FeatureCollection(gdf)
        fc.xy()
        # FeatureCollection IS a GeoDataFrame — read columns from self.
        assert "x" in fc.columns
        assert "y" in fc.columns
        for idx in range(len(fc)):
            assert fc.iloc[idx].geometry.geom_type == "Polygon"

    def test_no_multipolygon_unchanged(self, simple_polygon_gdf: GeoDataFrame):
        """A GDF without multi-geometries should be returned with same row count."""
        result = FeatureCollection._explode_gdf(
            simple_polygon_gdf.copy(), geometry="multipolygon"
        )
        assert len(result) == len(simple_polygon_gdf)


class TestMultiGeomHandler:
    """Tests for ``_multi_geom_handler``."""

    def test_multipoint_x(self, multipoint_geom: MultiPoint):
        result = FeatureCollection._multi_geom_handler(
            multipoint_geom, "x", "multipoint"
        )
        assert result == [1.0, 3.0, 5.0]

    def test_multipoint_y(self, multipoint_geom: MultiPoint):
        result = FeatureCollection._multi_geom_handler(
            multipoint_geom, "y", "multipoint"
        )
        assert result == [2.0, 4.0, 6.0]

    def test_multilinestring_x(self, multilinestring_geom: MultiLineString):
        result = FeatureCollection._multi_geom_handler(
            multilinestring_geom, "x", "multilinestring"
        )
        assert result == [[0.0, 1.0], [2.0, 3.0]]

    def test_multilinestring_y(self, multilinestring_geom: MultiLineString):
        result = FeatureCollection._multi_geom_handler(
            multilinestring_geom, "y", "multilinestring"
        )
        assert result == [[0.0, 1.0], [2.0, 3.0]]

    def test_multipolygon_x(self, multipolygon_geom: MultiPolygon):
        result = FeatureCollection._multi_geom_handler(
            multipolygon_geom, "x", "multipolygon"
        )
        assert len(result) == 2
        for coords in result:
            assert isinstance(coords, list)


class TestCenterPoint:
    """Tests for ``center_point`` method."""

    def test_center_point_single_polygon(self, simple_polygon_gdf: GeoDataFrame):
        """Center point of box(30,30,31,31) is ~(30.5, 30.5)."""
        fc = FeatureCollection(simple_polygon_gdf)
        result_gdf = fc.center_point()
        assert "center_point" in result_gdf.columns
        cp = result_gdf.loc[0, "center_point"]
        assert isinstance(cp, Point)
        assert abs(cp.x - 30.5) < 0.2
        assert abs(cp.y - 30.5) < 0.2


class TestDtypes:
    """Tests for the inherited ``dtypes`` property."""

    def test_gdf_dtypes(self, simple_polygon_gdf: GeoDataFrame):
        """dtypes should return a Series covering all columns."""
        fc = FeatureCollection(simple_polygon_gdf)
        dtypes = fc.dtypes
        assert "value" in dtypes.index
        assert "geometry" in dtypes.index


class TestReprojectPoints:
    """Tests for the static ``reproject_points2``."""

    def test_reproject_points2_roundtrip(self):
        """Reproject 4326 → 32636 → 4326 should round-trip."""
        lat = [30.0]
        lng = [31.0]
        x_utm, y_utm = FeatureCollection.reproject_points2(
            lat, lng, from_epsg=4326, to_epsg=32636
        )
        assert len(x_utm) == 1 and len(y_utm) == 1
        x_back, y_back = FeatureCollection.reproject_points2(
            y_utm, x_utm, from_epsg=32636, to_epsg=4326
        )
        assert abs(x_back[0] - 31.0) < 0.01
        assert abs(y_back[0] - 30.0) < 0.01

    def test_reproject_points2_multiple(self):
        lat = [30.0, 31.0]
        lng = [31.0, 32.0]
        x_out, y_out = FeatureCollection.reproject_points2(
            lat, lng, from_epsg=4326, to_epsg=32636
        )
        assert len(x_out) == 2 and len(y_out) == 2


class TestCreateSrFromProj:
    """Tests for ``_create_sr_from_proj``."""

    def test_wkt_default(self, wgs84_wkt: str):
        srs = FeatureCollection._create_sr_from_proj(wgs84_wkt)
        srs.AutoIdentifyEPSG()
        assert int(srs.GetAuthorityCode(None)) == 4326

    def test_proj4_string(self, utm_proj4: str):
        srs = FeatureCollection._create_sr_from_proj(utm_proj4, string_type="PROJ4")
        assert srs is not None

    def test_esri_wkt_string(self):
        srs_orig = osr.SpatialReference()
        srs_orig.ImportFromEPSG(4326)
        wkt_str = srs_orig.ExportToWkt()
        srs = FeatureCollection._create_sr_from_proj(wkt_str, string_type="ESRI wkt")
        assert srs is not None


class TestGetEpsgFromPrj:
    """Tests for ``get_epsg_from_prj``."""

    def test_valid_wkt(self, wgs84_wkt: str):
        assert FeatureCollection.get_epsg_from_prj(wgs84_wkt) == 4326

    def test_empty_prj(self):
        """Empty string defaults to 4326 (legacy behavior)."""
        assert FeatureCollection.get_epsg_from_prj("") == 4326


class TestGetCoords:
    """Tests for ``_get_coords`` dispatch."""

    @pytest.mark.parametrize(
        "geom, expected_type",
        [
            (Point(5, 10), float),
            (LineString([(0, 0), (1, 1)]), list),
            (Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), list),
        ],
        ids=["point", "linestring", "polygon"],
    )
    def test_basic_geometries(self, geom, expected_type):
        import pandas as pd

        row = pd.Series({"geometry": geom})
        result = FeatureCollection._get_coords(row, "geometry", "x")
        assert isinstance(result, expected_type)

    def test_multipolygon_returns_sentinel(self):
        """MultiPolygon rows return the -9999 sentinel so xy() can drop them."""
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        assert FeatureCollection._get_coords(row, "geometry", "x") == -9999

    def test_geometry_collection(self):
        import pandas as pd

        gc = GeometryCollection([Point(7, 8)])
        row = pd.Series({"geometry": gc})
        result = FeatureCollection._get_coords(row, "geometry", "x")
        assert 7.0 in result


class TestFeatureCollectionStr:
    """Tests for the pyramids-branded ``__str__``."""

    def test_str_returns_string(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        result = str(fc)
        assert isinstance(result, str)
        assert "FeatureCollection" in result


class TestToDatasetErrors:
    """Tests for the error paths of :meth:`FeatureCollection.to_dataset`."""

    def test_no_cell_size_no_dataset_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises(ValueError, match="cell size"):
            fc.to_dataset(cell_size=None, dataset=None)

    def test_mismatched_epsg_raises(self, simple_polygon_gdf: GeoDataFrame):
        from pyramids.dataset import Dataset

        fc = FeatureCollection(simple_polygon_gdf)
        ds = Dataset.create(
            cell_size=1000.0,
            rows=3,
            columns=3,
            dtype="float32",
            bands=1,
            top_left_corner=(500000.0, 3500000.0),
            epsg=32636,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="not the same EPSG"):
            fc.to_dataset(dataset=ds)

    def test_non_dataset_object_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises((TypeError, AttributeError)):
            fc.to_dataset(dataset="not_a_dataset")


class TestGetXyCoordsInvalidType:
    def test_invalid_coord_type_raises(self):
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="'x' or 'y'"):
            FeatureCollection._get_xy_coords(line, "z")


class TestGetPointCoordsInvalidType:
    def test_invalid_coord_type_raises(self):
        point = Point(5, 10)
        with pytest.raises(ValueError, match="'x' or 'y'"):
            FeatureCollection._get_point_coords(point, "z")
