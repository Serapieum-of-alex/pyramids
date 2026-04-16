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

from pyramids.dataset import Dataset
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


class TestEpsgCaching:
    """ARC-6: the ``epsg`` property caches its value per CRS identity."""

    def test_epsg_returns_same_value_on_repeated_access(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """Stable CRS → stable EPSG value across repeated access."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        assert fc.epsg == 4326
        # Cached state is set after first access.
        assert fc._epsg_cache_value == 4326

    def test_epsg_invalidates_when_crs_changes(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """Changing CRS via to_crs() must refresh the cached EPSG."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.epsg == 4326
        reproj = fc.to_crs(3857)
        assert reproj.epsg == 3857
        # Original FC is unchanged and still reports 4326.
        assert fc.epsg == 4326

    def test_epsg_none_when_no_crs(self):
        """EPSG is None when the GDF has no CRS set."""
        poly = box(0.0, 0.0, 1.0, 1.0)
        gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[poly])  # no crs=
        fc = FeatureCollection(gdf)
        assert fc.epsg is None

    def test_cached_attrs_are_in_metadata(self):
        """The cache attrs must be in ``_metadata`` so pandas preserves them.

        Without this, a ``fc.copy()`` would strip the cached values
        silently and subsequent access would silently re-resolve. We
        check the declaration directly so adding new cache attrs
        requires updating ``_metadata`` too.
        """
        assert "_epsg_cache_crs" in FeatureCollection._metadata
        assert "_epsg_cache_value" in FeatureCollection._metadata


class TestContextManager:
    """ARC-5: FeatureCollection supports the ``with`` protocol."""

    def test_with_yields_self(self, simple_polygon_gdf: GeoDataFrame):
        """``with fc as x`` must yield ``fc`` itself (not a copy)."""
        fc = FeatureCollection(simple_polygon_gdf)
        with fc as entered:
            assert entered is fc

    def test_close_is_noop_and_idempotent(self, simple_polygon_gdf: GeoDataFrame):
        """``close()`` is a safe no-op and can be called multiple times."""
        fc = FeatureCollection(simple_polygon_gdf)
        assert fc.close() is None
        assert fc.close() is None
        # FC remains usable after close (no resources were actually released).
        assert len(fc) == 1

    def test_exit_propagates_exceptions(self, simple_polygon_gdf: GeoDataFrame):
        """``__exit__`` returns False so exceptions propagate out."""
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises(RuntimeError, match="inside"):
            with fc:
                raise RuntimeError("inside")

    def test_read_file_usable_with_context(self, tmp_path):
        """``with FeatureCollection.read_file(path) as fc:`` is the idiom."""
        poly = box(0.0, 0.0, 1.0, 1.0)
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[poly], crs="EPSG:4326"
        )
        p = tmp_path / "x.geojson"
        gdf.to_file(p, driver="GeoJSON")
        with FeatureCollection.read_file(p) as fc:
            assert isinstance(fc, FeatureCollection)
            assert len(fc) == 1


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

    def test_reproject_points_no_future_warning(self):
        """ARC-2: reproject_points must not emit pyproj FutureWarning.

        Before ARC-2 the method called ``Proj(init="epsg:…")`` which
        pyproj deprecated in 2.0; the old code hid the resulting
        ``FutureWarning`` with ``warnings.filterwarnings``. ARC-2
        switched to ``pyproj.Transformer.from_crs`` and removed the
        suppression. Guard against regression.
        """
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            FeatureCollection.reproject_points(
                [30.0], [31.0], from_epsg=4326, to_epsg=3857
            )
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert not future, (
            f"reproject_points must not emit FutureWarning; got: "
            f"{[str(w.message) for w in future]}"
        )

    def test_reproject_points_roundtrip(self):
        """ARC-2: 4326 → 3857 → 4326 returns the original lat/lon."""
        y_merc, x_merc = FeatureCollection.reproject_points(
            [30.0], [31.0], from_epsg=4326, to_epsg=3857
        )
        y_back, x_back = FeatureCollection.reproject_points(
            y_merc, x_merc, from_epsg=3857, to_epsg=4326
        )
        assert abs(y_back[0] - 30.0) < 1e-4
        assert abs(x_back[0] - 31.0) < 1e-4


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

    def test_empty_prj_raises(self):
        """ARC-7: empty string raises ValueError instead of silently returning 4326.

        The old behavior masked real configuration errors (a vector
        with a missing projection would be assumed WGS84). Callers
        that want a fallback should catch the error explicitly.
        """
        with pytest.raises(ValueError, match="empty projection string"):
            FeatureCollection.get_epsg_from_prj("")


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

    def test_multipolygon_raises(self):
        """ARC-9: MultiPolygon rows raise; callers must explode first.

        Before ARC-9 the function returned the magic value -9999 as a
        sentinel, which silently dropped rows whose x coordinate
        happened to equal -9999 in projected CRSes. The function now
        raises ValueError; xy() guarantees this branch is unreachable
        by calling explode_gdf first.
        """
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        with pytest.raises(ValueError, match="MultiPolygon"):
            FeatureCollection._get_coords(row, "geometry", "x")

    def test_xy_does_not_drop_real_minus_9999(self):
        """ARC-9 regression: a polygon with x=-9999 is not silently dropped.

        Under the old sentinel design a polygon whose x coordinate
        equaled -9999 would have matched the drop filter and vanished.
        The explode-first design keeps it.
        """
        poly = box(-9999.0, -9999.0, -9998.0, -9998.0)
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[poly], crs="EPSG:32636"
        )
        fc = FeatureCollection(gdf)
        fc.xy()
        assert len(fc) == 1
        # The x column contains the literal -9999.0 (not dropped).
        xs = fc.loc[0, "x"]
        assert -9999.0 in xs, f"Real -9999 coord missing: {xs}"

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
            Dataset.from_features(fc, cell_size=None, template=None)

    def test_mismatched_epsg_raises(self, simple_polygon_gdf: GeoDataFrame):
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
            Dataset.from_features(fc, template=ds)

    def test_non_dataset_object_raises(self, simple_polygon_gdf: GeoDataFrame):
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises((TypeError, AttributeError)):
            Dataset.from_features(fc, template="not_a_dataset")


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
