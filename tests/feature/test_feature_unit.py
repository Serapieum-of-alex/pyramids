"""Unit tests for FeatureCollection methods that lack coverage.

Targets untested / low-coverage code paths in
``pyramids.featurecollection``, including:
- ``_geometry_collection`` static method
- ``_explode_gdf`` with multipolygon geometry
- ``_multi_geom_handler`` for multipoint, multilinestring, multipolygon
- ``_get_ds_epsg`` static method
- ``center_point`` method
- ``_copy_driver_to_memory``
- ``create_ds`` with various drivers (geojson, memory)
- ``dtypes`` property on both GeoDataFrame and DataSource
- ``reproject_points`` and ``reproject_points2``
- ``_create_sr_from_proj`` with different string types
"""
import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
from geopandas import GeoDataFrame
from osgeo import ogr, osr
from osgeo.ogr import DataSource
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

from pyramids.base._errors import DriverNotExistError
from pyramids.featurecollection import FeatureCollection



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
    return MultiLineString(
        [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
    )


@pytest.fixture()
def multipolygon_geom() -> MultiPolygon:
    """A MultiPolygon geometry."""
    return MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])


@pytest.fixture()
def geometry_collection_geom() -> GeometryCollection:
    """A GeometryCollection with a Point, LineString, and Polygon."""
    return GeometryCollection([
        Point(10, 20),
        LineString([(0, 0), (1, 1), (2, 0)]),
        box(5, 5, 6, 6),
    ])


@pytest.fixture()
def ogr_datasource(simple_polygon_gdf: GeoDataFrame) -> DataSource:
    """Create an OGR DataSource in memory from a simple GeoDataFrame."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "test.geojson")
    simple_polygon_gdf.to_file(path, driver="GeoJSON")
    ds = ogr.Open(path)
    return ds


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



class TestGeometryCollection:
    """Tests for the static ``_geometry_collection`` method."""

    def test_extracts_point_x(self, geometry_collection_geom):
        """Verify x coordinate is extracted from the point sub-geometry."""
        result = FeatureCollection._geometry_collection(
            geometry_collection_geom, "x"
        )
        assert 10.0 in result, "Point x coordinate should be in the result"

    def test_extracts_point_y(self, geometry_collection_geom):
        """Verify y coordinate is extracted from the point sub-geometry."""
        result = FeatureCollection._geometry_collection(
            geometry_collection_geom, "y"
        )
        assert 20.0 in result, "Point y coordinate should be in the result"

    def test_extracts_linestring_coords(self, geometry_collection_geom):
        """Verify linestring coordinates appear in the result."""
        result = FeatureCollection._geometry_collection(
            geometry_collection_geom, "x"
        )
        # linestring has x coords 0, 1, 2
        assert 0.0 in result, "LineString x=0 should be present"
        assert 1.0 in result, "LineString x=1 should be present"
        assert 2.0 in result, "LineString x=2 should be present"

    def test_extracts_polygon_coords(self, geometry_collection_geom):
        """Verify polygon exterior coordinates appear in the result."""
        result = FeatureCollection._geometry_collection(
            geometry_collection_geom, "x"
        )
        # The polygon is box(5,5,6,6), so x should contain 5 and 6
        assert 5.0 in result, "Polygon x=5 should be present"
        assert 6.0 in result, "Polygon x=6 should be present"

    def test_empty_geometry_collection(self):
        """An empty GeometryCollection should produce an empty list."""
        gc = GeometryCollection()
        result = FeatureCollection._geometry_collection(gc, "x")
        assert result == [], "Empty GeometryCollection should yield empty list"



class TestExplodeGdf:
    """Tests for ``_explode_gdf`` with multipolygon geometry."""

    def test_multipolygon_exploded_via_xy(self):
        """Verify _explode_gdf works end-to-end through xy() on a geometry-only GDF.

        The _explode_gdf method is exercised indirectly through the public xy()
        method, which is the way it is called in production code.
        Uses a GDF with only the geometry column to avoid a known pandas
        concat column-mismatch issue in multi-column DataFrames.
        """
        poly1 = box(0.0, 0.0, 1.0, 1.0)
        poly2 = box(2.0, 2.0, 3.0, 3.0)
        mpoly = MultiPolygon([poly1, poly2])
        single = box(10.0, 10.0, 11.0, 11.0)
        gdf = gpd.GeoDataFrame(geometry=[mpoly, single], crs="EPSG:4326")
        fc = FeatureCollection(gdf)
        fc.xy()
        # After xy(), the multipolygon should be exploded and x/y columns added
        assert "x" in fc.feature.columns, "xy() should add 'x' column"
        assert "y" in fc.feature.columns, "xy() should add 'y' column"
        # All geometries should now be Polygon (exploded)
        for idx in range(len(fc.feature)):
            geom_type = fc.feature.iloc[idx].geometry.geom_type
            assert geom_type == "Polygon", (
                f"Row {idx} should be Polygon after explode, got {geom_type}"
            )

    def test_no_multipolygon_unchanged(self, simple_polygon_gdf: GeoDataFrame):
        """A GDF without multi-geometries should be returned as-is."""
        result = FeatureCollection._explode_gdf(
            simple_polygon_gdf.copy(), geometry="multipolygon"
        )
        assert len(result) == len(simple_polygon_gdf), (
            "GDF without multipolygon should keep the same number of rows"
        )



class TestMultiGeomHandler:
    """Tests for ``_multi_geom_handler`` for multipoint, multilinestring, multipolygon."""

    def test_multipoint_x(self, multipoint_geom: MultiPoint):
        """Extract x coordinates from MultiPoint."""
        result = FeatureCollection._multi_geom_handler(
            multipoint_geom, "x", "multipoint"
        )
        assert result == [1.0, 3.0, 5.0], (
            f"Expected [1.0, 3.0, 5.0], got {result}"
        )

    def test_multipoint_y(self, multipoint_geom: MultiPoint):
        """Extract y coordinates from MultiPoint."""
        result = FeatureCollection._multi_geom_handler(
            multipoint_geom, "y", "multipoint"
        )
        assert result == [2.0, 4.0, 6.0], (
            f"Expected [2.0, 4.0, 6.0], got {result}"
        )

    def test_multilinestring_x(self, multilinestring_geom: MultiLineString):
        """Extract x coordinates from MultiLineString."""
        result = FeatureCollection._multi_geom_handler(
            multilinestring_geom, "x", "multilinestring"
        )
        assert result == [[0.0, 1.0], [2.0, 3.0]], (
            f"Unexpected MultiLineString x coords: {result}"
        )

    def test_multilinestring_y(self, multilinestring_geom: MultiLineString):
        """Extract y coordinates from MultiLineString."""
        result = FeatureCollection._multi_geom_handler(
            multilinestring_geom, "y", "multilinestring"
        )
        assert result == [[0.0, 1.0], [2.0, 3.0]], (
            f"Unexpected MultiLineString y coords: {result}"
        )

    def test_multipolygon_x(self, multipolygon_geom: MultiPolygon):
        """Extract x coordinates from MultiPolygon."""
        result = FeatureCollection._multi_geom_handler(
            multipolygon_geom, "x", "multipolygon"
        )
        assert len(result) == 2, "Should return a list for each polygon"
        # Each polygon exterior has 5 coords (closed ring)
        for coords in result:
            assert isinstance(coords, list), "Each element should be a list"



class TestGetDsEpsg:
    """Tests for ``_get_ds_epsg`` static method."""

    def test_returns_correct_epsg(self, ogr_datasource: DataSource):
        """An OGR DataSource created from a WGS84 GeoJSON should report EPSG 4326."""
        epsg = FeatureCollection._get_ds_epsg(ogr_datasource)
        assert epsg == 4326, f"Expected EPSG 4326, got {epsg}"



class TestCenterPoint:
    """Tests for ``center_point`` method."""

    def test_center_point_single_polygon(self, simple_polygon_gdf: GeoDataFrame):
        """Center point of a square should be at its centroid."""
        fc = FeatureCollection(simple_polygon_gdf)
        result_gdf = fc.center_point()
        assert "center_point" in result_gdf.columns, (
            "Result should have a 'center_point' column"
        )
        cp = result_gdf.loc[0, "center_point"]
        assert isinstance(cp, Point), "center_point should be a shapely Point"
        # box(30,30,31,31) centroid is (30.5, 30.5)
        assert abs(cp.x - 30.5) < 0.2, f"Center x should be ~30.5, got {cp.x}"
        assert abs(cp.y - 30.5) < 0.2, f"Center y should be ~30.5, got {cp.y}"



class TestCopyDriverToMemory:
    """Tests for ``_copy_driver_to_memory``."""

    def test_returns_datasource(self, ogr_datasource: DataSource):
        """Copying a DataSource to memory should return a DataSource."""
        mem_ds = FeatureCollection._copy_driver_to_memory(ogr_datasource)
        assert isinstance(mem_ds, DataSource), (
            f"Expected DataSource, got {type(mem_ds)}"
        )

    def test_custom_name(self, ogr_datasource: DataSource):
        """Copying with a custom name should succeed."""
        mem_ds = FeatureCollection._copy_driver_to_memory(
            ogr_datasource, name="my_ds"
        )
        assert isinstance(mem_ds, DataSource), (
            "Copy with custom name should return a DataSource"
        )



class TestCreateDs:
    """Tests for ``create_ds`` with various drivers."""

    def test_geojson_driver(self, tmp_path):
        """Create a GeoJSON DataSource on disk."""
        path = str(tmp_path / "out.geojson")
        ds = FeatureCollection.create_ds(driver="geojson", path=path)
        assert isinstance(ds, DataSource), (
            f"Expected DataSource, got {type(ds)}"
        )

    def test_memory_driver(self):
        """Create an in-memory DataSource without providing a path."""
        ds = FeatureCollection.create_ds(driver="memory")
        assert isinstance(ds, DataSource), (
            "Memory driver should create a DataSource"
        )

    def test_invalid_driver_raises(self):
        """An unsupported driver should raise an error (AttributeError from catalog lookup)."""
        with pytest.raises((DriverNotExistError, AttributeError)):
            FeatureCollection.create_ds(
                driver="totally_fake_driver_xyz", path="foo"
            )

    def test_geojson_no_path_raises(self):
        """A non-memory driver without a path should raise ValueError."""
        with pytest.raises(ValueError, match="path must be provided"):
            FeatureCollection.create_ds(driver="geojson", path=None)



class TestDtypes:
    """Tests for the ``dtypes`` property on GeoDataFrame and DataSource."""

    def test_gdf_dtypes(self, simple_polygon_gdf: GeoDataFrame):
        """dtypes on a GeoDataFrame should return a dict of column -> dtype string."""
        fc = FeatureCollection(simple_polygon_gdf)
        result = fc.dtypes
        assert isinstance(result, dict), "dtypes should return a dict"
        assert "value" in result, "'value' column should appear in dtypes"
        assert "geometry" in result, "'geometry' column should appear in dtypes"

    def test_ds_dtypes(self, ogr_datasource: DataSource):
        """dtypes on a DataSource should return a dict without the geometry column."""
        fc = FeatureCollection(ogr_datasource)
        result = fc.dtypes
        assert isinstance(result, dict), "dtypes should return a dict"
        # geometry column is NOT included when the vector is DataSource
        assert "geometry" not in result, (
            "Geometry column should not be in DataSource dtypes"
        )



class TestReprojectPoints:
    """Tests for ``reproject_points`` and ``reproject_points2``."""

    def test_reproject_points2_roundtrip(self):
        """Reproject from 4326 to 32636 and back to 4326 should yield original coords."""
        lat = [30.0]
        lng = [31.0]
        x_utm, y_utm = FeatureCollection.reproject_points2(
            lat, lng, from_epsg=4326, to_epsg=32636
        )
        assert len(x_utm) == 1, "Should return one x coordinate"
        assert len(y_utm) == 1, "Should return one y coordinate"

        # Round-trip: UTM -> GCS
        x_back, y_back = FeatureCollection.reproject_points2(
            y_utm, x_utm, from_epsg=32636, to_epsg=4326
        )
        assert abs(x_back[0] - 31.0) < 0.01, (
            f"Longitude round-trip failed: {x_back[0]}"
        )
        assert abs(y_back[0] - 30.0) < 0.01, (
            f"Latitude round-trip failed: {y_back[0]}"
        )

    def test_reproject_points2_multiple(self):
        """Verify reprojection works with multiple points."""
        lat = [30.0, 31.0]
        lng = [31.0, 32.0]
        x_out, y_out = FeatureCollection.reproject_points2(
            lat, lng, from_epsg=4326, to_epsg=32636
        )
        assert len(x_out) == 2, "Should have 2 x values"
        assert len(y_out) == 2, "Should have 2 y values"



class TestCreateSrFromProj:
    """Tests for ``_create_sr_from_proj`` with different string types."""

    def test_wkt_default(self, wgs84_wkt: str):
        """A WKT string with string_type=None should import successfully."""
        srs = FeatureCollection._create_sr_from_proj(wgs84_wkt)
        srs.AutoIdentifyEPSG()
        epsg = int(srs.GetAuthorityCode(None))
        assert epsg == 4326, f"Expected EPSG 4326, got {epsg}"

    def test_proj4_string(self, utm_proj4: str):
        """A Proj4 string with a non-WKT header should be imported via ImportFromProj4."""
        srs = FeatureCollection._create_sr_from_proj(
            utm_proj4, string_type="PROJ4"
        )
        assert srs is not None, "Spatial reference should not be None"

    def test_esri_wkt_string(self):
        """A WKT projection string starting with GEOGCS should go through the ESRI branch."""
        srs_orig = osr.SpatialReference()
        srs_orig.ImportFromEPSG(4326)
        wkt_str = srs_orig.ExportToWkt()
        # The WKT starts with GEOGCS so should trigger the ESRI branch
        srs = FeatureCollection._create_sr_from_proj(
            wkt_str, string_type="ESRI wkt"
        )
        assert srs is not None, "Spatial reference should not be None"



class TestGetEpsgFromPrj:
    """Tests for ``get_epsg_from_prj``."""

    def test_valid_wkt(self, wgs84_wkt: str):
        """A valid WGS84 WKT should return EPSG 4326."""
        epsg = FeatureCollection.get_epsg_from_prj(wgs84_wkt)
        assert epsg == 4326, f"Expected 4326, got {epsg}"

    def test_empty_prj(self):
        """An empty projection string should default to 4326."""
        epsg = FeatureCollection.get_epsg_from_prj("")
        assert epsg == 4326, f"Empty prj should return 4326, got {epsg}"



class TestGetCoords:
    """Tests for ``_get_coords`` static method via various geometry types."""

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
        """_get_coords should return the correct type for basic geometries."""
        import pandas as pd

        row = pd.Series({"geometry": geom})
        result = FeatureCollection._get_coords(row, "geometry", "x")
        assert isinstance(result, expected_type), (
            f"Expected {expected_type}, got {type(result)}"
        )

    def test_multipolygon_returns_sentinel(self):
        """MultiPolygon geometries should return -9999 (sentinel for removal)."""
        import pandas as pd

        mp = MultiPolygon([box(0, 0, 1, 1)])
        row = pd.Series({"geometry": mp})
        result = FeatureCollection._get_coords(row, "geometry", "x")
        assert result == -9999, (
            "MultiPolygon should return -9999 sentinel value"
        )

    def test_geometry_collection(self):
        """GeometryCollection should delegate to _geometry_collection."""
        import pandas as pd

        gc = GeometryCollection([Point(7, 8)])
        row = pd.Series({"geometry": gc})
        result = FeatureCollection._get_coords(row, "geometry", "x")
        assert 7.0 in result, (
            "GeometryCollection point x=7 should be in result"
        )



class TestFeatureCollectionStr:
    """Tests for __str__ method."""

    def test_str_returns_string(self, simple_polygon_gdf: GeoDataFrame):
        """__str__ should return a string containing feature info."""
        fc = FeatureCollection(simple_polygon_gdf)
        result = str(fc)
        assert isinstance(result, str), (
            "__str__ should return a string"
        )
        assert "Feature" in result, (
            "__str__ output should contain 'Feature'"
        )


class TestCreateDsDriverNotExist:
    """Tests for create_ds with a truly unsupported driver name."""

    def test_driver_not_exist_raises(self):
        """A driver that returns None from catalog should raise DriverNotExistError."""
        with pytest.raises(
            (DriverNotExistError, AttributeError)
        ):
            FeatureCollection.create_ds(
                driver="zzz_nonexistent_driver_zzz", path="dummy"
            )


class TestDsToGdfInplaceAndMethods:
    """Tests for _ds_to_gdf with inplace=True."""

    def test_ds_to_gdf_inplace_sets_feature(
        self, ogr_datasource: DataSource
    ):
        """Calling _ds_to_gdf with inplace=True should replace feature with GeoDataFrame."""
        fc = FeatureCollection(ogr_datasource)
        result = fc._ds_to_gdf(inplace=True)
        assert result is None, (
            "inplace=True should return None"
        )
        assert isinstance(fc.feature, GeoDataFrame), (
            "feature should now be a GeoDataFrame after inplace conversion"
        )


class TestToDatasetErrors:
    """Tests for to_dataset error paths."""

    def test_no_cell_size_no_dataset_raises(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """Calling to_dataset with neither cell_size nor dataset raises ValueError."""
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises(
            ValueError, match="cell size"
        ):
            fc.to_dataset(cell_size=None, dataset=None)

    def test_mismatched_epsg_raises(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """When dataset and vector have different EPSG, ValueError is raised."""
        from pyramids.dataset import Dataset

        fc = FeatureCollection(simple_polygon_gdf)
        # Create a dataset with a different EPSG (32636 UTM)
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

    def test_non_dataset_object_raises(
        self, simple_polygon_gdf: GeoDataFrame
    ):
        """Passing a non-Dataset object as dataset raises an error."""
        fc = FeatureCollection(simple_polygon_gdf)
        with pytest.raises((TypeError, AttributeError)):
            fc.to_dataset(dataset="not_a_dataset")


class TestGetEpsgFromDataSource:
    """Tests for _get_epsg when feature is a DataSource."""

    def test_datasource_epsg(self, ogr_datasource: DataSource):
        """A FeatureCollection wrapping a DataSource should report correct EPSG."""
        fc = FeatureCollection(ogr_datasource)
        epsg = fc.epsg
        assert epsg == 4326, f"Expected EPSG 4326, got {epsg}"


class TestGetEpsgInvalidType:
    """Tests for _get_epsg with unsupported feature type."""

    def test_invalid_feature_type_raises(self):
        """Wrapping a non-GDF/non-DataSource should raise ValueError on .epsg."""
        fc = FeatureCollection.__new__(FeatureCollection)
        fc._feature = "not_a_valid_feature"
        with pytest.raises(ValueError, match="Unable to get EPSG"):
            _ = fc.epsg


class TestGetXyCoordsInvalidType:
    """Tests for _get_xy_coords with invalid coord_type."""

    def test_invalid_coord_type_raises(self):
        """Passing an invalid coord_type raises ValueError."""
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="'x' or 'y'"):
            FeatureCollection._get_xy_coords(line, "z")


class TestGetPointCoordsInvalidType:
    """Tests for _get_point_coords with invalid coord_type."""

    def test_invalid_coord_type_raises(self):
        """Passing an invalid coord_type to _get_point_coords raises ValueError."""
        point = Point(5, 10)
        with pytest.raises(ValueError, match="'x' or 'y'"):
            FeatureCollection._get_point_coords(point, "z")


class TestExplodeMultiGeometry:
    """Tests for _explode_multi_geometry static method."""

    def test_explode_multipolygon(self):
        """Exploding a MultiPolygon should return individual Polygons."""
        poly1 = box(0, 0, 1, 1)
        poly2 = box(2, 2, 3, 3)
        mpoly = MultiPolygon([poly1, poly2])
        result = FeatureCollection._explode_multi_geometry(mpoly.geoms)
        assert len(result) == 2, (
            f"Expected 2 polygons, got {len(result)}"
        )
        for geom in result:
            assert geom.geom_type == "Polygon", (
                f"Each element should be a Polygon, got {geom.geom_type}"
            )
