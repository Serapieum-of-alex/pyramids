"""Tests for FeatureCollection dunder methods.

Covers __len__, __iter__, __getitem__, __contains__, __bool__,
__eq__, __repr__, __str__ with unit tests, edge cases, and
end-to-end integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr
from shapely.geometry import LineString, Point, Polygon

from pyramids.feature import FeatureCollection


@pytest.fixture
def point_fc():
    """FeatureCollection with 3 Point features and attributes.

    Returns:
        FeatureCollection with columns: name, value, geometry.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "value": [10, 20, 30]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def polygon_fc():
    """FeatureCollection with 2 Polygon features.

    Returns:
        FeatureCollection with columns: id, area_m2, geometry.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2], "area_m2": [100.0, 200.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def empty_fc():
    """Empty FeatureCollection with no features.

    Returns:
        FeatureCollection with 0 rows.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"name": [], "value": []},
        geometry=[],
    )
    return FeatureCollection(gdf)


@pytest.fixture
def single_fc():
    """FeatureCollection with exactly 1 feature.

    Returns:
        FeatureCollection with 1 Point feature.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"label": ["only"]},
        geometry=[Point(5, 5)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def ogr_fc():
    """FeatureCollection backed by an OGR DataSource.

    Returns:
        FeatureCollection wrapping an ogr.DataSource from a real file.
    """
    ds = ogr.Open("tests/data/coello-gauges.geojson")
    return FeatureCollection(ds)


class TestLen:
    """Tests for FeatureCollection.__len__."""

    def test_len_multiple_features(self, point_fc):
        """Test len returns correct count for multiple features.

        Test scenario:
            A 3-feature collection should return 3.
        """
        assert len(point_fc) == 3, f"Expected 3, got {len(point_fc)}"

    def test_len_empty(self, empty_fc):
        """Test len returns 0 for empty collection.

        Test scenario:
            An empty GeoDataFrame should yield len=0.
        """
        assert len(empty_fc) == 0, f"Expected 0, got {len(empty_fc)}"

    def test_len_single(self, single_fc):
        """Test len returns 1 for single-feature collection.

        Test scenario:
            Boundary case: exactly 1 feature.
        """
        assert len(single_fc) == 1, f"Expected 1, got {len(single_fc)}"

    def test_len_ogr_datasource(self, ogr_fc):
        """Test len works with OGR DataSource backend.

        Test scenario:
            coello-gauges.geojson has known feature count.
        """
        result = len(ogr_fc)
        assert result > 0, f"Expected positive len for OGR DataSource, got {result}"

    def test_len_after_getitem_slice(self, point_fc):
        """Test len on a sliced subset.

        Test scenario:
            fc[0:2] should produce a 2-feature collection.
        """
        subset = point_fc[0:2]
        assert len(subset) == 2, f"Expected 2, got {len(subset)}"


class TestIter:
    """Tests for FeatureCollection.__iter__."""

    def test_iter_yields_all_rows(self, point_fc):
        """Test iteration yields all features.

        Test scenario:
            Iterating a 3-feature collection should produce 3 tuples.
        """
        rows = list(point_fc)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"

    def test_iter_tuple_structure(self, point_fc):
        """Test each iteration yields (index, Series).

        Test scenario:
            Each yielded item should be a (index, pandas.Series) tuple
            with access to column values and geometry.
        """
        for idx, row in point_fc:
            assert "name" in row.index, f"Row should have 'name' column"
            assert hasattr(row, "geometry"), "Row should have geometry"
            break

    def test_iter_empty(self, empty_fc):
        """Test iteration over empty collection yields nothing.

        Test scenario:
            list(empty_fc) should be an empty list.
        """
        rows = list(empty_fc)
        assert rows == [], f"Expected empty list, got {rows}"

    def test_iter_preserves_values(self, point_fc):
        """Test iteration preserves attribute values.

        Test scenario:
            The 'name' values should be A, B, C in order.
        """
        names = [row["name"] for _, row in point_fc]
        assert names == ["A", "B", "C"], f"Expected ['A', 'B', 'C'], got {names}"

    def test_iter_ogr_datasource(self, ogr_fc):
        """Test iteration works with OGR DataSource.

        Test scenario:
            Should iterate without error; each row has geometry.
        """
        count = 0
        for idx, row in ogr_fc:
            assert row.geometry is not None, "Each row should have geometry"
            count += 1
            if count >= 3:
                break
        assert count >= 1, "Should iterate at least 1 row"


class TestGetitem:
    """Tests for FeatureCollection.__getitem__."""

    def test_getitem_int(self, point_fc):
        """Test integer indexing returns single-feature FeatureCollection.

        Test scenario:
            fc[0] should return a FeatureCollection with 1 feature.
        """
        result = point_fc[0]
        assert isinstance(result, FeatureCollection), (
            f"Expected FeatureCollection, got {type(result)}"
        )
        assert len(result) == 1, f"Expected 1 feature, got {len(result)}"

    def test_getitem_negative_int(self, point_fc):
        """Test negative integer indexing.

        Test scenario:
            fc[-1] should return the last feature.
        """
        result = point_fc[-1]
        assert isinstance(result, FeatureCollection), (
            f"Expected FeatureCollection, got {type(result)}"
        )
        assert len(result) == 1, f"Expected 1 feature, got {len(result)}"

    def test_getitem_slice(self, point_fc):
        """Test slice indexing returns subset FeatureCollection.

        Test scenario:
            fc[0:2] should return 2 features.
        """
        result = point_fc[0:2]
        assert isinstance(result, FeatureCollection), (
            f"Expected FeatureCollection, got {type(result)}"
        )
        assert len(result) == 2, f"Expected 2 features, got {len(result)}"

    def test_getitem_string_column(self, point_fc):
        """Test string key returns column Series.

        Test scenario:
            fc["name"] should return a pandas Series with values A, B, C.
        """
        result = point_fc["name"]
        assert list(result) == ["A", "B", "C"], (
            f"Expected ['A', 'B', 'C'], got {list(result)}"
        )

    def test_getitem_list_of_ints(self, point_fc):
        """Test list-of-int indexing returns subset.

        Test scenario:
            fc[[0, 2]] should return features at index 0 and 2.
        """
        result = point_fc[[0, 2]]
        assert isinstance(result, FeatureCollection), (
            f"Expected FeatureCollection, got {type(result)}"
        )
        assert len(result) == 2, f"Expected 2 features, got {len(result)}"

    def test_getitem_numpy_array(self, point_fc):
        """Test numpy array indexing.

        Test scenario:
            fc[np.array([0, 1])] should return 2 features.
        """
        result = point_fc[np.array([0, 1])]
        assert isinstance(result, FeatureCollection), (
            f"Expected FeatureCollection, got {type(result)}"
        )
        assert len(result) == 2, f"Expected 2 features, got {len(result)}"

    def test_getitem_preserves_crs(self, point_fc):
        """Test that slicing preserves the CRS.

        Test scenario:
            The subset should retain EPSG:4326 from the parent.
        """
        result = point_fc[0]
        assert result.epsg == 4326, f"Expected EPSG 4326, got {result.epsg}"

    def test_getitem_does_not_mutate_original(self, point_fc):
        """Test that getitem returns a copy, not a view.

        Test scenario:
            Modifying the subset should not affect the original.
        """
        original_len = len(point_fc)
        _ = point_fc[0:1]
        assert len(point_fc) == original_len, "Original should be unchanged"


class TestContains:
    """Tests for FeatureCollection.__contains__."""

    def test_contains_existing_column(self, point_fc):
        """Test 'in' returns True for existing columns.

        Test scenario:
            'name' and 'value' are columns in the collection.
        """
        assert "name" in point_fc, "'name' should be in fc"
        assert "value" in point_fc, "'value' should be in fc"
        assert "geometry" in point_fc, "'geometry' should be in fc"

    def test_contains_missing_column(self, point_fc):
        """Test 'in' returns False for non-existent columns.

        Test scenario:
            'nonexistent' is not a column.
        """
        assert "nonexistent" not in point_fc, (
            "'nonexistent' should not be in fc"
        )

    def test_contains_empty_string(self, point_fc):
        """Test 'in' with empty string.

        Test scenario:
            '' is not a column name.
        """
        assert "" not in point_fc, "Empty string should not be a column"


class TestBool:
    """Tests for FeatureCollection.__bool__."""

    def test_bool_nonempty(self, point_fc):
        """Test bool is True for non-empty collection.

        Test scenario:
            A 3-feature collection is truthy.
        """
        assert bool(point_fc) is True, "Non-empty fc should be truthy"

    def test_bool_empty(self, empty_fc):
        """Test bool is False for empty collection.

        Test scenario:
            An empty collection is falsy.
        """
        assert bool(empty_fc) is False, "Empty fc should be falsy"

    def test_bool_in_if_statement(self, point_fc, empty_fc):
        """Test bool works naturally in if/else.

        Test scenario:
            if fc: should enter the block for non-empty, skip for empty.
        """
        entered = False
        if point_fc:
            entered = True
        assert entered is True, "Non-empty fc should enter if block"

        entered = False
        if empty_fc:
            entered = True
        assert entered is False, "Empty fc should not enter if block"


class TestEq:
    """Tests for FeatureCollection.__eq__."""

    def test_eq_identical(self, point_fc):
        """Test equality with identical GeoDataFrame.

        Test scenario:
            Two FCs from the same GeoDataFrame should be equal.
        """
        import geopandas as gpd

        gdf_copy = point_fc.feature.copy()
        fc2 = FeatureCollection(gdf_copy)
        assert point_fc == fc2, "Identical FCs should be equal"

    def test_eq_different_data(self, point_fc, polygon_fc):
        """Test inequality with different data.

        Test scenario:
            FCs with different geometry/attributes should not be equal.
        """
        assert not (point_fc == polygon_fc), "Different FCs should not be equal"

    def test_eq_non_feature_collection(self, point_fc):
        """Test equality with non-FeatureCollection returns NotImplemented.

        Test scenario:
            Comparing with a string should return NotImplemented (not raise).
        """
        result = point_fc.__eq__("not a fc")
        assert result is NotImplemented, (
            f"Expected NotImplemented, got {result}"
        )

    def test_eq_self(self, point_fc):
        """Test self-equality.

        Test scenario:
            fc == fc should be True.
        """
        assert point_fc == point_fc, "FC should equal itself"


class TestStr:
    """Tests for FeatureCollection.__str__."""

    def test_str_contains_feature_count(self, point_fc):
        """Test __str__ includes the feature count.

        Test scenario:
            str(fc) should contain '3 features'.
        """
        s = str(point_fc)
        assert "3 features" in s, f"Expected '3 features' in '{s}'"

    def test_str_contains_columns(self, point_fc):
        """Test __str__ includes column names.

        Test scenario:
            str(fc) should mention 'name' and 'value'.
        """
        s = str(point_fc)
        assert "name" in s, f"Expected 'name' in '{s}'"
        assert "value" in s, f"Expected 'value' in '{s}'"

    def test_str_contains_epsg(self, point_fc):
        """Test __str__ includes EPSG code.

        Test scenario:
            str(fc) should contain 'epsg=4326'.
        """
        s = str(point_fc)
        assert "4326" in s, f"Expected '4326' in '{s}'"

    def test_str_empty(self, empty_fc):
        """Test __str__ for empty collection.

        Test scenario:
            Should show '0 features'.
        """
        s = str(empty_fc)
        assert "0 features" in s, f"Expected '0 features' in '{s}'"


class TestRepr:
    """Tests for FeatureCollection.__repr__."""

    def test_repr_format(self, point_fc):
        """Test __repr__ has the expected format.

        Test scenario:
            repr should start with 'FeatureCollection(' and include n_features.
        """
        r = repr(point_fc)
        assert r.startswith("FeatureCollection("), (
            f"Expected repr to start with 'FeatureCollection(', got '{r[:30]}'"
        )
        assert "n_features=3" in r, f"Expected 'n_features=3' in '{r}'"

    def test_repr_includes_columns(self, point_fc):
        """Test __repr__ includes column list.

        Test scenario:
            repr should list column names.
        """
        r = repr(point_fc)
        assert "name" in r, f"Expected 'name' in repr: {r}"


class TestEndToEnd:
    """End-to-end tests for dunder methods working together."""

    def test_read_len_iterate_slice(self):
        """Test full workflow: read file -> len -> iterate -> slice.

        Test scenario:
            Read a real GeoJSON, check length, iterate to collect names,
            slice a subset, and verify it.
        """
        fc = FeatureCollection.read_file("tests/data/coello-gauges.geojson")
        n = len(fc)
        assert n > 0, f"Expected positive len, got {n}"

        names = [row.get("id", row.get("name", idx)) for idx, row in fc]
        assert len(names) == n, f"Iteration count {len(names)} != len {n}"

        subset = fc[0:2]
        assert len(subset) == 2, f"Subset should have 2, got {len(subset)}"
        assert bool(subset) is True, "Non-empty subset should be truthy"

    def test_create_check_contains_compare(self):
        """Test workflow: create -> contains -> compare -> str.

        Test scenario:
            Create two identical FCs, check column containment,
            verify equality, and ensure str works.
        """
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"depth": [1.0, 2.0]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
        fc1 = FeatureCollection(gdf)
        fc2 = FeatureCollection(gdf.copy())

        assert "depth" in fc1, "'depth' should be a column"
        assert "missing" not in fc1, "'missing' should not be a column"
        assert fc1 == fc2, "Identical FCs should be equal"
        assert "2 features" in str(fc1), "str should show feature count"

    def test_empty_fc_all_dunders(self):
        """Test all dunder methods on empty FeatureCollection.

        Test scenario:
            len=0, bool=False, iter empty, contains False, str works.
        """
        import geopandas as gpd

        fc = FeatureCollection(gpd.GeoDataFrame())
        assert len(fc) == 0, "Empty len should be 0"
        assert bool(fc) is False, "Empty should be falsy"
        assert list(fc) == [], "Empty iteration should yield nothing"
        assert "geometry" not in fc, "Empty has no columns"
        s = str(fc)
        assert "0 features" in s, f"str should show 0 features: {s}"

    def test_ogr_datasource_all_dunders(self, ogr_fc):
        """Test dunder methods on OGR DataSource-backed FC.

        Test scenario:
            len > 0, iteration works, bool is True.
        """
        assert len(ogr_fc) > 0, "OGR FC should have features"
        assert bool(ogr_fc) is True, "OGR FC should be truthy"

        count = 0
        for _ in ogr_fc:
            count += 1
        assert count == len(ogr_fc), (
            f"Iteration count {count} should match len {len(ogr_fc)}"
        )

    def test_getitem_chain(self, point_fc):
        """Test chaining getitem operations.

        Test scenario:
            fc[0:2][0] should return a single-feature FC.
        """
        subset = point_fc[0:2]
        single = subset[0]
        assert len(single) == 1, f"Chained getitem should yield 1, got {len(single)}"
        assert isinstance(single, FeatureCollection), (
            f"Expected FeatureCollection, got {type(single)}"
        )

    def test_polygon_fc_workflow(self, polygon_fc):
        """Test dunder methods on polygon geometry FC.

        Test scenario:
            Full workflow on polygons: len, iterate, slice, contains.
        """
        assert len(polygon_fc) == 2, f"Expected 2, got {len(polygon_fc)}"
        assert "area_m2" in polygon_fc, "'area_m2' should be a column"

        areas = list(polygon_fc["area_m2"])
        assert areas == [100.0, 200.0], f"Expected [100, 200], got {areas}"

        for idx, row in polygon_fc:
            assert row.geometry.geom_type == "Polygon", (
                f"Expected Polygon, got {row.geometry.geom_type}"
            )
