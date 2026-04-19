"""ARC-27: FeatureCollection.schema property + list_layers classmethod.

``schema`` reports the fiona-style (geometry type + field types) dict.
``list_layers`` introspects a multi-layer file without loading its
contents.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from pyramids.feature import FeatureCollection


@pytest.fixture
def points_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"id": [1, 2], "score": [0.5, 1.5]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def mixed_geom_fc() -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"kind": ["pt", "ln", "poly"]},
            geometry=[
                Point(0, 0),
                LineString([(0, 0), (1, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def two_layer_gpkg(tmp_path: Path) -> Path:
    rivers = gpd.GeoDataFrame(
        {"name": ["r1", "r2"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    lakes = gpd.GeoDataFrame(
        {"name": ["l1"]}, geometry=[box(10, 10, 11, 11)], crs="EPSG:4326"
    )
    p = tmp_path / "multi.gpkg"
    rivers.to_file(p, driver="GPKG", layer="rivers")
    lakes.to_file(p, driver="GPKG", layer="lakes")
    return p


class TestSchema:
    """``fc.schema`` returns fiona-style {"geometry": ..., "properties": ...}."""

    def test_points_schema(self, points_fc: FeatureCollection):
        s = points_fc.schema
        assert s["geometry"] == "Point"
        assert set(s["properties"]) == {"id", "score"}
        assert "geometry" not in s["properties"]
        # Values are dtype strings, not dtype objects.
        assert all(isinstance(v, str) for v in s["properties"].values())

    def test_mixed_geom_schema(self, mixed_geom_fc: FeatureCollection):
        """Mixed geometry types report 'Unknown' (fiona convention)."""
        assert mixed_geom_fc.schema["geometry"] == "Unknown"

    def test_empty_schema(self):
        """Empty FC reports 'Unknown' geom but still exposes properties."""
        empty = FeatureCollection(
            gpd.GeoDataFrame({"x": []}, geometry=[])
        )
        s = empty.schema
        assert s["geometry"] == "Unknown"
        # The dtype inferred by pandas for an empty float column is
        # 'float64'; we just assert the column is there (don't pin
        # the exact dtype string, which varies across pandas versions).
        assert "x" in s["properties"]

    def test_schema_subclass_preserved(self, points_fc: FeatureCollection):
        """The returned dict is a plain dict (easy to json.dump)."""
        assert type(points_fc.schema) is dict


class TestListLayers:
    """``FeatureCollection.list_layers(path)`` enumerates layers."""

    def test_multi_layer_gpkg(self, two_layer_gpkg: Path):
        layers = FeatureCollection.list_layers(two_layer_gpkg)
        assert isinstance(layers, list)
        assert all(isinstance(n, str) for n in layers)
        assert set(layers) == {"rivers", "lakes"}

    def test_single_layer_geojson(self, tmp_path: Path):
        """A GeoJSON is single-layer; list_layers returns one name."""
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
        p = tmp_path / "solo.geojson"
        gdf.to_file(p, driver="GeoJSON")
        layers = FeatureCollection.list_layers(p)
        assert len(layers) == 1

    def test_list_layers_is_cheap(
        self, two_layer_gpkg: Path, monkeypatch
    ):
        """``list_layers`` must NOT load the whole dataset.

        Introspection only — intercept ``gpd.read_file`` and assert it
        is never called by ``list_layers``. This defends against
        someone replacing the implementation with
        ``gpd.read_file(p).name`` in a hurry.
        """
        import geopandas

        called: list = []

        def _fail_read_file(*a, **kw):
            called.append((a, kw))
            raise AssertionError("read_file must not be called")

        monkeypatch.setattr(geopandas, "read_file", _fail_read_file)
        layers = FeatureCollection.list_layers(two_layer_gpkg)
        assert set(layers) == {"rivers", "lakes"}
        assert not called, "list_layers triggered a full read"


class TestListLayersCache:
    """C15: ``list_layers`` memoises results behind an LRU cache.

    Repeated calls on the same path must hit the cache and avoid a
    fresh ``pyogrio.list_layers`` call. ``list_layers_cache_clear``
    evicts the cache so out-of-band writes become visible.
    """

    def test_repeated_calls_hit_the_cache(
        self, two_layer_gpkg: Path, monkeypatch
    ):
        import pyogrio

        FeatureCollection.list_layers_cache_clear()

        call_count = [0]
        real_list_layers = pyogrio.list_layers

        def _counting_list_layers(path):
            call_count[0] += 1
            return real_list_layers(path)

        monkeypatch.setattr(pyogrio, "list_layers", _counting_list_layers)
        FeatureCollection.list_layers(two_layer_gpkg)
        FeatureCollection.list_layers(two_layer_gpkg)
        FeatureCollection.list_layers(two_layer_gpkg)
        assert call_count[0] == 1, (
            f"expected exactly one pyogrio call, got {call_count[0]}"
        )

    def test_cache_clear_invalidates_entries(
        self, two_layer_gpkg: Path, monkeypatch
    ):
        import pyogrio

        FeatureCollection.list_layers_cache_clear()

        call_count = [0]
        real_list_layers = pyogrio.list_layers

        def _counting_list_layers(path):
            call_count[0] += 1
            return real_list_layers(path)

        monkeypatch.setattr(pyogrio, "list_layers", _counting_list_layers)
        FeatureCollection.list_layers(two_layer_gpkg)
        FeatureCollection.list_layers_cache_clear()
        FeatureCollection.list_layers(two_layer_gpkg)
        assert call_count[0] == 2

    def test_different_paths_get_separate_cache_entries(
        self, tmp_path: Path, two_layer_gpkg: Path
    ):
        """Two distinct paths must not collide in the cache."""
        FeatureCollection.list_layers_cache_clear()

        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
        solo = tmp_path / "solo.geojson"
        gdf.to_file(solo, driver="GeoJSON")

        multi = FeatureCollection.list_layers(two_layer_gpkg)
        single = FeatureCollection.list_layers(solo)
        assert set(multi) == {"rivers", "lakes"}
        assert len(single) == 1
        # Re-querying the first path still returns the multi-layer list.
        again = FeatureCollection.list_layers(two_layer_gpkg)
        assert set(again) == {"rivers", "lakes"}
