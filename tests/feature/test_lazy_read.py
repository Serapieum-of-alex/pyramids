"""Tests for :meth:`FeatureCollection.read_file(backend="dask")`.

DASK-22: when ``backend="dask"`` is passed, delegate to
:func:`dask_geopandas.read_file` and return the lazy
:class:`dask_geopandas.GeoDataFrame` directly. Users get the
partitioned-geometry API for clip / buffer / to_crs / sjoin without
a second class surface in pyramids.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection


try:
    import dask_geopandas  # noqa: F401

    HAS_DASK_GP = True
except ImportError:  # pragma: no cover
    HAS_DASK_GP = False


requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


@pytest.fixture
def small_geojson(tmp_path):
    """Write a 5-feature GeoJSON."""
    gdf = gpd.GeoDataFrame(
        {"id": list(range(5)), "name": list("abcde")},
        geometry=[Point(i, i) for i in range(5)],
        crs="EPSG:4326",
    )
    p = tmp_path / "pts.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return str(p)


class TestDefaultBackend:
    """Default ``backend="pandas"`` preserves existing behavior."""

    def test_returns_feature_collection(self, small_geojson):
        fc = FeatureCollection.read_file(small_geojson)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 5


class TestDaskBackend:
    """``backend="dask"`` returns a lazy dask_geopandas.GeoDataFrame."""

    @requires_dask_geopandas
    def test_returns_dask_geodataframe(self, small_geojson):
        lazy = FeatureCollection.read_file(small_geojson, backend="dask")
        assert hasattr(lazy, "npartitions")

    @requires_dask_geopandas
    def test_npartitions_respected(self, small_geojson):
        lazy = FeatureCollection.read_file(
            small_geojson, backend="dask", npartitions=2,
        )
        assert lazy.npartitions == 2

    @requires_dask_geopandas
    def test_compute_recovers_features(self, small_geojson):
        lazy = FeatureCollection.read_file(
            small_geojson, backend="dask", npartitions=2,
        )
        gdf = lazy.compute()
        assert len(gdf) == 5

    @requires_dask_geopandas
    def test_chunksize_alternative(self, small_geojson):
        lazy = FeatureCollection.read_file(
            small_geojson, backend="dask", chunksize=3,
        )
        assert hasattr(lazy, "npartitions")


class TestFilterKwargsRejected:
    """M7: pushing unsupported filter kwargs into backend='dask' must error."""

    def test_bbox_with_dask_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="filter kwargs"):
            FeatureCollection.read_file(
                small_geojson, backend="dask", bbox=(0, 0, 1, 1),
            )

    def test_where_with_dask_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="filter kwargs"):
            FeatureCollection.read_file(
                small_geojson, backend="dask", where="id > 0",
            )


class TestBackendValidation:
    def test_unknown_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="backend"):
            FeatureCollection.read_file(small_geojson, backend="bogus")

    def test_import_error_without_dask_geopandas(self, small_geojson, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask_geopandas":
                raise ImportError("no dask-geopandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[parquet-lazy\\]"):
            FeatureCollection.read_file(small_geojson, backend="dask")
