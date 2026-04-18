"""Tests for :meth:`FeatureCollection.spatial_shuffle`.

DASK-24: thin passthrough to :meth:`dask_geopandas.GeoDataFrame.spatial_shuffle`.
Validation paths (bad input type / missing dep) run without
dask-geopandas.
"""

from __future__ import annotations

import pytest

from pyramids.feature import FeatureCollection


try:
    import dask_geopandas  # noqa: F401

    HAS_DASK_GP = True
except ImportError:  # pragma: no cover
    HAS_DASK_GP = False


requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


class TestTypeValidation:
    """Non-dask inputs surface a clear TypeError."""

    @requires_dask_geopandas
    def test_feature_collection_input_rejected(self):
        import geopandas as gpd
        from shapely.geometry import Point

        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326",
            )
        )
        with pytest.raises(TypeError, match="dask_geopandas.GeoDataFrame"):
            FeatureCollection.spatial_shuffle(fc)


class TestImportError:
    """Missing dask-geopandas surfaces actionable ImportError."""

    def test_import_error_without_dask_geopandas(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask_geopandas":
                raise ImportError("no dask-geopandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[parquet-lazy\\]"):
            FeatureCollection.spatial_shuffle(None)


class TestPassthrough:
    """Actual shuffle round-trip: requires dask-geopandas + a real fixture."""

    @requires_dask_geopandas
    def test_shuffle_returns_dask_geodataframe(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"id": list(range(10))},
            geometry=[Point(i, i) for i in range(10)],
            crs="EPSG:4326",
        )
        p = tmp_path / "pts.geojson"
        gdf.to_file(p, driver="GeoJSON")
        lazy = FeatureCollection.read_file(
            str(p), backend="dask", npartitions=2,
        )
        shuffled = FeatureCollection.spatial_shuffle(lazy, by="hilbert")
        assert hasattr(shuffled, "npartitions")
        assert len(shuffled.compute()) == 10
