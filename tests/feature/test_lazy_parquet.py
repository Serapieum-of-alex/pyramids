"""Tests for :meth:`FeatureCollection.read_parquet(backend="dask")`.

DASK-23: extend the GeoParquet reader with split_row_groups / filters
/ blocksize kwargs, routed to :func:`dask_geopandas.read_parquet`
when ``backend="dask"``.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection


try:
    import dask_geopandas  # noqa: F401

    HAS_DASK_GP = True
except ImportError:  # pragma: no cover
    HAS_DASK_GP = False


try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:  # pragma: no cover
    HAS_PYARROW = False


requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)
requires_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow not installed"
)


@pytest.fixture
def small_parquet(tmp_path):
    """GeoParquet fixture — pyarrow is asserted at the class level via
    the ``requires_pyarrow`` mark on every class that uses this fixture,
    so no per-fixture skip is needed (N4)."""
    gdf = gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["water"] * 5 + ["land"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )
    p = tmp_path / "pts.parquet"
    gdf.to_parquet(p)
    return str(p)


# N4: pyarrow-dependent classes carry the skip as a class-level mark so
# pytest emits a single "[N] skipped: pyarrow not installed" entry per
# class rather than one SKIPPED per test method with no shared reason.
@requires_pyarrow
class TestDefaultBackend:
    def test_returns_feature_collection(self, small_parquet):
        fc = FeatureCollection.read_parquet(small_parquet)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 10


@requires_pyarrow
@requires_dask_geopandas
class TestDaskBackend:
    def test_returns_dask_geodataframe(self, small_parquet):
        lazy = FeatureCollection.read_parquet(small_parquet, backend="dask")
        assert hasattr(lazy, "npartitions")

    def test_filters_pushed_down(self, small_parquet):
        lazy = FeatureCollection.read_parquet(
            small_parquet, backend="dask",
            filters=[("class", "=", "water")],
        )
        gdf = lazy.compute()
        assert set(gdf["class"].unique()) == {"water"}

    def test_columns_projection(self, small_parquet):
        lazy = FeatureCollection.read_parquet(
            small_parquet, backend="dask", columns=["id", "geometry"],
        )
        gdf = lazy.compute()
        assert list(gdf.columns) == ["id", "geometry"]


class TestValidation:
    """Validation tests work without pyarrow — path is never read."""

    def test_unknown_backend_raises(self, tmp_path):
        dummy = str(tmp_path / "ignored.parquet")
        with pytest.raises(ValueError, match="backend"):
            FeatureCollection.read_parquet(dummy, backend="bogus")

    def test_import_error_without_dask_geopandas(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask_geopandas":
                raise ImportError("no dask-geopandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        dummy = str(tmp_path / "ignored.parquet")
        with pytest.raises(ImportError, match="pyramids-gis\\[parquet-lazy\\]"):
            FeatureCollection.read_parquet(dummy, backend="dask")
