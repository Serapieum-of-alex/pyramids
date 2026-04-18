"""Tests for :meth:`Dataset.zonal_stats`.

DASK-25: single-pass rasterize + numpy groupby zonal statistics over
a polygon FeatureCollection. Optional ``method="exactextract"``.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection


@pytest.fixture
def raster(tmp_path):
    """10x10 raster, value = row * 10 + col (0..99)."""
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326,
    )
    path = str(tmp_path / "grid.tif")
    ds.to_file(path)
    return Dataset.read_file(path)


@pytest.fixture
def two_boxes():
    """Two non-overlapping 2x2-cell boxes at the top-left of the raster."""
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[
            box(0.0, 8.0, 2.0, 10.0),
            box(2.0, 8.0, 4.0, 10.0),
        ],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestRasterizeMean:
    def test_returns_dataframe(self, raster, two_boxes):
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["mean"]
        assert len(result) == 2

    def test_mean_of_box_zero_correct(self, raster, two_boxes):
        """Box 0 covers cells (0,0), (0,1), (1,0), (1,1) = values 0, 1, 10, 11."""
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        first = result.iloc[0]["mean"]
        assert first == pytest.approx(np.mean([0, 1, 10, 11]))

    def test_mean_of_box_one_correct(self, raster, two_boxes):
        """Box 1 covers cells (0,2), (0,3), (1,2), (1,3) = 2, 3, 12, 13."""
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        second = result.iloc[1]["mean"]
        assert second == pytest.approx(np.mean([2, 3, 12, 13]))


class TestMultipleStats:
    def test_mean_sum_min_max(self, raster, two_boxes):
        result = raster.zonal_stats(
            two_boxes, stats=("mean", "sum", "min", "max"),
        )
        first = result.iloc[0]
        vals = [0, 1, 10, 11]
        assert first["mean"] == pytest.approx(np.mean(vals))
        assert first["sum"] == pytest.approx(np.sum(vals))
        assert first["min"] == pytest.approx(np.min(vals))
        assert first["max"] == pytest.approx(np.max(vals))


class TestStatValidation:
    def test_unknown_stat_raises(self, raster, two_boxes):
        with pytest.raises(ValueError, match="unknown stat"):
            raster.zonal_stats(two_boxes, stats=("bogus",))

    def test_unknown_method_raises(self, raster, two_boxes):
        with pytest.raises(ValueError, match="method"):
            raster.zonal_stats(two_boxes, method="invalid")


class TestExactExtractImport:
    def test_exactextract_import_error(self, raster, two_boxes, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "exactextract":
                raise ImportError("no exactextract")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[zonal\\]"):
            raster.zonal_stats(two_boxes, method="exactextract")
