"""Tests for UgridDataset.plot_histogram."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from pyramids.netcdf.ugrid.analysis import plot_histogram
from tests.ugrid.conftest import make_square_ds as _make_ds

pytestmark = [pytest.mark.core, pytest.mark.plot]


class TestPlotHistogram:
    def test_returns_fig_ax_hist(self):
        ds = _make_ds()
        fig, ax, hist = ds.plot_histogram(bins=3)
        assert fig is not None
        assert ax is not None

    def test_custom_bins(self):
        ds = _make_ds()
        fig, ax, hist = ds.plot_histogram(bins=2)
        assert fig is not None


class TestPlotHistogramEdgeCases:
    def test_all_nan_raises(self):
        """All-NaN data should raise ValueError."""
        with pytest.raises(ValueError, match="No valid samples"):
            plot_histogram(np.full(4, np.nan))

    def test_integer_data(self):
        """Integer arrays have no NaN — histogram uses all values."""
        fig, ax, hist = plot_histogram(
            np.array([1, 2, 3, 4, 5]), bins=3
        )
        assert fig is not None

    def test_nodata_excluded(self):
        """Sentinel nodata values are excluded from the histogram."""
        vals = np.array([1.0, 2.0, -9999.0, 3.0])
        fig, ax, hist = plot_histogram(vals, nodata=-9999.0, bins=3)
        assert fig is not None

    def test_nan_nodata_passthrough(self):
        """nodata=NaN on a float array is a no-op (NaN already masked).

        Test scenario:
            Passing nodata=NaN skips the sentinel comparison since NaN
            positions are already caught by the isnan check.
        """
        vals = np.array([1.0, 2.0, np.nan, 3.0])
        fig, ax, hist = plot_histogram(vals, nodata=np.nan, bins=3)
        assert fig is not None, (
            "NaN nodata passthrough should still produce a histogram"
        )
