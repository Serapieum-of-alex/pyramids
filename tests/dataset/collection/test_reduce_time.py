"""Tests for :meth:`DatasetCollection.reduce_time` (PY-P).

Buckets a COG time-stack by a pandas frequency, reduces each window through the
existing ``groupby`` reducer, and returns grid-attached ``Dataset`` objects.
Requires the ``[lazy]`` extra (the reduction runs through ``DatasetCollection.data``
→ dask), so the module is marked ``lazy`` and each test skips without dask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy

_GEO = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
_NODATA = -9999.0


@pytest.fixture
def daily_files(tmp_path):
    """4 daily single-band rasters with constant values 0, 1, 2, 3.

    Each is a 3x3 EPSG:4326 raster on the same grid (top-left (0, 3), cell 1)
    with no-data -9999.

    Returns:
        list[str]: the four file paths in day order.
    """
    paths = []
    for i in range(4):
        ds = Dataset.from_array(
            np.full((3, 3), float(i), dtype="float32"),
            no_data_value=_NODATA,
            geo_ref=GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326),
        )
        p = str(tmp_path / f"d{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.fixture
def daily_times():
    """Four consecutive days starting 2022-01-01."""
    return pd.date_range("2022-01-01", periods=4, freq="1D")


class TestReduceTime:
    """Tests for ``DatasetCollection.reduce_time``."""

    @requires_dask
    def test_monthly_collapses_to_one_window(self, daily_files, daily_times):
        """All four January days fall in one month → a single mean window.

        Test scenario:
            freq='1MS', op='mean' over values 0..3 yields one window with
            mean 1.5.
        """
        coll = DatasetCollection.from_files(daily_files)
        result = coll.reduce_time(daily_times, freq="1MS", op="mean")
        assert len(result) == 1, f"Jan-only data is one month, got {len(result)}"
        _, ds = result[0]
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert float(ds.read_array()[0, 0]) == pytest.approx(1.5), (
            "mean(0,1,2,3) should be 1.5"
        )

    @requires_dask
    def test_window_grid_is_preserved(self, daily_files, daily_times):
        """The reduced Dataset carries the collection's geotransform/CRS/no-data.

        Test scenario:
            The output raster matches the source grid exactly.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op="mean")[0]
        assert ds.geotransform == _GEO, f"geotransform not preserved: {ds.geotransform}"
        assert ds.epsg == 4326, f"epsg not preserved: {ds.epsg}"
        assert ds.no_data_value[0] == _NODATA, (
            f"no-data not preserved: {ds.no_data_value[0]}"
        )

    @requires_dask
    def test_two_day_windows_split_and_sorted(self, daily_files, daily_times):
        """A 2-day frequency makes two windows, sorted by window label.

        Test scenario:
            Days 0-1 → mean 0.5, days 2-3 → mean 2.5, in chronological order.
        """
        coll = DatasetCollection.from_files(daily_files)
        result = coll.reduce_time(daily_times, freq="2D", op="mean")
        assert len(result) == 2, f"Expected 2 windows, got {len(result)}"
        labels = [label for label, _ in result]
        assert labels == sorted(labels), "windows must be sorted by label"
        values = [float(ds.read_array()[0, 0]) for _, ds in result]
        assert values == pytest.approx([0.5, 2.5]), f"Unexpected window means: {values}"

    @requires_dask
    def test_window_labels_are_timestamps(self, daily_files, daily_times):
        """Each returned label is the window's left-edge pandas Timestamp.

        Test scenario:
            The monthly window is labelled 2022-01-01.
        """
        coll = DatasetCollection.from_files(daily_files)
        label, _ = coll.reduce_time(daily_times, freq="1MS", op="mean")[0]
        assert isinstance(label, pd.Timestamp), (
            f"Expected a Timestamp, got {type(label)}"
        )
        assert label == pd.Timestamp("2022-01-01"), f"Unexpected window label: {label}"

    @requires_dask
    def test_sum_op(self, daily_files, daily_times):
        """The sum reduction sums each window's timesteps.

        Test scenario:
            freq='1MS', op='sum' over 0..3 gives 6.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op="sum")[0]
        assert float(ds.read_array()[0, 0]) == pytest.approx(6.0), (
            "sum(0,1,2,3) should be 6"
        )

    @requires_dask
    @pytest.mark.parametrize("op, expected", [("min", 0.0), ("max", 3.0)])
    def test_min_max_ops(self, daily_files, daily_times, op, expected):
        """min/max reductions pick the extreme over the window.

        Args:
            op: The reduction under test.
            expected: Expected value over the single monthly window.

        Test scenario:
            Over 0..3, min=0 and max=3.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op=op)[0]
        assert float(ds.read_array()[0, 0]) == pytest.approx(expected), f"{op} mismatch"

    @requires_dask
    def test_accepts_string_times(self, daily_files):
        """times may be plain date strings (anything pandas.to_datetime accepts).

        Test scenario:
            A list of ISO date strings groups the same as a DatetimeIndex.
        """
        coll = DatasetCollection.from_files(daily_files)
        times = ["2022-01-01", "2022-01-02", "2022-02-01", "2022-02-02"]
        result = coll.reduce_time(times, freq="1MS", op="mean")
        assert len(result) == 2, "two distinct months → two windows"
        assert float(result[0][1].read_array()[0, 0]) == pytest.approx(0.5), (
            "Jan mean of 0,1"
        )
        assert float(result[1][1].read_array()[0, 0]) == pytest.approx(2.5), (
            "Feb mean of 2,3"
        )

    @requires_dask
    def test_invalid_op_raises(self, daily_files, daily_times):
        """An unsupported op raises ValueError before any reduction.

        Test scenario:
            op='bogus' is not in the supported set (median, which was the old
            placeholder here, is now a valid reducer — see #1301).
        """
        coll = DatasetCollection.from_files(daily_files)
        with pytest.raises(ValueError, match="op must be one of"):
            coll.reduce_time(daily_times, freq="1MS", op="bogus")

    @requires_dask
    def test_times_length_mismatch_raises(self, daily_files):
        """A times length not matching time_length raises ValueError.

        Test scenario:
            Three timestamps for a four-timestep collection is rejected.
        """
        coll = DatasetCollection.from_files(daily_files)
        with pytest.raises(ValueError, match="times has 3 entries"):
            coll.reduce_time(
                ["2022-01-01", "2022-01-02", "2022-01-03"], freq="1MS", op="mean"
            )

    @requires_dask
    def test_nat_times_raises(self, daily_files):
        """A NaT entry in times raises a clear ValueError, not an opaque TypeError.

        Test scenario:
            One timestep with pd.NaT would otherwise be dropped by the grouper,
            leaving an unlabelled timestep — caught up front instead.
        """
        coll = DatasetCollection.from_files(daily_files)
        times = [pd.NaT, "2022-01-02", "2022-01-03", "2022-01-04"]
        with pytest.raises(ValueError, match="unparseable / NaT"):
            coll.reduce_time(times, freq="1MS", op="mean")


#: Per-pixel time series for the 2x2 holistic-op fixture, in day order. The
#: four windows share one January month, so ``freq="1MS"`` yields one window.
_SERIES = {
    (0, 0): [0.0, 1.0, 2.0, 3.0],  # strictly increasing, no gap
    (0, 1): [10.0, 10.0, 20.0, 10.0],  # 10 is the clear mode
    (1, 0): [np.nan, 2.0, 4.0, 6.0],  # one leading gap
    (1, 1): [np.nan, np.nan, np.nan, np.nan],  # all-gap pixel
}


def _stack() -> np.ndarray:
    """The `(T, R, C)` numpy reference stack for :data:`_SERIES`."""
    return np.stack(
        [
            np.array([[_SERIES[(r, c)][i] for c in (0, 1)] for r in (0, 1)])
            for i in range(4)
        ]
    )


def _ref_slope(series: list[float], x: np.ndarray, *, sen: bool) -> float:
    """Reference OLS / Theil-Sen slope of one pixel's series over `x`."""
    y = np.asarray(series)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return np.nan
    xv, yv = x[valid], y[valid]
    if sen:
        pairs = [
            (yv[j] - yv[i]) / (xv[j] - xv[i])
            for i in range(len(xv))
            for j in range(i + 1, len(xv))
        ]
        return float(np.median(pairs))
    return float(np.polyfit(xv, yv, 1)[0])


@pytest.fixture
def varied_files(tmp_path):
    """4 daily 2x2 float rasters whose per-pixel series follow :data:`_SERIES`.

    Gaps are stored as NaN (the collection cube does not mask the -9999
    sentinel, so NaN is the gap the reducers skip). All four days are in
    January, so a ``freq="1MS"`` reduction yields a single window.

    Returns:
        list[str]: the four file paths in day order.
    """
    stack = _stack()
    paths = []
    for i in range(4):
        ds = Dataset.from_array(
            stack[i].astype("float32"),
            no_data_value=np.nan,
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        p = str(tmp_path / f"v{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
class TestReduceTimeOps:
    """The holistic ops #1301 adds: median, mode, percentile/quantile, trend.

    The all-gap pixel (1, 1) makes NaN-aware numpy reducers emit the benign
    ``All-NaN slice`` / ``Mean of empty slice`` warnings (the existing ``mean``
    path warns identically on an all-gap window); they are filtered here since
    ``all-gap -> NaN`` is the asserted behaviour.
    """

    def _one_window(self, paths, times, op, **kw):
        """Reduce the single January window and return its array."""
        coll = DatasetCollection.from_files(paths)
        result = coll.reduce_time(times, freq="1MS", op=op, **kw)
        assert len(result) == 1, f"one January window expected, got {len(result)}"
        return result[0][1].read_array().astype(float)

    @requires_dask
    def test_median_matches_numpy(self, varied_files, daily_times):
        """median equals np.nanmedian per pixel; the all-gap pixel is NaN.

        Test scenario:
            A stack with a leading gap and an all-gap pixel reduces to the
            NaN-aware per-pixel median.
        """
        got = self._one_window(varied_files, daily_times, "median")
        np.testing.assert_allclose(got, np.nanmedian(_stack(), axis=0), equal_nan=True)
        assert np.isnan(got[1, 1]), "all-gap pixel must be NaN"

    @requires_dask
    def test_mode_smallest_value_on_tie(self, varied_files, daily_times):
        """mode is the modal value, NaN-aware, smallest value breaking a tie.

        Test scenario:
            Pixel (0,1) repeats 10 (mode 10); pixel (0,0) is all-distinct so
            every value ties at count 1 and the smallest (0.0) wins; the all-gap
            pixel is NaN.
        """
        got = self._one_window(varied_files, daily_times, "mode")
        assert got[0, 1] == 10.0, f"mode of [10,10,20,10] is 10, got {got[0, 1]}"
        assert got[0, 0] == 0.0, f"all-distinct tie -> smallest 0.0, got {got[0, 0]}"
        assert got[1, 0] == 2.0, f"all-distinct (gap skipped) -> 2.0, got {got[1, 0]}"
        assert np.isnan(got[1, 1]), "all-gap pixel must be NaN"

    @requires_dask
    def test_percentile_matches_numpy(self, varied_files, daily_times):
        """percentile with q=90 equals np.nanpercentile per pixel.

        Test scenario:
            q is given on the 0-100 scale; gaps are skipped.
        """
        got = self._one_window(varied_files, daily_times, "percentile", q=90)
        ref = np.nanpercentile(_stack(), 90, axis=0)
        np.testing.assert_allclose(got, ref, equal_nan=True)

    @requires_dask
    def test_quantile_spelling_matches_percentile(self, varied_files, daily_times):
        """op='quantile' with q in (0,1] equals op='percentile' with q*100.

        Test scenario:
            quantile=0.9 and percentile=90 give the same raster.
        """
        q = self._one_window(varied_files, daily_times, "quantile", q=0.9)
        p = self._one_window(varied_files, daily_times, "percentile", q=90)
        np.testing.assert_allclose(q, p, equal_nan=True)

    @requires_dask
    def test_trend_matches_least_squares(self, varied_files, daily_times):
        """trend is the per-pixel OLS slope in value-units per day.

        Test scenario:
            Day offsets are 0..3; each pixel's slope matches numpy.polyfit over
            its non-gap timesteps; the all-gap pixel is NaN.
        """
        got = self._one_window(varied_files, daily_times, "trend")
        x = np.arange(4.0)
        ref = np.array(
            [
                [_ref_slope(_SERIES[(r, c)], x, sen=False) for c in (0, 1)]
                for r in (0, 1)
            ]
        )
        np.testing.assert_allclose(got, ref, equal_nan=True)

    @requires_dask
    def test_sens_slope_matches_theil_sen(self, varied_files, daily_times):
        """sens_slope is the per-pixel Theil-Sen median pairwise slope.

        Test scenario:
            Each pixel's slope matches a reference Theil-Sen over its non-gap
            timesteps; the all-gap pixel is NaN.
        """
        got = self._one_window(varied_files, daily_times, "sens_slope")
        x = np.arange(4.0)
        ref = np.array(
            [[_ref_slope(_SERIES[(r, c)], x, sen=True) for c in (0, 1)] for r in (0, 1)]
        )
        np.testing.assert_allclose(got, ref, equal_nan=True)

    @requires_dask
    def test_trend_skipna_false_makes_gap_pixels_nodata(
        self, varied_files, daily_times
    ):
        """With skipna=False any gap makes that pixel's slope NaN.

        Test scenario:
            The no-gap pixel (0,0) still gets a slope; the one-gap pixel (1,0)
            and the all-gap pixel (1,1) both become NaN.
        """
        got = self._one_window(varied_files, daily_times, "trend", skipna=False)
        assert got[0, 0] == pytest.approx(1.0), "no-gap pixel keeps its slope"
        assert np.isnan(got[1, 0]), "one-gap pixel -> NaN under skipna=False"
        assert np.isnan(got[1, 1]), "all-gap pixel -> NaN"

    @requires_dask
    def test_percentile_requires_q(self, varied_files, daily_times):
        """percentile without q raises ValueError before any reduction."""
        coll = DatasetCollection.from_files(varied_files)
        with pytest.raises(ValueError, match="requires a level q"):
            coll.reduce_time(daily_times, freq="1MS", op="percentile")

    @requires_dask
    @pytest.mark.parametrize(
        "op, q", [("percentile", 150), ("percentile", 0), ("quantile", 9)]
    )
    def test_q_out_of_range_raises(self, varied_files, daily_times, op, q):
        """An out-of-range q raises ValueError (0<q<=100 / 0<q<=1)."""
        coll = DatasetCollection.from_files(varied_files)
        with pytest.raises(ValueError, match="q must satisfy"):
            coll.reduce_time(daily_times, freq="1MS", op=op, q=q)

    @requires_dask
    def test_q_refused_for_non_quantile_op(self, varied_files, daily_times):
        """Passing q to an op that takes none raises ValueError."""
        coll = DatasetCollection.from_files(varied_files)
        with pytest.raises(ValueError, match="only valid for"):
            coll.reduce_time(daily_times, freq="1MS", op="mean", q=50)
