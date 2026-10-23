"""Tests for UgridDataset.equals and UgridDataset.identical."""

from __future__ import annotations

import numpy as np
import pytest

from tests.ugrid.conftest import make_two_triangle_ds as _make_ds

pytestmark = pytest.mark.core


class TestEquals:
    def test_identical_datasets_are_equal(self):
        assert _make_ds().equals(_make_ds())

    def test_different_values_not_equal(self):
        ds1 = _make_ds(np.array([1.0, 2.0]))
        ds2 = _make_ds(np.array([1.0, 3.0]))
        assert not ds1.equals(ds2)

    def test_nan_positions_must_match(self):
        ds1 = _make_ds(np.array([1.0, np.nan]))
        ds2 = _make_ds(np.array([1.0, np.nan]))
        assert ds1.equals(ds2)

    def test_different_nan_positions_not_equal(self):
        ds1 = _make_ds(np.array([np.nan, 2.0]))
        ds2 = _make_ds(np.array([1.0, np.nan]))
        assert not ds1.equals(ds2)

    def test_non_ugrid_returns_false(self):
        assert not _make_ds().equals("not a dataset")


class TestEqualsDifferentVarNames:
    """Equals when datasets have different variable names."""

    def test_different_variable_names_not_equal(self):
        ds1 = _make_ds()
        ds2 = _make_ds(var_name="temp")
        assert not ds1.equals(ds2), (
            "Datasets with different variable names should not be equal"
        )


class TestEqualsDataNone:
    """Equals when one variable has data=None."""

    def test_one_data_none_not_equal(self):
        ds1 = _make_ds()
        ds2 = _make_ds(load_data=False)
        assert not ds1.equals(ds2), (
            "Dataset with loaded data should not equal one without"
        )

    def test_both_data_none_are_equal(self):
        ds1 = _make_ds(load_data=False)
        ds2 = _make_ds(load_data=False)
        assert ds1.equals(ds2), (
            "Both-None-data datasets should be equal"
        )


class TestIdentical:
    def test_identical_datasets(self):
        assert _make_ds().identical(_make_ds())

    def test_different_dtype_not_identical(self):
        ds1 = _make_ds(np.array([1.0, 2.0]))
        ds2 = _make_ds(np.array([1, 2]))
        assert not ds1.identical(ds2)

    def test_nan_nodata_identical(self):
        """Two datasets with NaN nodata sentinels should be identical."""
        ds_a = _make_ds(nodata=float("nan"))
        ds_b = _make_ds(nodata=float("nan"))
        assert ds_a.identical(ds_b)

    def test_different_global_attributes_not_identical(self):
        ds1 = _make_ds()
        ds2 = _make_ds(global_attributes={"source": "test"})
        assert not ds1.identical(ds2), (
            "Different global_attributes should be non-identical"
        )

    def test_different_crs_wkt_not_identical(self):
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        assert not ds1.identical(ds2), (
            "Different CRS WKT should be non-identical"
        )

    def test_different_nodata_not_identical(self):
        ds1 = _make_ds(nodata=-9999.0)
        ds2 = _make_ds(nodata=-1.0)
        assert not ds1.identical(ds2), (
            "Different nodata sentinels should be non-identical"
        )

    def test_identical_both_data_none(self):
        ds1 = _make_ds(load_data=False)
        ds2 = _make_ds(load_data=False)
        assert ds1.identical(ds2), (
            "Both-None-data datasets should be identical"
        )

    def test_integer_nodata_mismatch_not_identical(self):
        """Non-float nodata comparison takes the else branch."""
        ds1 = _make_ds(
            np.array([10, 20], dtype=np.int32), nodata=-9999
        )
        ds2 = _make_ds(
            np.array([10, 20], dtype=np.int32), nodata=-1
        )
        assert not ds1.identical(ds2), (
            "Integer nodata mismatch should be non-identical"
        )

    def test_identical_one_data_none_not_identical(self):
        ds1 = _make_ds()
        ds2 = _make_ds(load_data=False)
        assert not ds1.identical(ds2), (
            "One-None-data should be non-identical"
        )

    def test_same_crs_both_set_identical(self):
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        ds2.set_crs(epsg=4326)
        assert ds1.identical(ds2), (
            "Same EPSG should be identical"
        )

    def test_different_crs_both_set_not_identical(self):
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        ds2.set_crs(epsg=32631)
        assert not ds1.identical(ds2), (
            "Different EPSG should not be identical"
        )

    def test_both_crs_none_identical(self):
        ds1 = _make_ds()
        ds2 = _make_ds()
        assert ds1.identical(ds2), (
            "Both None CRS should be identical"
        )

    def test_non_ugrid_returns_false(self):
        assert not _make_ds().identical(42)
