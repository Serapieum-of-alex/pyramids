"""Tests for element-wise array operations on UgridDataset.

Covers fillna, isnull, notnull, clip_values, round, astype, isin.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _make_ds(
    values: np.ndarray | None = None,
    nodata: float | None = None,
) -> UgridDataset:
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=fnc,
    )
    if values is None:
        values = np.array([10.0, 20.0])
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=values.shape,
        _data=values,
    )
    if nodata is not None:
        var.nodata = nodata
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


class TestFillna:
    def test_replaces_nan(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [1.0, 0.0])

    def test_replaces_nodata_sentinel(self):
        ds = _make_ds(np.array([1.0, -9999.0]), nodata=-9999.0)
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [1.0, 0.0])

    def test_no_missing_unchanged(self):
        ds = _make_ds(np.array([5.0, 10.0]))
        filled = ds.fillna(0.0)
        np.testing.assert_array_equal(filled["depth"].data, [5.0, 10.0])


class TestIsnull:
    def test_nan_detected(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        result = ds.isnull()
        np.testing.assert_array_equal(result["depth"].data, [False, True])

    def test_nodata_detected(self):
        ds = _make_ds(np.array([1.0, -9999.0]), nodata=-9999.0)
        result = ds.isnull()
        np.testing.assert_array_equal(result["depth"].data, [False, True])


class TestNotnull:
    def test_inverse_of_isnull(self):
        ds = _make_ds(np.array([1.0, np.nan]))
        result = ds.notnull()
        np.testing.assert_array_equal(
            result["depth"].data, [True, False]
        )


class TestClipValues:
    def test_clamps_values(self):
        ds = _make_ds(np.array([5.0, 25.0]))
        result = ds.clip_values(8.0, 22.0)
        np.testing.assert_array_equal(
            result["depth"].data, [8.0, 22.0]
        )

    def test_within_range_unchanged(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.clip_values(0.0, 100.0)
        np.testing.assert_array_equal(
            result["depth"].data, [10.0, 20.0]
        )


class TestRound:
    def test_rounds_to_integer(self):
        ds = _make_ds(np.array([1.567, 2.345]))
        result = ds.round(0)
        np.testing.assert_array_equal(result["depth"].data, [2.0, 2.0])

    def test_rounds_to_one_decimal(self):
        ds = _make_ds(np.array([1.567, 2.345]))
        result = ds.round(1)
        np.testing.assert_array_almost_equal(
            result["depth"].data, [1.6, 2.3]
        )


class TestAstype:
    def test_float_to_int(self):
        ds = _make_ds(np.array([1.0, 2.0]))
        result = ds.astype(np.int32)
        assert result["depth"].data.dtype == np.int32
        np.testing.assert_array_equal(result["depth"].data, [1, 2])

    def test_preserves_values(self):
        ds = _make_ds(np.array([1.5, 2.5]))
        result = ds.astype(np.float32)
        assert result["depth"].data.dtype == np.float32


class TestIsin:
    def test_membership(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.isin([10.0, 30.0])
        np.testing.assert_array_equal(
            result["depth"].data, [True, False]
        )
