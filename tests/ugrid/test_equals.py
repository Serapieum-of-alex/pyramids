"""Tests for UgridDataset.equals and UgridDataset.identical."""

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
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


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


class TestIdentical:
    def test_identical_datasets(self):
        assert _make_ds().identical(_make_ds())

    def test_different_dtype_not_identical(self):
        ds1 = _make_ds(np.array([1.0, 2.0]))
        ds2 = _make_ds(np.array([1, 2]))
        assert not ds1.identical(ds2)

    def test_non_ugrid_returns_false(self):
        assert not _make_ds().identical(42)
