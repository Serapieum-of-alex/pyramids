"""Tests for UgridDataset.proximity and UgridDataset.sieve."""

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


class TestProximity:
    def test_target_face_is_zero(self):
        ds = _make_ds()
        dist = ds.proximity(np.array([0]))
        assert dist[0] == 0

    def test_neighbor_is_one(self):
        ds = _make_ds()
        dist = ds.proximity(np.array([0]))
        assert dist[1] == 1

    def test_all_targets_are_zero(self):
        ds = _make_ds()
        dist = ds.proximity(np.array([0, 1]))
        np.testing.assert_array_equal(dist, [0, 0])


class TestSieve:
    def test_small_region_blanked(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.sieve(threshold=3)
        assert np.all(np.isnan(result["depth"].data))

    def test_large_region_kept(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.sieve(threshold=1)
        assert not np.any(np.isnan(result["depth"].data))

    def test_temporal_data_raises(self):
        """sieve must reject temporal (2-D) data."""
        ds = _make_ds(np.array([[10.0, 20.0], [30.0, 40.0]]))
        with pytest.raises(ValueError, match="temporal"):
            ds.sieve(threshold=3)
