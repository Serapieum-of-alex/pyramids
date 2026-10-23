"""Tests for UgridDataset.proximity and UgridDataset.sieve."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from tests.ugrid.conftest import make_two_triangle_ds as _make_ds

pytestmark = pytest.mark.core


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


class TestProximityEdgeCases:
    """Edge cases for the proximity BFS algorithm."""

    def test_out_of_range_target_ignored(self):
        """Target face indices outside [0, n_face) are silently skipped.

        Test scenario:
            Passing target_faces=[-1, 999] to a 2-face mesh should
            leave all distances at -1 (no valid targets seeded).
        """
        ds = _make_ds()
        dist = ds.proximity(np.array([-1, 999]))
        assert np.all(dist == -1), (
            f"Expected all -1 for out-of-range targets, got {dist}"
        )

    def test_disconnected_mesh_unreachable_faces(self):
        """Faces in a disconnected component with no target get -1.

        Test scenario:
            A mesh with two disconnected triangles where only the
            first triangle has a target face; the second triangle's
            face should have distance -1.
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 10.0, 11.0, 10.5]),
            node_y=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var},
            global_attributes={},
        )
        dist = ds.proximity(np.array([0]))
        assert dist[0] == 0, f"Target face should be 0, got {dist[0]}"
        assert dist[1] == -1, f"Disconnected face should be -1, got {dist[1]}"


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
