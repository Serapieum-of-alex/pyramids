"""The fill functions must not write to the mesh they are handed.

`UgridDataset.copy()` shares its `Mesh2d` with the original, so a fill that derives a connectivity
and stores it on the mesh mutates the caller's object — and, through the shared mesh, the dataset
the copy came from. These tests pin that the mesh arrives unchanged, and that a degenerate mesh
(no adjacency, or coincident coordinates) neither warns nor returns infinities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, fill
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

FACE_NODES = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])


def _grid_mesh() -> Mesh2d:
    """A 2x2 quad grid carrying no derived connectivity."""
    grid_x, grid_y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    return Mesh2d(
        node_x=grid_x.ravel(),
        node_y=grid_y.ravel(),
        face_node_connectivity=Connectivity(
            data=FACE_NODES,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


@pytest.fixture
def gappy() -> np.ndarray:
    """Face values with one gap."""
    return np.array([1.0, np.nan, 3.0, 4.0])


class TestLaplaceLeavesTheMeshAlone:
    def test_face_face_connectivity_is_not_built_on_the_caller_mesh(self, gappy):
        mesh = _grid_mesh()
        assert mesh.face_face_connectivity is None
        fill.laplace_interpolate(mesh, gappy, "face")
        assert mesh.face_face_connectivity is None

    def test_edge_connectivity_is_not_built_for_a_node_fill(self):
        mesh = _grid_mesh()
        assert mesh.edge_node_connectivity is None
        values = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0])
        fill.laplace_interpolate(mesh, values, "node")
        assert mesh.edge_node_connectivity is None

    def test_the_gap_is_still_filled(self, gappy):
        mesh = _grid_mesh()
        filled = fill.laplace_interpolate(mesh, gappy, "face")
        assert np.isfinite(filled).all()
        assert filled[0] == pytest.approx(1.0)
        assert filled[2] == pytest.approx(3.0)

    def test_filling_a_copy_does_not_touch_the_original(self):
        # copy() shares the Mesh2d, so a mutating fill would reach back into `original`.
        original = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            face_node_connectivity=FACE_NODES,
            data={"v": np.array([1.0, np.nan, 3.0, 4.0])},
        )
        assert original.mesh.face_face_connectivity is None
        original.copy().laplace_interpolate("v")
        assert original.mesh.face_face_connectivity is None


class TestDegenerateMeshes:
    def test_a_mesh_with_no_adjacency_does_not_warn(self):
        # Two faces sharing no edge: the adjacency is empty, so the weight mean is taken over an
        # empty slice unless that is guarded.
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 5.0, 6.0, 6.0]),
            node_y=np.array([0.0, 0.0, 1.0, 5.0, 5.0, 6.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2], [3, 4, 5]]),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = fill.laplace_interpolate(mesh, np.array([1.0, 2.0]), "face")
        assert result.tolist() == [1.0, 2.0]

    def test_coincident_coordinates_stay_finite(self, gappy):
        # Two faces with the same centroid make the inverse distance infinite unless floored.
        mesh = _grid_mesh()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            filled = fill.laplace_interpolate(
                mesh, gappy, "face", use_weights=True, direct_solve=True
            )
        assert np.isfinite(filled).all()
