"""Sampling a point that lies exactly on a face boundary.

Point-in-face used shapely's `within`, which is strict: a point on a polygon's boundary is not
`within` it. So every point on a face edge fell outside *every* face and came back as nodata —
including points on an interior edge shared by two faces, and the mesh's own nodes. A shared
interior edge is inside the mesh by any reasonable reading, so those points must resolve to one
of the faces that meet there.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.regrid import sample
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

# A unit square split by the diagonal running from (0, 0) to (1, 1).
NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def mesh() -> UgridDataset:
    """The two-triangle unit square with one value per face."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array([10.0, 20.0])},
    )


class TestPointsOnABoundaryAreInside:
    def test_a_point_on_the_shared_diagonal_resolves_to_a_face(self, mesh):
        index = MeshSpatialIndex(mesh.mesh)
        found = index.locate_faces(np.array([0.5]), np.array([0.5]))
        assert found[0] in (0, 1)

    def test_a_mesh_node_resolves_to_a_face(self, mesh):
        index = MeshSpatialIndex(mesh.mesh)
        found = index.locate_faces(np.array([0.0]), np.array([0.0]))
        assert found[0] in (0, 1)

    def test_a_point_on_the_outer_edge_resolves_to_a_face(self, mesh):
        index = MeshSpatialIndex(mesh.mesh)
        found = index.locate_faces(np.array([0.5]), np.array([0.0]))
        assert found[0] in (0, 1)

    def test_a_point_outside_is_still_refused(self, mesh):
        index = MeshSpatialIndex(mesh.mesh)
        found = index.locate_faces(np.array([10.0]), np.array([10.0]))
        assert found[0] == -1

    def test_the_resolution_is_deterministic(self, mesh):
        index = MeshSpatialIndex(mesh.mesh)
        repeated = [
            int(index.locate_faces(np.array([0.5]), np.array([0.5]))[0])
            for _ in range(5)
        ]
        assert len(set(repeated)) == 1


class TestSamplingOnABoundary:
    def test_contains_sampling_returns_a_value_on_the_diagonal(self, mesh):
        got = sample(mesh.mesh, np.array([10.0, 20.0]), 0.5, 0.5, method="contains")
        assert np.isfinite(got).all()
        assert got[0] in (10.0, 20.0)

    def test_barycentric_sampling_is_exact_on_the_diagonal(self, mesh):
        # A field linear in x and y must be reproduced exactly, boundary or not.
        node_values = NODE_X + 2.0 * NODE_Y
        got = sample(mesh.mesh, node_values, 0.5, 0.5, method="linear", location="node")
        assert got[0] == pytest.approx(1.5)

    def test_barycentric_sampling_at_a_node_is_that_nodes_value(self, mesh):
        node_values = NODE_X + 2.0 * NODE_Y
        got = sample(mesh.mesh, node_values, 1.0, 1.0, method="linear", location="node")
        assert got[0] == pytest.approx(3.0)

    def test_the_dataset_facade_samples_a_boundary_point(self, mesh):
        got = mesh.sample("v", 0.5, 0.5, method="contains")
        assert np.isfinite(np.asarray(got, dtype="float64")).all()
