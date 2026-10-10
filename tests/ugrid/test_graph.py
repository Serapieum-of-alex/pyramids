"""Unit tests for pyramids.netcdf.ugrid.graph.

Covers the connectivity builders, the sparse adjacency primitive, facet remapping, and the
graph algorithms (connected components, reverse Cuthill-McKee, binary morphology, Voronoi /
circumcenters, and the boundary helpers). All fixtures are small hand-built meshes or meshes
assembled with ``UgridDataset.from_arrays``.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.graph import (
    binary_dilation,
    binary_erosion,
    bounding_polygon,
    circumcenters,
    connected_components,
    edge_edge_connectivity,
    edge_face_connectivity,
    edge_node_connectivity,
    exterior_edges,
    exterior_faces,
    face_edge_connectivity,
    node_edge_connectivity,
    node_face_connectivity,
    node_node_connectivity,
    permute_faces,
    reverse_cuthill_mckee,
    sparse_adjacency,
    to_edge,
    to_face,
    to_node,
    voronoi,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.core


def _row_mesh(n_quads: int) -> Mesh2d:
    """Build a 1-row chain of ``n_quads`` edge-adjacent quads.

    Args:
        n_quads: Number of quads in the row.

    Returns:
        Mesh2d: The chain mesh (every quad shares a vertical edge with its neighbour).
    """
    cols = n_quads + 1
    xs = np.tile(np.arange(cols, dtype=np.float64), 2)
    ys = np.concatenate([np.zeros(cols), np.ones(cols)])
    faces = np.array(
        [[k, k + 1, cols + k + 1, cols + k] for k in range(n_quads)], dtype=np.intp
    )
    return UgridDataset.from_arrays(xs, ys, faces).mesh


def _grid_quads_with_hole() -> Mesh2d:
    """Build a 3x3 block of unit quads with the centre quad removed (a one-cell hole).

    Returns:
        Mesh2d: An 8-face mesh whose dissolved outline has one interior ring.
    """
    node_id = np.arange(16).reshape(4, 4)
    xs = np.tile(np.arange(4, dtype=np.float64), 4)
    ys = np.repeat(np.arange(4, dtype=np.float64), 4)
    faces = []
    for r in range(3):
        for c in range(3):
            if (r, c) == (1, 1):
                continue
            faces.append(
                [
                    node_id[r, c],
                    node_id[r, c + 1],
                    node_id[r + 1, c + 1],
                    node_id[r + 1, c],
                ]
            )
    return UgridDataset.from_arrays(xs, ys, np.array(faces, dtype=np.intp)).mesh


def _two_disjoint_triangles() -> Mesh2d:
    """Build two triangles with no shared node (two components).

    Returns:
        Mesh2d: A 2-face, 6-node mesh.
    """
    node_x = np.array([0.0, 1.0, 0.5, 10.0, 11.0, 10.5])
    node_y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.intp)
    return UgridDataset.from_arrays(node_x, node_y, faces).mesh


class TestConnectivityBuilders:
    """The derived connectivity builders against a known hand-built mesh."""

    def test_node_face_connectivity_maps_nodes_to_faces(self, mixed_mesh):
        """node_face lists exactly the faces incident to each node."""
        conn = node_face_connectivity(mixed_mesh)
        incident = {r: sorted(int(f) for f in conn.get_element(r)) for r in range(6)}
        assert incident == {0: [0], 1: [0, 1, 2], 2: [1], 3: [0], 4: [0, 2], 5: [1, 2]}

    def test_node_node_connectivity_from_edges(self, mixed_mesh):
        """node_node lists the edge-connected neighbour nodes of each node."""
        conn = node_node_connectivity(mixed_mesh)
        assert sorted(int(n) for n in conn.get_element(1)) == [0, 2, 4, 5]
        assert sorted(int(n) for n in conn.get_element(0)) == [1, 3]

    def test_edge_face_connectivity_marks_interior_and_exterior(self, mixed_mesh):
        """Interior edges carry two faces; boundary edges carry a -1 fill."""
        conn = edge_face_connectivity(mixed_mesh)
        n_faces_per_edge = np.sum(conn.data != -1, axis=1)
        assert set(n_faces_per_edge.tolist()) == {1, 2}
        assert np.count_nonzero(n_faces_per_edge == 2) == 2

    def test_node_edge_round_trips_through_edge_node(self, mixed_mesh):
        """Each edge appears in the node_edge rows of exactly its two endpoints."""
        enc = edge_node_connectivity(mixed_mesh)
        node_edge = node_edge_connectivity(mixed_mesh)
        for e in range(enc.n_elements):
            a, b = (int(x) for x in enc.get_element(e))
            assert e in set(int(x) for x in node_edge.get_element(a))
            assert e in set(int(x) for x in node_edge.get_element(b))

    def test_face_edge_and_edge_edge_are_consistent(self, mixed_mesh):
        """face_edge has one edge per face side; edge_edge neighbours share a node."""
        face_edge = face_edge_connectivity(mixed_mesh)
        assert face_edge.get_element(0).size == 4
        assert face_edge.get_element(1).size == 3
        enc = edge_node_connectivity(mixed_mesh)
        edge_edge = edge_edge_connectivity(mixed_mesh)
        nodes_of = {
            e: set(int(x) for x in enc.get_element(e)) for e in range(enc.n_elements)
        }
        for e in range(enc.n_elements):
            for other in edge_edge.get_element(e):
                assert nodes_of[e] & nodes_of[int(other)]

    def test_builders_do_not_mutate_mesh(self, mixed_mesh):
        """Building connectivity leaves the mesh's own connectivity untouched."""
        before = mixed_mesh.face_node_connectivity.data.copy()
        node_face_connectivity(mixed_mesh)
        edge_face_connectivity(mixed_mesh)
        assert mixed_mesh.edge_node_connectivity is None
        np.testing.assert_array_equal(mixed_mesh.face_node_connectivity.data, before)


class TestSparseAdjacency:
    """The shared sparse adjacency primitive."""

    def test_face_adjacency_is_symmetric(self, mixed_mesh):
        """The face adjacency matrix is symmetric and square on n_face."""
        adjacency = sparse_adjacency(mixed_mesh, kind="face")
        assert adjacency.shape == (3, 3)
        assert (adjacency != adjacency.T).nnz == 0

    def test_node_adjacency_is_symmetric(self, mixed_mesh):
        """The node adjacency matrix is symmetric and square on n_node."""
        adjacency = sparse_adjacency(mixed_mesh, kind="node")
        assert adjacency.shape == (6, 6)
        assert (adjacency != adjacency.T).nnz == 0

    def test_invalid_kind_raises(self, mixed_mesh):
        """An unknown kind is rejected."""
        with pytest.raises(ValueError, match="kind must be"):
            sparse_adjacency(mixed_mesh, kind="edge")


class TestFacetRemap:
    """Facet remapping between locations by averaging."""

    def test_face_to_node_to_face_round_trip_preserves_constant(self, mixed_mesh):
        """A constant face field survives a face -> node -> face round trip."""
        face_values = np.full(mixed_mesh.n_face, 7.0)
        node_values = to_node(mixed_mesh, face_values, source="face")
        back = to_face(mixed_mesh, node_values, source="node")
        np.testing.assert_allclose(back, 7.0)

    def test_to_node_averages_incident_faces(self, mixed_mesh):
        """A node's value is the mean of its incident faces' values."""
        face_values = np.array([0.0, 3.0, 9.0])
        node_values = to_node(mixed_mesh, face_values, source="face")
        assert node_values[0] == pytest.approx(0.0)
        assert node_values[1] == pytest.approx((0.0 + 3.0 + 9.0) / 3.0)
        assert node_values[5] == pytest.approx((3.0 + 9.0) / 2.0)

    def test_to_edge_preserves_leading_time_axis(self, mixed_mesh):
        """Remapping keeps the element axis last and leaves leading axes intact."""
        n_node = mixed_mesh.n_node
        node_values = np.vstack([np.arange(n_node), np.arange(n_node) + 10.0])
        edge_values = to_edge(mixed_mesh, node_values, source="node")
        enc = edge_node_connectivity(mixed_mesh)
        assert edge_values.shape == (2, enc.n_elements)

    def test_invalid_source_raises(self, mixed_mesh):
        """An unknown source location is rejected."""
        with pytest.raises(ValueError, match="source must be"):
            to_node(mixed_mesh, np.zeros(mixed_mesh.n_face), source="bogus")


class TestConnectedComponents:
    """Per-face connected-component labelling."""

    def test_connected_mesh_is_single_component(self, mixed_mesh):
        """A fully connected mesh yields all-zero labels."""
        labels = connected_components(mixed_mesh)
        np.testing.assert_array_equal(labels, np.zeros(3, dtype=labels.dtype))

    def test_two_disjoint_triangles_yield_two_labels(self):
        """Two triangles with no shared edge are two distinct components."""
        labels = connected_components(_two_disjoint_triangles())
        assert len(set(labels.tolist())) == 2
        assert labels[0] != labels[1]


class TestReverseCuthillMckee:
    """Face reordering via reverse Cuthill-McKee."""

    def test_permutation_is_valid(self, mixed_mesh):
        """The returned ordering is a permutation of range(n_face)."""
        perm = reverse_cuthill_mckee(mixed_mesh)
        np.testing.assert_array_equal(np.sort(perm), np.arange(mixed_mesh.n_face))

    def test_permute_faces_is_congruent_and_data_follows(self):
        """Permuting faces keeps each face's nodes and lets a face field follow it."""
        mesh = _row_mesh(4)
        values = np.arange(mesh.n_face, dtype=np.float64)
        perm = reverse_cuthill_mckee(mesh)
        reordered = permute_faces(mesh, perm)
        for k in range(mesh.n_face):
            np.testing.assert_array_equal(
                reordered.face_node_connectivity.data[k],
                mesh.face_node_connectivity.data[perm[k]],
            )
        np.testing.assert_array_equal(
            values[perm], np.array([perm], dtype=float).ravel()
        )

    def test_round_trip_recovers_original_order(self):
        """Applying the permutation then its inverse recovers the original mesh."""
        mesh = _row_mesh(4)
        perm = reverse_cuthill_mckee(mesh)
        inverse = np.argsort(perm)
        recovered = permute_faces(permute_faces(mesh, perm), inverse)
        np.testing.assert_array_equal(
            recovered.face_node_connectivity.data,
            mesh.face_node_connectivity.data,
        )

    def test_permute_faces_does_not_mutate_input(self):
        """The source mesh is left unchanged by permute_faces."""
        mesh = _row_mesh(3)
        before = mesh.face_node_connectivity.data.copy()
        permute_faces(mesh, reverse_cuthill_mckee(mesh))
        np.testing.assert_array_equal(mesh.face_node_connectivity.data, before)


class TestBinaryMorphology:
    """Edge-neighbour binary dilation and erosion on a face mask."""

    def test_dilation_grows_by_one_ring_per_iteration(self):
        """One True face grows along the chain by one neighbour per iteration."""
        mesh = _row_mesh(4)
        mask = np.array([True, False, False, False])
        one = binary_dilation(mesh, mask, iterations=1)
        two = binary_dilation(mesh, mask, iterations=2)
        np.testing.assert_array_equal(one, [True, True, False, False])
        np.testing.assert_array_equal(two, [True, True, True, False])

    def test_erosion_shrinks_by_one_ring_with_true_border(self):
        """Interior erosion (border held True) shrinks the True region by one ring."""
        mesh = _row_mesh(6)
        mask = np.array([False, True, True, True, True, False])
        result = binary_erosion(mesh, mask, iterations=1, border_value=True)
        np.testing.assert_array_equal(result, [False, False, True, True, False, False])

    def test_border_value_controls_boundary_erosion(self):
        """With border_value False every boundary face erodes; True leaves them intact."""
        mesh = _row_mesh(4)
        mask = np.array([True, True, True, True])
        eroded = binary_erosion(mesh, mask, iterations=1, border_value=False)
        kept = binary_erosion(mesh, mask, iterations=1, border_value=True)
        np.testing.assert_array_equal(eroded, [False, False, False, False])
        np.testing.assert_array_equal(kept, [True, True, True, True])

    def test_input_mask_not_mutated(self):
        """The input mask array is never modified in place."""
        mesh = _row_mesh(4)
        mask = np.array([True, False, False, False])
        before = mask.copy()
        binary_dilation(mesh, mask, iterations=2)
        np.testing.assert_array_equal(mask, before)


class TestVoronoiAndCircumcenters:
    """Voronoi dual and triangular circumcenters."""

    def test_circumcenters_of_known_triangles(self, triangle_mesh):
        """Circumcenters match the analytic values for the fixture triangles."""
        centers = circumcenters(triangle_mesh)
        np.testing.assert_allclose(centers[0], [0.5, 0.375])
        np.testing.assert_allclose(centers[1], [1.5, 0.375])

    def test_circumcenters_reject_non_triangular(self, mixed_mesh):
        """A mesh with a quad face cannot produce circumcenters."""
        with pytest.raises(ValueError, match="triangular"):
            circumcenters(mixed_mesh)

    def test_voronoi_interior_node_is_bounded(self):
        """A 3x3 node grid gives the centre node a bounded dual cell, corners unbounded."""
        node_id = np.arange(9).reshape(3, 3)
        xs = np.tile(np.arange(3, dtype=np.float64), 3)
        ys = np.repeat(np.arange(3, dtype=np.float64), 3)
        faces = []
        for r in range(2):
            for c in range(2):
                faces.append([node_id[r, c], node_id[r, c + 1], node_id[r + 1, c + 1]])
                faces.append([node_id[r, c], node_id[r + 1, c + 1], node_id[r + 1, c]])
        mesh = UgridDataset.from_arrays(xs, ys, np.array(faces, dtype=np.intp)).mesh
        vor = voronoi(mesh)
        assert vor.points.shape == (9, 2)
        center_region = vor.regions[vor.point_region[4]]
        corner_region = vor.regions[vor.point_region[0]]
        assert -1 not in center_region and len(center_region) > 0
        assert -1 in corner_region


class TestBoundary:
    """Exterior edges/faces and the dissolved bounding polygon."""

    def test_exterior_edges_and_faces_on_mixed_mesh(self, mixed_mesh):
        """The mixed mesh has six boundary edges and all three faces on the boundary."""
        assert exterior_edges(mixed_mesh).size == 6
        np.testing.assert_array_equal(exterior_faces(mixed_mesh), [0, 1, 2])

    def test_bounding_polygon_area_equals_footprint(self, mixed_mesh):
        """The dissolved outline area equals the sum of the face areas."""
        poly = bounding_polygon(mixed_mesh)
        assert isinstance(poly, (shapely.Polygon, shapely.MultiPolygon))
        assert poly.area == pytest.approx(float(mixed_mesh.face_areas.sum()))

    def test_bounding_polygon_preserves_a_hole(self):
        """A mesh with a one-cell hole dissolves to a polygon with one interior ring."""
        mesh = _grid_quads_with_hole()
        poly = bounding_polygon(mesh)
        assert poly.area == pytest.approx(8.0)
        assert len(poly.interiors) == 1
