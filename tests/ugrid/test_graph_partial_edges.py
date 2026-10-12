"""Graph derivation against a file that lists only *some* of its edges.

UGRID lets a file declare an `edge_node_connectivity` covering only the edges it cares about —
boundary edges only is a common, legal shape. The derived face topology must still cover the whole
mesh: before this was pinned, a partial edge list silently emptied the face adjacency, so
`connected_components` reported four isolated faces on a connected 2x2 grid and dilation no-opped.

The file's own edge numbering has to survive too, because edge-located variables index into it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

FACE_NODES = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
BOUNDARY_EDGES = np.array([[0, 1], [1, 2], [0, 3], [2, 5]])


def _nodes() -> tuple[np.ndarray, np.ndarray]:
    """The 3x3 node lattice of a 2x2 quad grid."""
    grid_x, grid_y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    return grid_x.ravel().astype(float), grid_y.ravel().astype(float)


def _connectivity(data: np.ndarray, cf_role: str) -> Connectivity:
    """A 0-indexed connectivity with -1 fill."""
    return Connectivity(
        data=data, fill_value=-1, cf_role=cf_role, original_start_index=0
    )


def _mesh(edges: np.ndarray | None) -> Mesh2d:
    """The 2x2 quad grid, optionally carrying a partial edge list."""
    node_x, node_y = _nodes()
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=_connectivity(FACE_NODES, "face_node_connectivity"),
        edge_node_connectivity=(
            None if edges is None else _connectivity(edges, "edge_node_connectivity")
        ),
    )


@pytest.fixture
def derived() -> Mesh2d:
    """The grid with no edge list at all, so every edge is derived."""
    return _mesh(None)


@pytest.fixture
def partial() -> Mesh2d:
    """The grid with only its four corner boundary edges listed."""
    return _mesh(BOUNDARY_EDGES)


class TestPartialEdgeListStillYieldsFullAdjacency:
    def test_adjacency_matches_the_fully_derived_mesh(self, derived, partial):
        assert (
            graph.sparse_adjacency(partial).nnz == graph.sparse_adjacency(derived).nnz
        )

    def test_the_grid_is_one_connected_component(self, partial):
        assert graph.connected_components(partial).tolist() == [0, 0, 0, 0]

    def test_dilation_still_spreads(self, partial):
        seed = np.array([True, False, False, False])
        grown = graph.binary_dilation(partial, seed)
        assert grown.sum() > seed.sum()

    def test_the_files_own_edge_numbering_is_preserved(self, partial):
        # Edge-located variables index into the file's edge order, so the listed edges must keep
        # positions 0..3; the derived ones are appended after them.
        edge_node = graph.edge_node_connectivity(partial).data
        listed = [sorted(pair) for pair in BOUNDARY_EDGES.tolist()]
        assert [sorted(pair) for pair in edge_node[:4].tolist()] == listed

    def test_every_face_implied_edge_is_present(self, derived, partial):
        edge_node = graph.edge_node_connectivity(partial).data
        full = graph.edge_node_connectivity(derived).data
        as_set = {tuple(sorted(pair)) for pair in edge_node.tolist()}
        assert as_set == {tuple(sorted(pair)) for pair in full.tolist()}

    def test_each_face_knows_all_four_of_its_edges(self, partial):
        face_edge = graph.face_edge_connectivity(partial)
        assert (face_edge.data >= 0).sum(axis=1).tolist() == [4, 4, 4, 4]


class TestMalformedEdgeRows:
    def test_an_all_fill_edge_row_is_skipped(self):
        # A ragged edge array padded with the fill value must not crash the derivation.
        mesh = _mesh(np.array([[0, 1], [-1, -1], [1, 2]]))
        assert graph.connected_components(mesh).tolist() == [0, 0, 0, 0]

    def test_a_one_node_edge_row_is_skipped(self):
        mesh = _mesh(np.array([[0, 1], [3, -1]]))
        assert graph.sparse_adjacency(mesh).nnz == 8
