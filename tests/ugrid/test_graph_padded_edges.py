"""A padded edge row must not reach anything that uses its node indices.

A file's `edge_node_connectivity` may carry a row that strips to fewer than two nodes. Its
position has to survive, because edge-located variables index into the file's edge order, so the
derivation keeps a `(-1, -1)` placeholder there. That pair is not a node pair: `np.bincount` and
scipy's COO builder both refuse negative indices, and anything else would read node `-1` as the
last node. These tests cover every node-side entry point, which the first round's tests did not —
they only probed the face side, so none of them could see the crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

FACE_NODES = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
PADDED_EDGES = np.array([[0, 1], [-1, -1], [1, 2]])


def _mesh(edges: np.ndarray | None) -> Mesh2d:
    """The 2x2 quad grid, optionally carrying a padded edge list."""
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
        edge_node_connectivity=(
            None
            if edges is None
            else Connectivity(
                data=edges,
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            )
        ),
    )


@pytest.fixture
def padded() -> Mesh2d:
    """The grid whose edge list holds one all-fill row."""
    return _mesh(PADDED_EDGES)


@pytest.fixture
def derived() -> Mesh2d:
    """The same grid with every edge derived, as the reference answer."""
    return _mesh(None)


class TestNodeSideEntryPoints:
    def test_node_edge_connectivity_does_not_raise(self, padded):
        assert graph.node_edge_connectivity(padded).n_elements == 9

    def test_node_node_connectivity_does_not_raise(self, padded):
        assert graph.node_node_connectivity(padded).n_elements == 9

    def test_edge_edge_connectivity_gives_the_padded_row_no_neighbours(self, padded):
        """The padded row is present and adjacent to nothing.

        Test scenario:
            This asserted `is not None`, which holds for anything the function can return --
            including the crash-adjacent output the H1-class defects produced. What matters is
            that the row keeps its position (so an edge variable stays aligned) and that it
            names no neighbour, since it is not an edge.
        """
        table = graph.edge_edge_connectivity(padded)
        rows = np.asarray(table.data)
        derived = np.asarray(graph.edge_node_connectivity(padded).data)

        assert rows.shape[0] == derived.shape[0], (
            f"one row per derived edge, got {rows.shape[0]} for {derived.shape[0]}"
        )
        # PADDED_EDGES puts the fill at position 1, and the derivation keeps it there.
        assert derived[1].tolist() == [-1, -1], (
            f"position 1 should still be the placeholder, got {derived[1].tolist()}"
        )
        assert (rows[1] == table.fill_value).all(), (
            f"the padded row has no neighbours, got {rows[1].tolist()}"
        )

    def test_node_adjacency_matches_the_derived_mesh(self, padded, derived):
        assert (
            graph.sparse_adjacency(padded, kind="node").nnz
            == graph.sparse_adjacency(derived, kind="node").nnz
        )

    def test_to_node_from_edges_does_not_raise(self, padded):
        edge_count = graph.edge_node_connectivity(padded).data.shape[0]
        values = np.ones(edge_count, dtype="float64")
        assert graph.to_node(padded, values, source="edge").shape == (9,)

    def test_the_placeholder_never_appears_as_a_node(self, padded):
        assert (graph.node_node_connectivity(padded).data >= -1).all()
        assert graph.sparse_adjacency(padded, kind="node").shape == (9, 9)


class TestPlaceholderIsNotABoundaryEdge:
    def test_exterior_edges_excludes_it(self, padded):
        placeholder = 1
        assert placeholder not in graph.exterior_edges(padded).tolist()

    def test_exterior_edge_count_matches_the_derived_mesh(self, padded, derived):
        assert len(graph.exterior_edges(padded)) == len(graph.exterior_edges(derived))


class TestFaceSideIsStillCorrect:
    def test_face_adjacency_is_unaffected(self, padded, derived):
        assert graph.sparse_adjacency(padded).nnz == graph.sparse_adjacency(derived).nnz

    def test_the_grid_is_one_component(self, padded):
        assert graph.connected_components(padded).tolist() == [0, 0, 0, 0]

    def test_the_files_edge_numbering_still_survives(self, padded):
        # The placeholder keeps position 1, so a variable indexed by the file's edge order
        # still lines up.
        edge_node = graph.edge_node_connectivity(padded).data
        assert edge_node[0].tolist() == [0, 1]
        assert edge_node[2].tolist() == [1, 2]

    def test_the_facade_reaches_it_too(self):
        dataset = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            face_node_connectivity=FACE_NODES,
            data={"v": np.array([1.0, 2.0, 3.0, 4.0])},
        )
        dataset.mesh._edge_node_connectivity = Connectivity(
            data=PADDED_EDGES,
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        assert dataset.sparse_adjacency(kind="node").shape == (9, 9)
