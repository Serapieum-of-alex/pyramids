"""Ownership and degenerate-input contracts in the graph and network modules.

Three findings the second review round raised. `permute_faces` claimed in a comment to copy what
it carries over but aliased the edge connectivity, so a write through the result reached the input
mesh. `Network1d.topology_subset` let the `-1` fill through `np.unique`, and numpy's negative
indexing then addressed the *last* node — turning a padded edge row into a real edge. And an empty
mesh reached scipy, which refuses a 0x0 matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Network1d, graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d


def _connectivity(data: np.ndarray, cf_role: str) -> Connectivity:
    """A 0-indexed connectivity with -1 fill."""
    return Connectivity(
        data=data, fill_value=-1, cf_role=cf_role, original_start_index=0
    )


@pytest.fixture
def square() -> Mesh2d:
    """Two triangles with an explicit edge list and edge coordinates."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity(
            np.array([[0, 1, 2], [0, 2, 3]]), "face_node_connectivity"
        ),
        edge_node_connectivity=_connectivity(
            np.array([[0, 1], [1, 2]]), "edge_node_connectivity"
        ),
        edge_x=np.array([0.5, 1.0]),
        edge_y=np.array([0.0, 0.5]),
    )


@pytest.fixture
def chain() -> Network1d:
    """A four-node chain whose last edge row is padded."""
    return Network1d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0]),
        node_y=np.zeros(4),
        edge_node_connectivity=np.array([[0, 1], [1, 2], [2, -1]]),
    )


class TestPermuteFacesOwnsItsArrays:
    def test_the_edge_connectivity_is_a_different_object(self, square):
        result = graph.permute_faces(square, np.array([1, 0]))
        assert result.edge_node_connectivity is not square.edge_node_connectivity

    def test_the_edge_connectivity_shares_no_memory(self, square):
        result = graph.permute_faces(square, np.array([1, 0]))
        assert not np.shares_memory(
            result.edge_node_connectivity.data, square.edge_node_connectivity.data
        )

    def test_a_write_through_the_result_does_not_reach_the_input(self, square):
        result = graph.permute_faces(square, np.array([1, 0]))
        result.edge_node_connectivity.data[0, 0] = 99
        assert square.edge_node_connectivity.data[0].tolist() == [0, 1]

    def test_the_edge_coordinates_are_detached_too(self, square):
        result = graph.permute_faces(square, np.array([1, 0]))
        assert not np.shares_memory(result.edge_x, square.edge_x)

    def test_the_faces_are_actually_permuted(self, square):
        result = graph.permute_faces(square, np.array([1, 0]))
        assert result.face_node_connectivity.data[0].tolist() == [0, 2, 3]


class TestTopologySubsetDropsTheFill:
    def test_a_padded_row_pulls_in_no_extra_node(self, chain):
        subset = chain.topology_subset([1, 2])
        assert np.asarray(subset.node_x).tolist() == [1.0, 2.0]

    def test_the_padded_entry_stays_a_fill(self, chain):
        subset = chain.topology_subset([1, 2])
        assert np.asarray(subset.edge_node_connectivity.data)[1].tolist() == [1, -1]

    def test_a_clean_subset_is_unchanged(self, chain):
        subset = chain.topology_subset([0, 1])
        assert np.asarray(subset.node_x).tolist() == [0.0, 1.0, 2.0]
        assert np.asarray(subset.edge_node_connectivity.data).tolist() == [
            [0, 1],
            [1, 2],
        ]


class TestEmptyInputs:
    def test_reverse_cuthill_mckee_on_an_empty_mesh(self):
        empty = Mesh2d(
            node_x=np.array([]),
            node_y=np.array([]),
            face_node_connectivity=_connectivity(
                np.empty((0, 3), dtype=np.intp), "face_node_connectivity"
            ),
        )
        assert graph.reverse_cuthill_mckee(empty).tolist() == []

    def test_an_empty_network_refuses_bounds(self):
        empty = Network1d(
            node_x=np.array([]),
            node_y=np.array([]),
            edge_node_connectivity=np.empty((0, 2), dtype=np.intp),
        )
        with pytest.raises(ValueError, match="no nodes"):
            _ = empty.bounds
