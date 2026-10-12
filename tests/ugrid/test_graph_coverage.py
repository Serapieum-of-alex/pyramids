"""The graph branches the parity suites never reach: remap sources and degenerate tables.

`test_graph.py` and friends exercise the ordinary topology work on ordinary meshes, which leaves
three classes of branch untested:

- the **second source** of each facet remap (`to_node(source="edge")`, `to_face(source="edge")`)
  and the `ValueError` each one raises for a source the kernel does not offer;
- a **one-dimensional** connectivity table — a face described by a single node — which
  `_invert_connectivity` and `_average_over` both have to promote to a column before using it;
- a mesh where **no node joins two edges**, so `edge_edge_connectivity` collects no pairs at all
  and the ragged grouping has to answer an all-fill table rather than divide by zero.

These are the cases a malformed or minimal file produces, so they must answer rather than crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d


def _connectivity(data: np.ndarray, cf_role: str) -> Connectivity:
    """A 0-indexed connectivity with -1 fill."""
    return Connectivity(
        data=np.asarray(data, dtype=np.intp),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


@pytest.fixture
def quad() -> Mesh2d:
    """The unit square as one quad, whose four derived edges are 0:(0,1) 1:(1,2) 2:(2,3) 3:(0,3)."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity([[0, 1, 2, 3]], "face_node_connectivity"),
    )


@pytest.fixture
def one_node_faces() -> Mesh2d:
    """Three faces each described by a single node, so the connectivity table is 1-D."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 1.0, 0.0]),
        face_node_connectivity=_connectivity([0, 1, 2], "face_node_connectivity"),
    )


class TestRemapFromTheSecondSource:
    def test_to_node_averages_the_edges_around_each_node(self, quad):
        """A node takes the mean of the two quad edges that meet at it."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert graph.to_node(quad, values, source="edge").tolist() == [
            2.5,
            1.5,
            2.5,
            3.5,
        ]

    def test_to_face_averages_the_faces_boundary_edges(self, quad):
        """A face takes the mean of its four boundary edges."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert graph.to_face(quad, values, source="edge").tolist() == [2.5]

    def test_to_face_rejects_an_unknown_source(self, quad):
        """`to_face` offers nodes and edges only; anything else is refused by name."""
        with pytest.raises(ValueError, match="source must be 'node' or 'edge'"):
            graph.to_face(quad, np.array([1.0]), source="face")

    def test_to_edge_rejects_an_unknown_source(self, quad):
        """`to_edge` offers nodes and faces only; anything else is refused by name."""
        with pytest.raises(ValueError, match="source must be 'node' or 'face'"):
            graph.to_edge(quad, np.array([1.0]), source="edge")


class TestOneDimensionalConnectivity:
    def test_the_inverse_gives_each_node_its_single_face(self, one_node_faces):
        """Inverting a 1-D face-node table yields one face per node, not a crash."""
        inverted = graph.node_face_connectivity(one_node_faces)
        assert inverted.data.tolist() == [[0], [1], [2]]

    def test_remapping_to_the_faces_copies_the_single_node_value(self, one_node_faces):
        """Averaging over a one-member row is that member's value."""
        values = np.array([1.0, 2.0, 3.0])
        assert graph.to_face(one_node_faces, values).tolist() == [1.0, 2.0, 3.0]

    def test_no_node_joins_two_edges_so_edge_edge_is_all_fill(self, one_node_faces):
        """With every node on one edge only, no edge has a neighbour and every row is fill."""
        neighbours = graph.edge_edge_connectivity(one_node_faces)
        assert neighbours.n_elements == 3
        assert neighbours.data.tolist() == [[-1], [-1], [-1]]
