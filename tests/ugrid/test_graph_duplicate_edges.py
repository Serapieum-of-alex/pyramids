"""A duplicated edge row in a file must not become a second edge.

UGRID does not forbid a file from listing the same node pair twice, and a round trip through
a tool that appends rather than merges produces exactly that. The row has to keep its
position, because edge-located variables index by position -- but it describes an edge the
file already has, so counting it again inflates `n_edge` and reports a boundary edge that is
not on the boundary.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import graph as g
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.core

_CLEAN_EDGES = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]]
_DUPLICATED_EDGES = [*_CLEAN_EDGES, [1, 0]]


def _connectivity(data, cf_role):
    """A connectivity over `data` tagged with `cf_role`, zero-based and `-1`-padded."""
    return Connectivity(
        data=np.asarray(data, dtype="int64"),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


def _mesh(edge_rows):
    """A unit square split into two triangles, with `edge_rows` as its edge list."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity(
            [[0, 1, 2], [0, 2, 3]], "face_node_connectivity"
        ),
        edge_node_connectivity=_connectivity(edge_rows, "edge_node_connectivity"),
    )


class TestDuplicatedEdgeRow:
    """The derived edge tables of a file that lists one edge twice."""

    def test_the_duplicate_is_not_a_boundary_edge(self):
        """`exterior_edges` reports the same four edges as the clean file.

        Test scenario:
            The square's four outer edges are its boundary; the diagonal is interior. Adding a
            sixth row that repeats `[0, 1]` reversed cannot change which edges are on the
            boundary, but the duplicate used to arrive as a face-less edge and so counted as a
            fifth exterior edge.
        """
        clean = g.exterior_edges(_mesh(_CLEAN_EDGES))
        duplicated = g.exterior_edges(_mesh(_DUPLICATED_EDGES))

        assert len(clean) == 4, (
            f"the clean square has four boundary edges, got {len(clean)}"
        )
        assert len(duplicated) == 4, (
            f"the duplicate row is not a fifth boundary edge, got {len(duplicated)}: "
            f"{duplicated.tolist()}"
        )

    def test_the_original_row_keeps_its_face(self):
        """The first occurrence stays the edge the faces resolve to.

        Test scenario:
            `edge_id` mapped the node pair to the *last* row carrying it, so the faces resolved
            their boundary to row 5 and row 0 -- the row an edge variable's first value sits on
            -- was left with `edge_face == [-1, -1]`, reading as an edge touching no face at
            all.
        """
        _, edge_face, _ = g._edge_tables(_mesh(_DUPLICATED_EDGES))

        assert edge_face[0].tolist() == [0, -1], (
            f"row 0 is the (0, 1) edge of face 0, so it keeps that face; got "
            f"{edge_face[0].tolist()}"
        )

    def test_the_duplicate_row_describes_no_edge(self):
        """The repeated row becomes a placeholder, like a padded row.

        Test scenario:
            The row must keep its position so edge-located variables stay aligned, so it cannot
            simply be dropped. Marking it `(-1, -1)` is how this module already represents a
            declared-but-undefined edge, and every node-side consumer filters those out.
        """
        edge_node, _, _ = g._edge_tables(_mesh(_DUPLICATED_EDGES))

        assert edge_node.shape[0] == 6, (
            f"the file's six rows all keep their positions, got {edge_node.shape[0]}"
        )
        assert edge_node[5].tolist() == [-1, -1], (
            f"row 5 repeats row 0, so it defines no edge of its own; got "
            f"{edge_node[5].tolist()}"
        )
        assert g._real_edges(edge_node).sum() == 5, (
            "five of the six rows describe an actual edge, got "
            f"{int(g._real_edges(edge_node).sum())}"
        )

    def test_the_node_side_tables_are_unchanged(self):
        """A duplicate row adds no node adjacency.

        Test scenario:
            `node_node_connectivity` is built from the real edges, so the duplicate must not
            add an edge between nodes 0 and 1 that is already there -- and must not crash on
            the `-1` placeholder, which is the H1 failure mode this reuses.
        """
        clean = g.node_node_connectivity(_mesh(_CLEAN_EDGES))
        duplicated = g.node_node_connectivity(_mesh(_DUPLICATED_EDGES))

        assert np.array_equal(np.asarray(clean.data), np.asarray(duplicated.data)), (
            f"node adjacency changed: clean {np.asarray(clean.data).tolist()} vs duplicated "
            f"{np.asarray(duplicated.data).tolist()}"
        )


class TestDanglingEdgeRow:
    """An edge the file lists that no face implies."""

    def test_it_is_reported_as_exterior(self):
        """A dangling edge bounds no face, so it counts as boundary.

        Test scenario:
            Row 5 joins nodes 1 and 3 -- a real edge across the square that neither triangle
            uses, so its `edge_face` row is `[-1, -1]`. Reporting it is the only answer
            `exterior_edges` can give from face adjacency alone, and it was neither documented
            nor tested, which is what let the placeholder rows look like the same case.
        """
        mesh = _mesh([*_CLEAN_EDGES, [1, 3]])

        exterior = g.exterior_edges(mesh)

        assert len(exterior) == 5, (
            f"the four boundary edges plus the dangling one, got {len(exterior)}: "
            f"{exterior.tolist()}"
        )
        assert 5 in exterior.tolist(), (
            f"row 5 is the dangling edge and should be in {exterior.tolist()}"
        )

    def test_it_has_no_face_on_either_side(self):
        """The `edge_face` row behind that answer."""
        _, edge_face, _ = g._edge_tables(_mesh([*_CLEAN_EDGES, [1, 3]]))

        assert edge_face[5].tolist() == [-1, -1], (
            f"no face uses the 1-3 edge, got {edge_face[5].tolist()}"
        )

    def test_a_placeholder_row_is_not_reported_though_it_also_has_two_fills(self):
        """The distinction the docstring now draws.

        Test scenario:
            A padded row and a dangling edge both show `[-1, -1]` in `edge_face`. One
            describes an edge with no face; the other describes no edge. Only the first is a
            boundary edge, which is why `exterior_edges` filters on `_real_edges` rather than
            on `edge_face` alone.
        """
        dangling = g.exterior_edges(_mesh([*_CLEAN_EDGES, [1, 3]]))
        duplicated = g.exterior_edges(_mesh(_DUPLICATED_EDGES))

        assert len(dangling) == 5, f"the dangling edge counts, got {len(dangling)}"
        assert len(duplicated) == 4, f"the placeholder does not, got {len(duplicated)}"
