"""A padded edge row must not become a real edge in any of `Network1d`'s geometry members.

`topology_subset` was taught to drop the fill before `np.unique` so a padded row could not
become an edge joining the last node. Every other consumer of the edge table still indexed it
raw, and numpy's negative indexing resolves `-1` to the *last* node -- so the placeholder row
acquired a length, a midpoint, a position in the KD-tree and a LineString.

The KD-tree case is the dangerous one: it is built from the midpoints, so the fabricated point
sits exactly on the last node and **wins** any nearest-edge query near it, at distance 0.0. That
is a silently wrong answer rather than a crash.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.network import Network1d

pytestmark = pytest.mark.core


@pytest.fixture
def padded_chain():
    """A 4-node chain of 3 edges, plus a fourth row that is all fill."""
    return Network1d.from_arrays(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([[0, 1], [1, 2], [2, 3], [-1, -1]]),
    )


@pytest.fixture
def clean_chain():
    """The same chain without the padded row, as the reference answer."""
    return Network1d.from_arrays(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([[0, 1], [1, 2], [2, 3]]),
    )


class TestPaddedRowGeometry:
    """The per-edge geometric members."""

    def test_the_padded_row_has_no_length(self, padded_chain):
        """A row that describes no edge has no length, so it is NaN rather than 0.0.

        Test scenario:
            `-1` indexed the last node from both ends, giving a zero-length edge from node 3 to
            node 3. Zero is a plausible length, so nothing downstream could tell it from a real
            degenerate edge; NaN says "not an edge".
        """
        lengths = padded_chain.edge_lengths

        assert lengths.shape == (4,), (
            f"one length per declared row, got {lengths.shape}"
        )
        np.testing.assert_allclose(lengths[:3], [1.0, 1.0, 1.0])
        assert np.isnan(lengths[3]), (
            f"the padded row should have no length, got {lengths[3]!r}"
        )

    def test_the_padded_row_has_no_midpoint(self, padded_chain):
        """Its midpoint is NaN, not the last node's coordinates."""
        mx, my = padded_chain.edge_midpoints

        np.testing.assert_allclose(mx[:3], [0.5, 1.5, 2.5])
        assert np.isnan(mx[3]) and np.isnan(my[3]), (
            f"expected a NaN midpoint, got ({mx[3]!r}, {my[3]!r})"
        )

    def test_a_nearest_edge_query_cannot_return_the_padded_row(
        self, padded_chain, clean_chain
    ):
        """The KD-tree answers the same edge as the chain without the padded row.

        Test scenario:
            The fabricated midpoint landed exactly on node 3, so querying `(3.0, 0.0)` returned
            the placeholder at distance 0.0 -- preferred over the real edge 2, whose midpoint is
            0.5 away. A caller asking "which edge is nearest this point" got a row that is not
            an edge.
        """
        point = np.array([[3.0, 0.0]])

        padded_distance, tree_position = padded_chain.edge_tree.query(point)
        padded_edge = int(padded_chain.real_edge_indices[int(tree_position[0])])
        clean_distance, clean_index = clean_chain.edge_tree.query(point)

        assert padded_edge == int(clean_index[0]) == 2, (
            f"both should find real edge 2, got padded={padded_edge} "
            f"clean={int(clean_index[0])}"
        )
        np.testing.assert_allclose(padded_distance, clean_distance)

    def test_the_tree_holds_only_real_edges(self, padded_chain):
        """The padded row occupies no position in the tree."""
        assert padded_chain.edge_tree.n == 3, (
            f"three real edges, so three points in the tree, got {padded_chain.edge_tree.n}"
        )
        np.testing.assert_array_equal(padded_chain.real_edge_indices, [0, 1, 2])

    def test_the_index_map_is_needed_when_the_padding_is_not_last(self):
        """A padded row in the middle shifts every later edge's tree position.

        Test scenario:
            This is what makes `real_edge_indices` load-bearing rather than decorative. With
            the fill at row 1, the tree holds rows 0, 2 and 3, so tree position 1 is edge 2 --
            reading the tree's index as an edge index would name the wrong edge, which is the
            silent-wrong-answer failure mode this whole finding is about.
        """
        network = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([[0, 1], [-1, -1], [1, 2], [2, 3]]),
        )

        assert network.edge_tree.n == 3
        np.testing.assert_array_equal(network.real_edge_indices, [0, 2, 3])

        _, tree_position = network.edge_tree.query(np.array([[2.5, 0.0]]))
        edge = int(network.real_edge_indices[int(tree_position[0])])

        assert edge == 3, f"the midpoint 2.5 belongs to edge 3, got {edge}"

    def test_get_edge_coords_refuses_the_padded_row(self, padded_chain):
        """Asking for the padded row's endpoints is an error with a clear message.

        Test scenario:
            It previously raised a bare `IndexError: index 0 is out of bounds for axis 0 with
            size 0` from inside numpy, which names nothing a caller can act on.
        """
        with pytest.raises(ValueError, match="describes no edge"):
            padded_chain.get_edge_coords(3)

    def test_get_edge_coords_still_works_for_a_real_row(self, padded_chain):
        """The guard does not disturb the real rows."""
        start, end = padded_chain.get_edge_coords(2)

        np.testing.assert_allclose(np.asarray(start), [2.0, 0.0])
        np.testing.assert_allclose(np.asarray(end), [3.0, 0.0])


class TestSubsetKeepsThePaddedRowInert:
    """The fix's own case: a subset that carries a padded row forward."""

    def test_the_subset_has_no_phantom_length(self, padded_chain):
        """`topology_subset` keeps the padded row, and it stays length-less.

        Test scenario:
            The round 2 fix stopped the padded row becoming a node of the subset, but the
            subset's own `edge_lengths` then reported `[1.0, 0.0]` -- the phantom survived the
            very subset the fix was written for.
        """
        subset = padded_chain.topology_subset([0, 3])
        lengths = subset.edge_lengths

        assert lengths.shape == (2,), f"two rows kept, got {lengths.shape}"
        np.testing.assert_allclose(lengths[0], 1.0)
        assert np.isnan(lengths[1]), (
            f"the carried-over padded row has no length, got {lengths[1]!r}"
        )


class TestCleanNetworkUnaffected:
    """Nothing changes for a network with no padding."""

    def test_lengths_midpoints_and_tree_are_unchanged(self, clean_chain):
        """The common case keeps exactly the answers it had."""
        np.testing.assert_allclose(clean_chain.edge_lengths, [1.0, 1.0, 1.0])
        mx, my = clean_chain.edge_midpoints
        np.testing.assert_allclose(mx, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(my, [0.0, 0.0, 0.0])
        assert clean_chain.edge_tree.n == 3


class TestHalfPaddedRow:
    """A row with exactly one real node, which GEOS refuses outright."""

    @pytest.fixture
    def half_padded(self):
        """A chain whose second row names one node and pads the other slot."""
        return Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([[0, 1], [2, -1]]),
        )

    def test_to_geodataframe_does_not_raise(self, half_padded):
        """A one-node row becomes an empty LineString, like a fully padded one.

        Test scenario:
            `LineString([(x, y)])` is not a geometry GEOS accepts, so the conversion died with
            `GEOSException: IllegalArgumentException: point array must contain 0 or >1
            elements` -- a raw GEOS error out of a public method, naming neither the row nor
            the network. The docstring's claim that a padded row "contributes an empty
            LineString" was written for the fully padded case and tested only there.
        """
        frame = half_padded.to_geodataframe("edge")

        assert len(frame) == 2, f"one row per declared edge, got {len(frame)}"
        assert frame.geometry.iloc[1].is_empty, (
            f"a row with one real node describes no edge, got "
            f"{frame.geometry.iloc[1].wkt}"
        )

    def test_the_real_row_is_unaffected(self, half_padded):
        """The complete row still draws its line."""
        frame = half_padded.to_geodataframe("edge")

        assert list(frame.geometry.iloc[0].coords) == [(0.0, 0.0), (1.0, 0.0)]

    def test_it_has_no_length_or_midpoint(self, half_padded):
        """The geometric members treat it as padded too, since one node is not an edge."""
        assert np.isnan(half_padded.edge_lengths[1])
        mx, my = half_padded.edge_midpoints
        assert np.isnan(mx[1]) and np.isnan(my[1])


class TestMeshEdgeLinestringsHalfPaddedRow:
    """`UgridDataset._edge_linestrings` has the identical fallback and the identical defect."""

    def test_a_one_node_edge_row_becomes_empty(self):
        """The mesh-side converter must not raise either.

        Test scenario:
            The two implementations mirror each other, so fixing one and leaving the other is
            the sibling problem this review round keeps finding.
        """
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"v": np.array([1.0, 2.0])},
            data_locations={"v": "face"},
        )
        # `edge_node_connectivity` is read-only on `Mesh2d`, so the half-padded table is set
        # on the private attribute the property reads.
        mesh.mesh._edge_node_connectivity = Connectivity(
            data=np.array([[0, 1], [2, -1]]),
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )

        frame = mesh.to_feature_collection(location="edge")

        assert len(frame) == 2, f"one row per declared edge, got {len(frame)}"
        assert frame.geometry.iloc[1].is_empty, (
            f"a row with one real node describes no edge, got "
            f"{frame.geometry.iloc[1].wkt}"
        )
