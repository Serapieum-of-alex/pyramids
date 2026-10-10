"""Unit tests for pyramids.netcdf.ugrid.network.Network1d.

Covers construction from arrays, topology/geometry properties (node/edge
counts, bounds, edge lengths and midpoints), the directed network sort
(upstream/downstream traversal), topology subsetting, and the GeoDataFrame
export (one LineString per edge).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.network import Network1d

pytestmark = pytest.mark.core


@pytest.fixture
def y_network():
    """A small Y-shaped river network (two headwaters -> confluence -> outlet).

    Layout (flow downward)::

        0       1       (headwaters)
         \\     /
          \\   /
            2           (confluence)
            |
            3           (outlet)

    Returns:
        Network1d: 4 nodes, 3 directed edges oriented downstream.
    """
    node_x = np.array([0.0, 2.0, 1.0, 1.0])
    node_y = np.array([2.0, 2.0, 1.0, 0.0])
    edges = np.array([[0, 2], [1, 2], [2, 3]], dtype=np.intp)
    return Network1d.from_arrays(node_x, node_y, edges, directed=True, epsg=4326)


class TestConstruction:
    """Construction and basic topology counts."""

    def test_from_arrays_counts(self, y_network):
        """from_arrays yields the expected node and edge counts."""
        assert y_network.n_node == 4, f"expected 4 nodes, got {y_network.n_node}"
        assert y_network.n_edge == 3, f"expected 3 edges, got {y_network.n_edge}"

    def test_directed_flag(self, y_network):
        """The directed flag is carried through from_arrays."""
        assert y_network.directed is True

    def test_crs_from_epsg(self, y_network):
        """An epsg argument builds a usable CRS."""
        assert y_network.crs is not None
        assert y_network.crs.to_epsg() == 4326

    def test_edge_node_connectivity_type(self, y_network):
        """The edge connectivity is a Connectivity wrapper of shape (n_edge, 2)."""
        enc = y_network.edge_node_connectivity
        assert isinstance(enc, Connectivity)
        assert enc.data.shape == (3, 2)

    def test_n_face_raises(self, y_network):
        """A 1D network has no faces; n_face raises a clear error."""
        with pytest.raises(AttributeError, match="no faces"):
            _ = y_network.n_face


class TestGeometry:
    """Derived geometric properties."""

    def test_bounds(self, y_network):
        """bounds spans the full node extent."""
        assert y_network.bounds == (0.0, 0.0, 2.0, 2.0)

    def test_edge_lengths(self, y_network):
        """Edge lengths match the Euclidean node distances."""
        expected = np.array([np.sqrt(2.0), np.sqrt(2.0), 1.0])
        np.testing.assert_array_almost_equal(
            y_network.edge_lengths, expected, err_msg="edge lengths wrong"
        )

    def test_edge_midpoints(self, y_network):
        """Edge midpoints are the mean of each edge's node coordinates."""
        mx, my = y_network.edge_midpoints
        np.testing.assert_array_almost_equal(mx, np.array([0.5, 1.5, 1.0]))
        np.testing.assert_array_almost_equal(my, np.array([1.5, 1.5, 0.5]))

    def test_get_edge_coords(self, y_network):
        """get_edge_coords returns the start and end node coordinates."""
        start, end = y_network.get_edge_coords(2)
        np.testing.assert_array_equal(start, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(end, np.array([1.0, 0.0]))

    def test_kd_trees_build(self, y_network):
        """Node and edge KD-trees build and have the right sizes."""
        assert y_network.node_tree.n == 4
        assert y_network.edge_tree.n == 3


class TestSort:
    """Directed network traversal / topological sort."""

    def test_downstream_order(self, y_network):
        """Downstream sort puts headwaters first and the outlet last."""
        order = y_network.sort("downstream")
        assert set(order.tolist()) == {0, 1, 2, 3}
        # headwaters (0, 1) before the confluence (2), confluence before outlet (3)
        pos = {int(n): i for i, n in enumerate(order)}
        assert pos[0] < pos[2]
        assert pos[1] < pos[2]
        assert pos[2] < pos[3]
        assert order[-1] == 3, "outlet must be last downstream"

    def test_upstream_order(self, y_network):
        """Upstream sort reverses the flow: the outlet comes first."""
        order = y_network.sort("upstream")
        assert order[0] == 3, "outlet must be first upstream"
        pos = {int(n): i for i, n in enumerate(order)}
        assert pos[3] < pos[2]
        assert pos[2] < pos[0]
        assert pos[2] < pos[1]

    def test_sort_requires_directed(self):
        """sort() on an undirected network raises."""
        net = Network1d.from_arrays(
            np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([[0, 1]])
        )
        with pytest.raises(ValueError, match="directed"):
            net.sort()

    def test_sort_bad_direction(self, y_network):
        """An unknown direction raises."""
        with pytest.raises(ValueError, match="downstream"):
            y_network.sort("sideways")

    def test_sort_detects_cycle(self):
        """A cyclic directed network cannot be sorted."""
        net = Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2], [2, 0]]),
            directed=True,
        )
        with pytest.raises(ValueError, match="cycle"):
            net.sort()


class TestTopologySubset:
    """Edge subsetting with node renumbering."""

    def test_subset_keeps_edges_and_renumbers(self, y_network):
        """Subsetting to one edge keeps 2 nodes renumbered to 0..1."""
        sub = y_network.topology_subset([2])
        assert sub.n_edge == 1
        assert sub.n_node == 2
        np.testing.assert_array_equal(
            sub.edge_node_connectivity.data, np.array([[0, 1]])
        )
        # nodes 2 and 3 of the parent become 0 and 1 in the subset
        np.testing.assert_array_equal(sub.node_x, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(sub.node_y, np.array([1.0, 0.0]))


class TestToGeoDataFrame:
    """Vector export."""

    def test_edges_to_linestrings(self, y_network):
        """to_geodataframe yields one LineString per edge with the network CRS."""
        gdf = y_network.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3, f"expected 3 edges, got {len(gdf)}"
        assert all(isinstance(g, LineString) for g in gdf.geometry)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326
        # the outlet edge (2 -> 3) runs from (1, 1) to (1, 0)
        np.testing.assert_array_almost_equal(
            np.array(gdf.geometry.iloc[2].coords),
            np.array([[1.0, 1.0], [1.0, 0.0]]),
        )

    def test_nodes_to_points(self, y_network):
        """to_geodataframe(location='node') yields one Point per node."""
        gdf = y_network.to_geodataframe(location="node")
        assert len(gdf) == 4
        assert all(isinstance(g, Point) for g in gdf.geometry)

    def test_bad_location(self, y_network):
        """An unknown location raises."""
        with pytest.raises(ValueError, match="edge.*node"):
            y_network.to_geodataframe(location="face")
