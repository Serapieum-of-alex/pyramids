"""1D network mesh topology class.

Provides the :class:`Network1d` class for UGRID ``topology_dimension == 1``
data (river networks, 1D hydraulic models): node coordinates plus an
edge-node connectivity table, with **no faces**. The class is a standalone
sibling of :class:`pyramids.netcdf.ugrid.mesh.Mesh2d` — it holds pure numpy
arrays and a :class:`~pyramids.netcdf.ugrid.connectivity.Connectivity`
wrapper, with no reference to GDAL objects and no dependency on ``Mesh2d``.

**In-memory only.** There is no reader or writer for 1-D networks yet, so a network has to
be built from arrays. Not because the topology parser rejects them -- it has no
``topology_dimension`` filter at all, and reads a file declaring ``1`` without complaint,
recording the dimension as given. The limitation is that the reader has only one mesh type to
build: :class:`UgridDataset` carries a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`, and
nothing maps a declared 1-D topology onto a :class:`Network1d`. A genuinely 1-D file -- edges
and no faces -- therefore comes back as a face-less ``Mesh2d``. Reading and writing
``topology_dimension == 1`` files as networks is a separate piece of work.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS as PyProjCRS
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from pyramids.netcdf.ugrid.connectivity import Connectivity


class Network1d:
    """1D network mesh topology (nodes + edges, no faces).

    Holds node coordinates and an edge-node connectivity table describing a
    1D network such as a river tree or a 1D hydraulic model. Provides derived
    geometric properties (edge lengths, edge midpoints, bounds), nearest-neighbour
    KD-trees on nodes and edge midpoints, and a topological network sort for
    upstream/downstream ordering of a directed network.

    This class does NOT inherit from Dataset or RasterBase and is independent
    of :class:`Mesh2d`. The core element is the **edge**; there are no faces,
    centroids, or areas.

    Built from arrays, not read from a file: there is no 1-D reader or writer yet, and
    :class:`UgridDataset` holds a :class:`Mesh2d`, so a network does not travel through
    `read_file` / `to_file`.

    Attributes:
        _node_x: Node x-coordinates (n_node,), float64.
        _node_y: Node y-coordinates (n_node,), float64.
        _edge_node_connectivity: Edge-to-node connectivity (n_edge, 2).
        _directed: Whether the edge orientation (from-node -> to-node) is meaningful.
        _crs: Optional CRS object (pyproj.CRS) for the node coordinates.
    """

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        edge_node_connectivity: Connectivity,
        directed: bool = False,
        crs: Any = None,
    ):
        """Initialise a 1D network from node coordinates and edge connectivity.

        Args:
            node_x: Node x-coordinates, shape (n_node,).
            node_y: Node y-coordinates, shape (n_node,).
            edge_node_connectivity: Edge-to-node connectivity wrapper; its data
                is an (n_edge, 2) integer array of node indices.
            directed: When True, each edge's node order (from-node then to-node)
                encodes flow direction and the network sort uses it. Defaults to False.
            crs: Optional coordinate reference system (a ``pyproj.CRS`` or None).
        """
        self._node_x = np.asarray(node_x, dtype=np.float64)
        self._node_y = np.asarray(node_y, dtype=np.float64)
        self._edge_node_connectivity = edge_node_connectivity
        self._directed = bool(directed)
        self._crs = crs
        self._cached_node_tree: cKDTree | None = None
        self._cached_edge_tree: cKDTree | None = None

    @property
    def node_x(self) -> np.typing.NDArray:
        """Node x-coordinates array (n_node,)."""
        return self._node_x

    @property
    def node_y(self) -> np.typing.NDArray:
        """Node y-coordinates array (n_node,)."""
        return self._node_y

    @property
    def edge_node_connectivity(self) -> Connectivity:
        """Edge-to-node connectivity array (n_edge, 2)."""
        return self._edge_node_connectivity

    @property
    def directed(self) -> bool:
        """Whether edge orientation (from-node -> to-node) is meaningful."""
        return self._directed

    @property
    def crs(self) -> Any:
        """The coordinate reference system object, or None."""
        return self._crs

    @property
    def n_node(self) -> int:
        """Number of nodes in the network."""
        return len(self._node_x)

    @property
    def n_edge(self) -> int:
        """Number of edges in the network."""
        return self._edge_node_connectivity.n_elements

    @property
    def n_face(self) -> int:
        """A 1D network has no faces; accessing this raises.

        Raises:
            AttributeError: Always — a 1D network has only nodes and edges.
        """
        raise AttributeError(
            "Network1d is a 1D topology (nodes + edges) and has no faces; "
            "'n_face' is undefined. Use 'n_node' / 'n_edge'."
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Network bounding box as (xmin, ymin, xmax, ymax).

        Raises:
            ValueError: The network has no nodes, so it has no extent.
        """
        if self._node_x.size == 0:
            raise ValueError(
                "bounds: the network has no nodes, so it has no bounding box."
            )
        result = (
            float(self._node_x.min()),
            float(self._node_y.min()),
            float(self._node_x.max()),
            float(self._node_y.max()),
        )
        return result

    @property
    def _edge_nodes(self) -> np.typing.NDArray:
        """The (n_edge, 2) integer node-index pairs for every edge."""
        data = np.asarray(self._edge_node_connectivity.data)
        return data.reshape(-1, 2) if data.ndim == 1 else data

    @property
    def _real_edges(self) -> np.typing.NDArray:
        """Mask of the rows of the edge table that describe an actual edge.

        A ragged or partially populated edge table pads a row with the fill value, and that row
        declares no edge. It has to keep its position, because an edge-located variable indexes
        by position -- but every geometric member must step over it. Indexing the raw table
        instead let numpy's negative indexing resolve ``-1`` to the **last** node, so a
        placeholder acquired a zero length, a midpoint on that node, and a winning position in
        the nearest-edge tree.

        Returns:
            np.ndarray: One boolean per declared row, True where both node slots are real.
        """
        edges = self._edge_nodes
        fill = self._edge_node_connectivity.fill_value
        real: np.typing.NDArray = (edges[:, 0] != fill) & (edges[:, 1] != fill)
        # Any negative index would address a node from the wrong end just as silently as the
        # conventional -1 fill does, so they are all excluded rather than just the fill.
        real &= (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
        return real

    @property
    def real_edge_indices(self) -> np.typing.NDArray:
        """The positions of the rows that describe an actual edge.

        This is the index map for :attr:`edge_tree`, which holds only the real edges: a tree
        position ``p`` is edge ``real_edge_indices[p]``. The two differ as soon as a padded row
        is not last, so reading a tree index as an edge index would name the wrong edge.

        Returns:
            np.ndarray: Edge positions, ascending. ``arange(n_edge)`` when nothing is padded.
        """
        indices: np.typing.NDArray = np.flatnonzero(self._real_edges).astype(np.intp)
        return indices

    @property
    def edge_lengths(self) -> np.typing.NDArray:
        """Euclidean length of each edge, shape (n_edge,).

        Lengths are computed from the node coordinates in the network's own
        coordinate units (no geodesic correction).

        A padded row describes no edge, so its length is ``nan`` rather than ``0.0`` -- zero is
        a plausible length for a real degenerate edge, and nothing downstream could tell the two
        apart.

        Returns:
            np.ndarray: One length per declared row, ``nan`` where the row is padded.
        """
        edges = self._edge_nodes
        real = self._real_edges
        lengths: np.typing.NDArray = np.full(edges.shape[0], np.nan, dtype="float64")
        if real.any():
            kept = edges[real]
            x0 = self._node_x[kept[:, 0]]
            y0 = self._node_y[kept[:, 0]]
            x1 = self._node_x[kept[:, 1]]
            y1 = self._node_y[kept[:, 1]]
            lengths[real] = np.hypot(x1 - x0, y1 - y0)
        return lengths

    @property
    def edge_midpoints(self) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Midpoint coordinates of every edge as (mx, my), each shape (n_edge,).

        A padded row has no midpoint, so both coordinates are ``nan`` -- it previously reported
        the last node's position, a point the network really contains, which made the fabricated
        value indistinguishable from a real one.

        Returns:
            tuple: ``(mx, my)``, each one value per declared row, ``nan`` where padded.
        """
        edges = self._edge_nodes
        real = self._real_edges
        mx = np.full(edges.shape[0], np.nan, dtype="float64")
        my = np.full(edges.shape[0], np.nan, dtype="float64")
        if real.any():
            kept = edges[real]
            mx[real] = 0.5 * (self._node_x[kept[:, 0]] + self._node_x[kept[:, 1]])
            my[real] = 0.5 * (self._node_y[kept[:, 0]] + self._node_y[kept[:, 1]])
        return mx, my

    @property
    def node_tree(self) -> cKDTree:
        """Lazily built KD-tree on the node coordinates."""
        if self._cached_node_tree is None:
            coords = np.column_stack([self._node_x, self._node_y])
            self._cached_node_tree = cKDTree(coords)
        return self._cached_node_tree

    @property
    def edge_tree(self) -> cKDTree:
        """Lazily built KD-tree on the edge midpoint coordinates.

        Built over the **real** edges only. A padded row has a ``nan`` midpoint and `cKDTree`
        refuses non-finite data, so it cannot simply be included; nor may it be faked into a
        finite point, which is the fabrication this guards against -- placed on the last node it
        won every query near that node at distance 0.0.

        A query therefore answers a position into this tree, which is **not** an edge index when
        anything is padded. Map it through :attr:`real_edge_indices`.

        Returns:
            scipy.spatial.cKDTree: A tree over the real edges' midpoints.
        """
        if self._cached_edge_tree is None:
            mx, my = self.edge_midpoints
            real = self._real_edges
            self._cached_edge_tree = cKDTree(np.column_stack([mx[real], my[real]]))
        return self._cached_edge_tree

    def get_edge_nodes(self, edge_idx: int) -> np.typing.NDArray:
        """Return the node indices of a single edge.

        Args:
            edge_idx: Edge index.

        Returns:
            1D array of the (two) node indices defining the edge.

        Raises:
            ValueError: ``edge_idx`` names a padded row, which declares no edge. It previously
                answered an empty array, and the coordinate lookup built on it then raised a
                bare `IndexError` from inside numpy.
        """
        if not bool(self._real_edges[edge_idx]):
            raise ValueError(
                f"edge {edge_idx} is a padded row and describes no edge; its node slots hold "
                f"the fill value {self._edge_node_connectivity.fill_value!r}."
            )
        return self._edge_node_connectivity.get_element(edge_idx)

    def get_edge_coords(
        self, edge_idx: int
    ) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Return start and end coordinates for an edge.

        Args:
            edge_idx: Edge index.

        Returns:
            Tuple of (start_xy, end_xy) where each is a (2,) array.

        Raises:
            ValueError: ``edge_idx`` names a padded row, which has no endpoints. Routed through
                :meth:`get_edge_nodes` so the one guard covers both accessors -- reading the
                connectivity directly here is what let a padded row through to a bare
                `IndexError` from inside numpy.
        """
        nodes = self.get_edge_nodes(edge_idx)
        start = np.array([self._node_x[nodes[0]], self._node_y[nodes[0]]])
        end = np.array([self._node_x[nodes[1]], self._node_y[nodes[1]]])
        return start, end

    def sort(self, direction: str = "downstream") -> np.typing.NDArray:
        """Topologically order the nodes of a directed network.

        Uses Kahn's algorithm on the directed edges (each edge runs from its
        first node to its second). For ``"downstream"`` the result starts at the
        source nodes (headwaters, in-degree 0) and ends at the sink nodes
        (outlets); ``"upstream"`` reverses every edge first, so the result starts
        at the outlets. Either way the result is a valid topological sort of the
        edges as traversed: ``"downstream"`` puts an upstream node before the
        downstream node it drains into, and ``"upstream"`` -- walking the reversed
        edges -- puts the downstream node first. On a chain 0->1->2->3 that is
        ``[0, 1, 2, 3]`` and ``[3, 2, 1, 0]`` respectively.

        Args:
            direction: ``"downstream"`` (sources first) or ``"upstream"`` (sinks
                first). Defaults to ``"downstream"``.

        Returns:
            1D integer array of node indices in traversal order (length n_node).

        Raises:
            ValueError: If ``direction`` is not ``"downstream"``/``"upstream"``, if
                the network is not directed, or if the network contains a cycle
                (no valid topological order exists).
        """
        if direction not in ("downstream", "upstream"):
            raise ValueError(
                f"direction must be 'downstream' or 'upstream', got {direction!r}."
            )
        if not self._directed:
            raise ValueError(
                "sort() needs a directed network; construct Network1d(..., directed=True) "
                "with edges oriented from upstream to downstream."
            )
        edges = self._edge_nodes
        if direction == "upstream":
            edges = edges[:, ::-1]

        n = self.n_node
        indegree = np.zeros(n, dtype=np.intp)
        adjacency: list[list[int]] = [[] for _ in range(n)]
        for src, dst in edges:
            adjacency[int(src)].append(int(dst))
            indegree[int(dst)] += 1

        queue: deque[int] = deque(int(i) for i in np.nonzero(indegree == 0)[0])
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adjacency[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != n:
            raise ValueError(
                "network contains a cycle; a directed 1D network must be acyclic "
                "to be sorted upstream/downstream."
            )
        return np.array(order, dtype=np.intp)

    def topology_subset(self, edge_indices: np.ndarray) -> Network1d:
        """Return a sub-network holding only the selected edges.

        The surviving edges' nodes are deduplicated and renumbered to a compact
        ``0..k-1`` range, mirroring the 2D ``topology_subset`` contract. Isolated
        nodes (not referenced by any surviving edge) are dropped.

        Args:
            edge_indices: Indices of the edges to keep.

        Returns:
            Network1d: A new network with renumbered nodes and the kept edges.
        """
        edge_idx = np.asarray(edge_indices, dtype=np.intp).ravel()
        kept = self._edge_nodes[edge_idx]
        # Keep the fill out of the node set. `np.unique` would include -1, and numpy's
        # negative indexing then silently addresses the *last* node -- so a padded edge
        # row became a real edge joining the end of the network, with a length and a
        # LineString, and a node no selected edge references became node 0.
        # The connectivity may be a bare array (the constructor accepts either), so read
        # the fill defensively; -1 is this class's documented fill.
        fill = getattr(self._edge_node_connectivity, "fill_value", -1)
        used_nodes = np.unique(kept[kept != fill])
        remap = np.full(self.n_node, -1, dtype=np.intp)
        remap[used_nodes] = np.arange(len(used_nodes), dtype=np.intp)
        # A fill entry stays a fill entry rather than being remapped to a node.
        new_edges = np.where(kept == fill, -1, remap[np.where(kept == fill, 0, kept)])

        new_conn = Connectivity(
            data=new_edges.astype(np.intp),
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        return Network1d(
            node_x=self._node_x[used_nodes],
            node_y=self._node_y[used_nodes],
            edge_node_connectivity=new_conn,
            directed=self._directed,
            crs=self._crs,
        )

    def _edge_linestrings(self) -> list:
        """Build one LineString per edge, vectorized for 2-node edges.

        Mirrors :meth:`pyramids.netcdf.ugrid.dataset.UgridDataset._edge_linestrings`:
        the standard clean (n_edge, 2) path is built with ``shapely.linestrings``;
        a ragged / filled connectivity falls back to a per-edge build.
        """
        enc = self._edge_node_connectivity
        node_idx = np.asarray(enc.data)
        no_fill = enc.fill_value is None or bool(np.all(node_idx != enc.fill_value))
        if node_idx.ndim == 2 and node_idx.shape[1] == 2 and no_fill:
            xs = self._node_x[node_idx]
            ys = self._node_y[node_idx]
            return list(shapely.linestrings(np.stack([xs, ys], axis=-1)))
        # A row with fewer than two real nodes describes no edge, and GEOS refuses a
        # one-point LineString outright ("point array must contain 0 or >1 elements"), so it
        # became a raw GEOSException out of a public method. An empty LineString is the same
        # answer a fully padded row gets, which keeps one row per declared edge.
        return [
            LineString([(self._node_x[n], self._node_y[n]) for n in enc.get_element(i)])
            if len(enc.get_element(i)) >= 2
            else LineString()
            for i in range(enc.n_elements)
        ]

    def to_geodataframe(self, location: str = "edge") -> gpd.GeoDataFrame:
        """Convert the network to a GeoDataFrame.

        For edges each row is a LineString; for nodes each row is a Point.

        Args:
            location: ``"edge"`` (one LineString per edge, the default) or
                ``"node"`` (one Point per node).

        Returns:
            geopandas.GeoDataFrame with a ``geometry`` column, CRS-stamped when
            the network has a CRS. For ``"edge"`` there is one row per *declared* edge, so an
            edge-located variable stays aligned with the frame -- which means a padded row
            (one the connectivity declares but leaves as fill) contributes an **empty**
            LineString rather than being dropped or drawn as a degenerate line.

        Raises:
            ValueError: If ``location`` is neither ``"edge"`` nor ``"node"``.
        """
        if location == "edge":
            geometries: list = self._edge_linestrings()
        elif location == "node":
            geometries = list(shapely.points(self._node_x, self._node_y))
        else:
            raise ValueError(f"location must be 'edge' or 'node', got {location!r}.")
        gdf = gpd.GeoDataFrame(geometry=geometries)
        if self._crs is not None:
            gdf = gdf.set_crs(self._crs)
        return gdf

    @classmethod
    def from_arrays(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        edge_node_connectivity: np.ndarray,
        directed: bool = False,
        epsg: int | None = None,
        crs: Any = None,
    ) -> Network1d:
        """Create a Network1d programmatically from numpy arrays.

        Args:
            node_x: Node x-coordinates, shape (n_node,).
            node_y: Node y-coordinates, shape (n_node,).
            edge_node_connectivity: (n_edge, 2) array of node indices per edge.
            directed: Whether edge orientation encodes flow direction. Defaults to False.
            epsg: Optional EPSG code to build the CRS from. Ignored when ``crs`` is given.
            crs: Optional CRS object (``pyproj.CRS``). Takes precedence over ``epsg``.

        Returns:
            Network1d instance.
        """
        enc = Connectivity(
            data=np.asarray(edge_node_connectivity, dtype=np.intp).reshape(-1, 2),
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )
        resolved_crs = crs
        if resolved_crs is None and epsg is not None:
            resolved_crs = PyProjCRS.from_epsg(epsg)
        return cls(
            node_x=np.asarray(node_x, dtype=np.float64),
            node_y=np.asarray(node_y, dtype=np.float64),
            edge_node_connectivity=enc,
            directed=directed,
            crs=resolved_crs,
        )
