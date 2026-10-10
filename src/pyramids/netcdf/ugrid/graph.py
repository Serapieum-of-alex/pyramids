"""Mesh graph and connectivity operations for UGRID 2D meshes.

A collection of pure functions that each take a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`
(and/or plain numpy arrays) and return plain results — numpy arrays, :class:`Connectivity`
wrappers, scipy sparse matrices, shapely geometries, or a rebuilt ``Mesh2d`` for the one
topology-transforming helper. None of them mutate their input.

The module groups the graph-flavoured mesh operations behind a single import:

- Connectivity builders that pyramids does not derive today: ``node_node_connectivity``,
  ``node_face_connectivity``, ``node_edge_connectivity``, ``edge_node_connectivity``,
  ``edge_face_connectivity``, ``face_edge_connectivity`` and ``edge_edge_connectivity``.
- ``sparse_adjacency`` — a scipy sparse adjacency matrix over faces or nodes, the shared
  primitive the graph algorithms below build on.
- Facet remapping (``to_node`` / ``to_edge`` / ``to_face``) that averages a per-element
  field onto another mesh facet via connectivity.
- Graph algorithms: ``connected_components``, ``reverse_cuthill_mckee`` (+ ``permute_faces``),
  ``binary_dilation`` / ``binary_erosion``, ``voronoi`` / ``circumcenters`` and the boundary
  helpers ``exterior_edges`` / ``exterior_faces`` / ``bounding_polygon``.

scipy (``scipy.sparse``, ``scipy.sparse.csgraph``, ``scipy.spatial``) and shapely are hard
dependencies of the UGRID subsystem, so they are imported at module top.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import Voronoi
from shapely.ops import unary_union

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d


def _edge_tables(
    mesh: Mesh2d,
) -> tuple[np.typing.NDArray, np.typing.NDArray, Connectivity]:
    """Derive the edge tables of a mesh without mutating it.

    Builds, from ``face_node_connectivity`` (and the mesh's own edges when present):

    - ``edge_node`` of shape ``(n_edge, 2)`` — the two node indices of each edge.
    - ``edge_face`` of shape ``(n_edge, 2)`` — the (up to two) faces on either side of each
      edge, with ``-1`` marking a missing (exterior) side.
    - ``face_edge`` — a ragged :class:`Connectivity` mapping each face to its boundary edges.

    When ``mesh.edge_node_connectivity`` exists its edge ordering/identity is honoured, so
    edge-located variables keep their indexing; otherwise edges are derived from the faces in
    first-seen order.

    Args:
        mesh: The mesh to read.

    Returns:
        tuple: ``(edge_node, edge_face, face_edge)``.
    """
    fnc = mesh.face_node_connectivity
    n_face = mesh.n_face

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    face_boundary_edges: list[list[tuple[int, int]]] = []
    for i in range(n_face):
        nodes = [int(x) for x in fnc.get_element(i)]
        n = len(nodes)
        keys: list[tuple[int, int]] = []
        for j in range(n):
            a = nodes[j]
            b = nodes[(j + 1) % n]
            key = (a, b) if a < b else (b, a)
            keys.append(key)
            edge_to_faces.setdefault(key, []).append(i)
        face_boundary_edges.append(keys)

    enc = mesh.edge_node_connectivity
    if enc is not None:
        # The file's edge order has to survive: edge-located variables index into it, so a listed
        # edge keeps its own position. UGRID does not require the list to be complete, though
        # (boundary-edges-only is a common, legal shape), so the face-implied edges it omits are
        # appended after it -- otherwise every unlisted edge is dropped from the face topology and
        # the derived adjacency silently comes back empty.
        keys_in_order: list[tuple[int, int]] = []
        edge_id: dict[tuple[int, int], int] = {}
        for e in range(enc.n_elements):
            ns = [int(x) for x in enc.get_element(e)]
            if len(ns) < 2:
                # A ragged edge array pads short rows with the fill value; get_element strips
                # those, so a padded row can arrive with fewer than two nodes. It defines no edge.
                keys_in_order.append((-1, -1))
                continue
            a, b = ns[0], ns[1]
            key = (a, b) if a < b else (b, a)
            keys_in_order.append(key)
            edge_id[key] = e
        for key in edge_to_faces:
            if key not in edge_id:
                edge_id[key] = len(keys_in_order)
                keys_in_order.append(key)
        edge_node = (
            np.array(keys_in_order, dtype=np.intp)
            if keys_in_order
            else np.empty((0, 2), dtype=np.intp)
        )
    else:
        keys_sorted = list(edge_to_faces.keys())
        edge_node = (
            np.array(keys_sorted, dtype=np.intp)
            if keys_sorted
            else np.empty((0, 2), dtype=np.intp)
        )
        edge_id = {k: idx for idx, k in enumerate(keys_sorted)}

    n_edge = int(edge_node.shape[0])
    edge_face = np.full((n_edge, 2), -1, dtype=np.intp)
    for key, faces in edge_to_faces.items():
        edge_idx = edge_id.get(key)
        if edge_idx is None:
            continue
        for col, f in enumerate(faces[:2]):
            edge_face[edge_idx, col] = f

    max_width = max((len(keys) for keys in face_boundary_edges), default=1) or 1
    fe_data = np.full((n_face, max_width), -1, dtype=np.intp)
    for i, keys in enumerate(face_boundary_edges):
        for col, key in enumerate(keys):
            edge_idx = edge_id.get(key)
            if edge_idx is not None:
                fe_data[i, col] = edge_idx
    face_edge = Connectivity(
        data=fe_data,
        fill_value=-1,
        cf_role="face_edge_connectivity",
        original_start_index=0,
    )
    return edge_node, edge_face, face_edge


def _group_to_connectivity(
    keys: np.typing.NDArray,
    values: np.typing.NDArray,
    n_groups: int,
    cf_role: str,
) -> Connectivity:
    """Group ``(key, value)`` pairs into a ragged :class:`Connectivity`.

    Each distinct ``key`` becomes a row whose entries are the sorted unique ``values`` seen
    for it, left-packed and padded with ``-1``. Keys with no pairs produce an all-fill row, so
    the result always has ``n_groups`` rows.

    Args:
        keys: Group id per pair (``0 <= key < n_groups``).
        values: Member id per pair.
        n_groups: Number of rows in the output.
        cf_role: ``cf_role`` tag for the returned connectivity.

    Returns:
        Connectivity: The ragged grouping.
    """
    keys = np.asarray(keys, dtype=np.intp)
    values = np.asarray(values, dtype=np.intp)
    if keys.size:
        pairs = np.unique(np.stack([keys, values], axis=1), axis=0)
        keys = pairs[:, 0]
        values = pairs[:, 1]
    counts = (
        np.bincount(keys, minlength=n_groups)
        if keys.size
        else np.zeros(n_groups, dtype=np.intp)
    )
    width = max(int(counts.max()) if counts.size else 1, 1)
    out = np.full((n_groups, width), -1, dtype=np.intp)
    if keys.size:
        starts = np.cumsum(counts) - counts
        col = np.arange(keys.size) - starts[keys]
        out[keys, col] = values
    return Connectivity(
        data=out, fill_value=-1, cf_role=cf_role, original_start_index=0
    )


def _invert_connectivity(
    conn: Connectivity, n_source: int, cf_role: str
) -> Connectivity:
    """Invert a target-to-source connectivity into source-to-target.

    Args:
        conn: Connectivity mapping each row (target) to its source indices.
        n_source: Number of source elements (rows in the result).
        cf_role: ``cf_role`` tag for the returned connectivity.

    Returns:
        Connectivity: The inverted mapping.
    """
    data = conn.data
    if data.ndim == 1:
        data = data[:, None]
    mask = data != conn.fill_value
    n_target = data.shape[0]
    target_ids = np.broadcast_to(np.arange(n_target)[:, None], data.shape)[mask]
    source_ids = data[mask]
    return _group_to_connectivity(source_ids, target_ids, n_source, cf_role)


def node_face_connectivity(mesh: Mesh2d) -> Connectivity:
    """Faces incident to each node (inverse of ``face_node_connectivity``).

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: Ragged ``node -> faces`` mapping, ``-1`` padded.
    """
    return _invert_connectivity(
        mesh.face_node_connectivity, mesh.n_node, "node_face_connectivity"
    )


def edge_node_connectivity(mesh: Mesh2d) -> Connectivity:
    """The two nodes of each edge, derived from the faces when absent.

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: ``edge -> (node, node)`` mapping.
    """
    edge_node, _, _ = _edge_tables(mesh)
    return Connectivity(
        data=edge_node,
        fill_value=-1,
        cf_role="edge_node_connectivity",
        original_start_index=0,
    )


def node_edge_connectivity(mesh: Mesh2d) -> Connectivity:
    """Edges incident to each node (inverse of ``edge_node_connectivity``).

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: Ragged ``node -> edges`` mapping, ``-1`` padded.
    """
    edge_node, _, _ = _edge_tables(mesh)
    e_ids = np.arange(edge_node.shape[0], dtype=np.intp)
    keys = np.concatenate([edge_node[:, 0], edge_node[:, 1]])
    values = np.concatenate([e_ids, e_ids])
    return _group_to_connectivity(keys, values, mesh.n_node, "node_edge_connectivity")


def node_node_connectivity(mesh: Mesh2d) -> Connectivity:
    """Neighbouring nodes of each node, derived from the mesh edges.

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: Ragged ``node -> neighbour nodes`` mapping, ``-1`` padded.
    """
    edge_node, _, _ = _edge_tables(mesh)
    keys = np.concatenate([edge_node[:, 0], edge_node[:, 1]])
    values = np.concatenate([edge_node[:, 1], edge_node[:, 0]])
    return _group_to_connectivity(keys, values, mesh.n_node, "node_node_connectivity")


def edge_face_connectivity(mesh: Mesh2d) -> Connectivity:
    """The (up to two) faces on either side of each edge.

    Edges on the mesh boundary carry a ``-1`` fill on their missing side.

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: ``edge -> faces`` mapping, ``-1`` for an exterior side.
    """
    _, edge_face, _ = _edge_tables(mesh)
    return Connectivity(
        data=edge_face,
        fill_value=-1,
        cf_role="edge_face_connectivity",
        original_start_index=0,
    )


def face_edge_connectivity(mesh: Mesh2d) -> Connectivity:
    """The boundary edges of each face.

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: Ragged ``face -> edges`` mapping, ``-1`` padded.
    """
    _, _, face_edge = _edge_tables(mesh)
    return face_edge


def edge_edge_connectivity(mesh: Mesh2d) -> Connectivity:
    """Edges that share a node with each edge.

    Args:
        mesh: The mesh to read.

    Returns:
        Connectivity: Ragged ``edge -> neighbouring edges`` mapping, ``-1`` padded.
    """
    node_edge = node_edge_connectivity(mesh)
    edge_node, _, _ = _edge_tables(mesh)
    n_edge = int(edge_node.shape[0])
    keys_list: list[np.typing.NDArray] = []
    values_list: list[np.typing.NDArray] = []
    for r in range(node_edge.n_elements):
        edges = node_edge.get_element(r)
        if edges.size < 2:
            continue
        left, right = np.meshgrid(edges, edges)
        off_diag = left != right
        keys_list.append(left[off_diag])
        values_list.append(right[off_diag])
    if keys_list:
        keys = np.concatenate(keys_list)
        values = np.concatenate(values_list)
    else:
        keys = np.empty(0, dtype=np.intp)
        values = np.empty(0, dtype=np.intp)
    return _group_to_connectivity(keys, values, n_edge, "edge_edge_connectivity")


def sparse_adjacency(mesh: Mesh2d, kind: str = "face") -> Any:
    """Build a symmetric scipy sparse adjacency matrix for the mesh.

    For ``kind="face"`` two faces are adjacent when they share an edge (an edge whose
    ``edge_face_connectivity`` has two valid faces). For ``kind="node"`` two nodes are
    adjacent when an edge joins them. The result is the shared primitive consumed by
    :func:`connected_components`, :func:`reverse_cuthill_mckee` and the binary morphology.

    Args:
        mesh: The mesh to read.
        kind: ``"face"`` (default) or ``"node"``.

    Returns:
        scipy.sparse.csr_matrix: The symmetric ``(n, n)`` adjacency matrix.

    Raises:
        ValueError: If ``kind`` is neither ``"face"`` nor ``"node"``.
    """
    if kind == "face":
        _, edge_face, _ = _edge_tables(mesh)
        both_valid = (edge_face[:, 0] != -1) & (edge_face[:, 1] != -1)
        rows = edge_face[both_valid, 0]
        cols = edge_face[both_valid, 1]
        n = mesh.n_face
    elif kind == "node":
        edge_node, _, _ = _edge_tables(mesh)
        rows = edge_node[:, 0]
        cols = edge_node[:, 1]
        n = mesh.n_node
    else:
        raise ValueError(f"kind must be 'face' or 'node', got {kind!r}.")

    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    data = np.ones(sym_rows.size, dtype=np.int8)
    matrix = sparse.coo_matrix((data, (sym_rows, sym_cols)), shape=(n, n)).tocsr()
    matrix.data[:] = 1
    return matrix


def _average_over(conn: Connectivity, values: np.typing.NDArray) -> np.typing.NDArray:
    """Average ``values`` of each row's valid members onto that row.

    Args:
        conn: Connectivity mapping each target row to its source member indices.
        values: Source field, element axis last (``(..., n_source)``).

    Returns:
        np.ndarray: The averaged field on the target facet (element axis last). Rows with no
        valid member are ``nan``.
    """
    data = conn.data
    if data.ndim == 1:
        data = data[:, None]
    mask = data != conn.fill_value
    safe = np.where(mask, data, 0)
    source = np.asarray(values, dtype=np.float64)
    gathered = np.where(mask, source[..., safe], 0.0)
    counts = mask.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = gathered.sum(axis=-1) / counts
    return np.asarray(result)


def to_node(
    mesh: Mesh2d, values: np.typing.NDArray, source: str = "face"
) -> np.typing.NDArray:
    """Remap a per-face or per-edge field onto the nodes by averaging.

    Args:
        mesh: The mesh to read.
        values: Source field, element axis last.
        source: ``"face"`` (default) or ``"edge"``.

    Returns:
        np.ndarray: The node-located field.

    Raises:
        ValueError: If ``source`` is not ``"face"`` or ``"edge"``.
    """
    if source == "face":
        conn = node_face_connectivity(mesh)
    elif source == "edge":
        conn = node_edge_connectivity(mesh)
    else:
        raise ValueError(f"source must be 'face' or 'edge', got {source!r}.")
    return _average_over(conn, values)


def to_face(
    mesh: Mesh2d, values: np.typing.NDArray, source: str = "node"
) -> np.typing.NDArray:
    """Remap a per-node or per-edge field onto the faces by averaging.

    Args:
        mesh: The mesh to read.
        values: Source field, element axis last.
        source: ``"node"`` (default) or ``"edge"``.

    Returns:
        np.ndarray: The face-located field.

    Raises:
        ValueError: If ``source`` is not ``"node"`` or ``"edge"``.
    """
    if source == "node":
        conn = mesh.face_node_connectivity
    elif source == "edge":
        conn = face_edge_connectivity(mesh)
    else:
        raise ValueError(f"source must be 'node' or 'edge', got {source!r}.")
    return _average_over(conn, values)


def to_edge(
    mesh: Mesh2d, values: np.typing.NDArray, source: str = "node"
) -> np.typing.NDArray:
    """Remap a per-node or per-face field onto the edges by averaging.

    Args:
        mesh: The mesh to read.
        values: Source field, element axis last.
        source: ``"node"`` (default) or ``"face"``.

    Returns:
        np.ndarray: The edge-located field.

    Raises:
        ValueError: If ``source`` is not ``"node"`` or ``"face"``.
    """
    if source == "node":
        conn = edge_node_connectivity(mesh)
    elif source == "face":
        conn = edge_face_connectivity(mesh)
    else:
        raise ValueError(f"source must be 'node' or 'face', got {source!r}.")
    return _average_over(conn, values)


def connected_components(mesh: Mesh2d) -> np.typing.NDArray:
    """Label each face by the connected component it belongs to.

    Two faces are in the same component when a path of shared edges joins them — the mesh
    analogue of a raster cluster labelling.

    Args:
        mesh: The mesh to read.

    Returns:
        np.ndarray: Per-face component label of length ``n_face``.
    """
    adjacency = sparse_adjacency(mesh, kind="face")
    _, labels = csgraph.connected_components(adjacency, directed=False)
    return np.asarray(labels, dtype=np.intp)


def reverse_cuthill_mckee(mesh: Mesh2d) -> np.typing.NDArray:
    """Return a face permutation that reduces the face-adjacency bandwidth.

    Computes the reverse Cuthill-McKee ordering of the face adjacency. Apply it to the mesh
    with :func:`permute_faces` and to every face-located variable with ``arr[..., perm]``.

    Args:
        mesh: The mesh to read.

    Returns:
        np.ndarray: A permutation of ``range(n_face)``.
    """
    adjacency = sparse.csr_matrix(sparse_adjacency(mesh, kind="face"))
    perm = csgraph.reverse_cuthill_mckee(adjacency, symmetric_mode=True)
    return np.asarray(perm, dtype=np.intp)


def permute_faces(mesh: Mesh2d, perm: np.typing.NDArray) -> Mesh2d:
    """Rebuild a mesh with its faces reordered by ``perm``.

    Node coordinates and numbering are unchanged — only the face rows (and any face centre
    coordinates) are reordered — so the result is congruent to the input. New face ``k`` is
    old face ``perm[k]``; the input mesh is not mutated.

    Args:
        mesh: The mesh to reorder.
        perm: A permutation of ``range(n_face)``.

    Returns:
        Mesh2d: The reordered mesh.
    """
    perm = np.asarray(perm, dtype=np.intp)
    old_fnc = mesh.face_node_connectivity
    new_fnc = Connectivity(
        data=old_fnc.data[perm].copy(),
        fill_value=old_fnc.fill_value,
        cf_role=old_fnc.cf_role,
        original_start_index=old_fnc.original_start_index,
    )
    face_x = mesh.face_x[perm] if mesh.face_x is not None else None
    face_y = mesh.face_y[perm] if mesh.face_y is not None else None
    return Mesh2d(
        node_x=mesh.node_x.copy(),
        node_y=mesh.node_y.copy(),
        face_node_connectivity=new_fnc,
        edge_node_connectivity=mesh.edge_node_connectivity,
        face_x=face_x,
        face_y=face_y,
        edge_x=mesh.edge_x,
        edge_y=mesh.edge_y,
        crs=mesh._crs,
    )


def _binary_iterate(
    mesh: Mesh2d,
    mask: np.typing.NDArray,
    value: bool,
    iterations: int,
    border_value: bool,
    fixed: np.typing.NDArray | None,
) -> np.typing.NDArray:
    """Iterate the shared edge-neighbour morphology on a boolean face mask.

    In each pass, wherever two edge-adjacent faces disagree both are set to ``value`` (``True``
    grows the region — dilation; ``False`` shrinks it — erosion). The first pass treats the
    mesh exterior as a neighbour holding ``border_value`` (only when ``border_value == value``).
    Fixed cells, when given, are reset to their input value after every pass.

    Args:
        mesh: The mesh to read.
        mask: Input boolean face mask.
        value: ``True`` for dilation, ``False`` for erosion.
        iterations: Number of growth/shrink rings.
        border_value: Value imputed to the mesh exterior.
        fixed: Optional boolean faces held at their input value, or ``None``.

    Returns:
        np.ndarray: The resulting boolean face mask (a new array).
    """
    coo = sparse_adjacency(mesh, kind="face").tocoo()
    i = np.asarray(coo.row, dtype=np.intp)
    j = np.asarray(coo.col, dtype=np.intp)
    input_mask = np.asarray(mask, dtype=bool)
    output = input_mask.copy()
    fixed_arr = None if fixed is None else np.asarray(fixed, dtype=bool)
    exterior = exterior_faces(mesh) if border_value == value else None

    for iteration in range(iterations):
        if iteration == 0 and exterior is not None:
            output[exterior] = value
            if fixed_arr is not None:
                output[fixed_arr] = input_mask[fixed_arr]
        differ = output[i] != output[j]
        output[np.concatenate([i[differ], j[differ]])] = value
        if fixed_arr is not None:
            output[fixed_arr] = input_mask[fixed_arr]
    return output


def binary_dilation(
    mesh: Mesh2d,
    mask: np.typing.NDArray,
    iterations: int = 1,
    border_value: bool = False,
    fixed: np.typing.NDArray | None = None,
) -> np.typing.NDArray:
    """Grow a ``True`` region of a boolean face mask across shared edges.

    Args:
        mesh: The mesh to read.
        mask: Input boolean face mask (not mutated).
        iterations: Number of rings to grow.
        border_value: Value imputed to the mesh exterior (grows inward from the boundary
            when ``True``).
        fixed: Optional boolean faces held at their input value.

    Returns:
        np.ndarray: The dilated boolean face mask.
    """
    return _binary_iterate(mesh, mask, True, iterations, border_value, fixed)


def binary_erosion(
    mesh: Mesh2d,
    mask: np.typing.NDArray,
    iterations: int = 1,
    border_value: bool = False,
    fixed: np.typing.NDArray | None = None,
) -> np.typing.NDArray:
    """Shrink a ``True`` region of a boolean face mask across shared edges.

    Args:
        mesh: The mesh to read.
        mask: Input boolean face mask (not mutated).
        iterations: Number of rings to shrink.
        border_value: Value imputed to the mesh exterior (erodes from the boundary when
            ``False``).
        fixed: Optional boolean faces held at their input value.

    Returns:
        np.ndarray: The eroded boolean face mask.
    """
    return _binary_iterate(mesh, mask, False, iterations, border_value, fixed)


def circumcenters(mesh: Mesh2d) -> np.typing.NDArray:
    """Per-face circumcenter coordinates for a triangular mesh.

    Args:
        mesh: The mesh to read (every face must be a triangle).

    Returns:
        np.ndarray: ``(n_face, 2)`` array of circumcenter ``(x, y)`` coordinates.

    Raises:
        ValueError: If the mesh is not purely triangular.
    """
    fnc = mesh.face_node_connectivity
    if not fnc.is_triangular():
        raise ValueError("circumcenters is only defined for triangular meshes.")
    tri = fnc.data[:, :3]
    ax, ay = mesh.node_x[tri[:, 0]], mesh.node_y[tri[:, 0]]
    bx, by = mesh.node_x[tri[:, 1]], mesh.node_y[tri[:, 1]]
    cx, cy = mesh.node_x[tri[:, 2]], mesh.node_y[tri[:, 2]]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a_sq = ax * ax + ay * ay
    b_sq = bx * bx + by * by
    c_sq = cx * cx + cy * cy
    ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
    uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d
    return np.column_stack([ux, uy])


def voronoi(mesh: Mesh2d) -> Any:
    """Voronoi tessellation of the mesh nodes (the mesh dual).

    Uses :class:`scipy.spatial.Voronoi` over the node coordinates. For a triangular mesh that
    is Delaunay the Voronoi vertices coincide with the face :func:`circumcenters`, and interior
    nodes receive bounded dual cells while boundary nodes receive unbounded (``-1``) regions.

    Args:
        mesh: The mesh to read.

    Returns:
        scipy.spatial.Voronoi: The Voronoi object; ``.points`` are the node coordinates.
    """
    points = np.column_stack([mesh.node_x, mesh.node_y])
    return Voronoi(points)


def exterior_edges(mesh: Mesh2d) -> np.typing.NDArray:
    """Indices of the edges on the mesh boundary.

    A boundary edge is one whose ``edge_face_connectivity`` has a fill on one side (only one
    adjacent face).

    Args:
        mesh: The mesh to read.

    Returns:
        np.ndarray: Boundary edge indices (into the mesh's edge ordering).
    """
    _, edge_face, _ = _edge_tables(mesh)
    exterior = np.any(edge_face == -1, axis=1)
    return np.flatnonzero(exterior).astype(np.intp)


def exterior_faces(mesh: Mesh2d) -> np.typing.NDArray:
    """Indices of the faces that touch the mesh boundary.

    A boundary face is one attached to at least one exterior edge.

    Args:
        mesh: The mesh to read.

    Returns:
        np.ndarray: Sorted unique boundary face indices.
    """
    _, edge_face, _ = _edge_tables(mesh)
    exterior = np.any(edge_face == -1, axis=1)
    faces = edge_face[exterior]
    valid = faces[faces != -1]
    return np.unique(valid).astype(np.intp)


def _face_polygons(mesh: Mesh2d) -> Any:
    """Build one shapely polygon per face, vectorized.

    Args:
        mesh: The mesh to read.

    Returns:
        np.ndarray: Array of shapely ``Polygon`` objects, one per face.
    """
    fnc = mesh.face_node_connectivity
    masked = fnc.as_masked()
    valid = ~np.ma.getmaskarray(masked)
    flat_nodes = np.asarray(masked.data[valid], dtype=np.intp)
    coords = np.column_stack([mesh.node_x[flat_nodes], mesh.node_y[flat_nodes]])
    ring_index = np.repeat(np.arange(fnc.n_elements), fnc.nodes_per_element())
    rings = shapely.linearrings(coords, indices=ring_index)
    return shapely.polygons(rings)


def bounding_polygon(mesh: Mesh2d) -> Any:
    """Dissolved outline of the mesh as a shapely (Multi)Polygon, holes preserved.

    Unions the face polygons so shared interior edges dissolve away; the result's exterior
    ring is the mesh outline and any mesh hole appears as an interior ring. Its area equals the
    mesh footprint. The returned geometry carries no CRS — the caller attaches one.

    Args:
        mesh: The mesh to read.

    Returns:
        shapely.Polygon | shapely.MultiPolygon: The dissolved outline.
    """
    return unary_union(list(_face_polygons(mesh)))
