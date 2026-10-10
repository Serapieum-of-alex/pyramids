"""Mesh-space gap filling for UGRID unstructured meshes.

Pure functions that fill NaN gaps in a per-element data array defined on a
:class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`. Two strategies are provided:

- :func:`fill_na_spatial` — nearest-valid fill by geometric (centroid/node/edge)
  distance, using a :class:`scipy.spatial.cKDTree` over the valid elements.
- :func:`laplace_interpolate` — smooth fill by solving Laplace's equation on the
  mesh adjacency graph, with the valid elements as Dirichlet boundary conditions.

Naming distinction (mesh space vs the time axis): these functions fill across
mesh **space** using connectivity/geometry. Filling along the **time** axis is a
separate operation (``interpolate_na(dim="time")`` on the NetCDF side); it is not
what these functions do.

The element axis is always the last axis (``-1``); any leading time/layer axes are
broadcast over and filled **per slab** (each slab uses its own null mask). The
input array is never mutated — a filled copy is returned.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree

from pyramids.netcdf.ugrid import graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

__all__ = ["fill_na_spatial", "laplace_interpolate"]


def _location_coords(mesh: Mesh2d, location: str) -> np.typing.NDArray:
    """Return the ``(n_elements, 2)`` coordinate array for a mesh location.

    Face elements use their centroids, node elements use the node coordinates,
    and edge elements use the edge-center coordinates.

    Args:
        mesh: Source mesh topology.
        location: Mesh location, one of ``"face"``, ``"node"`` or ``"edge"``.

    Returns:
        Array of shape ``(n_elements, 2)`` with the ``(x, y)`` coordinate of each
        element of the requested location.

    Raises:
        ValueError: If the location is unknown, or edge coordinates are requested
            but not available on the mesh.
    """
    if location == "face":
        cx, cy = mesh.face_centroids
    elif location == "node":
        cx, cy = mesh.node_x, mesh.node_y
    elif location == "edge":
        if not mesh.has_edge_coords:
            raise ValueError("Edge coordinates are not available on the mesh.")
        cx = cast("np.typing.NDArray", mesh.edge_x)
        cy = cast("np.typing.NDArray", mesh.edge_y)
    else:
        raise ValueError(f"Unknown location: {location!r}")
    return np.column_stack([cx, cy])


def _fill_slab_nearest(
    coords: np.typing.NDArray,
    slab: np.typing.NDArray,
    max_distance: float | None,
) -> None:
    """Fill one 1D slab in place from the nearest valid element.

    Args:
        coords: ``(n_elements, 2)`` element coordinates.
        slab: 1D data for a single slab; NaN marks a gap. Mutated in place.
        max_distance: If set, gaps whose nearest valid element is farther than
            this distance are left as NaN. ``None`` means unbounded.
    """
    isnull = np.isnan(slab)
    i_src = np.flatnonzero(~isnull)
    i_tgt = np.flatnonzero(isnull)
    if i_src.size and i_tgt.size:
        tree = cKDTree(coords[i_src])
        if max_distance is None:
            _, idx = tree.query(coords[i_tgt])
        else:
            _, idx = tree.query(coords[i_tgt], distance_upper_bound=max_distance)
        # cKDTree returns len(i_src) (out of range) and inf distance for a target with
        # no neighbour within distance_upper_bound; that target keeps its NaN.
        keep = idx < len(i_src)
        slab[i_tgt[keep]] = slab[i_src[idx[keep]]]


def fill_na_spatial(
    mesh: Mesh2d,
    values: np.typing.NDArray,
    location: str = "face",
    *,
    max_distance: float | None = None,
) -> np.typing.NDArray:
    """Fill NaN gaps from the nearest valid element by geometric distance.

    Each NaN element takes the value of the nearest non-NaN element of the same
    location, where "nearest" is Euclidean distance between element coordinates
    (face centroids, node coordinates, or edge centers). A :class:`scipy.spatial.cKDTree`
    is built over the valid elements of each slab.

    This fills across mesh **space**; it is distinct from filling along the time
    axis (``interpolate_na(dim="time")``).

    Args:
        mesh: Source mesh topology.
        values: Data array whose **last** axis is the element axis; NaN marks a
            gap. Any leading axes (time/layers) are broadcast over and filled per
            slab (each slab uses its own null mask).
        location: Mesh location of ``values``: ``"face"``, ``"node"`` or ``"edge"``.
        max_distance: Maximum distance for a gap to be filled. ``None`` (default)
            is unbounded; when set, gaps with no valid element within this distance
            stay NaN.

    Returns:
        A filled copy of ``values`` (``float64``); the input is not mutated.

    Raises:
        ValueError: If ``location`` is unknown, edge coordinates are missing, or the
            element axis length does not match the mesh.
    """
    coords = _location_coords(mesh, location)
    data = np.asarray(values, dtype=np.float64)
    if data.shape[-1] != coords.shape[0]:
        raise ValueError(
            f"values last axis ({data.shape[-1]}) does not match the number of "
            f"{location} elements ({coords.shape[0]})."
        )
    out = data.copy()
    flat = out.reshape(-1, out.shape[-1])
    for slab in flat:
        _fill_slab_nearest(coords, slab, max_distance)
    return out


def _face_adjacency(mesh: Mesh2d) -> sparse.csr_matrix:
    """Build a symmetric face-to-face adjacency matrix as a CSR 0/1 graph.

    Faces are adjacent when they share an edge. Derives
    ``face_face_connectivity`` on the mesh if it is not already present.

    Args:
        mesh: Source mesh topology.

    Returns:
        ``(n_face, n_face)`` symmetric adjacency matrix with unit weights and no
        self-loops.
    """
    if mesh.face_face_connectivity is None:
        # Derive without writing to the mesh. `UgridDataset.copy()` shares its Mesh2d, so
        # building the connectivity here would mutate the caller's mesh -- and through the
        # shared reference, the dataset the copy was made from.
        return graph.sparse_adjacency(mesh, kind="face")
    ff = cast("Connectivity", mesh.face_face_connectivity)
    n = ff.n_elements
    valid = ff.data != ff.fill_value
    rows = np.broadcast_to(np.arange(n)[:, None], ff.data.shape)[valid]
    cols = ff.data[valid]
    adjacency = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n)
    ).tocsr()
    return adjacency.maximum(adjacency.T)


def _node_adjacency(mesh: Mesh2d) -> sparse.csr_matrix:
    """Build a symmetric node-to-node adjacency matrix as a CSR 0/1 graph.

    Nodes are adjacent when they are the two endpoints of a mesh edge. Derives
    ``edge_node_connectivity`` on the mesh if it is not already present.

    Args:
        mesh: Source mesh topology.

    Returns:
        ``(n_node, n_node)`` symmetric adjacency matrix with unit weights and no
        self-loops.
    """
    if mesh.edge_node_connectivity is None:
        # Same reasoning as in _face_adjacency: derive, do not store.
        return graph.sparse_adjacency(mesh, kind="node")
    enc = cast("Connectivity", mesh.edge_node_connectivity)
    pairs = enc.data
    i = pairs[:, 0]
    j = pairs[:, 1]
    keep = (i != enc.fill_value) & (j != enc.fill_value)
    i, j = i[keep], j[keep]
    n = mesh.n_node
    adjacency = sparse.coo_matrix(
        (np.ones(i.size, dtype=np.float64), (i, j)), shape=(n, n)
    ).tocsr()
    return adjacency.maximum(adjacency.T)


def _laplacian(
    adjacency: sparse.csr_matrix,
    coords: np.typing.NDArray,
    use_weights: bool,
) -> sparse.csr_matrix:
    """Build the graph Laplacian ``L = diag(degree) - W`` from an adjacency graph.

    Args:
        adjacency: Symmetric 0/1 adjacency matrix.
        coords: ``(n_elements, 2)`` element coordinates (used only for weights).
        use_weights: If ``True``, off-diagonal weights are inverse distance between
            adjacent element coordinates, normalized to a mean of ~1.0. If
            ``False``, uniform ``1.0`` weights are used.

    Returns:
        The ``(n, n)`` Laplacian as a CSR matrix.
    """
    coo = adjacency.tocoo()
    i = coo.row
    j = coo.col
    if use_weights and i.size:
        dist = np.sqrt(((coords[i] - coords[j]) ** 2).sum(axis=1))
        # Coincident element coordinates give distance 0 and an infinite weight, which makes
        # the whole solve non-finite. Floor the distance at the smallest positive one, so a
        # coincident pair gets the strongest *finite* coupling instead.
        positive = dist[dist > 0.0]
        dist = np.maximum(dist, positive.min() if positive.size else 1.0)
        weights = 1.0 / dist
        weights = weights / weights.mean()
    else:
        # An empty adjacency would otherwise take the mean of an empty slice.
        weights = np.ones(i.size, dtype=np.float64)
    n = adjacency.shape[0]
    w_matrix = sparse.csr_matrix((weights, (i, j)), shape=(n, n))
    degree = np.ravel(w_matrix.sum(axis=1))
    return (sparse.diags(degree) - w_matrix).tocsr()


def _solve_slab(
    laplacian: sparse.csr_matrix,
    labels: np.typing.NDArray,
    n_components: int,
    slab: np.typing.NDArray,
    direct_solve: bool,
    rtol: float,
    atol: float,
    maxiter: int,
) -> None:
    """Fill one 1D slab in place by a Laplace solve on its known/unknown split.

    Connected components that are entirely NaN in this slab are skipped (they stay
    NaN, avoiding a singular system). Within the remaining components the valid
    elements are Dirichlet boundary conditions and the NaN elements are solved for.

    Args:
        laplacian: Graph Laplacian for the whole mesh location.
        labels: Connected-component label per element.
        n_components: Number of connected components.
        slab: 1D data for a single slab; NaN marks a gap. Mutated in place.
        direct_solve: If ``True`` use :func:`scipy.sparse.linalg.spsolve`, else
            conjugate gradient.
        rtol: Relative tolerance for the CG solver.
        atol: Absolute tolerance for the CG solver.
        maxiter: Maximum CG iterations.
    """
    isnull = np.isnan(slab)
    if isnull.any():
        comp_size = np.bincount(labels, minlength=n_components)
        comp_null = np.bincount(
            labels, weights=isnull.astype(np.float64), minlength=n_components
        )
        fully_null = comp_null == comp_size
        all_null = fully_null[labels]
        known = (~isnull) & (~all_null)
        unknown = isnull & (~all_null)
        if unknown.any():
            a_matrix = laplacian[unknown][:, unknown].tocsr()
            rhs = -(laplacian[unknown][:, known] @ slab[known])
            # Jacobi diagonal scaling conditions the system before the solve.
            scale = 1.0 / np.sqrt(a_matrix.diagonal())
            scale_matrix = sparse.diags(scale)
            a_scaled = (scale_matrix @ a_matrix @ scale_matrix).tocsr()
            rhs_scaled = scale * rhs
            if direct_solve:
                solution = spsolve(a_scaled, rhs_scaled)
            else:
                solution, info = cg(
                    a_scaled, rhs_scaled, rtol=rtol, atol=atol, maxiter=maxiter
                )
                if info != 0:
                    warnings.warn(
                        f"Laplace CG solver did not converge (info={info}); "
                        f"result may be inaccurate.",
                        UserWarning,
                        stacklevel=2,
                    )
            slab[unknown] = scale * solution


def laplace_interpolate(
    mesh: Mesh2d,
    values: np.typing.NDArray,
    location: str = "face",
    *,
    use_weights: bool = True,
    direct_solve: bool = False,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int = 500,
) -> np.typing.NDArray:
    """Fill NaN gaps by solving Laplace's equation on the mesh adjacency graph.

    The valid elements act as Dirichlet boundary conditions and the NaN elements are
    the unknowns of a discrete Laplace problem (``∇² = 0``) on the face-adjacency
    graph (face variables) or node-adjacency graph (node variables). A sparse graph
    Laplacian ``L = diag(degree) - W`` is built with :mod:`scipy.sparse`, split by
    connected component, and the unknown block is solved with
    :func:`scipy.sparse.linalg.spsolve` (direct) or :func:`scipy.sparse.linalg.cg`.

    This fills across mesh **space**; it is distinct from filling along the time
    axis (``interpolate_na(dim="time")``).

    Args:
        mesh: Source mesh topology.
        values: Data array whose **last** axis is the element axis; NaN marks a
            gap. Any leading axes (time/layers) are broadcast over and filled per
            slab (each slab uses its own null mask).
        location: Mesh location of ``values``: ``"face"`` (face adjacency) or
            ``"node"`` (node adjacency). ``"edge"`` is not supported.
        use_weights: If ``True`` (default) use inverse-distance edge weights
            normalized to a mean of ~1.0; if ``False`` use uniform weights.
        direct_solve: If ``True`` solve with ``spsolve``; otherwise use CG.
        rtol: Relative tolerance for the CG solver.
        atol: Absolute tolerance for the CG solver.
        maxiter: Maximum CG iterations.

    Returns:
        A filled copy of ``values`` (``float64``); the input is not mutated.
        Elements in connected components that are entirely NaN in a slab stay NaN.

    Raises:
        ValueError: If ``location`` is ``"edge"`` or unknown, the element axis length
            does not match the mesh, or every value is NaN.
    """
    if location == "edge":
        raise ValueError(
            "Laplace interpolation is not supported for edge-located variables."
        )
    if location == "face":
        adjacency = _face_adjacency(mesh)
    elif location == "node":
        adjacency = _node_adjacency(mesh)
    else:
        raise ValueError(f"Unknown location: {location!r}")

    coords = _location_coords(mesh, location)
    data = np.asarray(values, dtype=np.float64)
    n_elements = adjacency.shape[0]
    if data.shape[-1] != n_elements:
        raise ValueError(
            f"values last axis ({data.shape[-1]}) does not match the number of "
            f"{location} elements ({n_elements})."
        )
    if np.isnan(data).all():
        raise ValueError("Cannot interpolate: every value is NaN.")

    laplacian = _laplacian(adjacency, coords, use_weights)
    n_components, labels = connected_components(adjacency, directed=False)

    out = data.copy()
    flat = out.reshape(-1, out.shape[-1])
    for slab in flat:
        _solve_slab(
            laplacian, labels, n_components, slab, direct_solve, rtol, atol, maxiter
        )
    return out
