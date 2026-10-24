"""Neighbourhood (focal) statistics over a mesh's face adjacency (issue #1376).

The raster ``focal_*`` family aggregates each cell over a ``(2*radius+1)``-side **pixel
window**, which has no meaning on an unstructured mesh -- faces have neither a row/column
address nor a fixed neighbour count. The concept does carry over, though: aggregate each face
over its **k-ring neighbourhood**, the faces reachable within ``k`` hops of the face adjacency.

Hence the parameter is named ``rings``, not ``radius``: it counts adjacency hops, and calling
it a radius would imply a distance in coordinate units or a pixel window, which it is not.

Two conventions, both chosen to mirror the raster family:

- **The centre face is included** in its own neighbourhood, as the centre pixel is included in
  a raster window.
- **Boundary faces aggregate over whatever neighbours they have.** There is no padding and no
  fabricated value, so a face on the mesh edge is averaged over a smaller neighbourhood rather
  than against invented data.

``nodata`` faces are excluded from every aggregation, and so are inactive ones when an active
(wet/dry) mask is supplied -- a dry neighbour should not drag a wet face's mean down.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pyramids.base._array_ops import nodata_mask
from pyramids.netcdf.ugrid.graph import sparse_adjacency
from pyramids.netcdf.ugrid.mesh import Mesh2d

__all__ = ["focal_apply", "focal_mean", "focal_std", "neighbourhood"]


def neighbourhood(mesh: Mesh2d, rings: int = 1) -> Any:
    """The boolean k-ring neighbourhood matrix of the mesh faces, centre included.

    Row ``i`` marks every face within ``rings`` adjacency hops of face ``i``, including ``i``
    itself. Built as a boolean power of ``A + I``: each multiplication extends the reach by one
    hop, which is both the definition of a ring and cheaper than a per-face traversal.

    Args:
        mesh: The mesh to read.
        rings: How many adjacency hops the neighbourhood spans. ``1`` is a face and its
            edge-sharing neighbours.

    Returns:
        scipy.sparse.csr_matrix: An ``(n_face, n_face)`` boolean matrix.

    Raises:
        ValueError: ``rings`` is less than 1.
    """
    if rings < 1:
        raise ValueError(f"focal: rings must be at least 1, got {rings}.")
    adjacency = sparse.csr_matrix(sparse_adjacency(mesh, kind="face"), dtype=bool)
    identity = sparse.identity(mesh.n_face, dtype=bool, format="csr")
    one_hop = (adjacency + identity).astype(bool)
    reach = one_hop
    for _ in range(rings - 1):
        reach = (reach @ one_hop).astype(bool)
    return sparse.csr_matrix(reach)


def _usable(
    values: np.ndarray, nodata: Any, active: np.ndarray | None, mesh: Mesh2d
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """The float values and the boolean mask of faces that may contribute to an aggregate.

    Args:
        values: Per-face values.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask, or ``None``.
        mesh: The mesh, for the expected length.

    Returns:
        tuple: ``(values_as_float, usable_mask)``.

    Raises:
        ValueError: ``values`` is not one value per face, or ``active`` does not match it.
    """
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1 or array.shape[0] != mesh.n_face:
        raise ValueError(
            f"focal: expected one value per face -- a 1-D array of {mesh.n_face}, got shape "
            f"{array.shape}. Select a single step or layer first, e.g. with isel()."
        )
    usable = ~nodata_mask(array, nodata)
    if active is not None:
        mask = np.asarray(active, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError(
                f"focal: the active mask shape {mask.shape} does not match the values "
                f"{array.shape}; reduce it to one value per face first."
            )
        usable &= mask
    return array, usable


def _sums(
    reach: Any, values: np.typing.NDArray, usable: np.typing.NDArray
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """Per-face neighbourhood count, sum and sum of squares, excluding unusable faces."""
    weights = usable.astype("float64")
    filled = np.where(usable, values, 0.0)
    count = np.asarray(reach @ weights, dtype="float64")
    total = np.asarray(reach @ filled, dtype="float64")
    total_sq = np.asarray(reach @ (filled * filled), dtype="float64")
    return count, total, total_sq


def focal_mean(
    mesh: Mesh2d,
    values: np.ndarray,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
) -> np.typing.NDArray:
    """The mean of each face's k-ring neighbourhood, centre included.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.

    Returns:
        np.ndarray: Per-face mean of length ``n_face``. A face whose neighbourhood holds no
        usable value at all yields ``NaN`` -- there is nothing to average.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
    """
    array, usable = _usable(values, nodata, active, mesh)
    reach = neighbourhood(mesh, rings)
    count, total, _ = _sums(reach, array, usable)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)
    return np.asarray(result, dtype="float64")


def focal_std(
    mesh: Mesh2d,
    values: np.ndarray,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
) -> np.typing.NDArray:
    """The population standard deviation of each face's k-ring neighbourhood.

    Computed from the neighbourhood sums rather than by gathering each neighbourhood, so the
    whole mesh is one sparse product per moment.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.

    Returns:
        np.ndarray: Per-face standard deviation of length ``n_face``; ``NaN`` where the
        neighbourhood holds no usable value.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
    """
    array, usable = _usable(values, nodata, active, mesh)
    reach = neighbourhood(mesh, rings)
    count, total, total_sq = _sums(reach, array, usable)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(count > 0, count, 1.0)
        mean = total / safe
        variance = total_sq / safe - mean * mean
        # Catmull's identity is one subtraction away from a tiny negative on a constant
        # neighbourhood; clamp rather than hand back a NaN from the square root.
        result = np.where(count > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)
    return np.asarray(result, dtype="float64")


def focal_apply(
    mesh: Mesh2d,
    values: np.ndarray,
    func: Any,
    *,
    rings: int = 1,
    nodata: Any = None,
    active: np.ndarray | None = None,
) -> np.typing.NDArray:
    """Aggregate each face's k-ring neighbourhood with a user-supplied callable.

    ``func`` receives the 1-D array of usable values in one neighbourhood and returns a scalar,
    so anything numpy-shaped works (``np.ptp``, ``np.median``, a lambda). Unlike
    :func:`focal_mean` and :func:`focal_std`, this cannot be expressed as a sparse product, so
    it visits each face in turn -- noticeably slower on a large mesh, which is the price of an
    arbitrary aggregation.

    Args:
        mesh: The mesh to read.
        values: Per-face values.
        func: Callable taking a 1-D array and returning a scalar.
        rings: Adjacency hops the neighbourhood spans. Defaults to 1.
        nodata: Sentinel to exclude, or ``None``.
        active: Per-face activity mask whose ``False`` faces are excluded, or ``None``.

    Returns:
        np.ndarray: Per-face aggregate of length ``n_face``; ``NaN`` where the neighbourhood
        holds no usable value, so ``func`` is never handed an empty array.

    Raises:
        ValueError: ``rings`` is less than 1, or the shapes do not line up.
    """
    array, usable = _usable(values, nodata, active, mesh)
    reach = neighbourhood(mesh, rings).tolil().rows
    out = np.full(mesh.n_face, np.nan, dtype="float64")
    for face in range(mesh.n_face):
        # The neighbourhood always contains the centre face (the identity term in
        # `neighbourhood`), so a row is never empty and needs no guard for that.
        members = np.asarray(reach[face], dtype=np.intp)
        picked = members[usable[members]]
        if picked.size == 0:
            continue
        out[face] = float(func(array[picked]))
    return out
