"""Standalone mesh regridding, resampling, and point/time interpolation.

This module is the compute kernel behind the UGRID regridding surface. It holds
pure ``numpy`` / ``scipy`` / ``shapely`` implementations with **no** heavy
geometry dependency (no numba, no cell-tree), so every function here runs on a
core install. The :class:`~pyramids.netcdf.ugrid.dataset.UgridDataset` facades
(``sample`` / ``rasterize_like`` / ``regrid`` / ``interp_time``) are thin
wrappers that marshal a :class:`MeshVariable`'s array and location into these
functions and rewrap the result.

The four capabilities, matching the B-epic regridding/interpolation tasks:

- :func:`sample` (B2) — value of a variable at arbitrary points, with
  barycentric-``linear`` recovery for triangular node data and
  ``contains`` / ``nearest`` lookups reusing
  :class:`~pyramids.netcdf.ugrid.spatial.MeshSpatialIndex`.
- :func:`rasterize_like` (B1) — snap the mesh-to-raster interpolation onto an
  existing :class:`~pyramids.dataset.Dataset`'s grid (geotransform + shape),
  rather than onto a fresh ``cell_size`` + ``bounds`` grid.
- :func:`interp_time` (B4) — interpolate a temporal variable onto new time
  coordinates via :func:`scipy.interpolate.interp1d`.
- :class:`Regridder` (B3) — overlap / conservative / barycentric / centroid
  regridding between two meshes, exposing reusable ``.weights`` and a
  :meth:`Regridder.from_weights` constructor.

Depends on:
    - mesh.py: Mesh2d (geometry, centroids, areas, triangulation)
    - spatial.py: MeshSpatialIndex (STRtree containment, cKDTree nearest)
    - interpolation.py: _get_source_data / _interpolate_nearest / _interpolate_linear
    - scipy.spatial: cKDTree, Delaunay
    - scipy.interpolate: interp1d
    - scipy.sparse: csr_matrix (weight storage / reuse)
    - shapely: vectorized polygon intersection + STRtree
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import shapely
from numpy.typing import NDArray
from pyproj import Transformer
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay, cKDTree

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid.interpolation import (
    _get_source_data,
    _interpolate_linear,
    _interpolate_nearest,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

#: Aggregation / weighting methods accepted by :class:`Regridder`.
#: Distance at or below which a query point counts as coincident with a source point.
_COINCIDENT_DISTANCE = 1e-12

OVERLAP_METHODS = ("overlap", "conservative")
REGRID_METHODS = OVERLAP_METHODS + ("centroid", "centroid_locator", "barycentric")


def sample(
    mesh: Mesh2d,
    values: np.ndarray,
    x: float | np.ndarray,
    y: float | np.ndarray,
    *,
    method: str = "nearest",
    location: str = "face",
    nodata: float = np.nan,
    index: MeshSpatialIndex | None = None,
) -> NDArray:
    """Sample mesh ``values`` at arbitrary points.

    The compute kernel the ``UgridDataset.sample`` facade wires into. It resolves each
    query point to a mesh element and gathers (or interpolates) the value there. It is
    temporal-aware: when ``values`` is 2-D ``(n_time, n_elements)`` the leading time axis
    is preserved and the result is ``(n_time, n_points)``.

    Args:
        mesh: Source mesh topology.
        values: Data on the mesh, ``(n_elements,)`` static or ``(..., n_elements)`` with
            the element axis last (a leading time/layer axis is preserved).
        x: Query x-coordinate(s), in the mesh CRS.
        y: Query y-coordinate(s), in the mesh CRS.
        method: One of ``"contains"`` (point-in-face, face data only), ``"nearest"``
            (nearest face centroid / nearest node), or ``"linear"`` (barycentric on
            triangular node data, falling back to ``"nearest"`` otherwise).
        location: Mesh location the values live on — ``"face"`` or ``"node"``.
        nodata: Value written where a point falls outside the mesh. Defaults to NaN.
        index: An existing spatial index to reuse; a new one is built when ``None``.

    Returns:
        numpy.ndarray: One value per query point (float64), shaped
        ``values.shape[:-1] + (n_points,)``. Points outside the mesh get ``nodata``.

    Raises:
        ValueError: An unknown ``method`` or ``location``, or ``"contains"`` on node data.
    """
    arr = np.asarray(values, dtype=np.float64)
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if location not in ("face", "node"):
        raise ValueError(
            f"sample supports 'face' and 'node' locations, got {location!r}."
        )
    idx = index if index is not None else MeshSpatialIndex(mesh)

    out_shape = arr.shape[:-1] + (len(xs),)
    out = np.full(out_shape, nodata, dtype=np.float64)

    use_barycentric = (
        method == "linear"
        and location == "node"
        and mesh.face_node_connectivity.is_triangular()
    )
    if use_barycentric:
        result = _sample_barycentric(mesh, arr, xs, ys, idx, out)
    else:
        element_idx = _locate_elements(idx, xs, ys, method, location)
        inside = element_idx >= 0
        out[..., inside] = arr[..., element_idx[inside]]
        result = out
    return result


def _locate_elements(
    index: MeshSpatialIndex,
    xs: np.ndarray,
    ys: np.ndarray,
    method: str,
    location: str,
) -> NDArray:
    """Resolve each point to an element index (``-1`` when outside), by method and location.

    Shared by the ``nearest`` / ``contains`` paths of :func:`sample`, and by the ``linear``
    fall-back for non-triangular or face data. ``"linear"`` on face data degrades to the
    nearest face centroid.
    """
    if location == "face":
        if method == "contains":
            result = index.locate_faces(xs, ys)
        elif method in ("nearest", "linear"):
            result = (
                np.atleast_1d(index.locate_nearest_face(xs, ys)).ravel().astype(np.intp)
            )
        else:
            raise ValueError(
                f"sample method must be 'contains', 'nearest' or 'linear', got {method!r}."
            )
    else:  # node
        if method == "contains":
            raise ValueError(
                "sample method='contains' is only valid for face data; use 'nearest'."
            )
        if method not in ("nearest", "linear"):
            raise ValueError(
                f"sample method must be 'nearest' or 'linear' for node data, got {method!r}."
            )
        result = (
            np.atleast_1d(index.locate_nearest_node(xs, ys)).ravel().astype(np.intp)
        )
    return result


def _sample_barycentric(
    mesh: Mesh2d,
    arr: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    index: MeshSpatialIndex,
    out: np.ndarray,
) -> NDArray:
    """Barycentric-linear interpolation of triangular node data at points.

    Locates the containing triangle of each point, computes the three barycentric weights
    from the triangle's node coordinates, and forms ``sum(weight * node_value)``. This
    reproduces any field linear in ``x``/``y`` exactly. Points outside every face keep whatever
    ``out`` arrives pre-filled with, which is the caller's no-data value -- the reason this takes
    no separate ``nodata`` argument. Leading (time/layer) axes of ``arr`` are preserved.
    """
    face_idx = index.locate_faces(xs, ys)
    inside = face_idx >= 0
    if not np.any(inside):
        return out

    tri_nodes = mesh.face_node_connectivity.data[face_idx[inside]]
    node_x = mesh.node_x
    node_y = mesh.node_y
    px = xs[inside]
    py = ys[inside]

    x1, x2, x3 = (node_x[tri_nodes[:, k]] for k in range(3))
    y1, y2, y3 = (node_y[tri_nodes[:, k]] for k in range(3))
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    w3 = 1.0 - w1 - w2
    weights = np.stack([w1, w2, w3], axis=-1)  # (n_inside, 3)

    node_values = arr[..., tri_nodes]  # (..., n_inside, 3)
    out[..., inside] = np.sum(node_values * weights, axis=-1)
    return out


def rasterize_like(
    mesh: Mesh2d,
    values: np.ndarray,
    other_grid: Dataset,
    *,
    location: str = "face",
    method: str = "nearest",
    nodata: float = -9999.0,
    source_epsg: int | None = None,
    max_distance: float | None = None,
) -> Dataset:
    """Rasterize mesh ``values`` onto an existing grid, co-registered with ``other_grid``.

    Unlike the ``cell_size`` + ``bounds`` path of
    :func:`~pyramids.netcdf.ugrid.interpolation.mesh_to_grid`, this snaps the interpolation
    onto the geotransform / shape / CRS of an existing :class:`~pyramids.dataset.Dataset`, so
    the output is byte-for-byte aligned with it. The underlying cKDTree / ``LinearNDInterpolator``
    machinery is reused via :func:`~pyramids.netcdf.ugrid.interpolation._get_source_data`.

    Args:
        mesh: Source mesh topology.
        values: Data on the mesh. A temporal ``(n_time, n_elements)`` array uses the first
            time step, matching ``UgridDataset.to_dataset``.
        other_grid: Reference raster whose grid the output matches.
        location: Mesh location the values live on — ``"face"``, ``"node"`` or ``"edge"``.
        method: ``"nearest"``, ``"linear"`` or ``"idw"`` (inverse-distance-weighted).
        nodata: Fill value for cells outside the mesh / convex hull.
        source_epsg: EPSG of the mesh coordinates. When it differs from ``other_grid``'s EPSG
            the mesh source points are reprojected to the reference CRS before sampling.
            ``None`` assumes the two already share a CRS.
        max_distance: Nearest/IDW cutoff distance. Defaults to twice the reference cell size.

    Returns:
        Dataset: A raster co-registered with ``other_grid`` (same geotransform, shape, CRS).

    Raises:
        ValueError: An unknown ``method`` or ``location``.
    """
    geo = cast(
        "tuple[float, float, float, float, float, float]",
        tuple(other_grid.geotransform),
    )
    rows = int(other_grid.rows)
    cols = int(other_grid.columns)
    target_epsg = other_grid.epsg

    # Cell centres from the full affine, rotation terms included. `x_axis` / `y_axis`
    # deliberately ignore rotation -- a rotated grid has no single x per column -- so
    # sampling on them while stamping the rotated geotransform onto the output put the
    # values at the wrong coordinates. This reduces to the same axis when geo[2] and geo[4]
    # are zero.
    columns_at_centre = np.arange(cols, dtype="float64") + 0.5
    rows_at_centre = np.arange(rows, dtype="float64") + 0.5
    centre_cols, centre_rows = np.meshgrid(columns_at_centre, rows_at_centre)
    xx = geo[0] + centre_cols * geo[1] + centre_rows * geo[2]
    yy = geo[3] + centre_cols * geo[4] + centre_rows * geo[5]
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    data = np.asarray(values)
    if data.ndim > 1:
        data = data[0]
    source_points, source_values = _get_source_data(mesh, data, location)

    if (
        source_epsg is not None
        and target_epsg is not None
        and int(source_epsg) != int(target_epsg)
    ):
        transformer = Transformer.from_crs(
            f"EPSG:{int(source_epsg)}", f"EPSG:{int(target_epsg)}", always_xy=True
        )
        rx, ry = transformer.transform(source_points[:, 0], source_points[:, 1])
        source_points = np.column_stack([rx, ry])

    # Pixel width along the row direction, which on a rotated grid is the hypotenuse of the
    # two x terms rather than geo[1] alone (identical when there is no rotation).
    cell_size = float(np.hypot(geo[1], geo[4]))
    if max_distance is None:
        max_distance = 2.0 * cell_size

    if method == "nearest":
        grid_values = _interpolate_nearest(
            source_points, source_values, grid_points, nodata, max_distance
        )
    elif method == "linear":
        grid_values = _interpolate_linear(
            source_points, source_values, grid_points, nodata
        )
    elif method == "idw":
        grid_values = _interpolate_idw(
            source_points, source_values, grid_points, nodata, max_distance
        )
    else:
        raise ValueError(
            f"rasterize_like method must be 'nearest', 'linear' or 'idw', got {method!r}."
        )

    grid_array = grid_values.reshape(rows, cols)
    result = Dataset.from_array(
        grid_array,
        no_data_value=nodata,
        geo_ref=GeoReference(geo=geo, epsg=target_epsg),
    )
    return result


def _interpolate_idw(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    nodata: float,
    max_distance: float,
    power: float = 2.0,
    k: int = 8,
) -> NDArray:
    """Inverse-distance-weighted interpolation with a ``max_distance`` cutoff.

    Each target cell averages its ``k`` nearest source points, weighting by
    ``1 / distance**power``. Cells whose nearest source lies beyond ``max_distance`` get
    ``nodata``; a cell coinciding with a source point takes that source value exactly.

    Args:
        source_points: ``(N, 2)`` source coordinates.
        source_values: ``(N,)`` source data values.
        target_points: ``(M, 2)`` target grid coordinates.
        nodata: Fill value for out-of-range cells.
        max_distance: Maximum distance from the nearest source for a cell to receive a value.
        power: Inverse-distance exponent.
        k: Number of nearest source points blended per cell.

    Returns:
        numpy.ndarray: ``(M,)`` interpolated values.
    """
    if len(source_points) == 0:
        # Nothing to interpolate from. The contract is an (M,) array, so answer all-nodata
        # rather than letting cKDTree.query(k=0) raise a zero-size reduction error.
        return np.full(len(target_points), nodata, dtype=np.float64)
    tree = cKDTree(source_points)
    k_eff = min(k, len(source_values))
    distances, indices = tree.query(target_points, k=k_eff)
    if k_eff == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    result = np.full(len(target_points), nodata, dtype=np.float64)
    valid = distances[:, 0] <= max_distance

    dist = distances[valid]
    neigh = indices[valid]
    # A query point sitting on a source point gives a zero distance and an infinite inverse
    # weight. Compare against a small tolerance rather than exactly 0.0, so a point a rounding
    # error away counts as coincident too instead of taking a huge finite weight.
    exact = dist[:, 0] <= _COINCIDENT_DISTANCE
    # Compute the inverse distance only where it is finite. `np.where` evaluates both arms,
    # so dividing first and selecting afterwards emitted a divide-by-zero warning on an
    # exactly coincident target -- noise today, an error under `-W error`.
    coincident = dist <= _COINCIDENT_DISTANCE
    weights = np.zeros_like(dist, dtype=np.float64)
    np.divide(1.0, np.power(dist, power), out=weights, where=~coincident)
    weights[exact] = 0.0
    weights[exact, 0] = 1.0
    numer = np.sum(weights * source_values[neigh], axis=1)
    denom = np.sum(weights, axis=1)
    result[valid] = numer / denom
    return result


def interp_time(
    times: np.ndarray,
    values: np.ndarray,
    new_times: np.ndarray,
    *,
    method: str = "linear",
    axis: int = 0,
    bounds: str = "nan",
) -> NDArray:
    """Interpolate a temporal variable onto new time coordinates.

    Distinct from gap-filling: this blends between existing time steps at requested labels.
    A field that is linear in time is reproduced exactly at any interior label. This is the
    kernel behind the ``UgridDataset.interp_time`` facade, which supplies the variable's own
    time axis as ``axis``.

    Args:
        times: Source time coordinate, 1-D and with no repeated labels (any order).
        values: Data with time along ``axis``.
        new_times: Requested time labels.
        method: ``"linear"`` or ``"nearest"``.
        axis: Axis of ``values`` holding time.
        bounds: Out-of-range policy — ``"nan"`` fills NaN outside the source range,
            ``"clamp"`` holds the first/last step.

    Returns:
        numpy.ndarray: ``values`` with its time axis replaced by one of length ``len(new_times)``.

    Raises:
        ValueError: ``times`` is not 1-D, has duplicate labels, its length does not match
            ``values`` along ``axis``, or ``method`` / ``bounds`` is unknown.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nt = np.asarray(new_times, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("interp_time: `times` must be 1-D.")
    if v.shape[axis] != t.shape[0]:
        raise ValueError(
            f"interp_time: `values` has {v.shape[axis]} steps on axis {axis}, "
            f"but `times` has {t.shape[0]}."
        )
    if np.unique(t).size != t.size:
        raise ValueError("interp_time: `times` has duplicate labels.")
    if method not in ("linear", "nearest"):
        raise ValueError(
            f"interp_time method must be 'linear' or 'nearest', got {method!r}."
        )
    if bounds not in ("nan", "clamp"):
        raise ValueError(
            f"interp_time bounds must be 'nan' or 'clamp', got {bounds!r}."
        )

    order = np.argsort(t)
    t = t[order]
    v = np.take(v, order, axis=axis)

    interpolator = interp1d(
        t,
        v,
        axis=axis,
        kind=method,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    out = np.asarray(interpolator(nt), dtype=np.float64)

    if bounds == "clamp":
        out = np.moveaxis(out, axis, 0)
        below = nt < t[0]
        above = nt > t[-1]
        out[below] = np.take(v, 0, axis=axis)
        out[above] = np.take(v, -1, axis=axis)
        out = np.moveaxis(out, 0, axis)
    return out


class Regridder:
    """Weight-based regridder between two meshes (or a mesh and a raster-like target).

    Builds a sparse weight matrix ``W`` of shape ``(n_target, n_source)`` once, then applies it
    to any number of data slabs via :meth:`regrid` (``target = W @ source`` per slab, broadcast
    over leading time/layer axes). The weights are reusable — compute once with ``__init__``,
    reuse via :attr:`weights` / :meth:`from_weights`.

    Methods (``method=``):
        - ``"overlap"`` — area-weighted mean: ``W[t, s] = area(t, s) / sum_s area(t, s)``. Rows
          sum to 1, so the overlap-mean of a constant field is that constant.
        - ``"conservative"`` — first-order conservative:
          ``W[t, s] = area(t, s) / target_area[t]``. Conserves the integral
          ``sum(value * area)`` **when the target covers the whole source** -- any source
          area the target does not reach has its mass dropped (see :meth:`regrid`).
        - ``"centroid"`` (alias ``"centroid_locator"``) — nearest-source copy: each target takes
          the value of the source face whose centroid is nearest its own (weight 1.0).
        - ``"barycentric"`` — barycentric interpolation of target centroids in the Delaunay
          triangulation of the source face centroids; targets outside the hull are NaN.

    Overlap weights are true polygon-intersection areas computed with :mod:`shapely` and indexed
    with :class:`shapely.STRtree` to avoid the O(n*m) all-pairs scan. This is the pure
    numpy/scipy/shapely route (no numba cell-tree): slower but correct.

    Args:
        source: The mesh the values come from.
        target: The mesh the values are moved onto.
        method: The weighting. ``"overlap"`` and ``"conservative"`` differ only in their
            normalisation and agree whenever the target is fully covered by the source;
            ``"centroid"`` copies from the nearest source centroid and ``"barycentric"``
            interpolates in the Delaunay triangulation. Defaults to ``"overlap"``.
        source_location: The facet the source values sit on. Only ``"face"`` is supported;
            anything else is refused. Defaults to ``"face"``.

    Attributes:
        weights: The sparse ``(n_target, n_source)`` weight matrix.

    Raises:
        ValueError: An unknown ``method``, or a ``source_location`` other than ``"face"``.
    """

    def __init__(
        self,
        source: Mesh2d,
        target: Mesh2d,
        *,
        method: str = "overlap",
        source_location: str = "face",
    ):
        if method not in REGRID_METHODS:
            raise ValueError(
                f"Regridder method must be one of {REGRID_METHODS}, got {method!r}."
            )
        if source_location != "face":
            raise ValueError(
                "Regridder currently supports only face-located source data."
            )
        self._source: Mesh2d | None = source
        self._target = target
        self._method = method
        self._source_location = source_location
        self._n_source = source.n_face
        self._n_target = target.n_face
        self._weights = self._build_weights()
        self._empty_rows = np.diff(self._weights.indptr) == 0

    @property
    def weights(self) -> csr_matrix:
        """The sparse ``(n_target, n_source)`` weight matrix (reusable / serializable)."""
        return self._weights

    @property
    def method(self) -> str:
        """The regridding method this instance was built with."""
        return self._method

    @property
    def n_source(self) -> int:
        """Number of source elements the weights expect."""
        return self._n_source

    @property
    def n_target(self) -> int:
        """Number of target elements the weights produce."""
        return self._n_target

    @classmethod
    def from_weights(
        cls,
        weights: Any,
        target: Mesh2d,
        *,
        source_location: str = "face",
    ) -> Regridder:
        """Build a regridder from a precomputed weight matrix, skipping the geometry pass.

        Lets weights computed once (and optionally persisted) be reused on a fresh process or a
        different data set without rebuilding the intersection/triangulation.

        Args:
            weights: Any ``scipy.sparse`` matrix or dense array of shape ``(n_target, n_source)``.
            target: The target mesh the weights map onto.
            source_location: Mesh location the source data live on.

        Returns:
            Regridder: An instance whose :meth:`regrid` applies ``weights`` directly.
        """
        obj = cls.__new__(cls)
        obj._source = None
        obj._target = target
        obj._method = "custom"
        obj._source_location = source_location
        matrix = csr_matrix(weights)
        obj._weights = matrix
        obj._n_target, obj._n_source = matrix.shape
        obj._empty_rows = np.diff(matrix.indptr) == 0
        return obj

    def regrid(self, values: np.ndarray) -> NDArray:
        """Apply the weights to source ``values``, returning the data on the target mesh.

        Broadcasts over leading time/layer axes: the element axis must be last and of length
        ``n_source``; every leading slab is regridded by the same sparse matmul. Target elements
        with no source contribution (empty weight rows) are NaN.

        Conservation (``method="conservative"``): because ``W[t, s] = area(t, s) / target_area[t]``,
        ``sum(target * target_area) == sum(source * source_area)`` **when the target covers the
        whole source** -- the target's footprint has to contain the source's, not the other way
        round. Source area outside the target contributes to no row, so its mass is dropped: a
        3x3 unit source over ``[0, 3]^2`` regridded onto a 2x2 unit target over ``[0, 2]^2``
        loses the uncovered ring (measured integrals 45.0 vs 12.0). And because an uncovered
        target element is NaN, the target integral is ``nan`` -- hence unevaluable -- whenever
        the target reaches beyond the source.

        Args:
            values: ``(n_source,)`` or ``(..., n_source)`` source data.

        Returns:
            numpy.ndarray: ``values.shape[:-1] + (n_target,)`` data on the target mesh.

        Raises:
            ValueError: ``values``' last axis is not ``n_source``.
        """
        data = np.asarray(values, dtype=np.float64)
        if data.shape[-1] != self._n_source:
            raise ValueError(
                f"regrid: last axis is {data.shape[-1]}, expected n_source={self._n_source}."
            )
        flat = data.reshape(-1, self._n_source)
        regridded = np.asarray(self._weights.dot(flat.T)).T  # (k, n_target)
        regridded[:, self._empty_rows] = np.nan
        result = regridded.reshape(data.shape[:-1] + (self._n_target,))
        return result

    def _build_weights(self) -> csr_matrix:
        """Dispatch to the method-specific weight builder, returning a CSR matrix."""
        if self._method in OVERLAP_METHODS:
            result = self._build_overlap_weights()
        elif self._method in ("centroid", "centroid_locator"):
            result = self._build_centroid_weights()
        else:  # barycentric
            result = self._build_barycentric_weights()
        return result

    def _build_overlap_weights(self) -> csr_matrix:
        """Build overlap / conservative weights from true polygon-intersection areas.

        Target and source face polygons are intersected with a ``shapely.STRtree`` index; the
        clipped-intersection area is the raw weight. ``"overlap"`` normalises each target row by
        its total overlapped area (area-weighted mean); ``"conservative"`` divides by the target
        face area, which is what conserves ``sum(value * area)``.
        """
        rows, cols, areas = self._overlap_triplets()
        if self._method == "overlap":
            totals = np.zeros(self._n_target, dtype=np.float64)
            np.add.at(totals, rows, areas)
            scale = np.zeros_like(totals)
            np.divide(1.0, totals, out=scale, where=totals > 0.0)
            data = areas * scale[rows]
        else:  # conservative
            target_area = self._target.face_areas
            data = areas / target_area[rows]
        return csr_matrix((data, (rows, cols)), shape=(self._n_target, self._n_source))

    def _overlap_triplets(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(target_idx, source_idx, overlap_area)`` for every intersecting face pair.

        Uses the source mesh's STRtree to find candidate source faces per target face, then
        shapely's vectorized intersection + area on just those candidate pairs. Zero-area touches
        (shared edges only) are dropped.
        """
        assert self._source is not None  # nosec B101 - only reached via the geometry-building __init__
        source_index = MeshSpatialIndex(self._source)
        source_polys = np.asarray(source_index.face_polygons, dtype=object)
        target_polys = np.asarray(
            MeshSpatialIndex(self._target).face_polygons, dtype=object
        )

        target_idx, source_idx = source_index.face_strtree.query(
            target_polys, predicate="intersects"
        )
        intersections = shapely.intersection(
            target_polys[target_idx], source_polys[source_idx]
        )
        areas = shapely.area(intersections)
        keep = areas > 0.0
        return (
            target_idx[keep].astype(np.intp),
            source_idx[keep].astype(np.intp),
            areas[keep],
        )

    def _build_centroid_weights(self) -> csr_matrix:
        """Build a 1:1 copy matrix: each target takes the nearest source face by centroid."""
        assert self._source is not None  # nosec B101 - only reached via the geometry-building __init__
        tcx, tcy = self._target.face_centroids
        source_idx = (
            np.atleast_1d(MeshSpatialIndex(self._source).locate_nearest_face(tcx, tcy))
            .ravel()
            .astype(np.intp)
        )
        rows = np.arange(self._n_target, dtype=np.intp)
        data = np.ones(self._n_target, dtype=np.float64)
        return csr_matrix(
            (data, (rows, source_idx)), shape=(self._n_target, self._n_source)
        )

    def _build_barycentric_weights(self) -> csr_matrix:
        """Build barycentric weights of target centroids in the source-centroid Delaunay mesh.

        Triangulates the source face centroids, locates each target centroid's containing simplex,
        and uses the simplex's three barycentric coordinates as weights onto the three source
        faces. Targets outside the convex hull contribute no weight (NaN on apply).
        """
        assert self._source is not None  # nosec B101 - only reached via the geometry-building __init__
        scx, scy = self._source.face_centroids
        triangulation = Delaunay(np.column_stack([scx, scy]))
        tcx, tcy = self._target.face_centroids
        points = np.column_stack([tcx, tcy])

        simplices = triangulation.find_simplex(points)
        inside = simplices >= 0
        hit = simplices[inside]

        transform = triangulation.transform[hit]
        offset = points[inside] - transform[:, 2]
        bary2 = np.einsum("ijk,ik->ij", transform[:, :2], offset)
        weights = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])  # (n_inside, 3)
        vertices = triangulation.simplices[hit]  # (n_inside, 3) source face indices

        rows = np.repeat(np.nonzero(inside)[0], 3).astype(np.intp)
        cols = vertices.ravel().astype(np.intp)
        data = weights.ravel().astype(np.float64)
        return csr_matrix((data, (rows, cols)), shape=(self._n_target, self._n_source))
