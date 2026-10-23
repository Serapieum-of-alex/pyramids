"""Mesh-native analysis: point extraction, random sampling, contouring.

Module-level functions taking :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`
plus plain arrays — no dataset import, keeping the import graph acyclic.
Same contract as ``fill.py``, ``graph.py``, ``vector_ops.py``.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex


def extract_points(
    mesh: Mesh2d,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    location: str = "face",
) -> np.ndarray:
    """Sample variable values at arbitrary coordinates.

    Args:
        mesh: The mesh topology.
        values: Variable data (``(n_elements,)`` or ``(n_time, n_elements)``).
        x: Query x-coordinates.
        y: Query y-coordinates.
        location: ``"face"`` (point-in-face) or ``"node"`` (nearest node).

    Returns:
        Extracted values — shape ``(n_points,)`` or ``(n_time, n_points)``.
        Points outside the mesh get ``NaN``.

    Raises:
        ValueError: If *location* is not ``"face"`` or ``"node"``.

    See Also:
        sample_random: Draw spatially random samples instead of
            querying specific coordinates.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    idx = MeshSpatialIndex(mesh)

    if location == "face":
        element_ids = idx.locate_faces(x, y)
        outside = element_ids < 0
        safe_ids = np.where(outside, 0, element_ids)
        result = values[..., safe_ids].astype(float)
        result[..., outside] = np.nan
    elif location == "node":
        element_ids = idx.locate_nearest_node(x, y).ravel()
        result = values[..., element_ids].astype(float)
    else:
        raise ValueError(
            f"location must be 'face' or 'node', got {location!r}"
        )
    return result


def sample_random(
    mesh: Mesh2d,
    values: np.ndarray,
    n: int,
    location: str = "face",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw *n* spatially random samples from the mesh domain.

    Uses rejection sampling within the mesh bounding box, keeping only
    points that fall inside a mesh face.

    Args:
        mesh: The mesh topology.
        values: Variable data (``(n_elements,)`` or ``(n_time, n_elements)``).
        n: Number of samples to draw.
        location: ``"face"`` (default).
        seed: RNG seed for reproducibility.

    Returns:
        ``(x, y, sampled_values)`` — arrays of length *n*.

    See Also:
        extract_points: Sample at specific coordinates rather than
            random ones.
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = mesh.bounds
    idx = MeshSpatialIndex(mesh)

    collected_x: list[float] = []
    collected_y: list[float] = []
    collected_ids: list[int] = []
    max_iters = 100

    iterations = 0
    while len(collected_x) < n:
        iterations += 1
        if iterations > max_iters:
            raise RuntimeError(
                f"sample_random failed to collect {n} samples after "
                f"{max_iters} batches — the mesh domain may be "
                f"degenerate or extremely sparse relative to its "
                f"bounding box."
            )
        batch = max(n * 3, 256)
        cx = rng.uniform(xmin, xmax, batch)
        cy = rng.uniform(ymin, ymax, batch)
        face_ids = idx.locate_faces(cx, cy)
        inside = face_ids >= 0
        cx, cy = cx[inside], cy[inside]
        face_ids = face_ids[inside]
        needed = n - len(collected_x)
        collected_x.extend(cx[:needed].tolist())
        collected_y.extend(cy[:needed].tolist())
        collected_ids.extend(face_ids[:needed].tolist())

    sx = np.array(collected_x[:n])
    sy = np.array(collected_y[:n])
    ids = np.array(collected_ids[:n], dtype=np.intp)
    sampled = values[..., ids].astype(float)
    return sx, sy, sampled


def contour_lines(
    mesh: Mesh2d,
    values: np.ndarray,
    levels: Any = None,
    location: str = "face",
) -> Any:
    """Generate contour lines from mesh data via tricontour.

    Args:
        mesh: The mesh topology.
        values: 1-D variable data (one value per element).
        levels: Contour levels (int for count, array for explicit).
        location: ``"face"`` or ``"node"``.

    Returns:
        GeoDataFrame: Contour lines with a ``level`` column.
    """
    from pyramids.base._utils import require_cleopatra

    require_cleopatra()
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri_arr = mesh.fan_triangles
    triangulation = mtri.Triangulation(
        mesh.node_x, mesh.node_y, tri_arr
    )

    if location == "face":
        node_values = _face_values_to_nodes(mesh, values)
    else:
        node_values = np.asarray(values)

    if levels is None:
        levels = 10
    fig, ax = plt.subplots()
    cs = ax.tricontour(triangulation, node_values, levels)

    geometries = []
    level_values = []
    for level_val, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 2:
                geometries.append(LineString(seg))
                level_values.append(float(level_val))
    plt.close(fig)

    result = gpd.GeoDataFrame(
        {"level": level_values}, geometry=geometries
    )
    return result


def _face_values_to_nodes(
    mesh: Mesh2d, face_values: np.ndarray
) -> np.ndarray:
    """Average face values to nodes for tricontour input.

    Each node gets the mean of its surrounding face values.
    """
    node_sums = np.zeros(mesh.n_node, dtype=float)
    node_counts = np.zeros(mesh.n_node, dtype=float)
    fnc = mesh.face_node_connectivity
    masked = fnc.as_masked()
    valid = ~np.ma.getmaskarray(masked)

    for fi in range(fnc.n_elements):
        row_valid = valid[fi]
        nodes = masked.data[fi, row_valid]
        node_sums[nodes] += face_values[fi]
        node_counts[nodes] += 1.0

    node_counts[node_counts == 0] = 1.0
    return node_sums / node_counts


def plot_histogram(
    values: np.ndarray,
    nodata: Any = None,
    bins: int = 15,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a variable's value distribution as a histogram.

    Delegates to cleopatra's ``HistogramGlyph``.

    Args:
        values: The flat data array.
        nodata: No-data sentinel to exclude.
        bins: Number of histogram bins.
        ax: Existing matplotlib axes to draw into.
        **kwargs: Forwarded to ``HistogramGlyph``.

    Returns:
        ``(fig, ax, hist)`` from ``HistogramGlyph.histogram``.

    Raises:
        ValueError: If all values are NaN / nodata and nothing
            remains to histogram.

    See Also:
        contour_lines: Spatial contour extraction from the same
            mesh variable.
    """
    from pyramids.base._utils import require_cleopatra

    require_cleopatra()
    from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph

    flat = np.asarray(values).ravel()
    mask = np.ones(flat.shape, dtype=bool)
    if np.issubdtype(flat.dtype, np.floating):
        mask &= ~np.isnan(flat)
    if nodata is not None:
        if np.issubdtype(flat.dtype, np.floating) and np.isnan(nodata):
            pass
        else:
            mask &= flat != nodata
    valid = flat[mask]
    if valid.size == 0:
        raise ValueError(
            "No valid samples to histogram after masking nodata / NaN."
        )
    glyph = HistogramGlyph(
        valid, ax=ax, **HistogramGlyph.filter_kwargs(kwargs)
    )
    result = glyph.histogram(bins=bins)
    return result
