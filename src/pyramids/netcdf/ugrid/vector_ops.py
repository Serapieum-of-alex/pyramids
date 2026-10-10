"""Pure mesh <-> vector operations for UGRID 2D meshes.

Standalone, side-effect-free functions that bridge a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d`
(plus a per-face value array and/or shapely / geopandas geometry) and ordinary vector geometry.
None of them mutate their inputs, and none import :mod:`pyramids.netcdf.ugrid.dataset`, so they can
be driven directly or wrapped by thin ``UgridDataset`` facades:

- :func:`polygonize` — dissolve adjacent faces that share an (approximately) equal value into merged
  polygons, one row per value group (the mesh analogue of raster ``to_feature_collection``).
- :func:`sample_line` — extract a transect: the face values crossed by a shapely ``LineString``,
  ordered along the line, with the along-line distance of each crossed face.
- :func:`burn` — rasterize vector geometry (points / lines / polygons) onto the faces, returning a
  per-face array set to a value wherever a face is touched / covered by the geometry.

All three reuse the lazily-built :class:`~pyramids.netcdf.ugrid.spatial.MeshSpatialIndex`
(``face_strtree`` / ``face_polygons``) rather than constructing a second spatial index.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

__all__ = ["polygonize", "sample_line", "burn"]


#: Area at or below which a geometry counts as having no area (a line or a point).
_ZERO_AREA = 0.0


def _as_single_geometry(geometry: Any) -> BaseGeometry:
    """Reduce any supported vector input to a single shapely geometry.

    Accepts a shapely geometry, a pyramids ``FeatureCollection`` (via its ``_gdf``), a geopandas
    ``GeoDataFrame`` / ``GeoSeries``, or any iterable of shapely geometries, and unions the parts
    into one geometry.

    Args:
        geometry: The vector input to normalise.

    Returns:
        BaseGeometry: The single (possibly multi-part) geometry.
    """
    if isinstance(geometry, BaseGeometry):
        result = geometry
    else:
        if isinstance(geometry, gpd.GeoDataFrame):
            geoms = geometry.geometry.to_numpy()
        elif isinstance(geometry, gpd.GeoSeries):
            geoms = geometry.to_numpy()
        else:
            geoms = np.asarray(list(geometry), dtype=object)
        result = shapely.unary_union(geoms)
    return result


def _as_linestring(line: Any) -> LineString:
    """Coerce a line input to a shapely ``LineString``.

    Args:
        line: A shapely ``LineString`` or a sequence of ``(x, y)`` coordinates.

    Returns:
        LineString: The line as a ``LineString``.

    Raises:
        TypeError: If ``line`` is a non-line shapely geometry.
    """
    if isinstance(line, LineString):
        result = line
    elif isinstance(line, BaseGeometry):
        raise TypeError(f"line must be a LineString, got {line.geom_type}.")
    else:
        result = LineString(line)
    return result


def polygonize(
    mesh: Mesh2d,
    values: np.typing.ArrayLike,
    *,
    decimals: int | None = None,
    crs: Any = None,
) -> gpd.GeoDataFrame:
    """Dissolve adjacent faces sharing an (approximately) equal value into merged polygons.

    Groups the mesh faces by their per-face value and unions each group's face polygons into one
    (possibly multi-part) geometry — the mesh analogue of the raster ``to_feature_collection``. The
    result has exactly one row per distinct value. Faces whose value is NaN are dropped.

    This is intended for fields with **few distinct values** (e.g. a classified / label field).
    Float values are compared by exact equality unless ``decimals`` is given, which rounds the
    values before grouping so near-equal floats dissolve together.

    Args:
        mesh: The mesh whose faces are polygonized.
        values: Per-face value array of length ``mesh.n_face`` (1-D). Pick a single time / layer
            step for a temporal or layered field before calling.
        decimals: If given, round the values to this many decimals before grouping (approximate
            equality). ``None`` groups by exact value.
        crs: CRS to attach to the output (anything geopandas accepts, e.g. an EPSG int or WKT).
            Defaults to the mesh's own CRS when it carries one.

    Returns:
        geopandas.GeoDataFrame: Columns ``value`` and ``geometry``, one row per dissolved value
        group, sorted by value.

    Raises:
        ValueError: If ``values`` is not 1-D or its length does not match ``mesh.n_face``.
    """
    values_arr = np.asarray(values)
    if values_arr.ndim != 1:
        raise ValueError(
            "polygonize expects a 1-D per-face value array; select a single time/layer step first."
        )
    if values_arr.shape[0] != mesh.n_face:
        raise ValueError(
            f"values length {values_arr.shape[0]} does not match n_face {mesh.n_face}."
        )

    polygons = MeshSpatialIndex(mesh).face_polygons
    numeric = np.issubdtype(values_arr.dtype, np.number)
    keys = (
        values_arr
        if decimals is None or not numeric
        else np.round(values_arr, decimals)
    )
    if numeric:
        finite = np.isfinite(values_arr.astype(np.float64))
    else:
        # A classified / label field -- which the docstring recommends this for -- cannot be
        # coerced to float, so "has a value" is tested directly instead: not None, and not a
        # NaN hiding in an object array (NaN is the only value unequal to itself).
        finite = np.array([not pd.isna(item) for item in values_arr], dtype=bool)

    target_crs = crs if crs is not None else getattr(mesh, "crs", None)
    gdf = gpd.GeoDataFrame({"value": keys}, geometry=list(polygons), crs=target_crs)
    gdf = gdf.loc[finite]
    result = gdf.dissolve(by="value", as_index=False)
    return result


def sample_line(
    mesh: Mesh2d,
    values: np.typing.ArrayLike,
    line: Any,
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """Sample face values along a line (a transect / cross-section).

    Finds the faces whose polygons intersect ``line`` (via the mesh STRtree), orders them along the
    line by the along-line distance of each crossing's midpoint, and returns the ordered face values
    with those distances.

    This is **sampling, not exact segment clipping**: each crossed face contributes one sample at
    the midpoint of the portion of the line that falls inside it. Parts of the line that fall off
    the mesh simply produce no samples (there is no face to sample there). The element axis of
    ``values`` is the last axis, so a leading time / layer axis is preserved in the returned values.

    Args:
        mesh: The mesh to sample.
        values: Per-face values with the face axis last (shape ``(..., n_face)``).
        line: A shapely ``LineString`` or a sequence of ``(x, y)`` coordinates.

    Returns:
        tuple: ``(distances, sampled_values, face_indices)`` where ``distances`` has shape
        ``(n_samples,)`` (along-line distance, ascending), ``sampled_values`` has shape
        ``(..., n_samples)``, and ``face_indices`` has shape ``(n_samples,)`` (the crossed face
        indices in the same order).

    Raises:
        ValueError: If the last axis of ``values`` does not match ``mesh.n_face``.
    """
    values_arr = np.asarray(values)
    if values_arr.shape[-1] != mesh.n_face:
        raise ValueError(
            f"values last axis {values_arr.shape[-1]} does not match n_face {mesh.n_face}."
        )

    line_geom = _as_linestring(line)
    index = MeshSpatialIndex(mesh)
    polygons = index.face_polygons
    candidates = np.asarray(index.face_strtree.query(line_geom, predicate="intersects"))

    ordered: list[tuple[float, int]] = []
    for face_i in candidates:
        try:
            intersection = line_geom.intersection(polygons[int(face_i)])
        except shapely.errors.GEOSException as exc:
            # A self-intersecting face is not a valid polygon, and GEOS answers with a bare
            # "side location conflict" that names no face. Say which one.
            raise ValueError(
                f"sample_line: face {int(face_i)} is not a valid polygon, so the line cannot be "
                f"intersected with it ({exc}). Check the mesh's face_node_connectivity."
            ) from exc
        if intersection.is_empty:
            continue
        distance = float(line_geom.project(intersection.centroid))
        ordered.append((distance, int(face_i)))

    ordered.sort(key=lambda pair: pair[0])
    distances = np.array([d for d, _ in ordered], dtype=np.float64)
    face_indices = np.array([f for _, f in ordered], dtype=np.intp)
    sampled_values = values_arr[..., face_indices]
    return distances, sampled_values, face_indices


def burn(
    mesh: Mesh2d,
    geometry: Any,
    value: float,
    *,
    fill: float = np.nan,
    all_touched: bool = True,
) -> np.typing.NDArray:
    """Burn vector geometry onto the mesh faces (rasterize-style).

    Returns a per-face array set to ``value`` wherever a face is touched / covered by ``geometry``
    and to ``fill`` everywhere else — the inverse of :func:`polygonize`. Faces are selected with the
    mesh STRtree and shapely predicates, so points, lines and polygons are all supported without a
    cell-tree dependency (this is boolean coverage, never area-weighted overlap).

    Selection rules:
        - Points / lines (zero-area geometry): every face the geometry intersects.
        - Polygons with ``all_touched=True``: every face the polygon intersects (any overlap or
          boundary touch).
        - Polygons with ``all_touched=False``: only faces whose centroid lies inside the polygon.

    Args:
        mesh: The mesh to burn onto.
        geometry: A shapely geometry, a pyramids ``FeatureCollection``, a geopandas
            ``GeoDataFrame`` / ``GeoSeries``, or an iterable of shapely geometries.
        value: The value assigned to burned faces.
        fill: The value for faces not touched by the geometry (default ``NaN``, giving a float
            output).
        all_touched: Polygon selection mode (see above). Ignored for zero-area geometry.

    Returns:
        numpy.ndarray: A ``(n_face,)`` array of the burned values.
    """
    # Size the output from BOTH the fill and the burned value: allocating from `fill` alone
    # made `burn(..., 1.5, fill=0)` an int64 array, which truncated the 1.5 to 1 on assignment.
    out = np.full(
        mesh.n_face, fill, dtype=np.result_type(np.asarray(fill), np.asarray(value))
    )
    geom = _as_single_geometry(geometry)
    if not geom.is_empty:
        index = MeshSpatialIndex(mesh)
        candidates = np.asarray(
            index.face_strtree.query(geom, predicate="intersects"), dtype=np.intp
        )
        if candidates.size:
            if all_touched or geom.area <= _ZERO_AREA:
                selected = candidates
            else:
                cx, cy = mesh.face_centroids
                centroid_points = shapely.points(
                    np.asarray(cx)[candidates], np.asarray(cy)[candidates]
                )
                selected = candidates[shapely.contains(geom, centroid_points)]
            out[selected] = value
    return out
