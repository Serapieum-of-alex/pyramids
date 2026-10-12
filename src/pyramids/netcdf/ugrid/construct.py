"""Mesh-construction helpers that flatten structured grids and polygons into UGRID topology.

Pure, dependency-light functions that turn a pyramids :class:`~pyramids.dataset.dataset.Dataset`
(regular raster grid) or a :class:`geopandas.GeoDataFrame` of polygons into the node coordinate
and ``face_node_connectivity`` arrays a UGRID 2D mesh needs. Each function returns a plain tuple
``(node_x, node_y, face_node_connectivity, data)`` so the caller can hand it straight to
:meth:`pyramids.netcdf.ugrid.dataset.UgridDataset.from_arrays` together with the CRS it reads from
the source (``Dataset.epsg`` / ``GeoDataFrame.crs``). Keeping the result as bare arrays — rather
than a built :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d` — avoids an import cycle with
``ugrid.dataset`` and lets the facade own the single ``from_arrays`` call.

The two entry points:

- :func:`from_structured` / :func:`from_dataset` — each raster cell becomes a quad face, cell
  corners become deduplicated nodes, and every band becomes a per-face data array.
- :func:`from_geodataframe` — each polygon becomes a face, its exterior vertices become
  deduplicated nodes, and every attribute column becomes a per-face data array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np

if (
    TYPE_CHECKING
):  # pragma: no cover - typing only, avoids importing the heavy Dataset at runtime
    from pyramids.dataset.dataset import Dataset

MeshArrays = tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]


def from_structured(
    dataset: Dataset,
    band_names: list[str] | None = None,
) -> MeshArrays:
    """Flatten a regular raster grid into a quad mesh.

    Every raster cell becomes one quad face: the cell's four corners become nodes on an
    ``(ny + 1) x (nx + 1)`` lattice (shared corners are naturally deduplicated by the lattice
    indexing, so no post-hoc ``np.unique`` is needed), and each band's values become a per-face
    data array in row-major order matching the face ordering.

    Faces are emitted counter-clockwise, as UGRID expects. Each face is built in
    counter-clockwise ``(column, row)`` index order and then reversed when the geotransform
    flips orientation, which the sign of its Jacobian determinant
    (``gt[1] * gt[5] - gt[2] * gt[4]``) decides. That covers a north-up raster (where ``y``
    descends down the rows), a mirrored one, and a rotated one alike -- including an exact
    90-degree rotation, where both pixel sizes are ``0.0`` and their signs say nothing.

    No-data cells are **kept** as faces carrying their raw band value (including the raster's
    no-data value) — they are not dropped — so ``n_face == rows * columns`` and face ordering
    stays aligned with the source grid.

    Args:
        dataset: Source pyramids :class:`~pyramids.dataset.dataset.Dataset`. Read through its
            public accessors only (``geotransform``, ``rows``, ``columns``, ``band_count``,
            ``read_array``); it is never mutated.
        band_names: Optional names for the per-band face variables, one per band. Defaults to
            ``["band_1", "band_2", ...]``. A single-band raster yields ``"band_1"``.

    Returns:
        tuple: ``(node_x, node_y, face_node_connectivity, data)`` where ``node_x`` / ``node_y`` are
        ``float64`` arrays of length ``(ny + 1) * (nx + 1)``, ``face_node_connectivity`` is an
        ``(rows * columns, 4)`` ``intp`` array of 0-indexed node indices, and ``data`` maps each
        band name to a ``(rows * columns,)`` array. Hand these to
        :meth:`UgridDataset.from_arrays` with ``epsg=dataset.epsg``.

    Raises:
        ValueError: If the dataset has no rows or columns.
    """
    gt = tuple(float(v) for v in dataset.geotransform)
    ny = int(dataset.rows)
    nx = int(dataset.columns)
    if ny < 1 or nx < 1:
        raise ValueError(
            f"Cannot build a mesh from a {ny}x{nx} raster: need at least one row and column."
        )

    i_idx = np.arange(nx + 1)
    j_idx = np.arange(ny + 1)
    ii, jj = np.meshgrid(i_idx, j_idx)
    node_xx = gt[0] + ii * gt[1] + jj * gt[2]
    node_yy = gt[3] + ii * gt[4] + jj * gt[5]
    node_x = node_xx.ravel().astype(np.float64)
    node_y = node_yy.ravel().astype(np.float64)

    linear_index = np.arange((ny + 1) * (nx + 1)).reshape(ny + 1, nx + 1)

    first = slice(None, -1)
    second = slice(1, None)

    # Emit each face counter-clockwise in (column, row) index space...
    face_node_connectivity = np.empty((ny * nx, 4), dtype=np.intp)
    face_node_connectivity[:, 0] = linear_index[first, first].ravel()
    face_node_connectivity[:, 1] = linear_index[first, second].ravel()
    face_node_connectivity[:, 2] = linear_index[second, second].ravel()
    face_node_connectivity[:, 3] = linear_index[second, first].ravel()

    # ...then reverse when the geotransform flips orientation. The sign of the Jacobian
    # determinant is what decides that, so it is the only thing worth testing: keying on the two
    # pixel sizes instead read an exact 90-degree rotation (where both are 0.0) as north-up and
    # emitted clockwise faces, breaking the UGRID counter-clockwise convention that every area
    # and polygon consumer relies on.
    determinant = gt[1] * gt[5] - gt[2] * gt[4]
    if determinant < 0.0:
        face_node_connectivity = face_node_connectivity[:, ::-1]

    data = _band_face_variables(dataset, ny, nx, band_names)

    return node_x, node_y, face_node_connectivity, data


def from_dataset(
    dataset: Dataset,
    band_names: list[str] | None = None,
) -> MeshArrays:
    """Alias for :func:`from_structured`.

    Provided so the UGRID facade can expose both ``UgridDataset.from_dataset`` and
    ``UgridDataset.from_structured`` names over the same quad-meshing logic.

    Args:
        dataset: Source pyramids :class:`~pyramids.dataset.dataset.Dataset`.
        band_names: Optional per-band face-variable names (see :func:`from_structured`).

    Returns:
        tuple: ``(node_x, node_y, face_node_connectivity, data)`` — see :func:`from_structured`.
    """
    return from_structured(dataset, band_names=band_names)


def _band_face_variables(
    dataset: Dataset,
    ny: int,
    nx: int,
    band_names: list[str] | None,
) -> dict[str, np.ndarray]:
    """Read every band and ravel it, row-major, into a per-face data array.

    Args:
        dataset: Source dataset to read bands from.
        ny: Number of raster rows.
        nx: Number of raster columns.
        band_names: Optional per-band names; defaults to ``band_1 ... band_n``.

    Returns:
        dict: Mapping of band name to a ``(ny * nx,)`` array ordered to match the face ordering.

    Raises:
        ValueError: If the number of supplied ``band_names`` does not match the band count.
    """
    band_count = int(dataset.band_count)
    if band_names is None:
        names = [f"band_{i + 1}" for i in range(band_count)]
    elif len(band_names) != band_count:
        raise ValueError(
            f"band_names has {len(band_names)} entries but the raster has {band_count} band(s)."
        )
    else:
        names = list(band_names)

    arr = np.asarray(dataset.read_array())
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    data: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        data[name] = arr[index].reshape(ny * nx)
    return data


def from_geodataframe(
    gdf: gpd.GeoDataFrame,
    include_attributes: bool = True,
) -> MeshArrays:
    """Build a face mesh from polygon geometries — the inverse of ``UgridDataset.to_geodataframe``.

    Each polygon becomes one face. Exterior-ring vertices (the closing duplicate dropped) are
    stacked across all polygons and deduplicated with ``np.unique(..., axis=0,
    return_inverse=True)``, so shared vertices collapse to a single node. Deduplication is
    **exact** — two vertices are shared only if their coordinates are bit-for-bit equal; no
    snapping tolerance is applied. Ragged faces (a mix of triangles, quads, ...) are padded with
    the ``-1`` fill value to the widest ring. Each face's vertex order is reoriented
    counter-clockwise (rings whose shapely ``is_ccw`` is false are reversed, which preserves the
    polygon exactly), and degenerate polygons (fewer than three distinct vertices, empty, or null)
    are dropped. Attribute columns become per-face data arrays, aligned to the kept faces -- a
    dropped row's attribute value goes with it, so the alignment holds.

    Only each polygon's exterior ring is used; interior rings (holes) are ignored, matching the
    single-ring face model of a UGRID mesh. A ``MultiPolygon`` has no single exterior ring, so
    such a row is **dropped** along with its attributes; call ``gdf.explode()`` first to turn
    each part into its own face.

    Args:
        gdf: GeoDataFrame of :class:`shapely.Polygon` geometries. The geometry column is taken from
            ``gdf.geometry``; every other column is treated as a face attribute.
        include_attributes: When ``True`` (default), non-geometry columns become per-face data
            arrays. When ``False``, ``data`` is empty.

    Returns:
        tuple: ``(node_x, node_y, face_node_connectivity, data)`` where ``node_x`` / ``node_y`` are
        ``float64`` node coordinates, ``face_node_connectivity`` is an ``(n_face, max_nodes)``
        ``intp`` array of 0-indexed node indices padded with ``-1``, and ``data`` maps each kept
        attribute column to an ``(n_face,)`` array. Hand these to
        :meth:`UgridDataset.from_arrays` with the ``epsg`` taken from ``gdf.crs``.

    Raises:
        ValueError: If the GeoDataFrame holds no valid (non-degenerate) polygon.
    """
    geometries = gdf.geometry
    rings: list[np.ndarray] = []
    kept_positions: list[int] = []
    for position, geom in enumerate(geometries):
        ring = _polygon_ring(geom)
        if ring is None:
            continue
        rings.append(ring)
        kept_positions.append(position)

    if not rings:
        raise ValueError(
            "The GeoDataFrame contains no valid polygon to build a mesh from."
        )

    ring_lengths = np.array([len(r) for r in rings], dtype=np.intp)
    all_coords = np.vstack(rings)
    unique_coords, inverse = np.unique(all_coords, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1).astype(np.intp)

    node_x = unique_coords[:, 0].astype(np.float64)
    node_y = unique_coords[:, 1].astype(np.float64)

    max_nodes = int(ring_lengths.max())
    face_node_connectivity = np.full((len(rings), max_nodes), -1, dtype=np.intp)
    offset = 0
    for face_index, length in enumerate(ring_lengths):
        length = int(length)
        face_node_connectivity[face_index, :length] = inverse[offset : offset + length]
        offset += length

    data: dict[str, np.ndarray] = {}
    if include_attributes:
        geometry_name = gdf.geometry.name
        for column in gdf.columns:
            if column == geometry_name:
                continue
            data[column] = np.asarray(gdf[column].to_numpy())[kept_positions]

    return node_x, node_y, face_node_connectivity, data


def _polygon_ring(geom: Any) -> np.ndarray | None:
    """Return a polygon's deduplicated, counter-clockwise exterior ring, or ``None`` if degenerate.

    The closing duplicate vertex that shapely appends is dropped, repeated vertices are
    collapsed, and the ring is reversed when shapely reports it clockwise so every face is
    emitted counter-clockwise. ``None`` is answered for a ring that cannot be a face: a null or
    empty geometry, one with no single exterior ring (a ``MultiPolygon``), or one left with fewer
    than three vertices once the repeats are gone.

    A face must name each of its nodes once. Counting the distinct vertices is not enough to
    guarantee that -- a ring of three distinct points spread over four coordinates, such as
    ``[(0,0), (1,0), (1,0), (0,1)]``, passes any count test and still emits ``[0, 2, 2, 1]``.
    That face has a self-loop edge, so the derived topology reports four edges for a triangle
    and calls all four exterior. The repeats are therefore removed rather than merely counted.

    Args:
        geom: A shapely geometry; expected to be a :class:`shapely.Polygon`.

    Returns:
        np.ndarray | None: An ``(M, 2)`` array of exterior vertices, ``M >= 3`` and every one
        distinct, or ``None``.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return None
    exterior = getattr(geom, "exterior", None)
    if exterior is None:
        # A MultiPolygon (or any geometry that is not a polygon) has no exterior ring.
        return None
    coords = _without_repeats(np.asarray(exterior.coords)[:-1, :2])
    if len(coords) < 3:
        # Either the ring was sub-triangular to begin with, or collapsing the repeats took it
        # there -- `[(0,0), (0,0), (1,1), (1,1)]` is four coordinates over two points, and has
        # no area to make a face from.
        return None
    if not exterior.is_ccw:
        coords = coords[::-1]
    return np.ascontiguousarray(coords, dtype=np.float64)


def _without_repeats(coords: np.ndarray) -> np.ndarray:
    """A ring's vertices with every repeat removed, order preserved.

    Consecutive repeats are redundant: dropping one leaves the same polygon. A non-consecutive
    repeat means the ring touches itself, which no single UGRID face can describe -- the face
    would name that node twice whichever way it is traversed -- so those vertices are dropped
    too, which takes such a ring under the three-vertex threshold and out of the mesh.

    Args:
        coords: The ``(N, 2)`` exterior vertices, closing duplicate already removed.

    Returns:
        np.ndarray: The distinct vertices in first-appearance order.
    """
    if len(coords) < 2:
        return coords
    # `np.unique(..., axis=0)` sorts, which would reorder the ring and change the polygon, so
    # first appearance is recovered from the inverse mapping instead.
    _, first_index = np.unique(coords, axis=0, return_index=True)
    kept: np.ndarray = coords[np.sort(first_index)]
    return kept
