"""Spatial indexing, point-in-face queries, mesh clipping, and CRS assignment.

Provides MeshSpatialIndex for KD-tree and STRtree based spatial
queries, point-in-face location, mesh clipping by polygon,
bounding box subsetting, and CRS declaration for meshes.

Depends on:
    - mesh.py: Mesh2d
    - connectivity.py: Connectivity
    - models.py: MeshVariable
    - pyramids.base.crs: sr_from_epsg
    - pyproj: CRS
    - scipy.spatial: cKDTree (optional, imported inline)
    - shapely: STRtree, Polygon (already a pyramids dependency)
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import shapely
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.ops import unary_union

from pyramids.base.crs import sr_from_epsg
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import _KEEP_ACTIVE, MeshVariable


class MeshSpatialIndex:
    """Lazy-built spatial index for mesh elements.

    Uses scipy.spatial.cKDTree for nearest-neighbor queries and
    shapely.STRtree for point-in-polygon queries. Both indexes
    are built on demand and cached.

    Attributes:
        _mesh: Reference to the Mesh2d topology.
        _node_tree: KD-tree on node coordinates (lazy).
        _face_tree: KD-tree on face centroids (lazy).
        _face_strtree: STRtree on face polygons (lazy).
        _face_polygons: List of Shapely polygons (lazy).
    """

    def __init__(self, mesh: Mesh2d):
        self._mesh = mesh
        self._node_tree: Any = None
        self._face_tree: Any = None
        self._face_strtree: Any = None
        self._face_polygons: list[Any] | None = None

    @property
    def node_tree(self) -> Any:
        """KD-tree on node coordinates. Lazy-built on first access."""
        if self._node_tree is None:
            self._node_tree = cKDTree(
                np.column_stack([self._mesh.node_x, self._mesh.node_y])
            )
        return self._node_tree

    @property
    def face_tree(self) -> Any:
        """KD-tree on face centroids. Lazy-built on first access."""
        if self._face_tree is None:
            cx, cy = self._mesh.face_centroids
            self._face_tree = cKDTree(np.column_stack([cx, cy]))
        return self._face_tree

    @property
    def face_strtree(self) -> Any:
        """Shapely STRtree on face polygons. Lazy-built on first access."""
        if self._face_strtree is None:
            self._face_polygons = self._build_face_polygons()
            self._face_strtree = STRtree(self._face_polygons)
        return self._face_strtree

    @property
    def face_polygons(self) -> list[Any]:
        """List of Shapely Polygon objects for all faces."""
        if self._face_polygons is None:
            self._face_polygons = self._build_face_polygons()
        return self._face_polygons

    def _build_face_polygons(self) -> list[Any]:
        """Build Shapely Polygon objects for all mesh faces.

        Vectorised: every face's vertices are gathered into one flat coordinate array
        and handed to ``shapely.linearrings`` / ``shapely.polygons`` with a per-vertex
        ring index, instead of constructing one ``Polygon`` per face in a Python loop.
        ``shapely.linearrings`` closes each ring automatically, so no explicit
        first-point append is needed.

        Returns:
            List of Shapely Polygon objects, one per face (in face-index order).
        """
        fnc = self._mesh.face_node_connectivity
        masked = fnc.as_masked()
        valid = ~np.ma.getmaskarray(masked)
        flat_nodes = np.asarray(masked.data[valid], dtype=np.intp)
        coords = np.column_stack(
            [self._mesh.node_x[flat_nodes], self._mesh.node_y[flat_nodes]]
        )
        ring_index = np.repeat(np.arange(fnc.n_elements), fnc.nodes_per_element())
        rings = shapely.linearrings(coords, indices=ring_index)
        return list(shapely.polygons(rings))

    def locate_nearest_node(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        k: int = 1,
    ) -> np.typing.NDArray:
        """Find k nearest nodes to query point(s).

        Args:
            x: Query x-coordinate(s) (scalar or array).
            y: Query y-coordinate(s) (scalar or array).
            k: Number of nearest neighbors to find.

        Returns:
            Array of node indices. Shape: (k,) for scalar input,
            (n_queries, k) for array input.
        """
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        _, indices = self.node_tree.query(points, k=k)
        result = np.asarray(indices)
        return result

    def locate_nearest_face(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        k: int = 1,
    ) -> np.typing.NDArray:
        """Find k nearest face centroids to query point(s).

        Args:
            x: Query x-coordinate(s) (scalar or array).
            y: Query y-coordinate(s) (scalar or array).
            k: Number of nearest neighbors to find.

        Returns:
            Array of face indices. Shape: (k,) for scalar input,
            (n_queries, k) for array input.
        """
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        _, indices = self.face_tree.query(points, k=k)
        result = np.asarray(indices)
        return result

    def locate_nodes_in_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.typing.NDArray:
        """Find all nodes within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            Array of node indices within the bounding box.
        """
        mask = (
            (self._mesh.node_x >= xmin)
            & (self._mesh.node_x <= xmax)
            & (self._mesh.node_y >= ymin)
            & (self._mesh.node_y <= ymax)
        )
        result = np.nonzero(mask)[0]
        return result

    def locate_faces_in_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.typing.NDArray:
        """Find all faces whose centroids fall within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            Array of face indices within the bounding box.
        """
        cx, cy = self._mesh.face_centroids
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        result = np.nonzero(mask)[0]
        return result

    def locate_faces(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.typing.NDArray:
        """Find which face contains each query point.

        Uses Shapely STRtree for exact containment testing.
        Returns -1 for points outside all faces.

        Args:
            x: Query x-coordinates (array).
            y: Query y-coordinates (array).

        Returns:
            Array of face indices, -1 for points outside mesh.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        result = np.full(len(x), -1, dtype=np.intp)

        points = shapely.points(x, y)
        strtree = self.face_strtree
        # `intersects`, not `within`: `within` excludes the polygon boundary, so a point sitting
        # exactly on a face edge -- an interior edge shared by two faces, or one of the mesh's own
        # nodes -- matched no face at all and was reported as outside the mesh. A boundary point
        # matches every face that meets there; the lowest index wins so the answer is stable
        # rather than dependent on tree order.
        input_idx, tree_idx = strtree.query(points, predicate="intersects")

        for pt_i, face_i in zip(input_idx, tree_idx):
            if result[pt_i] == -1 or face_i < result[pt_i]:
                result[pt_i] = face_i

        return result


def clip_mesh(
    dataset: Any,
    mask: Any,
    touch: bool = True,
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Clip a UGRID dataset to a polygon mask.

    Selects faces that intersect (touch=True) or are fully contained
    within (touch=False) the mask polygon. Renumbers nodes and edges
    to produce a compact, self-consistent mesh.

    Args:
        dataset: Source UgridDataset.
        mask: Polygon mask (GeoDataFrame, FeatureCollection, or Shapely geometry).
        touch: If True, include faces that touch the mask boundary.
            If False, only include faces fully inside.

    Returns:
        tuple: ``(mesh, data_variables)`` for the clipped faces — wrap with
        :meth:`UgridDataset.clip`, which is the public entry point.
    """
    mesh = dataset.mesh

    if hasattr(mask, "_gdf"):
        mask_geom = unary_union(mask._gdf.geometry)
    elif hasattr(mask, "geometry"):
        mask_geom = unary_union(mask.geometry)
    else:
        mask_geom = mask

    spatial_idx = MeshSpatialIndex(mesh)
    # Reuse the index's lazily-built face STRtree instead of constructing a second one
    # over the same polygons.
    tree = spatial_idx.face_strtree

    predicate = "intersects" if touch else "contains"
    candidates = tree.query(mask_geom, predicate=predicate)

    selected_faces = sorted(int(c) for c in candidates)

    result = _subset_mesh_by_face_indices(dataset, selected_faces)
    return result


def subset_by_bounds(
    dataset: Any,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Subset mesh to faces whose centroids fall within a bounding box.

    Faster than clip_mesh because it only checks face centroids
    against the box without building Shapely polygons or doing
    intersection tests. Uses vectorized numpy comparisons.

    Args:
        dataset: Source UgridDataset.
        xmin: Minimum x-coordinate.
        ymin: Minimum y-coordinate.
        xmax: Maximum x-coordinate.
        ymax: Maximum y-coordinate.

    Returns:
        tuple: ``(mesh, data_variables)`` for the faces whose centroids fall in the box —
        wrap with :meth:`UgridDataset.subset_by_bounds`, the public entry point.
    """
    mesh = dataset.mesh
    cx, cy = mesh.face_centroids
    mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
    selected_faces = np.nonzero(mask)[0].tolist()

    result = _subset_mesh_by_face_indices(dataset, selected_faces)
    return result


def subset_mesh_by_faces(
    dataset: Any,
    selected_faces: list[int],
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Public alias of :func:`_subset_mesh_by_face_indices` for index-based face selection.

    ``clip_mesh`` / ``subset_by_bounds`` resolve a *geometry* to face indices and then call the
    private helper. Element-axis ``isel`` / ``sel`` already hold the indices, so they need the
    same renumbering machinery without a geometry step — this is that entry point.

    Args:
        dataset: Source UgridDataset (read-only).
        selected_faces: Face indices to keep, in the order they should appear.

    Returns:
        tuple: ``(mesh, data_variables)`` for the selected faces.
    """
    return _subset_mesh_by_face_indices(dataset, selected_faces)


def _subset_mesh_by_face_indices(
    dataset: Any,
    selected_faces: list[int],
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Subset a mesh to a list of face indices, returning the rebuilt parts.

    Handles node renumbering, edge filtering, coordinate subsetting, and data variable
    slicing. Shared by :func:`clip_mesh` and :func:`subset_by_bounds`. Returns the rebuilt
    ``(mesh, data_variables)`` rather than a ``UgridDataset`` so this module stays
    independent of ``ugrid.dataset`` (STR-3 — breaks the import cycle); the caller
    (``UgridDataset.clip`` / ``.subset_by_bounds``) wraps the parts back into a dataset.

    Args:
        dataset: Source UgridDataset (read-only; its mesh / data variables are sliced).
        selected_faces: List of face indices to keep.

    Returns:
        tuple: ``(mesh, data_variables)`` for the selected faces.
    """
    mesh = dataset.mesh

    # A ragged 3-D (volume) mesh cannot be face-subset here: its data lives on a flat volume
    # axis indexed through a VolumeTopology whose CSR offsets would have to be rebuilt for the
    # kept columns, which is out of scope. Refuse clearly rather than return a dataset whose
    # mesh, volume variable and (dropped) topology silently disagree (issue #1330, review M3).
    volume_vars = [
        name
        for name, var in dataset._data_variables.items()
        if var.location == "volume"
    ]
    if volume_vars:
        raise ValueError(
            "spatial subsetting (crop / clip / subset_by_bounds / isel(element=...)) is not "
            "supported for a ragged 3-D (volume) mesh; collapse it to a 2-D field first with "
            f"collapse_layer(k). Volume-located variable(s): {volume_vars}."
        )

    if len(selected_faces) == 0:
        empty_fnc = Connectivity(
            data=np.empty((0, 1), dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        empty_mesh = Mesh2d(
            node_x=np.empty(0),
            node_y=np.empty(0),
            face_node_connectivity=empty_fnc,
        )
        return empty_mesh, {}

    selected_faces_arr = np.array(selected_faces, dtype=np.intp)

    # Renumber nodes with vectorised numpy instead of a Python set + dict + per-cell
    # loop. The kept (old) node ids are the sorted unique non-fill entries of the
    # selected faces; remapping old -> compact-new is a single ``searchsorted`` against
    # that sorted array (fill cells stay -1). UGRID stores fills trailing each row, so
    # preserving column positions matches the old left-packing behaviour.
    old_fnc = mesh.face_node_connectivity
    sel_rows = old_fnc.data[selected_faces_arr]
    sel_fill = sel_rows == old_fnc.fill_value
    kept_node_indices = np.unique(sel_rows[~sel_fill]).astype(np.intp)

    new_fnc_data = np.full(sel_rows.shape, -1, dtype=np.intp)
    new_fnc_data[~sel_fill] = np.searchsorted(kept_node_indices, sel_rows[~sel_fill])

    new_fnc = Connectivity(
        data=new_fnc_data,
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=old_fnc.original_start_index,
    )

    new_enc = None
    kept_edge_indices = None
    if mesh.edge_node_connectivity is not None:
        enc = mesh.edge_node_connectivity
        enc_fill = enc.data == enc.fill_value
        # Keep an edge iff every valid (non-fill) node it references survived the clip.
        keep_edge = np.all(np.isin(enc.data, kept_node_indices) | enc_fill, axis=1)
        kept_edge_indices = np.flatnonzero(keep_edge).astype(np.intp)

        kept_rows = enc.data[kept_edge_indices]
        kept_fill = kept_rows == enc.fill_value
        new_enc_data = np.full(kept_rows.shape, -1, dtype=np.intp)
        new_enc_data[~kept_fill] = np.searchsorted(
            kept_node_indices, kept_rows[~kept_fill]
        )

        new_enc = Connectivity(
            data=new_enc_data,
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=enc.original_start_index,
        )

    new_node_x = mesh.node_x[kept_node_indices]
    new_node_y = mesh.node_y[kept_node_indices]
    new_face_x = mesh.face_x[selected_faces_arr] if mesh.face_x is not None else None
    new_face_y = mesh.face_y[selected_faces_arr] if mesh.face_y is not None else None
    new_edge_x = None
    new_edge_y = None
    if kept_edge_indices is not None:
        if mesh.edge_x is not None:
            new_edge_x = mesh.edge_x[kept_edge_indices]
        if mesh.edge_y is not None:
            new_edge_y = mesh.edge_y[kept_edge_indices]

    new_mesh = Mesh2d(
        node_x=new_node_x,
        node_y=new_node_y,
        face_node_connectivity=new_fnc,
        edge_node_connectivity=new_enc,
        face_x=new_face_x,
        face_y=new_face_y,
        edge_x=new_edge_x,
        edge_y=new_edge_y,
    )

    new_data_vars: dict[str, MeshVariable] = {}
    for name, var in dataset._data_variables.items():
        data = var.data
        # The element index array this variable's location is subset by, so its active
        # (wet/dry) mask — which is element-indexed — is sliced the same way the data is
        # (issue #1331). A non-located variable keeps its mask as-is.
        elements: np.ndarray | None = None
        if var.location == "face":
            elements = selected_faces_arr
            new_data = data[..., selected_faces_arr] if data is not None else None
        elif var.location == "node":
            elements = kept_node_indices
            new_data = data[..., kept_node_indices] if data is not None else None
        elif var.location == "edge":
            if kept_edge_indices is None:
                # An edge-located variable cannot be subset without edge topology
                # (no edge_node_connectivity to tell which edges survive). Drop it
                # rather than silently carrying a full-length edge array onto the
                # clipped mesh, which would leave the dataset internally inconsistent.
                warnings.warn(
                    f"dropping edge-located variable {name!r} from the subset: the "
                    f"mesh has no edge_node_connectivity, so its edges cannot be "
                    f"subset to match the clipped mesh.",
                    stacklevel=2,
                )
                continue
            elements = kept_edge_indices
            new_data = data[..., kept_edge_indices] if data is not None else None
        else:
            # Non-located variables (e.g. scalars or time-only) carry through whole.
            new_data = data

        active = (
            _KEEP_ACTIVE if elements is None else var._element_sliced_active(elements)
        )
        new_data_vars[name] = var.with_data(new_data, active=active)

    return new_mesh, new_data_vars


def resolve_crs(
    mesh: Mesh2d,
    crs: str | None = None,
    epsg: int | None = None,
) -> tuple[str, CRS]:
    """Resolve a CRS declaration into a WKT string and a pyproj CRS object.

    Validates the input, converts it to both representations, and assigns
    the CRS to the mesh.  The caller stores the returned ``(wkt, crs_obj)``
    on the dataset.

    Args:
        mesh: The mesh whose :attr:`Mesh2d.crs` is updated in place.
        crs: WKT string describing the CRS.
        epsg: EPSG code (e.g. ``4326``).

    Returns:
        tuple: ``(wkt, crs_obj)`` — the WKT string and the pyproj CRS.

    Raises:
        ValueError: If neither *crs* nor *epsg* is provided, or both are.

    Examples:
        - Assign a CRS by EPSG code and inspect the result:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> from pyramids.netcdf.ugrid.spatial import resolve_crs
            >>> fnc = Connectivity(
            ...     np.array([[0, 1, 2], [1, 3, 2]]),
            ...     -1, "face_node_connectivity", 0,
            ... )
            >>> mesh = Mesh2d(
            ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            ...     face_node_connectivity=fnc,
            ... )
            >>> wkt, crs_obj = resolve_crs(mesh, epsg=4326)
            >>> crs_obj.to_epsg()
            4326
            >>> mesh.crs.to_epsg()
            4326

            ```
        - Omitting both arguments raises an error:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> from pyramids.netcdf.ugrid.spatial import resolve_crs
            >>> fnc = Connectivity(
            ...     np.array([[0, 1, 2], [1, 3, 2]]),
            ...     -1, "face_node_connectivity", 0,
            ... )
            >>> mesh = Mesh2d(
            ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            ...     face_node_connectivity=fnc,
            ... )
            >>> resolve_crs(mesh)
            Traceback (most recent call last):
                ...
            ValueError: Exactly one of `crs` or `epsg` must be provided.

            ```

    See Also:
        MeshSpatialIndex: Spatial index for point-in-face and nearest
            queries on the same mesh topology.

    """
    if (crs is None) == (epsg is None):
        raise ValueError(
            "Exactly one of `crs` or `epsg` must be provided."
        )

    if crs is not None:
        crs_obj = CRS.from_wkt(crs)
        wkt = crs
    else:
        srs = sr_from_epsg(cast(int, epsg))
        wkt = srs.ExportToWkt()
        crs_obj = CRS.from_epsg(cast(int, epsg))

    mesh.crs = crs_obj
    return wkt, crs_obj
