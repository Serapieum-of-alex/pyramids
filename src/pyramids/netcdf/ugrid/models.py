"""Data models for UGRID unstructured mesh metadata and variables.

This module defines the core data structures used throughout the
UGRID subpackage: topology metadata, mesh variables (data on the
mesh), and dataset-level metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from osgeo import gdal

from pyramids.netcdf._mdim import open_mdarray

#: Default UGRID mesh-topology variable name used when a file/in-memory mesh has no
#: explicit name (the de-facto convention written by Deltares D-Flow FM and others).
DEFAULT_MESH_NAME = "mesh2d"

#: The ``Conventions`` string a pyramids-written UGRID file declares: CF for the
#: coordinate/attribute conventions, UGRID for the mesh topology on top of them. One
#: definition, because the reader, the writer and both bridges have to agree on it.
UGRID_CONVENTIONS = "CF-1.8 UGRID-1.0"

#: Attribute marking a companion variable as a variable's per-element active (wet/dry)
#: mask — the on-disk role for :attr:`MeshVariable.active` (issue #1331). The main variable
#: points at the companion through the CF ``ancillary_variables`` attribute; the companion
#: itself carries ``{ACTIVE_MASK_ATTR: ACTIVE_MASK_ROLE}`` so a reader recognises it without
#: guessing from the name. ``True``/non-zero means active/wet, so a consumer never conflates
#: a dry element with a missing (``nodata``/NaN) one.
ACTIVE_MASK_ATTR = "pyramids_role"
ACTIVE_MASK_ROLE = "active_mask"

#: Suffix of the companion variable that holds a data variable's active mask on disk
#: (``<var>_active``), and the attribute on that companion naming the variable it masks.
ACTIVE_MASK_SUFFIX = "_active"
ACTIVE_MASK_PARENT_ATTR = "ancillary_for"

#: ``pyramids_role`` values marking the companion arrays that store a ragged
#: :class:`VolumeTopology` on disk (issue #1330). The reader keys off these so a round-trip
#: rebuilds the topology without guessing from names.
VOLUME_ROLE_LAYER_COUNT = "layer_count"
VOLUME_ROLE_VOLUME_OFFSET = "volume_offset"
VOLUME_ROLE_LAYER_Z = "layer_z"
VOLUME_ROLE_INTERFACE_Z = "interface_z"

#: Sentinel distinguishing "keep the source mask" from "set the mask to ``None``" in
#: :meth:`MeshVariable.with_data`, where ``None`` is a valid caller-supplied value (clear the
#: mask) and so cannot double as "unspecified".
_KEEP_ACTIVE: Any = object()


def _consistent_active(
    active: np.ndarray | None, data: np.ndarray | None
) -> np.ndarray | None:
    """Keep ``active`` only when it still lines up with ``data``, else drop it to ``None``.

    The guard behind the *implicit* mask carry (issue #1331): a shape-preserving derivation
    keeps the source mask automatically, but a time-aggregating or reshaping one
    (``reduce``/``diff``/``squeeze``/``concat``/…) would otherwise carry a now wrong-length
    mask. The mask is element-indexed (last axis) with an optional leading time axis, so it is
    consistent with ``data`` when its last axis matches ``data``'s element axis and — for a
    per-step mask — its leading axis matches ``data``'s leading axis. When it does not, the
    operation changed the shape in a way the carried mask cannot follow, so it is dropped
    rather than left mismatched. An **explicitly** supplied mask is trusted and never passes
    through here.

    Args:
        active: The source mask being carried implicitly, or ``None``.
        data: The derivation's new data array, or ``None``.

    Returns:
        ``active`` when it still matches ``data``, otherwise ``None``.
    """
    if active is None or data is None:
        return active
    mask = np.asarray(active)
    if mask.shape[-1] != data.shape[-1]:
        return None
    if mask.ndim >= 2 and mask.shape[0] != data.shape[0]:
        return None
    return active


@dataclass(frozen=True)
class MeshTopologyInfo:
    """Parsed UGRID topology metadata from a NetCDF file.

    Represents the structure of a single mesh topology variable,
    including references to coordinate variables, connectivity
    arrays, and data variables defined on the mesh.

    Attributes:
        mesh_name: Name of the topology variable (e.g., "mesh2d").
        topology_dimension: Mesh dimensionality (1=network, 2=surface, 3=volume).
        node_x_var: Name of the node x-coordinate variable.
        node_y_var: Name of the node y-coordinate variable.
        face_node_var: Name of the face-node connectivity variable.
        edge_node_var: Name of the edge-node connectivity variable.
        face_edge_var: Name of the face-edge connectivity variable.
        face_face_var: Name of the face-face connectivity variable.
        edge_face_var: Name of the edge-face connectivity variable.
        boundary_node_var: Name of the boundary-node connectivity variable.
        face_x_var: Name of the face center x-coordinate variable.
        face_y_var: Name of the face center y-coordinate variable.
        edge_x_var: Name of the edge center x-coordinate variable.
        edge_y_var: Name of the edge center y-coordinate variable.
        data_variables: Mapping of variable name to mesh location
            (e.g., {"water_level": "face"}).
        crs_wkt: Well-Known Text representation of the CRS, if available.
    """

    mesh_name: str
    topology_dimension: int
    node_x_var: str
    node_y_var: str
    face_node_var: str | None = None
    edge_node_var: str | None = None
    face_edge_var: str | None = None
    face_face_var: str | None = None
    edge_face_var: str | None = None
    boundary_node_var: str | None = None
    face_x_var: str | None = None
    face_y_var: str | None = None
    edge_x_var: str | None = None
    edge_y_var: str | None = None
    data_variables: dict[str, str] = field(default_factory=dict)
    crs_wkt: str | None = None


@dataclass(frozen=True)
class VolumeTopology:
    """Ragged (per-column) vertical layering for a genuinely 3-D unstructured mesh.

    A uniform 3-D output forces every column to the same layer count, so its data is a dense
    ``(time?, n_layers, n_face)`` block. A real hydrodynamic mesh does not: a deep channel
    column may carry ten layers and a shallow one two. That data is *ragged* — a flat
    ``(time?, n_volume)`` array indexed through CSR-style offsets — and this value object
    carries the map from faces to volumes so a ``location="volume"`` variable can be sliced
    per column without padding absent layers (issue #1330).

    The layout is the compressed-sparse-row one the finite-volume codes use: face ``f`` owns
    the volumes ``volume_offset[f] : volume_offset[f + 1]``, bottom layer first, so
    ``layer_count[f] == volume_offset[f + 1] - volume_offset[f]`` and ``n_volume`` is the last
    offset. When present, ``layer_z`` / ``interface_z`` give the vertical coordinate per volume
    (``n_volume``) and per interface (``n_volume + n_face``) respectively, since those vary by
    column too.

    Attributes:
        layer_count: ``(n_face,)`` int — active layers per column. Non-negative.
        volume_offset: ``(n_face + 1,)`` int — CSR offsets into the flat volume axis. Starts
            at 0, non-decreasing, and ends at ``n_volume``.
        layer_z: Optional ``(n_volume,)`` float — the vertical coordinate of each volume
            (layer centre), or ``None``.
        interface_z: Optional ``(n_volume + n_face,)`` float — the vertical coordinate of each
            layer interface (one more interface than layers per column), or ``None``.

    Examples:
        - Three columns of 3, 1 and 2 layers describe six volumes:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid import VolumeTopology
            >>> vt = VolumeTopology(
            ...     layer_count=np.array([3, 1, 2]),
            ...     volume_offset=np.array([0, 3, 4, 6]),
            ... )
            >>> vt.n_face, vt.n_volume, vt.max_layers
            (3, 6, 3)
            >>> vt.volumes_of_face(2)
            slice(4, 6, None)

            ```
        - The face/layer maps lay the flat volume axis out per column:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid import VolumeTopology
            >>> vt = VolumeTopology(
            ...     layer_count=np.array([3, 1, 2]),
            ...     volume_offset=np.array([0, 3, 4, 6]),
            ... )
            >>> vt.face_of_volume().tolist()
            [0, 0, 0, 1, 2, 2]
            >>> vt.layer_of_volume().tolist()
            [0, 1, 2, 0, 0, 1]

            ```
        - A malformed offset array is refused at construction:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.ugrid import VolumeTopology
            >>> VolumeTopology(
            ...     layer_count=np.array([3, 1, 2]),
            ...     volume_offset=np.array([0, 3, 6]),
            ... )
            Traceback (most recent call last):
                ...
            ValueError: VolumeTopology: volume_offset must have n_face + 1 entries (got 3 for 3 face(s)).

            ```
    """

    layer_count: np.ndarray
    volume_offset: np.ndarray
    layer_z: np.ndarray | None = None
    interface_z: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the CSR invariants so a malformed topology fails at build, not at use."""
        layer_count = np.asarray(self.layer_count)
        volume_offset = np.asarray(self.volume_offset)
        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "volume_offset", volume_offset)
        if layer_count.ndim != 1 or volume_offset.ndim != 1:
            raise ValueError(
                "VolumeTopology: layer_count and volume_offset must be 1-D arrays."
            )
        if volume_offset.shape[0] != layer_count.shape[0] + 1:
            raise ValueError(
                "VolumeTopology: volume_offset must have n_face + 1 entries "
                f"(got {volume_offset.shape[0]} for {layer_count.shape[0]} face(s))."
            )
        if layer_count.shape[0] and int(volume_offset[0]) != 0:
            raise ValueError("VolumeTopology: volume_offset must start at 0.")
        if np.any(np.diff(volume_offset) < 0):
            raise ValueError("VolumeTopology: volume_offset must be non-decreasing.")
        if np.any(np.diff(volume_offset) != layer_count):
            raise ValueError(
                "VolumeTopology: layer_count must equal the gaps in volume_offset "
                "(layer_count[f] == volume_offset[f + 1] - volume_offset[f])."
            )
        if self.layer_z is not None:
            object.__setattr__(self, "layer_z", np.asarray(self.layer_z))
        if self.interface_z is not None:
            object.__setattr__(self, "interface_z", np.asarray(self.interface_z))

    @property
    def n_face(self) -> int:
        """Number of columns (faces) the topology describes."""
        return int(self.layer_count.shape[0])

    @property
    def n_volume(self) -> int:
        """Total number of volumes (the flat volume-axis length)."""
        return int(self.volume_offset[-1]) if self.volume_offset.shape[0] else 0

    @property
    def max_layers(self) -> int:
        """The deepest column's layer count — the number of layer slots any column has."""
        return int(self.layer_count.max()) if self.layer_count.shape[0] else 0

    def volumes_of_face(self, face: int) -> slice:
        """The slice of the flat volume axis owned by ``face`` (bottom layer first)."""
        return slice(int(self.volume_offset[face]), int(self.volume_offset[face + 1]))

    def face_of_volume(self) -> np.ndarray:
        """``(n_volume,)`` int — the face each volume belongs to."""
        return np.repeat(np.arange(self.n_face), self.layer_count)

    def layer_of_volume(self) -> np.ndarray:
        """``(n_volume,)`` int — the 0-based layer position of each volume within its column."""
        return np.concatenate(
            [np.arange(int(count)) for count in self.layer_count]
            if self.n_face
            else [np.empty(0, dtype=np.intp)]
        ).astype(np.intp)

    def volume_index(self, face: int, layer: int) -> int | None:
        """The flat volume position of ``(face, layer)``, or ``None`` when that layer is absent.

        Args:
            face: Column index.
            layer: 0-based layer position within the column, bottom first.

        Returns:
            int | None: The flat volume index, or ``None`` when ``layer`` is past the column's
            ``layer_count`` (that column is shallower than ``layer + 1`` layers).

        Examples:
            - A layer that exists maps to its flat volume position; one that does not is ``None``:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import VolumeTopology
                >>> vt = VolumeTopology(
                ...     layer_count=np.array([3, 1, 2]),
                ...     volume_offset=np.array([0, 3, 4, 6]),
                ... )
                >>> vt.volume_index(0, 2)
                2
                >>> vt.volume_index(1, 1) is None
                True

                ```
        """
        if layer < 0 or layer >= int(self.layer_count[face]):
            return None
        return int(self.volume_offset[face]) + layer


def read_mesh_variable(
    path: str,
    var_name: str,
    *,
    window: tuple[int, int | None] | None = None,
    context: str = "variable read",
) -> np.typing.NDArray | None:
    """Re-open ``path`` and read one mesh MDArray, optionally a leading-axis window.

    The shared body of the two UGRID lazy readers — the whole-variable loader built by
    :func:`pyramids.netcdf.ugrid.dataset._make_variable_loader` and the windowed time-slab
    reader :func:`_read_time_slab`. Both re-open the (already-closed) store on demand and
    read a single MDArray; this is the common open → root group → resolve → read → close.

    ``ReadAsArray`` returns a fresh, numpy-owned array, so the result stays valid after the
    dataset is closed in ``finally``. The deterministic close keeps Windows from holding a
    read handle on the file (review N3).

    Args:
        path: File to re-open (MDIM mode).
        var_name: Name of the MDArray to read.
        window: ``(start, count)`` along the **leading** (time) axis, or ``None`` to read
            the whole array. ``count`` is ``None`` for a single leading step; the caller
            drops the leading axis itself when it wants a single step squeezed out.
        context: Short phrase naming the caller, used in the re-open error message.

    Returns:
        The array as read (leading axis **not** squeezed), or ``None`` when the MDArray has
        no readable values.

    Raises:
        ValueError: If the file cannot be re-opened, or the variable is no longer present.
    """
    ds = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR)
    if ds is None:
        raise ValueError(f"GDAL cannot re-open {path!r} for a {context}.")
    try:
        rg = ds.GetRootGroup()
        md = open_mdarray(rg, var_name) if rg is not None else None
        if md is None:
            raise ValueError(f"Variable {var_name!r} is no longer present in {path!r}.")
        if window is None:
            data: np.typing.NDArray | None = md.ReadAsArray()
        else:
            start, count = window
            sizes = [d.GetSize() for d in md.GetDimensions()]
            start_idx = [0] * len(sizes)
            start_idx[0] = start
            counts = list(sizes)
            counts[0] = 1 if count is None else count
            data = md.ReadAsArray(array_start_idx=start_idx, count=counts)
    finally:
        ds = None
    return data


def _read_time_slab(
    path: str, var_name: str, start: int, stop: int | None
) -> np.typing.NDArray | None:
    """Read only a time slab of ``var_name`` straight from ``path`` (axis-0 time).

    Reads ``[start]`` (when ``stop`` is ``None``) or ``[start:stop]`` along the leading axis via a
    windowed MDArray read, so a single-step / short-range selection never materialises the whole
    ``(n_time, n_elements)`` array (issue #982). The leading axis is dropped for a single step, so the
    result matches ``data[index]``.
    """
    count = None if stop is None else stop - start
    slab = read_mesh_variable(
        path, var_name, window=(start, count), context="windowed variable read"
    )
    if slab is None:
        result = None
    else:
        result = slab[0] if stop is None else slab
    return result


@dataclass
class MeshVariable:
    """Data variable defined on a mesh location.

    Wraps a numpy array of values associated with mesh elements
    (nodes, faces, or edges). Supports lazy loading via a loader
    callable that defers reading until data is first accessed.

    Attributes:
        name: Variable name in the NetCDF file.
        location: Mesh location ("node", "face", or "edge").
        mesh_name: Name of the associated mesh topology variable.
        shape: Shape of the data array.
        attributes: Dictionary of NetCDF variable attributes.
        nodata: No-data / fill value for masked elements.
        units: Physical units string (e.g., "m", "m/s").
        standard_name: CF standard name (e.g., "sea_surface_height").
        active: Optional boolean active (wet/dry) mask, broadcastable to the variable's
            element axis — either ``(n_element,)`` (static) or ``(n_time, n_element)``
            (per step). ``True`` means active/wet. Kept **separate** from ``nodata``/NaN so a
            dry element is never conflated with a missing one; values are blanked by the mask
            only when a caller opts in (issue #1331). ``None`` when the variable carries no
            such mask.
        _data: Eagerly loaded data array, or None if using lazy loading.
        _loader: Callable that returns the data array on first access.
    """

    name: str
    location: str
    mesh_name: str
    shape: tuple[int, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    nodata: float | None = None
    units: str | None = None
    standard_name: str | None = None
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    active: np.ndarray | None = None
    _data: np.ndarray | None = field(default=None, repr=False)
    _loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _dtype: np.dtype | None = field(default=None, repr=False)
    # File this variable was read from, threaded so a single-step / short-range time selection can
    # read only that slab from disk instead of the whole `(n_time, n_elements)` array (issue #982).
    _source_path: str | None = field(default=None, repr=False)

    @property
    def data(self) -> np.typing.NDArray | None:
        """Return the data array, triggering lazy load if needed."""
        if self._data is None and self._loader is not None:
            self._data = self._loader()
        return self._data

    @property
    def has_data_source(self) -> bool:
        """True when the variable can produce an array — eager data or a lazy loader.

        Metadata-only, so callers can decide whether to read a variable's values without forcing
        the load a bare ``.data`` access would trigger.
        """
        return self._data is not None or self._loader is not None

    @property
    def n_elements(self) -> int:
        """Number of mesh elements (last dimension of shape)."""
        result = self.shape[-1] if self.shape else 0
        return result

    @property
    def time_index(self) -> int | None:
        """Axis of the real time dimension, or `None` when the variable is not temporal.

        Identifies the time axis by name from `dimensions` (a dimension named `time`/`t`, or
        `time…`/`…_time`/`…_time_…` per the CF/UGRID convention) rather than assuming any extra axis
        is time. The word-boundary match avoids substring false-positives like `runtime` / `lifetime`
        / `daytime`, and a non-temporal multi-dimensional variable such as `(n_layers, n_face)`
        reports `None` (ARC-19). When `dimensions` is unknown — e.g. an array-built variable that
        carries no dimension names — it falls back to the historical heuristic (leading axis of a
        >1-D array).

        Returns:
            The 0-based index of the time axis, or `None` if the variable has no time dimension.
        """
        if self.dimensions:
            for axis, dim in enumerate(self.dimensions):
                lowered = dim.lower()
                if (
                    lowered in ("t", "time")
                    or lowered.startswith("time")
                    or lowered.endswith("_time")
                    or "_time_" in lowered
                ):
                    return axis
            return None
        return 0 if len(self.shape) > 1 else None

    @property
    def has_time(self) -> bool:
        """True when the variable has a real time dimension.

        Unlike a bare `ndim > 1` test, this is driven by `time_index`, so a non-temporal
        `(n_layers, n_face)` variable is not mis-flagged as temporal (ARC-19).
        """
        return self.time_index is not None

    @property
    def n_time_steps(self) -> int:
        """Number of time steps. Returns 0 if the variable has no time dimension."""
        idx = self.time_index
        result = self.shape[idx] if idx is not None and self.shape else 0
        return result

    @property
    def dtype(self) -> np.dtype:
        """Data type of the variable.

        Returns the explicitly set dtype if available, falls back to
        the loaded data's dtype, and defaults to float64.
        """
        if self._dtype is not None:
            result = self._dtype
        elif self._data is not None:
            result = self._data.dtype
        else:
            result = np.dtype("float64")
        return result

    def load_array(self) -> np.typing.NDArray | None:
        """Return the full array **without** memoising it on this variable.

        ``.data`` caches the loaded array on the instance; ``load_array`` reads it for a one-shot
        consumer (e.g. writing every variable to a file) without retaining it, so a bulk operation
        over many variables need not hold them all resident at once (issue #982). Returns an already
        loaded array when present.

        Returns:
            numpy.ndarray | None: The array already held, or the one the loader reads -- which
            is **not** memoised, so a second call reads again. ``None`` when the variable has
            neither data nor a loader, which is a metadata-only variable rather than an error.
        """
        if self._data is not None:
            result = self._data
        elif self._loader is not None:
            result = self._loader()
        else:
            result = None
        return result

    def _can_window_time(self, start: int, stop: int | None) -> bool:
        """True when a time slab can be read straight from disk instead of loading everything.

        Requires an on-disk source, an axis-0 time dimension (what the slab reader indexes), and
        non-negative in-range indices; otherwise the caller falls back to a full load then slice.
        A reversed **or empty** range (``stop <= start``) is excluded so the in-memory slice — which
        yields a consistent empty array — handles it, rather than passing a zero/negative ``count``
        to GDAL, which raises under ``gdal.UseExceptions()`` (reviews L2, R2-M1).
        """
        n_steps = self.n_time_steps
        return (
            self._source_path is not None
            and self.time_index == 0
            and start >= 0
            and start < n_steps
            and (stop is None or (start < stop <= n_steps))
        )

    def _read_time_window(
        self, start: int, stop: int | None
    ) -> np.typing.NDArray | None:
        """Return the ``[start]`` step (``stop is None``) or ``[start:stop]`` range of the time axis.

        Reads only the requested slab from disk when possible (:meth:`_can_window_time`); otherwise
        slices an already loaded array, or falls back to a full load then slice.
        """
        if self._data is not None:
            result = self._data[start] if stop is None else self._data[start:stop]
        elif self._can_window_time(start, stop):
            result = _read_time_slab(
                cast("str", self._source_path), self.name, start, stop
            )
        else:
            data = self.data
            if data is None:
                result = None
            else:
                result = data[start] if stop is None else data[start:stop]
        return result

    def sel_time(self, index: int) -> np.typing.NDArray:
        """Select a single time step, reading only that slab from disk when possible.

        Args:
            index: Time step index.

        Returns:
            1D array of values at the given time step.

        Raises:
            IndexError: If index is out of range.
            ValueError: If the variable has no time dimension, or has no
                loaded data array.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        slab = self._read_time_window(index, None)
        if slab is None:
            raise ValueError(f"Variable '{self.name}' has no loaded data.")
        return cast("np.typing.NDArray", slab)

    def sel_time_range(self, start: int, stop: int) -> MeshVariable:
        """Select a time range, reading only that slab from disk when possible.

        Args:
            start: Start time index (inclusive).
            stop: Stop time index (exclusive).

        Returns:
            New MeshVariable with the selected time range.

        Raises:
            ValueError: If the variable has no time dimension, or has no
                loaded data array.
        """
        if not self.has_time:
            raise ValueError(f"Variable '{self.name}' has no time dimension.")
        slab = self._read_time_window(start, stop)
        if slab is None:
            raise ValueError(f"Variable '{self.name}' has no loaded data.")
        return self.with_data(slab, active=self._time_sliced_active(start, stop))

    @property
    def active_has_time(self) -> bool:
        """True when the active mask carries its own leading time axis (``(n_time, n_element)``).

        A static mask is ``(n_element,)`` and applies to every step; a per-step mask is 2-D with
        time on axis 0. Returns ``False`` when there is no mask.

        Examples:
            - A per-step mask has a time axis; a static one does not:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid.models import MeshVariable
                >>> per_step = MeshVariable(
                ...     name="wl", location="face", mesh_name="mesh2d", shape=(2, 3),
                ...     active=np.array([[True, False, True], [True, True, False]]),
                ... )
                >>> per_step.active_has_time
                True
                >>> static = MeshVariable(
                ...     name="wl", location="face", mesh_name="mesh2d", shape=(3,),
                ...     active=np.array([True, False, True]),
                ... )
                >>> static.active_has_time
                False

                ```
        """
        return self.active is not None and np.asarray(self.active).ndim >= 2

    def _time_sliced_active(self, start: int, stop: int | None) -> np.ndarray | None:
        """The active mask reduced the same way :meth:`_read_time_window` reduces the data.

        A per-step mask (:attr:`active_has_time`) is indexed on its leading time axis — ``[start]``
        for a single step (``stop is None``), ``[start:stop]`` for a range — so it stays aligned
        with the sliced data. A static mask has no time axis and is returned unchanged. ``None``
        stays ``None``.
        """
        if self.active is None:
            return None
        mask = np.asarray(self.active)
        if not self.active_has_time:
            return mask
        return mask[start] if stop is None else mask[start:stop]

    def _time_indexed_active(self, indices: np.ndarray) -> np.ndarray | None:
        """The active mask taking the time steps at ``indices`` (for :meth:`np.take` on axis 0).

        The counterpart of :meth:`_time_sliced_active` for an arbitrary index sequence rather
        than a contiguous range. A static mask is returned unchanged; ``None`` stays ``None``.
        """
        if self.active is None:
            return None
        mask = np.asarray(self.active)
        if not self.active_has_time:
            return mask
        return np.take(mask, np.asarray(indices, dtype=np.intp), axis=0)

    def _element_sliced_active(self, elements: np.ndarray) -> np.ndarray | None:
        """The active mask restricted to ``elements`` along its (last) element axis.

        Used when a derivation keeps a subset of mesh elements (e.g. a face selection): the mask
        is element-indexed, so it is taken on its last axis in the given order, which matches the
        renumbered element order. ``None`` stays ``None``.
        """
        if self.active is None:
            return None
        return np.asarray(self.active)[..., elements]

    def with_data(
        self, data: np.ndarray | None, *, active: np.ndarray | None = _KEEP_ACTIVE
    ) -> MeshVariable:
        """Return a copy of this variable carrying ``data``, keeping all other metadata.

        The new variable is eager (``_data`` set, no loader); its ``shape`` is taken from
        ``data`` when provided, else the original ``shape`` is retained. Used wherever a
        derived dataset (time selection, spatial clip, …) replaces a variable's array while
        preserving its name / location / mesh / attributes.

        Args:
            data: The replacement data array, or ``None`` to keep the original shape with
                no eager data.
            active: The active (wet/dry) mask the copy carries. Left unset, the source's mask
                is carried through unchanged (right for a shape-preserving derivation such as a
                layer selection); pass a sliced mask when the element or time axis changed, or
                ``None`` to drop it.

        Returns:
            MeshVariable: The copy carrying ``data``.
        """
        return MeshVariable(
            name=self.name,
            location=self.location,
            mesh_name=self.mesh_name,
            shape=data.shape if data is not None else self.shape,
            # Copy, not alias: a derived variable owns its own attributes dict, so a later
            # edit to the derivation can never leak back into the source (these derivations
            # are meant to be immutable).
            attributes=dict(self.attributes),
            nodata=self.nodata,
            units=self.units,
            standard_name=self.standard_name,
            dimensions=self.dimensions,
            # Implicit carry is validated against the new shape (dropped if it no longer
            # lines up); an explicitly supplied mask is trusted as given.
            active=(
                _consistent_active(self.active, data)
                if active is _KEEP_ACTIVE
                else active
            ),
            _data=data,
        )


@dataclass(frozen=True)
class UgridMetadata:
    """Full metadata summary for a UGRID dataset.

    Aggregates topology information, data variable inventory,
    global attributes, and mesh element counts for display
    and inspection purposes.

    Attributes:
        mesh_topologies: List of parsed mesh topologies in the file.
        data_variables: Mapping of variable name to location.
        global_attributes: File-level NetCDF attributes.
        conventions: Conventions string (e.g., "CF-1.8 UGRID-1.0").
        n_nodes: Total number of mesh nodes.
        n_faces: Total number of mesh faces.
        n_edges: Total number of mesh edges.
    """

    mesh_topologies: tuple[MeshTopologyInfo, ...] = ()
    data_variables: dict[str, str] = field(default_factory=dict)
    global_attributes: dict[str, Any] = field(default_factory=dict)
    conventions: str | None = None
    n_nodes: int = 0
    n_faces: int = 0
    n_edges: int = 0
