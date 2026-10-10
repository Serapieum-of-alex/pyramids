"""Standalone xarray / Zarr / dask interop for UGRID meshes.

Pure functions that translate between a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d` plus a
``{name: MeshVariable}`` mapping and the UGRID-CF encoding carried by an ``xarray.Dataset`` --
the in-memory mirror of the on-disk encoding written by
:mod:`pyramids.netcdf.ugrid.io`. Keeping these functions mesh/variable-shaped (never a
``UgridDataset``) keeps the module free of the dataset import cycle; the ``UgridDataset`` facade
wraps the results.

``xarray``, ``zarr`` and ``dask`` are **optional** dependencies. None of them is imported at module
scope: every guarded import happens inside the function that needs it, through
:func:`pyramids.base._utils.import_xarray` / :func:`~pyramids.base._utils.require_optional` /
:func:`~pyramids.base._utils.import_dask`, mirroring ``NetCDF.to_xarray``. A missing package raises
:class:`pyramids.errors.OptionalPackageDoesNotExist`.

The one encoding trap -- the connectivity ``start_index`` / ``_FillValue`` round-trip -- is handled
by :func:`_connectivity_to_disk` / :func:`_connectivity_from_disk`, which mirror the write-side
reverse mapping in :func:`pyramids.netcdf.ugrid.io._write_connectivity_array` and the read-side
normalisation in :meth:`pyramids.netcdf.ugrid.connectivity.Connectivity.from_gdal_array`. The
internal 0-indexed / ``-1``-fill arrays are never emitted to xarray.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import CRS as PyProjCRS

from pyramids.base._utils import (
    extra_hint,
    import_dask,
    import_xarray,
    require_optional,
)
from pyramids.netcdf._lazy import build_lazy_array
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import DEFAULT_MESH_NAME, MeshVariable

__all__ = [
    "to_xarray",
    "from_xarray",
    "to_zarr",
    "from_zarr",
    "chunked_variables",
]

#: File fill value stamped into connectivity arrays on disk / in the exported dataset. Matches
#: :func:`pyramids.netcdf.ugrid.io._write_connectivity_array`.
_FILE_FILL = -999

#: CF conventions string stamped on the exported dataset, matching the file writer.
_CONVENTIONS = "CF-1.8 UGRID-1.0"

_XARRAY_HINT = extra_hint("xarray is required for UGRID xarray interop.", "interop")
_ZARR_HINT = extra_hint("Zarr IO requires the optional 'zarr' dependency.", "lazy")
_DASK_HINT = extra_hint(
    "A chunked mesh read requires the optional 'dask' dependency.", "lazy"
)


def _location_dim(mesh_name: str, location: str) -> str:
    """Return the element dimension name for a mesh ``location``.

    Mirrors the ``f"{mesh_name}_n{Location}s"`` dimension names
    :func:`pyramids.netcdf.ugrid.io.write_ugrid_data_variable` writes.

    Args:
        mesh_name: Name of the mesh topology variable.
        location: Mesh location -- ``"node"``, ``"face"`` or ``"edge"``.

    Returns:
        The element dimension name, e.g. ``"mesh2d_nFaces"``.
    """
    return f"{mesh_name}_n{location.capitalize()}s"


def _connectivity_to_disk(conn: Connectivity) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode a connectivity array back to the on-disk UGRID convention.

    Reverses the internal 0-indexed / ``-1``-fill representation to the array's original
    ``start_index`` and the file ``_FillValue``, so a reader (xugrid or any UGRID consumer) sees the
    topology it expects. This is the single most error-prone step of the export; it reuses the exact
    reverse mapping of :func:`pyramids.netcdf.ugrid.io._write_connectivity_array`.

    Args:
        conn: The connectivity wrapper, 0-indexed internally with ``-1`` fill.

    Returns:
        A ``(array, attrs)`` pair: the re-indexed integer array and the UGRID attributes
        (``cf_role`` / ``start_index`` / ``_FillValue``) to stamp on the exported variable.
    """
    out_data = conn.data.astype(np.int64)
    out_data[out_data == conn.fill_value] = _FILE_FILL
    if conn.original_start_index != 0:
        valid = out_data != _FILE_FILL
        out_data[valid] += conn.original_start_index
    attrs = {
        "cf_role": conn.cf_role,
        "start_index": conn.original_start_index,
        "_FillValue": _FILE_FILL,
    }
    return out_data, attrs


def _connectivity_from_disk(
    raw_data: np.ndarray, attrs: dict[str, Any], cf_role: str
) -> Connectivity:
    """Decode an on-disk connectivity array to the internal 0-indexed / ``-1``-fill form.

    The inverse of :func:`_connectivity_to_disk`, mirroring
    :meth:`pyramids.netcdf.ugrid.connectivity.Connectivity.from_gdal_array`: subtract the
    ``start_index`` from the valid entries and remap the file ``_FillValue`` to ``-1``.

    Args:
        raw_data: The connectivity array as stored (original start index + file fill).
        attrs: The variable's attributes, read for ``start_index`` / ``_FillValue``.
        cf_role: The UGRID ``cf_role`` to tag the connectivity with.

    Returns:
        A :class:`Connectivity` with 0-indexed data and ``-1`` fill.
    """
    start_index = int(attrs.get("start_index", 0))
    raw_fill = attrs.get("_FillValue")
    raw_fill = int(raw_fill) if raw_fill is not None else _FILE_FILL

    data = np.asarray(raw_data).astype(np.intp)
    mask = data == raw_fill
    if start_index != 0:
        data[~mask] -= start_index
    data[mask] = -1
    return Connectivity(
        data=data,
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=start_index,
    )


def _variable_dims(var: MeshVariable, mesh_name: str) -> tuple[str, ...]:
    """Resolve the xarray dimension names for a mesh data variable.

    Prefers the variable's own ``dimensions`` when they match its rank (file-read variables carry
    them), so a round trip preserves the exact dimension names -- and therefore
    :attr:`MeshVariable.time_index` / :attr:`~MeshVariable.has_time`. Otherwise it synthesises names:
    the element axis last (keyed to the mesh ``location`` so same-mesh variables share it), a leading
    ``"time"`` axis for a temporal variable, and generic names for any intervening layer axes. The
    element axis is always last, honouring the mesh-variable invariant.

    Args:
        var: The mesh variable to name dimensions for.
        mesh_name: Name of the mesh topology variable.

    Returns:
        One dimension name per axis of ``var.data``.
    """
    data = var.data
    ndim = 0 if data is None else data.ndim
    if var.dimensions and len(var.dimensions) == ndim:
        return tuple(var.dimensions)

    elem_dim = _location_dim(mesh_name, var.location)
    result: tuple[str, ...]
    if ndim == 0:
        # A 0-d variable has no axes to name. Handing back the element dimension made the
        # export fail with "different number of dimensions on data and dims: 0 vs 1".
        result = ()
    elif ndim == 1:
        result = (elem_dim,)
    else:
        leading = [
            "time" if axis == var.time_index else f"{var.name}_axis{axis}"
            for axis in range(ndim - 1)
        ]
        result = (*leading, elem_dim)
    return result


def _build_data_array(xr: Any, var: MeshVariable, mesh_name: str) -> Any:
    """Build the exported ``xarray.DataArray`` for one mesh data variable.

    Args:
        xr: The imported :mod:`xarray` module.
        var: The mesh variable to export.
        mesh_name: Name of the mesh topology variable.

    Returns:
        An ``xarray.DataArray`` carrying the values, UGRID ``mesh`` / ``location`` attributes, and the
        variable's ``units`` / ``standard_name`` / ``_FillValue`` where set.
    """
    attrs: dict[str, Any] = {
        # Name the topology actually being written, not the one the variable happens to
        # remember: exporting under a different `mesh_name` otherwise left every data
        # variable pointing at a topology the file does not contain, and a reader then
        # skipped them all as belonging to another mesh.
        "mesh": mesh_name,
        "location": var.location,
    }
    if var.units:
        attrs["units"] = var.units
    if var.standard_name:
        attrs["standard_name"] = var.standard_name
    if var.nodata is not None:
        attrs["_FillValue"] = var.nodata
    return xr.DataArray(
        np.asarray(var.data),
        dims=_variable_dims(var, mesh_name),
        attrs=attrs,
    )


def _topology_attrs(mesh: Mesh2d, mesh_name: str) -> dict[str, Any]:
    """Assemble the CF UGRID attributes for the dummy ``mesh_topology`` variable.

    Mirrors the topology attribute set written by
    :func:`pyramids.netcdf.ugrid.io.write_ugrid_topology`.

    Args:
        mesh: The mesh being exported.
        mesh_name: Name of the mesh topology variable.

    Returns:
        The ``cf_role`` / ``topology_dimension`` / coordinate / connectivity attributes.
    """
    attrs: dict[str, Any] = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": f"{mesh_name}_node_x {mesh_name}_node_y",
        "face_node_connectivity": f"{mesh_name}_face_nodes",
    }
    if mesh.edge_node_connectivity is not None:
        attrs["edge_node_connectivity"] = f"{mesh_name}_edge_nodes"
    if mesh.has_face_coords:
        attrs["face_coordinates"] = f"{mesh_name}_face_x {mesh_name}_face_y"
    return attrs


def to_xarray(
    mesh: Mesh2d,
    data_variables: dict[str, MeshVariable],
    mesh_name: str = DEFAULT_MESH_NAME,
    crs_wkt: str | None = None,
) -> Any:
    """Encode a mesh and its variables as a UGRID-CF ``xarray.Dataset``.

    Produces a plain ``xarray.Dataset`` carrying the mesh topology the UGRID convention defines -- a
    dummy ``mesh_topology`` variable, the node coordinates, the ``face_node_connectivity`` (with its
    ``start_index`` and file ``_FillValue`` restored to the on-disk convention), optional
    edge connectivity and face-centre coordinates, a CRS grid-mapping variable, and each data variable
    on its ``location`` dimension with ``mesh`` / ``location`` attributes. The result is readable by
    ecosystem tools such as ``xugrid.UgridDataset.from_dataset`` and round-trips losslessly through
    :func:`from_xarray`.

    ``xarray`` is optional; it is imported through
    :func:`pyramids.base._utils.import_xarray` inside this function -- never at module scope.

    Args:
        mesh: The mesh topology to export.
        data_variables: Mapping of variable name to :class:`MeshVariable`.
        mesh_name: Name for the exported topology variable. Defaults to ``"mesh2d"``.

    Returns:
        xarray.Dataset: The UGRID-CF encoded dataset.

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``xarray`` is not installed.
    """
    xr = import_xarray(_XARRAY_HINT)

    data_vars: dict[str, Any] = {}

    # The dummy topology and grid-mapping variables carry a length-1 dimension rather than
    # being true 0-d scalars. GDAL's multidimensional reader does not expose a dimensionless
    # variable as an array, so writing this dataset to NetCDF and reading it back -- with
    # `read_file`, or with any other GDAL-based UGRID reader -- found no topology variable at
    # all and reported the file as not being UGRID. pyramids' own writer uses the same
    # size-1 encoding.
    scalar_dim = f"{mesh_name}_scalar"
    data_vars[mesh_name] = xr.DataArray(
        np.array([0], dtype=np.int32),
        dims=(scalar_dim,),
        attrs=_topology_attrs(mesh, mesh_name),
    )

    data_vars[f"{mesh_name}_node_x"] = xr.DataArray(
        np.asarray(mesh.node_x, dtype=np.float64),
        dims=(f"{mesh_name}_nNodes",),
        attrs={
            "standard_name": "projection_x_coordinate",
            "long_name": "node x-coordinate",
        },
    )
    data_vars[f"{mesh_name}_node_y"] = xr.DataArray(
        np.asarray(mesh.node_y, dtype=np.float64),
        dims=(f"{mesh_name}_nNodes",),
        attrs={
            "standard_name": "projection_y_coordinate",
            "long_name": "node y-coordinate",
        },
    )

    fnc_data, fnc_attrs = _connectivity_to_disk(mesh.face_node_connectivity)
    data_vars[f"{mesh_name}_face_nodes"] = xr.DataArray(
        fnc_data,
        dims=(f"{mesh_name}_nFaces", f"{mesh_name}_nMaxFaceNodes"),
        attrs=fnc_attrs,
    )

    if mesh.edge_node_connectivity is not None:
        enc_data, enc_attrs = _connectivity_to_disk(mesh.edge_node_connectivity)
        data_vars[f"{mesh_name}_edge_nodes"] = xr.DataArray(
            enc_data,
            dims=(f"{mesh_name}_nEdges", "Two"),
            attrs=enc_attrs,
        )

    if mesh.has_face_coords:
        data_vars[f"{mesh_name}_face_x"] = xr.DataArray(
            np.asarray(mesh.face_x, dtype=np.float64), dims=(f"{mesh_name}_nFaces",)
        )
        data_vars[f"{mesh_name}_face_y"] = xr.DataArray(
            np.asarray(mesh.face_y, dtype=np.float64), dims=(f"{mesh_name}_nFaces",)
        )

    # The caller's WKT wins: a UgridDataset keeps its CRS on the dataset (from_arrays stores an
    # epsg there), not on the Mesh2d, so relying on the mesh alone silently dropped the CRS from
    # every export. Fall back to the mesh's own CRS for a bare-Mesh2d caller.
    exported_wkt = crs_wkt if crs_wkt is not None else _mesh_crs_wkt(mesh)
    if exported_wkt is not None:
        data_vars["crs"] = xr.DataArray(
            np.array([0], dtype=np.int32),
            dims=(scalar_dim,),
            attrs={"crs_wkt": exported_wkt, "spatial_ref": exported_wkt},
        )

    for name, var in data_variables.items():
        if var.data is None:
            continue
        data_vars[name] = _build_data_array(xr, var, mesh_name)

    return xr.Dataset(data_vars=data_vars, attrs={"Conventions": _CONVENTIONS})


def _mesh_crs_wkt(mesh: Mesh2d) -> str | None:
    """Return the mesh CRS as a WKT string, or ``None`` when it has none / cannot export.

    Args:
        mesh: The mesh whose CRS to serialise.

    Returns:
        The WKT string, or ``None``.
    """
    crs = mesh.crs
    result: str | None = None
    if crs is not None:
        try:
            result = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
        except Exception:  # nosec B110 - best-effort CRS export; falls back to no CRS
            result = None
    return result


def _topology_name(ds: Any) -> str:
    """Return the name of the ``mesh_topology`` variable in an exported dataset.

    Args:
        ds: The ``xarray.Dataset`` to scan.

    Returns:
        The topology variable name, defaulting to ``"mesh2d"`` when none is tagged.
    """
    for name, var in ds.variables.items():
        if var.attrs.get("cf_role") == "mesh_topology":
            return str(name)
    return DEFAULT_MESH_NAME


def _split_pair(value: Any) -> tuple[str | None, str | None]:
    """Split a UGRID ``"<x> <y>"`` coordinate-list attribute into its two variable names.

    Args:
        value: The attribute value (a space-separated pair, or anything else).

    Returns:
        ``(x_name, y_name)``, padding with ``None`` where a name is absent.
    """
    parts = value.split() if isinstance(value, str) else []
    x = parts[0] if len(parts) > 0 else None
    y = parts[1] if len(parts) > 1 else None
    return x, y


def _reconstruct_mesh(ds: Any, mesh_name: str) -> Mesh2d:
    """Rebuild a :class:`Mesh2d` from the topology variables of an exported dataset.

    Args:
        ds: The UGRID-CF ``xarray.Dataset``.
        mesh_name: Name of the topology variable.

    Returns:
        The reconstructed mesh.

    Raises:
        ValueError: The dataset carries no ``mesh_topology`` variable, so it is not a UGRID
            dataset — raised instead of letting the lookup surface a bare ``KeyError`` naming an
            internal variable the caller never chose.
        KeyError: A referenced coordinate / connectivity variable is absent.
    """
    if mesh_name not in ds.variables:
        carriers = [
            str(name)
            for name, var in ds.variables.items()
            if var.attrs.get("cf_role") == "mesh_topology"
        ]
        raise ValueError(
            "not a UGRID dataset: no variable carries cf_role='mesh_topology' "
            f"(looked for {mesh_name!r}; topology variables found: {carriers or 'none'}). "
            "A UGRID-CF dataset declares its topology on a dummy variable."
        )
    topo_attrs = ds[mesh_name].attrs
    node_x_var, node_y_var = _split_pair(topo_attrs.get("node_coordinates"))
    node_x = np.asarray(ds[node_x_var].values, dtype=np.float64)
    node_y = np.asarray(ds[node_y_var].values, dtype=np.float64)

    fnc_var = topo_attrs.get("face_node_connectivity", f"{mesh_name}_face_nodes")
    fnc = _connectivity_from_disk(
        ds[fnc_var].values, ds[fnc_var].attrs, "face_node_connectivity"
    )

    enc = None
    enc_var = topo_attrs.get("edge_node_connectivity")
    if enc_var is not None and enc_var in ds.variables:
        enc = _connectivity_from_disk(
            ds[enc_var].values, ds[enc_var].attrs, "edge_node_connectivity"
        )

    face_x = face_y = None
    face_x_var, face_y_var = _split_pair(topo_attrs.get("face_coordinates"))
    if face_x_var in ds.variables and face_y_var in ds.variables:
        face_x = np.asarray(ds[face_x_var].values, dtype=np.float64)
        face_y = np.asarray(ds[face_y_var].values, dtype=np.float64)

    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=fnc,
        edge_node_connectivity=enc,
        face_x=face_x,
        face_y=face_y,
        crs=_crs_from_dataset(ds),
    )


def _crs_from_dataset(ds: Any) -> Any:
    """Reconstruct a pyproj CRS from an exported dataset's ``crs`` variable, or ``None``.

    Args:
        ds: The ``xarray.Dataset`` to read.

    Returns:
        A ``pyproj.CRS`` when a parseable WKT is present, else ``None``.
    """
    result = None
    if "crs" in ds.variables:
        wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
        if isinstance(wkt, str):
            try:
                result = PyProjCRS.from_wkt(wkt)
            except Exception:  # nosec B110 - best-effort CRS parse; falls back to None
                result = None
    return result


def from_xarray(ds: Any) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Reconstruct a mesh and its variables from a UGRID-CF ``xarray.Dataset``.

    The inverse of :func:`to_xarray`: reads the ``mesh_topology`` variable, the node coordinates, the
    connectivity arrays (normalising ``start_index`` / ``_FillValue`` back to the internal
    0-indexed / ``-1``-fill form), and every data variable (a variable carrying a ``location``
    attribute). Round-tripping ``to_xarray`` then ``from_xarray`` reproduces the same nodes, faces and
    values.

    ``xarray`` is optional; it is imported inside this function, never at module scope.

    Args:
        ds: A UGRID-CF ``xarray.Dataset``, e.g. one produced by :func:`to_xarray`.

    Returns:
        A ``(mesh, data_variables)`` tuple -- a :class:`Mesh2d` and a ``{name: MeshVariable}`` mapping.

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``xarray`` is not installed.
        TypeError: ``ds`` is not an ``xarray.Dataset``.
    """
    xr = import_xarray(_XARRAY_HINT)
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xarray.Dataset, got {type(ds).__name__}")

    mesh_name = _topology_name(ds)
    mesh = _reconstruct_mesh(ds, mesh_name)

    data_variables: dict[str, MeshVariable] = {}
    for name in ds.data_vars:
        da = ds[name]
        attrs = dict(da.attrs)
        if "location" not in attrs:
            continue
        nodata = attrs.get("_FillValue")
        data_variables[str(name)] = MeshVariable(
            name=str(name),
            location=str(attrs["location"]),
            mesh_name=str(attrs.get("mesh", mesh_name)),
            shape=tuple(da.shape),
            attributes=attrs,
            nodata=None if nodata is None else float(nodata),
            units=attrs.get("units"),
            standard_name=attrs.get("standard_name"),
            dimensions=tuple(str(d) for d in da.dims),
            _data=np.asarray(da.values),
        )

    return mesh, data_variables


def read_crs_wkt(ds: Any) -> str | None:
    """The CRS WKT a UGRID-CF dataset carries, or ``None``.

    ``from_xarray`` answers the mesh and its variables; the CRS lives beside them on the grid-mapping
    variable, and the caller needs it to stamp the dataset it builds. Exposed separately so
    ``from_xarray``'s two-value contract stays as it is.

    Args:
        ds: The ``xarray.Dataset`` to read.

    Returns:
        The WKT string, or ``None`` when the dataset declares no CRS.
    """
    result: str | None = None
    if "crs" in ds.variables:
        wkt = ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref")
        if isinstance(wkt, str):
            result = wkt
    return result


def to_zarr(
    mesh: Mesh2d,
    data_variables: dict[str, MeshVariable],
    store: Any,
    mesh_name: str = DEFAULT_MESH_NAME,
    crs_wkt: str | None = None,
    **to_zarr_kwargs: Any,
) -> Any:
    """Write a mesh and its variables to a UGRID-encoded Zarr store.

    Routes through :func:`to_xarray` and ``xarray.Dataset.to_zarr``, so the connectivity
    ``start_index`` / ``_FillValue`` encoding is handled once in :func:`to_xarray` and never
    re-encoded here. ``xarray`` and ``zarr`` are optional and imported inside this function.

    Args:
        mesh: The mesh topology to write.
        data_variables: Mapping of variable name to :class:`MeshVariable`.
        store: A Zarr store -- a path, ``str``, or any ``MutableMapping`` ``to_zarr`` accepts.
        mesh_name: Name for the exported topology variable. Defaults to ``"mesh2d"``.
        **to_zarr_kwargs: Extra keyword arguments forwarded to ``xarray.Dataset.to_zarr``
            (``mode`` defaults to ``"w"``).

    Returns:
        Whatever ``xarray.Dataset.to_zarr`` returns (a ``ZarrStore`` / delayed object).

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``xarray`` or ``zarr`` is not installed.
    """
    require_optional("zarr", _ZARR_HINT)
    ds = to_xarray(mesh, data_variables, mesh_name=mesh_name, crs_wkt=crs_wkt)
    to_zarr_kwargs.setdefault("mode", "w")
    return ds.to_zarr(store, **to_zarr_kwargs)


def from_zarr(
    store: Any, **open_zarr_kwargs: Any
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Read a mesh and its variables back from a UGRID-encoded Zarr store.

    The inverse of :func:`to_zarr`: opens the store with ``xarray.open_zarr`` and hands the dataset to
    :func:`from_xarray`. ``xarray`` and ``zarr`` are optional and imported inside this function.

    Args:
        store: A Zarr store -- a path, ``str``, or ``MutableMapping`` ``open_zarr`` accepts.
        **open_zarr_kwargs: Extra keyword arguments forwarded to ``xarray.open_zarr``.

    Returns:
        A ``(mesh, data_variables)`` tuple, as :func:`from_xarray` returns.

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``xarray`` or ``zarr`` is not installed.
    """
    return from_xarray(open_zarr_dataset(store, **open_zarr_kwargs))


def open_zarr_dataset(store: Any, **open_zarr_kwargs: Any) -> Any:
    """Open a Zarr store as an ``xarray.Dataset``, without decoding the mesh.

    :func:`from_zarr` answers the mesh and its variables, which leaves the CRS (a grid-mapping
    variable beside them) unreachable to a caller that needs to stamp it onto a dataset. Opening is
    therefore exposed on its own, so a caller can read both the mesh and
    :func:`read_crs_wkt` from one open.

    Args:
        store: A Zarr store -- a path, ``str``, or ``MutableMapping`` ``open_zarr`` accepts.
        **open_zarr_kwargs: Extra keyword arguments forwarded to ``xarray.open_zarr``.

    Returns:
        The opened ``xarray.Dataset``.

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``xarray`` or ``zarr`` is not installed.
    """
    require_optional("zarr", _ZARR_HINT)
    xr = import_xarray(_XARRAY_HINT)
    return xr.open_zarr(store, **open_zarr_kwargs)


def chunked_variables(
    data_variables: dict[str, MeshVariable], chunks: Any = "auto"
) -> dict[str, Any]:
    """Return dask-backed arrays for the given mesh variables, for out-of-core reads.

    The lazy counterpart of reading ``MeshVariable.data`` eagerly: each variable becomes a
    ``dask.array`` so a large temporal variable can be reduced block by block without materialising
    the whole ``(n_time, n_elements)`` array. A file-backed variable (one carrying a source path) is
    wrapped with :func:`pyramids.netcdf._lazy.build_lazy_array` so its chunks are read straight from
    the store; an in-memory variable is wrapped with ``dask.array.from_array``.

    A variable with neither -- a metadata-only variable, carrying a declared shape and dtype but no
    array and no source -- has nothing to chunk and is left out of the result, so the mapping can be
    smaller than the input.

    ``dask`` is optional and imported inside this function; a missing dask raises the ``[lazy]``
    install hint rather than failing with a bare import error.

    Args:
        data_variables: Mapping of variable name to :class:`MeshVariable`.
        chunks: Chunk spec forwarded to the lazy reader -- ``"auto"`` (default), an ``int``, a
            ``tuple``, or a ``dict``.

    Returns:
        A ``{name: dask.array}`` mapping, one entry per variable that has readable data.

    Raises:
        pyramids.errors.OptionalPackageDoesNotExist: ``dask`` is not installed.
    """
    dask = import_dask(_DASK_HINT)
    da_mod = dask.array

    result: dict[str, Any] = {}
    for name, var in data_variables.items():
        if not var.has_data_source:
            continue
        source_path = var._source_path
        if source_path is not None:
            result[name] = build_lazy_array(source_path, var.name, chunks, orient=False)
        else:
            data = var.data
            if data is None:
                continue
            result[name] = da_mod.from_array(np.asarray(data), chunks=chunks)
    return result
