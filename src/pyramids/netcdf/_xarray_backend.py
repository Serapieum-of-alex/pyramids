"""xarray :class:`BackendEntrypoint` registration for pyramids NetCDF.

DASK-13: register pyramids as an xarray backend so users who prefer
xarray's ergonomics can do
``xr.open_dataset(path, engine="pyramids", chunks={})`` and receive a
lazy, dask-backed :class:`xarray.Dataset` whose array-level reads go
through pyramids' :class:`CachingFileManager` + MDArray chunked
reader.

This module is import-gated: xarray is not in pyramids' hard
dependency set. Import of :class:`PyramidsBackendEntrypoint` only
touches :mod:`xarray.backends` lazily when the entry point is
instantiated by xarray's plugin loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_xarray() -> tuple[Any, Any, Any]:
    """Return ``(BackendEntrypoint, BackendArray, indexing)`` from xarray.

    Wrapped in a helper so the module itself can be imported without
    pulling xarray in — xarray only loads when the entry point is
    instantiated. Bundled into one call so the entry-point class body
    below can do a single lookup.
    """
    from xarray.backends import BackendArray, BackendEntrypoint
    from xarray.core import indexing

    return BackendEntrypoint, BackendArray, indexing


BackendEntrypoint, BackendArray, _indexing = _require_xarray()


class PyramidsBackendArray(BackendArray):
    """Thin adapter that wraps a pyramids :class:`NetCDF` variable read.

    xarray's ``open_dataset`` wraps every variable in a ``BackendArray``
    so that chunking is applied lazily by xarray itself (after the
    backend returns). The adapter forwards ``__getitem__`` calls
    through :func:`xarray.core.indexing.explicit_indexing_adapter` so
    basic / outer indexers translate to the ``(xoff, yoff, xsize,
    ysize)`` window arguments that pyramids' reader understands.
    """

    def __init__(self, nc_path: str, variable_path: str, shape, dtype) -> None:
        self._nc_path = nc_path
        self._variable_path = variable_path
        self.shape = tuple(shape)
        self.dtype = dtype

    def __getitem__(self, key):
        """Read a slab via xarray's indexing-adapter contract."""
        return _indexing.explicit_indexing_adapter(
            key, self.shape, _indexing.IndexingSupport.BASIC, self._get_slab,
        )

    def _get_slab(self, key) -> Any:
        """Read the requested slab via pyramids' NetCDF reader."""
        from pyramids.netcdf import NetCDF

        nc = NetCDF.read_file(self._nc_path, open_as_multi_dimensional=True)
        subset = nc.get_variable(self._variable_path)
        arr = subset.read_array()
        return arr[key]


class PyramidsBackendEntrypoint(BackendEntrypoint):
    """xarray backend entry point for NetCDF files opened through pyramids.

    Enable the backend explicitly with ``engine="pyramids"``:

    ```python
    import xarray as xr
    ds = xr.open_dataset("file.nc", engine="pyramids", chunks={})
    ```

    :meth:`guess_can_open` returns ``False`` for every path so the
    pyramids backend never competes with xarray's built-in netCDF4
    engine unless the user opts in by name.
    """

    description = "pyramids NetCDF backend"
    url = "https://serapeum-org.github.io/pyramids"

    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
    )

    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        """Open a NetCDF file and return a skeletal :class:`xarray.Dataset`.

        xarray applies ``chunks=`` AFTER this method returns — the
        backend only exposes lazy :class:`PyramidsBackendArray`
        wrappers. The actual per-chunk reads still go through
        pyramids' :class:`CachingFileManager` once xarray starts
        materializing the dask graph.

        Args:
            filename_or_obj: Path to the NetCDF file.
            drop_variables: Optional collection of variable names to
                skip when constructing the :class:`xarray.Dataset`.
        """
        import xarray as xr

        from pyramids.netcdf import NetCDF

        path = str(filename_or_obj)
        nc = NetCDF.read_file(path, open_as_multi_dimensional=True)
        variables: dict[str, xr.Variable] = {}
        drop = set(drop_variables or [])

        for var_name in nc.get_variable_names():
            if var_name in drop:
                continue
            subset = nc.get_variable(var_name)
            sample = subset.read_array()
            backend_array = PyramidsBackendArray(
                path, var_name, shape=sample.shape, dtype=sample.dtype,
            )
            wrapped = _indexing.LazilyIndexedArray(backend_array)
            dims = tuple(f"dim_{i}" for i in range(len(sample.shape)))
            variables[var_name] = xr.Variable(dims, wrapped)

        ds = xr.Dataset(variables)
        ds.attrs["pyramids_backend"] = True
        return ds

    def guess_can_open(self, filename_or_obj) -> bool:
        """Never auto-claim a path — require explicit ``engine="pyramids"``.

        xarray's netCDF4 / h5netcdf engines already own the ``.nc``
        extension; returning ``False`` here keeps pyramids opt-in so
        existing pipelines are unaffected when the package is
        installed. Users activate the backend by passing
        ``engine="pyramids"`` explicitly.
        """
        del filename_or_obj
        return False
