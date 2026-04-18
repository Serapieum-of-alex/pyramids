"""Tests for the :class:`PyramidsBackendEntrypoint` xarray backend.

DASK-13: register pyramids as an xarray backend so
``xr.open_dataset(path, engine="pyramids")`` returns a lazy xarray
:class:`xarray.Dataset` whose reads go through pyramids' NetCDF path.
"""

from __future__ import annotations

import pytest


try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    HAS_XARRAY = False


requires_xarray = pytest.mark.skipif(
    not HAS_XARRAY, reason="xarray not installed"
)


FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


class TestBackendEntrypoint:
    """Direct instantiation of the backend class (no engine= dispatch)."""

    @requires_xarray
    def test_open_dataset_returns_xarray_dataset(self):
        from pyramids.netcdf._xarray_backend import PyramidsBackendEntrypoint

        entry = PyramidsBackendEntrypoint()
        ds = entry.open_dataset(FIXTURE)
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs.get("pyramids_backend") is True

    @requires_xarray
    def test_dataset_exposes_variables(self):
        from pyramids.netcdf._xarray_backend import PyramidsBackendEntrypoint

        entry = PyramidsBackendEntrypoint()
        ds = entry.open_dataset(FIXTURE)
        assert "values" in ds.data_vars

    @requires_xarray
    def test_drop_variables(self):
        from pyramids.netcdf._xarray_backend import PyramidsBackendEntrypoint

        entry = PyramidsBackendEntrypoint()
        ds = entry.open_dataset(FIXTURE, drop_variables=["values"])
        assert "values" not in ds.data_vars


class TestEngineDispatch:
    """``xr.open_dataset(..., engine="pyramids")`` dispatches to our backend."""

    @requires_xarray
    def test_engine_name_dispatches(self):
        try:
            import xarray.backends

            xarray.backends.plugins.refresh_engines()
        except Exception:
            pass
        ds = xr.open_dataset(FIXTURE, engine="pyramids")
        assert ds.attrs.get("pyramids_backend") is True


class TestGuessCanOpen:
    """Backend never auto-claims — entry is opt-in via engine=."""

    @requires_xarray
    def test_guess_always_false(self):
        from pyramids.netcdf._xarray_backend import PyramidsBackendEntrypoint

        entry = PyramidsBackendEntrypoint()
        assert entry.guess_can_open(FIXTURE) is False
        assert entry.guess_can_open("random.nc") is False


class TestChunksApplied:
    """xarray applies chunks= AFTER open_dataset returns — result is dask-backed."""

    @requires_xarray
    def test_chunks_produces_dask_backing(self):
        try:
            import dask  # noqa: F401
        except ImportError:
            pytest.skip("dask not installed")
        ds = xr.open_dataset(FIXTURE, engine="pyramids", chunks={})
        data = ds["values"].data
        assert hasattr(data, "dask")
