"""Which exception a reader actually raises for a file GDAL cannot open.

GDAL exceptions are enabled package-wide, so a failed `gdal.OpenEx` raises `RuntimeError`
from inside the binding and never returns `None`. Every reader here therefore surfaces
GDAL's own error, not the `ValueError` three of their docstrings promised -- and the
`if ... is None: raise ValueError(...)` guards behind those promises can never run.
"""

from pathlib import Path

import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core


@pytest.fixture
def not_a_netcdf(tmp_path) -> Path:
    """A file with an `.nc` name that no GDAL driver recognises."""
    path = tmp_path / "not-a-netcdf.nc"
    path.write_bytes(b"this is not a netcdf file at all, not even close\n")
    return path


class TestUnopenableFile:
    """Every reader's behaviour on a file GDAL refuses."""

    def test_read_file_raises_runtime_error(self, not_a_netcdf):
        """`read_file` surfaces GDAL's own error.

        Test scenario:
            The `if ds is None: raise ValueError(...)` line below the open cannot be reached,
            because `OpenEx` raises first.
        """
        with pytest.raises(RuntimeError, match="not recognized"):
            UgridDataset.read_file(str(not_a_netcdf))

    def test_mesh_names_raises_runtime_error(self, not_a_netcdf):
        """`mesh_names` surfaces GDAL's own error, as its Raises block says."""
        with pytest.raises(RuntimeError, match="not recognized"):
            UgridDataset.mesh_names(str(not_a_netcdf))

    def test_read_all_raises_runtime_error(self, not_a_netcdf):
        """`read_all` surfaces GDAL's own error.

        Test scenario:
            It delegates to `mesh_names`, so it cannot raise the `ValueError` its Raises block
            promised for an unopenable file.
        """
        with pytest.raises(RuntimeError, match="not recognized"):
            UgridDataset.read_all(str(not_a_netcdf))

    def test_from_bytes_raises_runtime_error(self, not_a_netcdf):
        """`from_bytes` surfaces GDAL's own error.

        Test scenario:
            It writes the buffer into `/vsimem` and reads it through `read_file`, so the same
            `RuntimeError` comes back -- naming the `/vsimem` path rather than the caller's
            bytes, which is worth pinning since that is what a user sees.
        """
        payload = not_a_netcdf.read_bytes()

        with pytest.raises(RuntimeError, match="not recognized") as exc:
            UgridDataset.from_bytes(payload)

        assert "/vsimem/" in str(exc.value), (
            f"the message should name the in-memory file it tried to read: {exc.value}"
        )


class TestValidFileWithoutATopology:
    """The `ValueError` the readers really do raise."""

    def test_a_netcdf_with_no_mesh_is_a_value_error(self):
        """A readable NetCDF that declares no UGRID topology raises `ValueError`.

        Test scenario:
            This is the contrast case that keeps `ValueError` in the Raises blocks honest: a
            plain CF gridded file opens fine, so GDAL raises nothing and the topology check is
            what fires.
        """
        cf_file = (
            Path(__file__).parents[1] / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"
        )

        with pytest.raises(ValueError, match="No UGRID mesh topology"):
            UgridDataset.read_file(str(cf_file))
