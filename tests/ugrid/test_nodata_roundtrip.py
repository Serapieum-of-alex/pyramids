"""A declared no-data value must survive a UGRID file round trip.

`to_file` wrote `_FillValue` *after* the variable's data, and NetCDF refuses to define a fill
value once data exists -- so the attribute write failed and the error was swallowed as a DEBUG
log by the shared attribute writer, whose docstring blamed a "GDAL limitation". The declaration
was therefore lost on every write, and a `-999` sentinel read back as ordinary data.

The reader had the mirror-image defect: it looked for `_FillValue` in the attribute dictionary,
and GDAL's netCDF driver does not list the fill value among an array's attributes -- it is
reachable only through the dedicated no-data accessors. So fixing the write alone would not have
restored the round trip.

This matters to this branch specifically: the round 2 L2 fix made `regrid`, `to_location` and
`burn` declare NaN so that an element with no contributor is distinguishable from data. That
guarantee was void the moment the result was written to a file.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(values, nodata=None, offset=0.0):
    """A two-face mesh carrying `values`, optionally declaring `nodata`."""
    dataset = UgridDataset.from_arrays(
        node_x=_NODE_X + offset,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"v": np.asarray(values)},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        dataset["v"].nodata = nodata
    return dataset


def _round_trip(dataset, tmp_path, name="m.nc"):
    """Write `dataset` to a file and read it back."""
    path = tmp_path / name
    dataset.to_file(str(path))
    return UgridDataset.read_file(str(path))


class TestDeclaredNodataSurvives:
    """The declaration a caller set before writing."""

    def test_a_sentinel_survives(self, tmp_path):
        """A `-999` fill value is still declared after a write and a read.

        Test scenario:
            Without it the sentinel reads back as an ordinary value: the array still holds
            `-999.0`, and nothing says it means "missing", so a mean over the variable is
            silently wrong by the sentinel's magnitude.
        """
        read_back = _round_trip(_mesh([1.0, -999.0], nodata=-999.0), tmp_path)

        assert read_back["v"].nodata == -999.0, (
            f"the declared fill value should survive the round trip, got "
            f"{read_back['v'].nodata!r}"
        )

    def test_nan_survives(self, tmp_path):
        """A NaN fill value survives too.

        Test scenario:
            NaN is what the facades declare, so it is the case that matters most here. It is
            also the one an equality test would miss, which is why this asserts through
            `np.isnan` rather than `==`.
        """
        read_back = _round_trip(_mesh([1.0, np.nan], nodata=float("nan")), tmp_path)
        declared = read_back["v"].nodata

        assert declared is not None and np.isnan(declared), (
            f"NaN should survive as the declared fill value, got {declared!r}"
        )

    def test_the_values_are_unchanged(self, tmp_path):
        """Restoring the declaration does not disturb the data.

        Test scenario:
            The fix reorders the attribute write ahead of the data write, so the data path is
            the thing most at risk of regressing.
        """
        read_back = _round_trip(_mesh([1.0, -999.0], nodata=-999.0), tmp_path)

        np.testing.assert_allclose(np.asarray(read_back["v"].data), [1.0, -999.0])

    def test_no_declaration_stays_undeclared(self, tmp_path):
        """A variable that declares nothing still declares nothing.

        Test scenario:
            The guard against the opposite failure: reading the fill through GDAL's accessor
            must not fabricate one for a variable that has none, which would mark a real value
            as missing.
        """
        read_back = _round_trip(_mesh([1.0, 2.0]), tmp_path)

        assert read_back["v"].nodata is None, (
            f"a variable with no declared fill value should read back as None, got "
            f"{read_back['v'].nodata!r}"
        )


class TestFacadeNodataSurvives:
    """The NaN the round 2 L2 fix made the facades declare."""

    def test_a_regridded_variable_keeps_its_nan_declaration(self, tmp_path):
        """`regrid` declares NaN, and the file keeps it.

        Test scenario:
            L2 made `regrid` declare NaN precisely so a target element the source does not
            cover is distinguishable from data. Writing the result dropped the declaration, so
            the guarantee held in memory and not on disk -- which is where it is needed, since
            that is what another tool reads.
        """
        source = _mesh([1.0, 2.0])
        target = _mesh([0.0, 0.0], offset=10.0)

        regridded = source.regrid(target)
        assert regridded["v"].nodata is not None and np.isnan(regridded["v"].nodata), (
            "precondition: the regrid facade declares NaN"
        )

        read_back = _round_trip(regridded, tmp_path, name="regridded.nc")
        declared = read_back["v"].nodata

        assert declared is not None and np.isnan(declared), (
            f"the regridded variable's NaN declaration should survive the write, got "
            f"{declared!r}"
        )

    def test_a_remapped_variable_keeps_its_nan_declaration(self, tmp_path):
        """`to_location` declares NaN for the same reason, and that survives too."""
        moved = _mesh([1.0, 2.0]).to_location("node")

        read_back = _round_trip(moved, tmp_path, name="moved.nc")
        declared = read_back["v"].nodata

        assert declared is not None and np.isnan(declared), (
            f"the remapped variable's NaN declaration should survive the write, got "
            f"{declared!r}"
        )
