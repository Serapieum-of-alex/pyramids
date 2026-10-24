"""Decoding the CF time coordinate of a UGRID mesh (issue #1381).

The reader collects data variables by their ``location`` attribute, and a CF ``time``
coordinate carries none — so the time axis used to read as a bare dimension and
``time_values`` fell back to positional indices even for a file that states its stamps. These
tests pin the decode on read, the coordinate on write, and the positional fallback that must
survive for a file carrying no CF time units.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.dataset import (
    TIME_CALENDAR_ATTR,
    TIME_UNITS_ATTR,
    _time_coordinate_name,
)

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _temporal_mesh(stamps=None, units=None, calendar="standard"):
    """A two-face mesh with a 3-step variable, optionally carrying CF time metadata."""
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"depth": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        data_locations={"depth": "face"},
    )
    ds["depth"].dimensions = ("time", "mesh2d_nFaces")
    if stamps is not None:
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: units,
            TIME_CALENDAR_ATTR: calendar,
        }
    return ds


class TestTimeCoordinateName:
    """Which dimension the decoder treats as the time axis."""

    def test_taken_from_the_variable_dimensions(self):
        """The name comes from the variable's own dimensions at its time_index."""
        ds = _temporal_mesh()
        assert _time_coordinate_name(ds.data_vars) == "time"

    def test_none_when_nothing_is_temporal(self):
        """A static-only dataset has no time coordinate to look for."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"bed": np.array([1.0, 2.0])},
            data_locations={"bed": "face"},
        )
        assert _time_coordinate_name(ds.data_vars) is None

    def test_honours_a_non_default_axis_name(self):
        """A file naming its axis something other than `time` is followed, not assumed."""
        ds = _temporal_mesh()
        ds["depth"].dimensions = ("t", "mesh2d_nFaces")
        assert _time_coordinate_name(ds.data_vars) == "t"


class TestRoundTrip:
    """Write then read must preserve the stamps, units and calendar."""

    def test_stamps_survive_a_round_trip(self, tmp_path):
        """Regression (#1381): the writer emitted the time dimension but no time array, so a
        round trip degraded real dates to positional indices."""
        stamps = np.array(
            ["2020-03-01T00:00:00", "2020-03-01T01:00:00", "2020-03-01T02:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "seconds since 2020-03-01 00:00:00")
        path = tmp_path / "stamped.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert [str(v) for v in back.time_values] == [str(v) for v in stamps]

    def test_units_and_calendar_survive(self, tmp_path):
        """The decoded axis carries its CF units and calendar, which convert_calendar needs."""
        stamps = np.array(
            ["2020-01-01T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "days since 2020-01-01 00:00:00")
        path = tmp_path / "units.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        attrs = back["depth"].attributes
        assert "days since" in str(attrs.get(TIME_UNITS_ATTR))
        assert attrs.get(TIME_CALENDAR_ATTR) == "standard"

    def test_a_non_standard_calendar_round_trips(self, tmp_path):
        """A model calendar decodes through the existing cftime path, not just `standard`."""
        import cftime

        stamps = [
            cftime.DatetimeNoLeap(2020, 1, 1),
            cftime.DatetimeNoLeap(2020, 1, 2),
            cftime.DatetimeNoLeap(2020, 1, 3),
        ]
        ds = _temporal_mesh(stamps, "days since 2020-01-01 00:00:00", calendar="noleap")
        path = tmp_path / "noleap.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back["depth"].attributes.get(TIME_CALENDAR_ATTR) == "noleap"
        assert len(back.time_values) == 3


class TestPositionalFallback:
    """A file with no CF time units must read exactly as it did before."""

    def test_no_time_metadata_reads_positionally(self, tmp_path):
        """Decoding is additive: it must never become a new way for read_file to fail."""
        ds = _temporal_mesh()  # no stamps, no units
        path = tmp_path / "plain.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.time_values == [0, 1, 2]

    def test_a_static_mesh_is_untouched(self, tmp_path):
        """No time dimension at all: nothing to decode, nothing to write."""
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"bed": np.array([1.0, 2.0])},
            data_locations={"bed": "face"},
        )
        path = tmp_path / "static.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.time_values is None


class TestResampleUnblocked:
    """The wider benefit: resample needs date-like coordinates, which decoding supplies."""

    def test_resample_works_once_stamps_are_decoded(self, tmp_path):
        """resample bins by a pandas offset alias, so it was unusable while the axis was
        positional. Locks in that decoding makes it work."""
        stamps = np.array(
            ["2020-01-01T00:00:00", "2020-01-01T00:30:00", "2020-01-01T01:00:00"],
            dtype="datetime64[ns]",
        )
        ds = _temporal_mesh(stamps, "seconds since 2020-01-01 00:00:00")
        path = tmp_path / "resample.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        hourly = back.resample("1h", how="mean")
        assert hourly["depth"].n_time_steps == 2
