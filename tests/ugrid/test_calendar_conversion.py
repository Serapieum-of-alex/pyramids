"""Moving a mesh time axis between CF calendars (issue #1381, step 3).

``convert_calendar`` restamps without touching a value and drops the dates the target calendar
has no day for; ``interp_calendar`` is its lossless counterpart, interpolating onto the
target's own stamps. Both need real dates, so both refuse a positional axis -- which is the
behaviour that makes the CF decode of step 1 worth having.
"""

import cftime
import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(stamps, values=None, units="days since 2000-01-01", calendar="standard"):
    """A two-face mesh whose time axis carries ``stamps`` and CF metadata."""
    data = (
        np.asarray(values, dtype="float64")
        if values is not None
        else np.arange(2.0 * len(stamps)).reshape(len(stamps), 2)
    )
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"depth": data},
        data_locations={"depth": "face"},
    )
    ds["depth"].dimensions = ("time", "mesh2d_nFaces")
    if units is not None:
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: units,
            TIME_CALENDAR_ATTR: calendar,
        }
    else:
        ds["depth"].attributes = {**ds["depth"].attributes, "time_values": list(stamps)}
    return ds


def _days(calendar, *dates):
    """cftime stamps on ``calendar`` from ``(year, month, day)`` triples."""
    return [cftime.datetime(*date, calendar=calendar) for date in dates]


class TestConvertCalendar:
    """Restamping, and what it costs."""

    def test_the_values_are_never_touched(self):
        """Only the coordinate changes - that is the whole contract."""
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2), (2001, 1, 3))
        ds = _mesh(stamps)
        converted = ds.convert_calendar("noleap")
        assert np.array_equal(converted["depth"].data, ds["depth"].data)

    def test_the_target_calendar_is_carried(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        converted = _mesh(stamps).convert_calendar("360_day")
        assert converted["depth"].attributes[TIME_CALENDAR_ATTR] == "360_day"
        assert all(s.calendar == "360_day" for s in converted.time_values)

    def test_the_units_string_is_kept(self):
        """The offsets are re-counted in the same unit and epoch."""
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        converted = _mesh(stamps, units="days since 1990-01-01").convert_calendar(
            "noleap"
        )
        assert converted["depth"].attributes[TIME_UNITS_ATTR] == "days since 1990-01-01"

    def test_29_february_is_dropped_moving_to_noleap(self):
        """The documented loss: noleap has no 29 February, so that step goes with the date."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        converted = _mesh(stamps).convert_calendar("noleap")
        assert converted["depth"].n_time_steps == 2, "the leap day's step is dropped"
        assert [(s.month, s.day) for s in converted.time_values] == [(2, 28), (3, 1)]

    def test_the_surviving_steps_keep_their_own_values(self):
        """The dropped step must take ITS row with it, not shift the others."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        values = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
        converted = _mesh(stamps, values=values).convert_calendar("noleap")
        assert converted["depth"].data.tolist() == [[10.0, 11.0], [30.0, 31.0]]

    def test_31_december_is_dropped_moving_to_360_day(self):
        """A 360-day year has no month longer than 30 days."""
        stamps = _days("standard", (2001, 12, 30), (2001, 12, 31))
        converted = _mesh(stamps).convert_calendar("360_day")
        assert converted["depth"].n_time_steps == 1

    def test_align_on_year_keeps_the_position_in_the_year(self):
        """Model bookkeeping: the day's place in the year matters, not its date."""
        stamps = _days("standard", (2001, 1, 1), (2001, 7, 2), (2001, 12, 31))
        converted = _mesh(stamps).convert_calendar("360_day", align_on="year")
        assert converted["depth"].n_time_steps == 3, (
            "align_on='year' drops nothing here"
        )
        assert converted.time_values[0].day == 1

    def test_dropping_every_step_is_refused(self):
        """Rather than returning an empty dataset."""
        # Every month on a 360_day calendar is 30 days, so no 31st survives. (29 February
        # does, which is why the leap day alone would not have emptied the axis.)
        stamps = _days("standard", (2001, 1, 31), (2001, 12, 31))
        with pytest.raises(ValueError, match="would drop every step"):
            _mesh(stamps).convert_calendar("360_day")

    def test_an_unknown_calendar_is_refused(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        with pytest.raises(ValueError, match="cftime does not know"):
            _mesh(stamps).convert_calendar("mayan")

    def test_an_unknown_alignment_is_refused(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        with pytest.raises(ValueError, match="align_on"):
            _mesh(stamps).convert_calendar("noleap", align_on="month")

    def test_a_positional_axis_is_refused_with_a_reason(self):
        """No CF units means the steps are positions, so there is no calendar to convert."""
        ds = _mesh([0, 1, 2], units=None)
        with pytest.raises(ValueError, match="no 'units' attribute"):
            ds.convert_calendar("noleap")

    def test_a_datetime64_axis_is_accepted(self):
        """The reader yields datetime64 for the Gregorian family; it must still convert."""
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        converted = _mesh(list(stamps)).convert_calendar("360_day")
        assert converted["depth"].n_time_steps == 3
        assert all(s.calendar == "360_day" for s in converted.time_values)

    def test_a_per_step_active_mask_loses_the_same_steps(self):
        """A dropped step must take its mask row with it, or the mask silently misaligns."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        ds = _mesh(stamps)
        ds["depth"].active = np.array(
            [[True, True], [False, False], [True, False]], dtype=bool
        )
        converted = ds.convert_calendar("noleap")
        assert converted["depth"].active.tolist() == [[True, True], [True, False]]


class TestInterpCalendar:
    """The lossless counterpart."""

    def test_it_lands_on_the_targets_stamps(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 3), (2001, 1, 5)))
        target = _mesh(_days("noleap", (2001, 1, 1), (2001, 1, 5)), calendar="noleap")
        result = source.interp_calendar(target)
        assert result["depth"].n_time_steps == 2
        assert [(s.month, s.day) for s in result.time_values] == [(1, 1), (1, 5)]

    def test_no_step_is_lost_where_convert_would_drop_one(self):
        """The point of the method: 29 February survives as an interpolated value."""
        source = _mesh(_days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1)))
        target = _mesh(_days("noleap", (2004, 2, 28), (2004, 3, 1)), calendar="noleap")
        assert source.convert_calendar("noleap")["depth"].n_time_steps == 2
        assert source.interp_calendar(target)["depth"].n_time_steps == 2

    def test_the_interpolation_is_linear_in_decimal_years(self):
        """Mid-way between two stamps is the mean of their values."""
        source = _mesh(
            _days("standard", (2001, 1, 1), (2001, 1, 3)),
            values=[[0.0, 0.0], [10.0, 20.0]],
        )
        target = _mesh(_days("standard", (2001, 1, 2)), values=[[0.0, 0.0]])
        result = source.interp_calendar(target)["depth"].data
        assert np.allclose(result, [[5.0, 10.0]]), "2 Jan is midway between 1 and 3 Jan"

    def test_the_targets_calendar_and_units_are_carried(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(
            _days("noleap", (2001, 1, 2), (2001, 1, 3)),
            calendar="noleap",
            units="hours since 1999-01-01",
        )
        result = source.interp_calendar(target)
        assert result["depth"].attributes[TIME_CALENDAR_ATTR] == "noleap"
        assert result["depth"].attributes[TIME_UNITS_ATTR] == "hours since 1999-01-01"

    def test_a_stamp_outside_the_span_is_a_gap_not_an_extrapolation(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(_days("standard", (2001, 1, 3), (2009, 1, 1)))
        result = source.interp_calendar(target)["depth"].data
        assert np.isfinite(result[0]).all(), "inside the span"
        assert np.isnan(result[1]).all(), "outside it"

    def test_a_target_that_does_not_overlap_at_all_is_refused(self):
        """Rather than returning a fully-masked dataset that looks like a result."""
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(_days("standard", (2050, 1, 1), (2050, 1, 5)))
        with pytest.raises(ValueError, match="does not overlap"):
            source.interp_calendar(target)

    def test_a_single_source_step_is_refused(self):
        source = _mesh(_days("standard", (2001, 1, 1)))
        target = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 2)))
        with pytest.raises(ValueError, match="at least two steps"):
            source.interp_calendar(target)

    def test_a_different_mesh_is_refused(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        other = UgridDataset.from_arrays(
            node_x=np.array([0.0, 9.0, 9.0]),
            node_y=np.array([0.0, 0.0, 9.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.zeros((2, 1))},
            data_locations={"depth": "face"},
        )
        with pytest.raises(Exception, match="(?i)mesh|topolog|align"):
            source.interp_calendar(other)
