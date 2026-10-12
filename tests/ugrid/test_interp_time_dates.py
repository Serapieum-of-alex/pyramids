"""`interp_time` must accept the date coordinates `interpolate_na` and `resample` accept.

The same commit series taught `interpolate_na(use_coordinate=True)` to convert date-like
coordinates to nanoseconds since the epoch, and taught `interp_time` to *refuse* them with a
politer message. Since `time_values` returns whatever the variable's attribute holds, a mesh
read from a dated file reached the refusal -- so `interp_time` was unusable on exactly the data
`resample` handles.

Worse, the two forms that were accepted disagreed by 1000x: a `datetime64[ns]` array came
through as nanoseconds and a `pd.DatetimeIndex` as microseconds, so `new_times` written against
one silently produced all-NaN against the other.
"""

import numpy as np
import pandas as pd
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
_STAMPS = ["2020-01-01", "2020-01-03"]


def _dated(stamps):
    """A two-face mesh with two time steps carrying `stamps` as their coordinates."""
    mesh = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"v": np.array([[0.0, 10.0], [2.0, 12.0]])},
        data_locations={"v": "face"},
    )
    mesh["v"].attributes["time_values"] = list(stamps)
    return mesh


class TestDateCoordinatesAreAccepted:
    """The three date-like spellings a caller can hold."""

    @pytest.mark.parametrize(
        "stamps",
        [
            list(pd.to_datetime(_STAMPS)),
            pd.DatetimeIndex(pd.to_datetime(_STAMPS)),
            np.asarray(pd.to_datetime(_STAMPS), dtype="datetime64[ns]"),
        ],
        ids=["timestamp-list", "datetime-index", "datetime64-array"],
    )
    def test_the_midpoint_interpolates(self, stamps):
        """Halfway between the two steps gives the mean of their values.

        Args:
            stamps: The time coordinates, in one of the accepted spellings.

        Test scenario:
            A `Timestamp` list raised `ValueError: interp_time needs numeric time
            coordinates`, which is the form a file-read dataset carries. All three must agree,
            so the same midpoint is asked for by date in each.
        """
        mesh = _dated(stamps)
        midpoint = pd.Timestamp("2020-01-02")

        out = mesh.interp_time([midpoint])
        values = np.asarray(out["v"].data)

        assert values.shape == (1, 2), f"one step, two faces, got {values.shape}"
        np.testing.assert_allclose(values[0], [1.0, 11.0])

    def test_the_scales_agree_across_spellings(self):
        """The three spellings must not differ by a factor of 1000.

        Test scenario:
            `datetime64[ns]` came through as nanoseconds and `pd.DatetimeIndex` as
            microseconds, so a `new_times` value valid for one landed far outside the other's
            axis and produced all-NaN. Asking for the same instant in each is what catches it.
        """
        midpoint = pd.Timestamp("2020-01-02")
        answers = [
            np.asarray(_dated(stamps).interp_time([midpoint])["v"].data)
            for stamps in (
                list(pd.to_datetime(_STAMPS)),
                pd.DatetimeIndex(pd.to_datetime(_STAMPS)),
                np.asarray(pd.to_datetime(_STAMPS), dtype="datetime64[ns]"),
            )
        ]

        for other in answers[1:]:
            np.testing.assert_allclose(answers[0], other)

    def test_a_label_outside_the_axis_is_still_nan(self):
        """The `bounds="nan"` default survives the date conversion."""
        mesh = _dated(list(pd.to_datetime(_STAMPS)))

        out = mesh.interp_time([pd.Timestamp("2021-06-01")])

        assert np.isnan(np.asarray(out["v"].data)).all()


class TestNumericCoordinatesUnchanged:
    """A numeric axis keeps working exactly as before."""

    def test_the_midpoint_interpolates(self):
        """The ordinary case, which must not regress."""
        mesh = _dated([0.0, 2.0])

        out = mesh.interp_time([1.0])

        np.testing.assert_allclose(np.asarray(out["v"].data)[0], [1.0, 11.0])


class TestCoordinatesThatCarryNoOrder:
    """Something that is neither a number nor a date is still refused."""

    def test_plain_labels_are_refused(self):
        """A written error, not a bare `TypeError` from inside numpy.

        Test scenario:
            The refusal still has a job: labels like "spring"/"summer" carry no distance to
            interpolate along, so there is nothing to measure between them.
        """
        mesh = _dated(["spring", "summer"])

        with pytest.raises(ValueError, match="interp_time"):
            mesh.interp_time([1.0])
