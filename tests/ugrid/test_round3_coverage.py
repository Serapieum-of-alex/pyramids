"""The branches round 3's fixes added that no other test reaches.

Each of these is a guard written during the round 3 burndown. A guard the suite never runs is
a guard that can stop working without anyone noticing, which is the whole reason the previous
round's half-applied fixes survived as long as they did.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid._facades import _representable
from pyramids.netcdf.ugrid.construct import _polygon_ring, _without_repeats

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


class TestRepresentableRemainingBranches:
    """The two arms of `_representable` the parametrisation does not reach."""

    def test_a_fractional_value_is_not_an_integer_sentinel(self):
        """`1.5` cannot be an `int32` fill: casting it would change it.

        Test scenario:
            The integer branch tests integrality before range, because `int(1.5)` is in range
            and would have looked representable.
        """
        assert _representable(1.5, np.dtype("int32")) is False

    def test_an_uncastable_object_is_answered_not_raised(self):
        """The `except` arm: something numpy cannot make an array of at all."""

        class Opaque:
            """An object with no array interface."""

            def __array__(self, *args, **kwargs):
                raise TypeError("no array interface")

        assert _representable(Opaque(), np.dtype("float64")) is False


class TestPolygonRingRemainingBranches:
    """The zero-area rejection and the short-ring shortcut."""

    def test_a_collinear_ring_is_rejected(self):
        """Three distinct vertices in a line enclose nothing.

        Test scenario:
            The count tests all pass -- three raw coordinates, three distinct points -- so the
            area test is the only thing that can catch it.
        """
        from shapely.geometry import Polygon

        assert _polygon_ring(Polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])) is None

    def test_a_one_point_ring_short_circuits(self):
        """`_without_repeats` returns a ring of fewer than two points untouched."""
        single = np.array([[1.0, 2.0]])

        np.testing.assert_array_equal(_without_repeats(single), single)


class TestTidyFrameAxisCollisions:
    """When the dimension names put time or layer in the element slot."""

    def _with_dims(self, values, dimensions):
        """A two-face mesh carrying `values` under `dimensions`."""
        dataset = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"anchor": np.array([1.0, 2.0])},
            data_locations={"anchor": "face"},
        )
        dataset = dataset.with_variable("v", np.asarray(values), location="face")
        dataset["v"].dimensions = dimensions
        return dataset

    def test_a_time_named_element_axis_is_read_as_elements(self):
        """The element count wins when the names disagree with the shape.

        Test scenario:
            A `(2,)` variable whose single axis is named "time" on a two-face mesh: the
            element reading wins, so there is no step axis to walk and the frame carries one
            row per face rather than per step.
        """
        frame = self._with_dims(np.array([5.0, 6.0]), ("time",)).to_dataframe(
            variables=["v"]
        )

        assert len(frame) == 2
        assert sorted(frame["element"].tolist()) == [0, 1]
        assert not frame["x"].isna().any()

    def test_a_layer_named_element_axis_is_read_as_elements(self):
        """The same collision for the layer axis."""
        frame = self._with_dims(np.array([5.0, 6.0]), ("mesh2d_nLayers",)).to_dataframe(
            variables=["v"]
        )

        assert len(frame) == 2
        assert sorted(frame["element"].tolist()) == [0, 1]
