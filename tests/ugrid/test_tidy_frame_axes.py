"""`to_dataframe` must address every value it emits, or refuse the variable.

The tidy frame walks one row per element and carries `time` and `layer` columns, so it can
address a variable laid out as `(time?, layer?, element)` and nothing else. Round 2 made the
transpose a real permutation, which stopped numpy raising -- but the consumer still treated
whatever ended up last as the element axis, so instead of a loud error two layouts produced a
frame that looked fine and was not:

* a variable whose element axis is **not** last silently lost the values beyond the element
  count and emitted `element` indices past it, with NaN coordinates;
* a variable carrying an **extra** dimension the frame has no column for kept every value but
  folded the extra axis into the element slot, again overrunning the element count.

A frame whose `element` column indexes faces that do not exist, and whose `x`/`y` are all NaN,
is worse than an exception, because a caller joins on it.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def two_faces():
    """A two-face mesh carrying one static per-face variable."""
    return UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"anchor": np.array([1.0, 2.0])},
        data_locations={"anchor": "face"},
    )


def _with_dims(dataset, name, values, dimensions):
    """Add `name` to `dataset` with explicit `dimensions`, returning the new dataset."""
    result = dataset.with_variable(name, np.asarray(values), location="face")
    result[name].dimensions = dimensions
    return result


class TestElementAxisNotLast:
    """A variable whose element axis comes before its time axis."""

    def test_every_value_is_emitted_once(self, two_faces):
        """A `(n_faces, time)` variable gives one row per element per step.

        Test scenario:
            The mesh has 2 faces and the variable 3 steps, so the frame must carry 6 rows. It
            carried 3: the last axis was read as the element axis, so three of the six values
            were dropped and the `element` column ran to 2 on a 2-face mesh.
        """
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (face, time)
        dataset = _with_dims(two_faces, "v", values, ("mesh2d_nFaces", "time"))

        frame = dataset.to_dataframe(variables=["v"])

        assert len(frame) == 6, f"2 faces x 3 steps = 6 rows, got {len(frame)}"
        assert sorted(frame["value"].tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (
            f"every value should appear exactly once, got {sorted(frame['value'].tolist())}"
        )

    def test_the_element_column_stays_within_the_element_count(self, two_faces):
        """`element` only ever names a face the mesh has."""
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = _with_dims(two_faces, "v", values, ("mesh2d_nFaces", "time"))

        frame = dataset.to_dataframe(variables=["v"])

        assert sorted(frame["element"].unique().tolist()) == [0, 1], (
            f"a 2-face mesh has elements 0 and 1, got "
            f"{sorted(frame['element'].unique().tolist())}"
        )

    def test_the_coordinates_are_real(self, two_faces):
        """Each row carries its face's centroid, not NaN.

        Test scenario:
            The coordinate table is indexed by `element`, so an out-of-range element produced
            NaN for every row -- 3 of 3 before the fix.
        """
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = _with_dims(two_faces, "v", values, ("mesh2d_nFaces", "time"))

        frame = dataset.to_dataframe(variables=["v"])

        assert not frame["x"].isna().any(), (
            f"no row should have a NaN x, got {frame['x'].isna().sum()} of {len(frame)}"
        )
        assert not frame["y"].isna().any()

    def test_each_value_keeps_its_element_and_step(self, two_faces):
        """The pairing survives the transpose, not just the row count.

        Test scenario:
            A correct row count can still come from a wrong arrangement, so this pins the
            actual association: face 0's three values are 1, 2, 3 in step order.
        """
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = _with_dims(two_faces, "v", values, ("mesh2d_nFaces", "time"))

        frame = dataset.to_dataframe(variables=["v"])
        face0 = frame[frame["element"] == 0].sort_values("time")["value"].tolist()
        face1 = frame[frame["element"] == 1].sort_values("time")["value"].tolist()

        assert face0 == [1.0, 2.0, 3.0], f"face 0 should hold 1,2,3 got {face0}"
        assert face1 == [4.0, 5.0, 6.0], f"face 1 should hold 4,5,6 got {face1}"


class TestUnaddressableExtraAxis:
    """A variable carrying a dimension the frame has no column for."""

    def test_it_is_refused_rather_than_folded_into_the_element_slot(self, two_faces):
        """A rank-4 variable raises, naming the dimension it cannot place.

        Test scenario:
            The frame carries `time` and `layer` and one row per element. A fourth dimension
            has nowhere to go, and folding it into the element slot produced 24 rows with
            `element` 0-5 on a 2-face mesh and every coordinate NaN. Refusing keeps the data
            and tells the caller which dimension is the problem.
        """
        values = np.zeros((2, 2, 3, 2))  # (time, layer, extra, face)
        dataset = _with_dims(
            two_faces,
            "v",
            values,
            ("time", "mesh2d_nLayers", "extra", "mesh2d_nFaces"),
        )

        with pytest.raises(ValueError, match="extra") as exc:
            dataset.to_dataframe(variables=["v"])

        assert "to_dataframe" in str(exc.value), (
            f"the message should name the operation, got {exc.value}"
        )

    def test_the_message_names_the_variable(self, two_faces):
        """A caller with many variables needs to know which one was refused."""
        values = np.zeros((2, 2, 3, 2))
        dataset = _with_dims(
            two_faces,
            "wide",
            values,
            ("time", "mesh2d_nLayers", "extra", "mesh2d_nFaces"),
        )

        with pytest.raises(ValueError, match="wide"):
            dataset.to_dataframe(variables=["wide"])


class TestSupportedLayoutsUnchanged:
    """The layouts the frame has always addressed keep working."""

    def test_a_static_variable(self, two_faces):
        """One row per element."""
        frame = two_faces.to_dataframe(variables=["anchor"])

        assert len(frame) == 2
        assert frame["value"].tolist() == [1.0, 2.0]

    def test_a_temporal_variable(self, two_faces):
        """`(time, element)`: one row per element per step."""
        dataset = _with_dims(
            two_faces,
            "v",
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            ("time", "mesh2d_nFaces"),
        )

        frame = dataset.to_dataframe(variables=["v"])

        assert len(frame) == 4
        assert sorted(frame["value"].tolist()) == [1.0, 2.0, 3.0, 4.0]

    def test_a_layered_temporal_variable(self, two_faces):
        """`(time, layer, element)`: the full three-axis layout."""
        dataset = _with_dims(
            two_faces,
            "v",
            np.arange(8.0).reshape(2, 2, 2),
            ("time", "mesh2d_nLayers", "mesh2d_nFaces"),
        )

        frame = dataset.to_dataframe(variables=["v"])

        assert len(frame) == 8, f"2 steps x 2 layers x 2 faces = 8, got {len(frame)}"
        assert sorted(frame["value"].tolist()) == [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ]
        assert sorted(frame["element"].unique().tolist()) == [0, 1]
        assert not frame["x"].isna().any()
