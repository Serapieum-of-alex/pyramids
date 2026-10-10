"""A6 — concat/merge: AlignmentError, merge compat, and the class-or-instance call shape."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _temporal(values: np.ndarray, name: str = "d") -> UgridDataset:
    """A two-face mesh carrying `values` of shape (n_time, 2) under `name`."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=FACES,
        data={name: values},
        data_locations={name: "face"},
    )


def _other_mesh() -> UgridDataset:
    """A mesh with different node coordinates, so topology comparison must fail."""
    return UgridDataset.from_arrays(
        node_x=NODE_X + 10.0,
        node_y=NODE_Y,
        face_node_connectivity=FACES,
        data={"d": np.array([[1.0, 1.0]])},
        data_locations={"d": "face"},
    )


class TestAlignmentError:
    def test_concat_on_a_different_mesh_raises_alignmenterror(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        with pytest.raises(AlignmentError, match="same mesh topology"):
            a.concat([_other_mesh()])

    def test_merge_on_a_different_mesh_raises_alignmenterror(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        with pytest.raises(AlignmentError, match="same mesh topology"):
            a.merge([_other_mesh()])

    def test_alignmenterror_is_not_a_valueerror(self):
        # Deliberate break, matching the raster side: AlignmentError derives from the package
        # error, NOT from ValueError, so an old `except ValueError` no longer catches it.
        a = _temporal(np.array([[1.0, 1.0]]))
        with pytest.raises(AlignmentError) as caught:
            a.concat([_other_mesh()])
        assert not isinstance(caught.value, ValueError)


class TestMergeCompat:
    def test_no_conflicts_refuses_a_duplicate(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]))
        with pytest.raises(ValueError, match="present in more than one dataset"):
            a.merge([b])

    def test_override_keeps_the_first(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]))
        out = a.merge([b], compat="override")
        assert out["d"].data.tolist() == [[1.0, 1.0]]

    def test_unknown_compat_refused(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]), name="e")
        with pytest.raises(ValueError, match="compat must be"):
            a.merge([b], compat="bogus")

    def test_distinct_names_still_merge(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]), name="e")
        out = a.merge([b])
        assert sorted(out.data_variable_names) == ["d", "e"]


class TestCallShapes:
    def test_concat_as_classmethod(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]))
        out = UgridDataset.concat([a, b])
        assert out["d"].data.tolist() == [[1.0, 1.0], [2.0, 2.0]]

    def test_concat_as_instance_method(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]))
        out = a.concat([b])
        assert out["d"].data.tolist() == [[1.0, 1.0], [2.0, 2.0]]

    def test_both_shapes_agree(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]))
        assert (
            UgridDataset.concat([a, b])["d"].data.tolist()
            == a.concat([b])["d"].data.tolist()
        )

    def test_merge_as_classmethod(self):
        a = _temporal(np.array([[1.0, 1.0]]))
        b = _temporal(np.array([[2.0, 2.0]]), name="e")
        out = UgridDataset.merge([a, b])
        assert sorted(out.data_variable_names) == ["d", "e"]

    def test_empty_class_call_refused(self):
        with pytest.raises(ValueError, match="at least one dataset"):
            UgridDataset.concat([])
