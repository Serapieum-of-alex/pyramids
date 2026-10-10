"""Every spelling of the class-or-instance joins.

`concat` / `merge` are reachable four ways: on the class with a list, on the class with a single
dataset, on an instance with a list, and on an instance with a single dataset. A `UgridDataset` is
iterable over its variable names, so the class path turned a bare dataset into a list of strings
before this was pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _dataset(values: list[list[float]], name: str = "v") -> UgridDataset:
    """A two-face mesh carrying one temporal variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={name: np.array(values)},
        data_locations={name: "face"},
    )


@pytest.fixture
def first() -> UgridDataset:
    """One time step of values."""
    return _dataset([[1.0, 2.0]])


@pytest.fixture
def second() -> UgridDataset:
    """A second, later time step."""
    return _dataset([[3.0, 4.0]])


def _steps(dataset: UgridDataset, name: str = "v") -> int:
    """The number of time steps a variable carries."""
    return int(np.asarray(dataset[name].data).shape[0])


class TestConcatSpellings:
    def test_class_call_with_a_list(self, first, second):
        assert _steps(UgridDataset.concat([first, second])) == 2

    def test_class_call_with_a_single_dataset(self, first):
        # A bare dataset is iterable over variable names; it must not be treated as a list.
        assert _steps(UgridDataset.concat(first)) == 1

    def test_instance_call_with_a_list(self, first, second):
        assert _steps(first.concat([second])) == 2

    def test_instance_call_with_a_single_dataset(self, first, second):
        assert _steps(first.concat(second)) == 2

    def test_an_empty_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one dataset"):
            UgridDataset.concat([])


class TestMergeSpellings:
    def test_class_call_with_a_single_dataset(self, first):
        merged = UgridDataset.merge(first)
        assert set(merged) == {"v"}

    def test_class_call_with_a_list(self, first):
        other = _dataset([[9.0, 9.0]], name="w")
        merged = UgridDataset.merge([first, other])
        assert set(merged) == {"v", "w"}

    def test_instance_call_with_a_single_dataset(self, first):
        other = _dataset([[9.0, 9.0]], name="w")
        assert set(first.merge(other)) == {"v", "w"}

    def test_compat_override_is_reachable_from_the_class(self, first):
        clashing = _dataset([[7.0, 7.0]])
        merged = UgridDataset.merge([first, clashing], compat="override")
        assert np.asarray(merged["v"].data).tolist() == [[1.0, 2.0]]
