"""Every spelling of the class-or-instance joins.

`concat` / `merge` are reachable four ways: on the class with a list, on the class with a single
dataset, on an instance with a list, and on an instance with a single dataset. A `UgridDataset` is
iterable over its variable names, so the class path turned a bare dataset into a list of strings
before this was pinned.
"""

from __future__ import annotations

import inspect

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

    def test_override_keeps_the_first_carrier_not_self(self, first):
        """When `self` does not carry the name, the first of `others` that does wins.

        Test scenario:
            The docstring said override "keeps `self`'s", which only holds when `self` has the
            variable. Merging a dataset carrying only `v` with two that both carry `w` keeps
            the earlier `w`, which belongs to neither `self` nor the last dataset.
        """
        earlier = _dataset([[3.0, 3.0]], name="w")
        later = _dataset([[8.0, 8.0]], name="w")

        merged = first.merge([earlier, later], compat="override")

        assert np.asarray(merged["w"].data).tolist() == [[3.0, 3.0]], (
            f"the first dataset carrying 'w' should win, got "
            f"{np.asarray(merged['w'].data).tolist()}"
        )


class TestJoinIntrospection:
    def test_the_instance_join_keeps_its_signature(self, first):
        """The bound method advertises exactly what it takes."""
        parameters = list(inspect.signature(first.merge).parameters)
        assert parameters == ["others", "compat"], (
            f"the instance call takes the datasets to join plus compat, got {parameters}"
        )

    def test_the_class_join_advertises_the_class_spelling(self):
        """The class call takes the whole join as one positional argument, and no `self`.

        Test scenario:
            `functools.wraps` sets `__wrapped__`, which `inspect.signature` follows, so the
            class call used to advertise the undecorated `(self, others, *, compat=...)`. That
            is the instance signature with a stray leading `self` -- wrong for
            `UgridDataset.merge([a, b])`, where the list *is* the first argument. Asserting
            only that "compat" appears passed against it, because `compat` is in both.
        """
        signature = inspect.signature(UgridDataset.merge)
        parameters = list(signature.parameters)

        assert "self" not in parameters, (
            f"a class call has no self to pass, yet the signature asks for one: {signature}"
        )
        assert parameters == ["others", "compat"], (
            f"expected the join list then compat, got {parameters}"
        )
        assert (
            signature.parameters["others"].kind
            is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ), "the join list is passed positionally, as UgridDataset.merge([a, b]) does"

    def test_the_bound_join_compares_equal_like_a_method(self, first):
        # Two separate attribute accesses, as an ordinary bound method would be compared.
        one = first.concat
        another = first.concat
        assert one == another

    def test_the_join_keeps_its_name(self, first):
        assert first.merge.__name__ == "merge"

    def test_the_join_keeps_its_docstring(self, first):
        """The bound join carries `merge`'s own text, not a wrapper's.

        Test scenario:
            `__doc__ is not None` holds for almost any callable, including a bare closure with
            an accidental docstring, so it could not see the descriptor losing the real one.
            Pinning a phrase from `merge`'s summary is what makes it fail if that happens.
        """
        doc = first.merge.__doc__

        assert doc is not None, "the bound join should carry a docstring"
        assert "Merge the variables" in doc, (
            f"expected merge's own summary, got {doc!r}"
        )
        assert doc == UgridDataset.merge.__doc__, (
            "the bound and class spellings should show the same documentation"
        )
