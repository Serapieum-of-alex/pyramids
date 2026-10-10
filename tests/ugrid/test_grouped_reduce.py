"""A2 — grouped `reduce(groupby=...)` and the `resample` sugar on UgridDataset."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


def _temporal(values: np.ndarray, stamps: list | None = None) -> UgridDataset:
    """A two-face mesh carrying `values` of shape (n_time, 2) along time."""
    mesh = UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": values},
        data_locations={"d": "face"},
    )
    if stamps is not None:
        mesh["d"].attributes = {**mesh["d"].attributes, "time_values": list(stamps)}
    return mesh


class TestGroupbyLabels:
    def test_label_sequence_groups_and_keeps_the_axis(self):
        values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
        out = _temporal(values).reduce("mean", groupby=["a", "a", "b", "b"])
        assert out["d"].has_time is True
        assert out["d"].data.tolist() == [[2.0, 20.0], [6.0, 60.0]]

    def test_group_labels_become_the_time_coordinate(self):
        values = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        out = _temporal(values).reduce("sum", groupby=["x", "y", "y"])
        assert out["d"].attributes["time_values"] == ["x", "y"]
        assert out["d"].data.tolist() == [[1.0, 1.0], [8.0, 8.0]]

    def test_first_appearance_order_is_kept(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = _temporal(values).reduce("max", groupby=["late", "early", "late"])
        assert out["d"].attributes["time_values"] == ["late", "early"]
        assert out["d"].data.tolist() == [[3.0, 3.0], [2.0, 2.0]]

    def test_gaps_are_skipped_within_a_group(self):
        values = np.array([[np.nan, 1.0], [4.0, 3.0]])
        out = _temporal(values).reduce("mean", groupby=["g", "g"])
        assert out["d"].data.tolist() == [[4.0, 2.0]]

    def test_wrong_label_count_raises(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(ValueError, match="one label per time step"):
            _temporal(values).reduce("mean", groupby=["only-one"])

    def test_ungrouped_reduce_still_collapses(self):
        values = np.array([[1.0, 1.0], [3.0, 3.0]])
        out = _temporal(values).reduce("mean")
        assert out["d"].has_time is False
        assert out["d"].data.tolist() == [2.0, 2.0]


class TestResample:
    def test_monthly_bins(self):
        stamps = ["2020-01-01", "2020-01-15", "2020-02-01"]
        values = np.array([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0]])
        out = _temporal(values, stamps=stamps).resample("1MS", "mean")
        assert out["d"].data.tolist() == [[2.0, 2.0], [10.0, 10.0]]
        assert len(out["d"].attributes["time_values"]) == 2

    def test_resample_defaults_to_mean(self):
        stamps = ["2020-01-01", "2020-01-02"]
        values = np.array([[2.0, 2.0], [4.0, 4.0]])
        out = _temporal(values, stamps=stamps).resample("1MS")
        assert out["d"].data.tolist() == [[3.0, 3.0]]

    def test_empty_bins_are_dropped(self):
        # Jan and Mar only: the Feb bin is empty and must not become an all-gap step.
        stamps = ["2020-01-01", "2020-03-01"]
        values = np.array([[1.0, 1.0], [5.0, 5.0]])
        out = _temporal(values, stamps=stamps).resample("1MS", "sum")
        assert out["d"].data.tolist() == [[1.0, 1.0], [5.0, 5.0]]

    def test_non_date_coordinates_raise(self):
        values = np.array([[1.0, 1.0], [2.0, 2.0]])
        with pytest.raises(ValueError, match="date-like time coordinates"):
            _temporal(values, stamps=["not-a-date", "also-not"]).resample("1MS")
