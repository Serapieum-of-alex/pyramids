"""A7 — argmin / argmax / idxmin / idxmax along the mesh time axis."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


def _temporal(values: np.ndarray, stamps: list[float] | None = None) -> UgridDataset:
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


class TestArgExtremum:
    def test_argmin_picks_the_step_of_the_minimum(self):
        values = np.array([[5.0, 1.0], [3.0, 2.0], [9.0, 8.0]])
        out = _temporal(values).argmin()
        assert out["d"].has_time is False
        assert out["d"].data.tolist() == [1, 0]

    def test_argmax_picks_the_step_of_the_maximum(self):
        values = np.array([[5.0, 1.0], [3.0, 2.0], [9.0, 8.0]])
        out = _temporal(values).argmax()
        assert out["d"].data.tolist() == [2, 2]

    def test_result_is_int64_and_declares_minus_one(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _temporal(values).argmin()
        assert out["d"].data.dtype == np.int64
        assert out["d"].nodata == -1

    def test_gaps_are_skipped(self):
        values = np.array([[np.nan, 5.0], [2.0, np.nan], [7.0, 1.0]])
        out = _temporal(values).argmin()
        assert out["d"].data.tolist() == [1, 2]

    def test_all_gap_element_is_minus_one(self):
        values = np.array([[np.nan, 5.0], [np.nan, 2.0]])
        out = _temporal(values).argmin()
        assert out["d"].data.tolist() == [-1, 1]

    def test_no_time_dimension_raises(self):
        static = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
        )
        with pytest.raises(ValueError, match="needs at least one variable with a time"):
            static.argmin()


class TestIdxExtremum:
    def test_idxmin_answers_the_time_coordinate(self):
        values = np.array([[5.0, 1.0], [3.0, 2.0], [9.0, 8.0]])
        out = _temporal(values, stamps=[100.0, 200.0, 300.0]).idxmin()
        assert out["d"].data.tolist() == [200.0, 100.0]

    def test_idxmax_answers_the_time_coordinate(self):
        values = np.array([[5.0, 1.0], [3.0, 2.0], [9.0, 8.0]])
        out = _temporal(values, stamps=[100.0, 200.0, 300.0]).idxmax()
        assert out["d"].data.tolist() == [300.0, 300.0]

    def test_all_gap_element_is_nan(self):
        values = np.array([[np.nan, 5.0], [np.nan, 2.0]])
        out = _temporal(values, stamps=[10.0, 20.0]).idxmin()
        row = out["d"].data
        assert np.isnan(row[0])
        assert row[1] == 20.0

    def test_falls_back_to_step_positions_without_stamps(self):
        values = np.array([[5.0, 1.0], [3.0, 2.0]])
        out = _temporal(values).idxmin()
        assert out["d"].data.tolist() == [1.0, 0.0]
