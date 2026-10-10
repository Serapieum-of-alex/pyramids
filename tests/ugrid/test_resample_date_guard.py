"""`resample` must refuse a positional time axis instead of binning it as 1970.

A mesh with no decoded time coordinate falls back to step positions `0, 1, 2, …`.
`pd.to_datetime([0, 1, 2])` happily reads those as *nanoseconds* since the epoch, so every step
landed inside the same calendar month and `resample("1MS")` silently collapsed the axis to a
single 1970 bin — while the docstring promised a `ValueError` for coordinates that are not
date-like. Binning by a calendar frequency is meaningless on a positional axis, so it has to say
so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
VALUES = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def _dataset(time_values: list | None = None) -> UgridDataset:
    """A three-step mesh variable, optionally stamped with real time coordinates."""
    dataset = UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": VALUES},
        data_locations={"v": "face"},
    )
    if time_values is not None:
        dataset["v"].attributes["time_values"] = time_values
    return dataset


@pytest.fixture
def positional() -> UgridDataset:
    """A dataset whose time coordinate is just step positions."""
    return _dataset()


@pytest.fixture
def dated() -> UgridDataset:
    """A dataset spanning two calendar months."""
    return _dataset(
        [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-15"),
            pd.Timestamp("2020-02-01"),
        ]
    )


class TestPositionalAxisIsRefused:
    def test_resample_raises_on_a_positional_axis(self, positional):
        assert positional.time_values == [0, 1, 2]
        with pytest.raises(ValueError, match="date-like"):
            positional.resample("1MS")

    def test_the_message_points_at_the_alternative(self, positional):
        with pytest.raises(ValueError, match="label sequence"):
            positional.resample("1MS")

    def test_reduce_with_an_offset_alias_is_refused_too(self, positional):
        with pytest.raises(ValueError, match="date-like"):
            positional.reduce("mean", groupby="1MS")

    def test_a_float_positional_axis_is_refused(self):
        dataset = _dataset([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="date-like"):
            dataset.resample("1MS")


class TestDateLikeAxisStillWorks:
    def test_two_months_give_two_bins(self, dated):
        resampled = dated.resample("1MS")
        assert np.asarray(resampled["v"].data).shape[0] == 2

    def test_the_bins_hold_the_right_means(self, dated):
        resampled = dated.resample("1MS")
        # January holds steps 0 and 1; February holds step 2.
        assert np.asarray(resampled["v"].data)[0].tolist() == pytest.approx([2.0, 3.0])
        assert np.asarray(resampled["v"].data)[1].tolist() == pytest.approx([5.0, 6.0])

    def test_a_label_sequence_works_on_a_positional_axis(self, positional):
        grouped = positional.reduce("mean", groupby=["a", "a", "b"])
        assert np.asarray(grouped["v"].data).shape[0] == 2
