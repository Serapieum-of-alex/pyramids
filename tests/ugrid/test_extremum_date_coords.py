"""`idxmin` / `idxmax` against date-like time coordinates.

The coordinate form answers float64, so a `Timestamp` coordinate cannot be expressed. It used to
surface as a raw `TypeError` from inside numpy's float cast — "float() argument must be ... not
'Timestamp'" — on an entirely ordinary dated dataset, while the docstrings declared only
`ValueError`. The position form (`argmin` / `argmax`) is unaffected and is the right answer there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
VALUES = np.array([[1.0, 9.0], [5.0, 2.0], [3.0, 7.0]])


def _dataset(time_values: list | None = None) -> UgridDataset:
    """A three-step mesh variable, optionally dated."""
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
def dated() -> UgridDataset:
    """Three consecutive days as the time coordinate."""
    return _dataset(
        [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
        ]
    )


class TestDateLikeCoordinatesAreRefusedClearly:
    def test_idxmin_raises_a_value_error(self, dated):
        with pytest.raises(ValueError, match="needs numeric"):
            dated.idxmin()

    def test_idxmax_raises_a_value_error(self, dated):
        with pytest.raises(ValueError, match="needs numeric"):
            dated.idxmax()

    def test_the_message_points_at_the_position_form(self, dated):
        with pytest.raises(ValueError, match="argmin"):
            dated.idxmin()


class TestPositionFormStillWorks:
    def test_argmin_is_unaffected(self, dated):
        assert np.asarray(dated.argmin()["v"].data).tolist() == [0, 1]

    def test_argmax_is_unaffected(self, dated):
        assert np.asarray(dated.argmax()["v"].data).tolist() == [1, 0]


class TestNumericCoordinatesStillWork:
    def test_idxmin_answers_the_coordinate(self):
        assert np.asarray(_dataset().idxmin()["v"].data).tolist() == [0.0, 1.0]

    def test_idxmax_answers_the_coordinate(self):
        assert np.asarray(_dataset().idxmax()["v"].data).tolist() == [1.0, 0.0]

    def test_explicit_numeric_stamps_are_used(self):
        dataset = _dataset([10.0, 20.0, 30.0])
        assert np.asarray(dataset.idxmin()["v"].data).tolist() == [10.0, 20.0]
