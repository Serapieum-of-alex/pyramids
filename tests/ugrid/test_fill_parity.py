"""A5 — the fill family's `dim` guard, dropna's drop-all guard, and `use_coordinate`."""

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


GAPPY = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])


class TestDimGuard:
    @pytest.mark.parametrize(
        "operation", ["ffill", "bfill", "interpolate_na", "dropna"]
    )
    def test_time_is_accepted(self, operation):
        mesh = _temporal(GAPPY)
        getattr(mesh, operation)(dim="time")

    @pytest.mark.parametrize(
        "operation", ["ffill", "bfill", "interpolate_na", "dropna"]
    )
    def test_element_dim_is_refused(self, operation):
        mesh = _temporal(GAPPY)
        with pytest.raises(ValueError, match="one along-able dimension, 'time'"):
            getattr(mesh, operation)(dim="nMesh2d_face")


class TestDropnaAllGuard:
    def test_dropping_every_step_is_refused(self):
        allnan = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        with pytest.raises(ValueError, match="would drop every one of the"):
            _temporal(allnan).dropna()

    def test_partial_drop_still_works(self):
        out = _temporal(GAPPY).dropna()
        assert out["d"].data.tolist() == [[1.0, 1.0], [3.0, 3.0]]


class TestUseCoordinate:
    def test_uneven_coordinate_weights_the_gap(self):
        # Stamps 0, 1, 5: the gap at t=1 sits one unit from 1.0 and four from 3.0, so a
        # distance-weighted fill gives 1.4, where even spacing would give 2.0.
        out = _temporal(GAPPY, stamps=[0.0, 1.0, 5.0]).interpolate_na()
        assert out["d"].data[1].tolist() == pytest.approx([1.4, 1.4])

    def test_use_coordinate_false_uses_positions(self):
        out = _temporal(GAPPY, stamps=[0.0, 1.0, 5.0]).interpolate_na(
            use_coordinate=False
        )
        assert out["d"].data[1].tolist() == pytest.approx([2.0, 2.0])

    def test_even_coordinate_matches_positions(self):
        out = _temporal(GAPPY, stamps=[0.0, 1.0, 2.0]).interpolate_na()
        assert out["d"].data[1].tolist() == pytest.approx([2.0, 2.0])

    def test_non_numeric_stamps_fall_back_to_positions(self):
        out = _temporal(GAPPY, stamps=["a", "b", "c"]).interpolate_na()
        assert out["d"].data[1].tolist() == pytest.approx([2.0, 2.0])
