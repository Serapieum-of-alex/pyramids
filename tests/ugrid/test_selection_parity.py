"""A1 — element-axis `isel` and nearest-by-label `sel` on UgridDataset."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


def _two_triangles(data: dict | None = None) -> UgridDataset:
    """A unit square split into two triangles, optionally carrying data."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data=data if data is not None else {"depth": np.array([2.0, 4.0])},
    )


def _temporal(stamps: list[float]) -> UgridDataset:
    """A two-face mesh with a temporal variable and explicit time coordinates."""
    mesh = UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.arange(len(stamps) * 2, dtype="float64").reshape(len(stamps), 2)},
        data_locations={"d": "face"},
    )
    mesh["d"].attributes = {**mesh["d"].attributes, "time_values": list(stamps)}
    return mesh


class TestIselElement:
    def test_single_face_renumbers_the_mesh(self):
        out = _two_triangles().isel(element=0)
        assert (out.n_face, out.n_node) == (1, 3)
        assert out["depth"].data.tolist() == [2.0]

    def test_sequence_keeps_order_and_values(self):
        out = _two_triangles().isel(element=[1, 0])
        assert out.n_face == 2
        assert out["depth"].data.tolist() == [4.0, 2.0]

    def test_slice_selects_faces(self):
        out = _two_triangles().isel(element=slice(0, 1))
        assert out.n_face == 1

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError, match="out of range"):
            _two_triangles().isel(element=5)

    def test_neither_axis_raises(self):
        with pytest.raises(ValueError, match="needs `time`, `element`, or both"):
            _two_triangles().isel()

    def test_time_still_positional_and_unbroken(self):
        # The pre-existing positional time call must keep working.
        out = _temporal([0.0, 1.0, 2.0]).isel(1)
        assert out["d"].has_time is False
        assert out["d"].data.tolist() == [2.0, 3.0]

    def test_element_then_time_compose(self):
        out = _temporal([0.0, 1.0, 2.0]).isel(1, element=0)
        assert out.n_face == 1
        assert out["d"].data.tolist() == [2.0]


class TestSelNearest:
    def test_exact_match_unchanged(self):
        out = _temporal([0.0, 10.0, 20.0]).sel(10.0)
        assert out["d"].data.tolist() == [2.0, 3.0]

    def test_missing_value_raises_valueerror(self):
        with pytest.raises(ValueError, match="not found in the time coordinate"):
            _temporal([0.0, 10.0, 20.0]).sel(7.0)

    def test_nearest_snaps(self):
        out = _temporal([0.0, 10.0, 20.0]).sel(7.0, method="nearest")
        assert out["d"].data.tolist() == [2.0, 3.0]

    def test_nearest_within_tolerance(self):
        out = _temporal([0.0, 10.0, 20.0]).sel(12.0, method="nearest", tolerance=5.0)
        assert out["d"].data.tolist() == [2.0, 3.0]

    def test_tolerance_breach_raises_keyerror(self):
        with pytest.raises(KeyError, match="within tolerance"):
            _temporal([0.0, 10.0, 20.0]).sel(7.0, method="nearest", tolerance=1.0)

    def test_sequence_of_values(self):
        out = _temporal([0.0, 10.0, 20.0]).sel([0.0, 20.0], method="nearest")
        assert out["d"].has_time is True
        assert out["d"].data.tolist() == [[0.0, 1.0], [4.0, 5.0]]

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method must be None or 'nearest'"):
            _temporal([0.0, 1.0]).sel(0.0, method="bogus")

    def test_tolerance_without_nearest_rejected(self):
        with pytest.raises(ValueError, match="only valid with method='nearest'"):
            _temporal([0.0, 1.0]).sel(0.0, tolerance=1.0)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="must not be negative"):
            _temporal([0.0, 1.0]).sel(0.0, method="nearest", tolerance=-1.0)
