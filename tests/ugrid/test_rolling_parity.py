"""A3 — rolling's `dim` guard, `q` for quantile, and the full-window `min_periods` default."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def mesh() -> UgridDataset:
    """A two-face mesh with four time steps of a face variable."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])},
        data_locations={"d": "face"},
    )


class TestRollingDim:
    def test_time_dim_is_accepted(self, mesh):
        out = mesh.rolling(2, "mean", dim="time")["d"].data
        assert out[1].tolist() == [2.0, 3.0]

    def test_element_dim_is_refused(self, mesh):
        with pytest.raises(ValueError, match="rolls along 'time' only"):
            mesh.rolling(2, "mean", dim="nMesh2d_face")


class TestRollingQuantile:
    def test_quantile_needs_q(self, mesh):
        with pytest.raises(ValueError, match="needs q in"):
            mesh.rolling(2, "quantile")

    def test_q_only_with_quantile(self, mesh):
        with pytest.raises(ValueError, match="only valid with how='quantile'"):
            mesh.rolling(2, "mean", q=0.5)

    def test_quantile_window(self, mesh):
        # A 2-step trailing window's median is the mean of the pair.
        out = mesh.rolling(2, "quantile", q=0.5)["d"].data
        assert out[1].tolist() == [2.0, 3.0]
        assert out[3].tolist() == [6.0, 7.0]


class TestMinPeriods:
    def test_default_is_the_full_window(self, mesh):
        out = mesh.rolling(3, "mean")["d"].data
        assert np.isnan(out[0]).all()
        assert np.isnan(out[1]).all()
        assert out[2].tolist() == [3.0, 4.0]

    def test_explicit_one_keeps_partials(self, mesh):
        out = mesh.rolling(3, "mean", min_periods=1)["d"].data
        assert out[0].tolist() == [1.0, 2.0]

    def test_above_window_is_refused(self, mesh):
        with pytest.raises(ValueError, match="between 1 and window"):
            mesh.rolling(2, "mean", min_periods=5)

    def test_below_one_is_refused(self, mesh):
        with pytest.raises(ValueError, match="between 1 and window"):
            mesh.rolling(2, "mean", min_periods=0)

    def test_window_below_one_is_refused(self, mesh):
        with pytest.raises(ValueError, match="window must be >= 1"):
            mesh.rolling(0, "mean")
