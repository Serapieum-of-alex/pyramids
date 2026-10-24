"""k-ring focal statistics over the mesh face adjacency (issue #1376).

The raster ``focal_*`` family aggregates over a ``(2*radius+1)`` pixel window, which has no
mesh equivalent. The mesh analogue is the k-ring face neighbourhood -- hence ``rings``, not
``radius``. These tests hand-compute every expected value on a four-face strip, where the
neighbourhoods are small enough to check by eye.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.focal import neighbourhood

pytestmark = pytest.mark.core

# A four-face strip: faces 0-1-2-3 in a chain, so the 1-ring neighbourhoods are
# {0,1}, {0,1,2}, {1,2,3}, {2,3} - small enough to verify by hand.
_NODE_X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])


def _strip(values, nodata=None, active=None):
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values, dtype="float64")},
        data_locations={"v": "face"},
    )
    if nodata is not None:
        ds["v"].nodata = nodata
    if active is not None:
        ds["v"].active = np.asarray(active, dtype=bool)
    return ds


class TestNeighbourhood:
    """The k-ring reach matrix the aggregations are built on."""

    def test_one_ring_includes_the_centre(self):
        """The centre face is in its own neighbourhood, as a raster window includes its centre."""
        reach = neighbourhood(_strip([0.0] * 4).mesh, 1).toarray().astype(int)
        assert reach.tolist() == [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]

    def test_two_rings_extends_by_one_hop(self):
        reach = neighbourhood(_strip([0.0] * 4).mesh, 2).toarray().astype(int)
        assert reach[0].tolist() == [1, 1, 1, 0]
        assert reach[1].tolist() == [1, 1, 1, 1]

    def test_rings_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            neighbourhood(_strip([0.0] * 4).mesh, 0)


class TestFocalMean:
    """Hand-computed means over the strip."""

    def test_one_ring(self):
        """mean{0,10}=5, mean{0,10,20}=10, mean{10,20,30}=20, mean{20,30}=25."""
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_mean().tolist() == pytest.approx(
            [5.0, 10.0, 20.0, 25.0]
        )

    def test_boundary_faces_use_only_what_they_have(self):
        """No padding: face 0 averages over two faces, not a fabricated three."""
        result = _strip([0.0, 10.0, 20.0, 30.0]).focal_mean()
        assert result[0] == pytest.approx(5.0), "padding would have skewed this"

    def test_two_rings(self):
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_mean(2).tolist() == pytest.approx(
            [10.0, 15.0, 15.0, 20.0]
        )

    def test_a_constant_field_is_unchanged(self):
        assert _strip([7.0] * 4).focal_mean().tolist() == pytest.approx([7.0] * 4)

    def test_nodata_neighbours_are_excluded(self):
        """Face 1's neighbourhood is {0,1,2}; with face 0 nodata it averages {10,20}."""
        ds = _strip([-999.0, 10.0, 20.0, 30.0], nodata=-999.0)
        assert ds.focal_mean()[1] == pytest.approx(15.0)

    def test_an_all_nodata_neighbourhood_is_nan(self):
        ds = _strip([-999.0, -999.0, 20.0, 30.0], nodata=-999.0)
        assert np.isnan(ds.focal_mean()[0]), "nothing usable to average"

    def test_inactive_neighbours_are_excluded(self):
        """A dry neighbour should not drag a wet face's mean down."""
        ds = _strip([0.0, 10.0, 20.0, 30.0], active=[False, True, True, True])
        assert ds.focal_mean()[1] == pytest.approx(15.0)

    def test_refuses_a_temporal_variable(self):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"v": np.zeros((3, 4))},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError, match="one value per face"):
            ds.focal_mean()


class TestFocalStd:
    """Population standard deviation over the same neighbourhoods."""

    def test_one_ring(self):
        """std{0,10}=5, std{0,10,20}=8.165 (population, not sample)."""
        result = _strip([0.0, 10.0, 20.0, 30.0]).focal_std()
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(np.std([0.0, 10.0, 20.0]))

    def test_a_constant_neighbourhood_is_exactly_zero(self):
        """The sums identity can land a hair below zero on a constant field; it is clamped,
        so the square root never sees a negative."""
        result = _strip([7.0] * 4).focal_std()
        assert result.tolist() == pytest.approx([0.0] * 4)
        assert not np.isnan(result).any()

    def test_all_nodata_is_nan(self):
        ds = _strip([-999.0, -999.0, -999.0, -999.0], nodata=-999.0)
        assert np.isnan(ds.focal_std()).all()


class TestFocalApply:
    """An arbitrary aggregation over the same neighbourhoods."""

    def test_peak_to_peak(self):
        """ptp{0,10}=10, ptp{0,10,20}=20, ptp{10,20,30}=20, ptp{20,30}=10."""
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_apply(
            np.ptp
        ).tolist() == pytest.approx([10.0, 20.0, 20.0, 10.0])

    def test_median(self):
        assert _strip([0.0, 10.0, 20.0, 30.0]).focal_apply(np.median)[
            1
        ] == pytest.approx(10.0)

    def test_matches_focal_mean_when_given_mean(self):
        """The arbitrary path and the vectorised one must agree on the same statistic."""
        ds = _strip([0.0, 10.0, 20.0, 30.0])
        assert ds.focal_apply(np.mean).tolist() == pytest.approx(
            ds.focal_mean().tolist()
        )

    def test_func_never_sees_an_empty_array(self):
        """An all-nodata neighbourhood yields NaN without calling func."""
        ds = _strip([-999.0, -999.0, -999.0, -999.0], nodata=-999.0)

        def explode(_values):
            raise AssertionError("func must not be called on an empty neighbourhood")

        assert np.isnan(ds.focal_apply(explode)).all()
