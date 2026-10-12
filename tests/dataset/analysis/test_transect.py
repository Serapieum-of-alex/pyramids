"""Tests for :meth:`Dataset.transect` / ``Analysis.transect`` and its helpers.

Covers the along-a-line sampler: `n`- and `step`-based densification, ground
distance measured geodesically (geographic) and via a WGS 84 reprojection
(projected), band selection, agreement with :meth:`Dataset.sample` /
:meth:`Dataset.point` at the same coordinates, out-of-bounds handling, the
returned :class:`FeatureCollection` shape and CRS, and every guard clause. All
fixtures are in-memory ``from_array`` rasters.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from pyramids.base.geodesy import geodesic_distance
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def ramp() -> Dataset:
    """A 1-band 5x5 ramp 0..24, top-left (0, 5), cell 1, EPSG:4326, nodata -9999.

    Row 0 (northernmost, centres at y=4.5) holds 0..4 west-to-east; the leftmost
    column holds 0, 5, 10, 15, 20 north-to-south.

    Returns:
        Dataset: the test raster.
    """
    arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
    return Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
    )


@pytest.fixture(scope="function")
def two_band() -> Dataset:
    """A 2-band 5x5 raster; band 0 holds 0..24, band 1 holds 25..49.

    Top-left (0, 5), cell 1, EPSG:4326, nodata -9999.

    Returns:
        Dataset: the test raster.
    """
    arr = np.arange(2 * 5 * 5, dtype="float32").reshape(2, 5, 5)
    return Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
    )


@pytest.fixture(scope="function")
def utm() -> Dataset:
    """A 1-band 5x5 ramp in UTM 36N (EPSG:32636), cell 100 m, top-left (0, 500000).

    Projected and metric, so a ground ``step`` reaches the reprojection path in
    ``_cumulative_ground_distance``.

    Returns:
        Dataset: the test raster.
    """
    arr = np.arange(25, dtype="float32").reshape(1, 5, 5)
    return Dataset.from_array(
        arr,
        no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0, 500000), cell_size=100.0, epsg=32636),
    )


class TestTransect:
    """Tests for ``Analysis.transect`` via the ``Dataset.transect`` facade."""

    def test_n_mode_values_match_ramp_row(self, ramp):
        """``n`` evenly samples the row; values are the ramp 0..4, distance from 0.

        Test scenario:
            A west-east line across row 0 with ``n=5`` lands on the five cell
            centres, so the values are ``[0, 1, 2, 3, 4]`` and the first distance
            is zero.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        tr = ramp.transect(line, n=5)
        assert isinstance(tr, FeatureCollection), (
            f"expected FeatureCollection, got {type(tr)}"
        )
        assert tr["value"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0], tr["value"].tolist()
        assert float(tr["distance"].iloc[0]) == 0.0, tr["distance"].iloc[0]
        assert len(tr) == 5, f"expected 5 positions, got {len(tr)}"

    def test_distance_is_monotonic_and_ground_metres(self, ramp):
        """Distance rises strictly and equals the geodesic length at the end.

        Test scenario:
            The last ``distance`` equals ``geodesic_distance`` between the line's
            endpoints, confirming the column is cumulative ground metres.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        tr = ramp.transect(line, n=5)
        diffs = tr["distance"].diff().iloc[1:]
        assert bool((diffs > 0).all()), (
            f"distance not strictly increasing: {diffs.tolist()}"
        )
        expected = geodesic_distance(0.5, 4.5, 4.5, 4.5, crs=4326)
        assert float(tr["distance"].iloc[-1]) == pytest.approx(expected), (
            f"end distance {tr['distance'].iloc[-1]} != geodesic {expected}"
        )

    def test_result_carries_raster_crs(self, ramp):
        """The returned points carry the raster's CRS (EPSG:4326).

        Test scenario:
            A geographic raster yields points whose CRS resolves to EPSG:4326.
        """
        tr = ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=3)
        assert tr.crs.to_epsg() == 4326, f"expected 4326, got {tr.crs.to_epsg()}"

    def test_values_match_sample_and_point(self, ramp):
        """The sampled values equal :meth:`sample`/:meth:`point` at the same coords.

        Test scenario:
            Re-sampling the returned points with :meth:`Dataset.sample` and
            reading an endpoint with :meth:`Dataset.point` both reproduce the
            ``value`` column, proving the DoD "values match point/sample".
        """
        tr = ramp.transect(LineString([(0.5, 4.5), (4.5, 0.5)]), n=5)
        resampled = ramp.sample(tr, bands=0)
        assert np.allclose(resampled, tr["value"].to_numpy()), (
            f"sample {resampled} != value column {tr['value'].tolist()}"
        )
        first = tr.geometry.iloc[0]
        assert float(ramp.point(first.x, first.y, band=0)) == tr["value"].iloc[0], (
            "point() disagrees with the first transect value"
        )

    def test_diagonal_line(self, ramp):
        """A diagonal line samples the raster's main diagonal.

        Test scenario:
            The NW-SE diagonal with ``n=5`` hits cells (0,0)..(4,4), giving the
            diagonal values ``[0, 6, 12, 18, 24]``.
        """
        tr = ramp.transect(LineString([(0.5, 4.5), (4.5, 0.5)]), n=5)
        values = tr["value"].tolist()
        assert values == [0.0, 6.0, 12.0, 18.0, 24.0], values

    def test_band_selection(self, two_band):
        """``band=1`` samples the second band.

        Test scenario:
            Band 1 holds 25..49, so row 0 reads ``[25, 26, 27, 28, 29]``.
        """
        tr = two_band.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=5, band=1)
        values = tr["value"].tolist()
        assert values == [25.0, 26.0, 27.0, 28.0, 29.0], values

    def test_step_includes_trailing_endpoint(self, ramp):
        """``step`` not dividing the length still ends exactly on the endpoint.

        Test scenario:
            With a step a little under half the line, the final position is the
            line's end, so the last distance equals the full geodesic length.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        total = float(geodesic_distance(0.5, 4.5, 4.5, 4.5, crs=4326))
        tr = ramp.transect(line, step=total * 0.4)
        assert float(tr["distance"].iloc[-1]) == pytest.approx(total), (
            f"last distance {tr['distance'].iloc[-1]} != total {total}"
        )
        assert len(tr) == 4, (
            f"expected 0, .4, .8, 1.0 of length -> 4 points, got {len(tr)}"
        )

    def test_step_exact_multiple_has_no_duplicate_endpoint(self, ramp):
        """A step that divides the length exactly does not append a duplicate.

        Test scenario:
            ``step = total / 4`` yields exactly five positions (0..total), with no
            repeated trailing distance.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        total = float(geodesic_distance(0.5, 4.5, 4.5, 4.5, crs=4326))
        tr = ramp.transect(line, step=total / 4)
        assert len(tr) == 5, f"expected 5 positions, got {len(tr)}"
        assert tr["distance"].is_monotonic_increasing, tr["distance"].tolist()
        assert tr["distance"].nunique() == 5, "a duplicate endpoint was appended"

    def test_step_longer_than_line_gives_endpoints(self, ramp):
        """A step larger than the line returns just the two endpoints.

        Test scenario:
            ``step`` bigger than the whole ground length still yields start and
            end, never a single point.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5)])
        tr = ramp.transect(line, step=1e9)
        assert len(tr) == 2, f"expected 2 endpoints, got {len(tr)}"
        assert tr["value"].tolist() == [0.0, 4.0], tr["value"].tolist()

    def test_projected_step_is_ground_metres(self, utm):
        """A projected raster's ``step`` is metres of ground via the reproject path.

        Test scenario:
            A 400 m west-east line in UTM sampled every 100 m lands on the five
            cell centres (values 0..4) at ~0, 100, 200, 300, 400 m.
        """
        line = LineString([(50, 499950), (450, 499950)])
        tr = utm.transect(line, step=100.0)
        assert tr["value"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0], tr["value"].tolist()
        # Ground, not planar: 400 m of UTM ~4.5 deg off the central meridian is
        # ~399 m on the ellipsoid (scale distortion), so allow ~1 % slack.
        assert float(tr["distance"].iloc[-1]) == pytest.approx(400.0, rel=0.01), (
            f"end distance {tr['distance'].iloc[-1]} not ~400 m of ground"
        )
        assert tr.crs.to_epsg() == 32636, f"expected 32636, got {tr.crs.to_epsg()}"

    def test_out_of_bounds_positions_get_nodata(self, ramp):
        """Positions past the extent carry the band's no-data value.

        Test scenario:
            A line running east off the raster samples -9999 (the band no-data)
            for the out-of-bounds positions.
        """
        tr = ramp.transect(LineString([(0.5, 4.5), (100.0, 4.5)]), n=3)
        assert tr["value"].iloc[0] == 0.0, tr["value"].iloc[0]
        assert (tr["value"].iloc[1:] == -9999.0).all(), tr["value"].tolist()

    def test_multi_vertex_line_cumulative_distance(self, ramp):
        """A 3-vertex line accumulates distance across both segments.

        Test scenario:
            An L-shaped line's final distance equals the sum of its two segments'
            geodesic lengths.
        """
        line = LineString([(0.5, 4.5), (4.5, 4.5), (4.5, 0.5)])
        tr = ramp.transect(line, n=7)
        seg1 = float(geodesic_distance(0.5, 4.5, 4.5, 4.5, crs=4326))
        seg2 = float(geodesic_distance(4.5, 4.5, 4.5, 0.5, crs=4326))
        assert float(tr["distance"].iloc[-1]) == pytest.approx(seg1 + seg2), (
            f"end {tr['distance'].iloc[-1]} != {seg1 + seg2}"
        )

    def test_minimum_two_positions(self, ramp):
        """``n=2`` returns exactly the two endpoints.

        Test scenario:
            The smallest valid count yields the line's start and end only, with
            the start value and the end value of the sampled row.
        """
        tr = ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=2)
        assert len(tr) == 2, f"expected 2 positions, got {len(tr)}"
        assert tr["value"].tolist() == [0.0, 4.0], tr["value"].tolist()
        assert float(tr["distance"].iloc[0]) == 0.0, tr["distance"].iloc[0]

    def test_three_dimensional_line_drops_z(self, ramp):
        """A LineString carrying z-coordinates is sampled on its x/y only.

        Test scenario:
            The z ordinate is ignored, so a 3-D west-east line gives the same
            values as its 2-D projection.
        """
        line3d = LineString([(0.5, 4.5, 100.0), (4.5, 4.5, 200.0)])
        tr = ramp.transect(line3d, n=5)
        assert tr["value"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0], tr["value"].tolist()
        assert not tr.geometry.iloc[0].has_z, (
            "z should have been dropped from the output"
        )

    def test_band_out_of_range_raises(self, ramp):
        """A band index past the raster is surfaced as the sampler's ValueError.

        Test scenario:
            ``band=5`` on a single-band raster propagates ``sample``'s band-range
            ValueError rather than being swallowed.
        """
        with pytest.raises(ValueError):
            ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=3, band=5)


class TestTransectValidation:
    """Guard clauses on ``transect`` arguments (``_validate_transect_args``)."""

    def test_neither_n_nor_step(self, ramp):
        """Omitting both ``n`` and ``step`` raises.

        Test scenario:
            Exactly one of the two is required; neither is a ValueError naming it.
        """
        with pytest.raises(ValueError, match="exactly one of n= or step=") as exc:
            ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]))
        assert "neither" in str(exc.value), str(exc.value)

    def test_both_n_and_step(self, ramp):
        """Giving both ``n`` and ``step`` raises.

        Test scenario:
            The mutually exclusive pair rejects being supplied together.
        """
        with pytest.raises(ValueError, match="exactly one of n= or step=") as exc:
            ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=5, step=30)
        assert "both" in str(exc.value), str(exc.value)

    def test_n_below_two(self, ramp):
        """``n < 2`` raises.

        Test scenario:
            A single position cannot form a transect.
        """
        with pytest.raises(ValueError, match="at least 2"):
            ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), n=1)

    def test_non_positive_step(self, ramp):
        """``step <= 0`` raises.

        Test scenario:
            A zero or negative ground spacing is refused.
        """
        with pytest.raises(ValueError, match="must be positive"):
            ramp.transect(LineString([(0.5, 4.5), (4.5, 4.5)]), step=0)

    def test_not_a_linestring(self, ramp):
        """A non-LineString geometry raises.

        Test scenario:
            A shapely Point is not a line and is rejected by type.
        """
        with pytest.raises(ValueError, match="needs a shapely LineString"):
            ramp.transect(Point(0.5, 4.5), n=5)

    def test_empty_line(self, ramp):
        """An empty LineString raises.

        Test scenario:
            A line with no vertices cannot be densified.
        """
        with pytest.raises(ValueError, match="at least two vertices"):
            ramp.transect(LineString(), n=5)

    def test_zero_ground_length_line(self, ramp):
        """A line whose vertices are coincident raises on zero ground length.

        Test scenario:
            Two identical points have length zero, so densification is impossible.
        """
        with pytest.raises(ValueError, match="zero ground length"):
            ramp.transect(LineString([(2.5, 2.5), (2.5, 2.5)]), n=5)
