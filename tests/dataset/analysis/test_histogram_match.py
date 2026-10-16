"""Tests for :meth:`Dataset.match_histogram` and :meth:`Dataset.percentile_stretch`.

Covers the histogram-matching remap (CDF alignment onto a reference band, band
selection, no-data / NaN exclusion from both distributions and preservation in
the output, facade/engine equivalence) and the single-raster percentile stretch
(full-range and partial stretch, clipping, no-data handling), plus every guard
clause. All fixtures are in-memory ``from_array`` rasters.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(31.0, 30.0), cell_size=0.01, epsg=4326)


@pytest.fixture
def scene_pair() -> tuple[Dataset, Dataset]:
    """A reference raster and a brighter, differently-calibrated copy of one scene.

    Returns:
        tuple[Dataset, Dataset]: ``(reference, target)`` over the same grid.
    """
    rng = np.random.default_rng(0)
    scene = rng.normal(loc=0.35, scale=0.05, size=(64, 64))
    reference = Dataset.from_array(
        arr=np.clip(scene, 0, 1).astype("float32"), geo_ref=GEO
    )
    target = Dataset.from_array(
        arr=np.clip(scene * 1.8 + 0.15, 0, 1).astype("float32"), geo_ref=GEO
    )
    return reference, target


def test_match_histogram_aligns_the_mean(scene_pair: tuple[Dataset, Dataset]):
    """Matching pulls the target's mean onto the reference's distribution."""
    reference, target = scene_pair
    matched = target.match_histogram(reference)
    assert matched.band_count == 1
    assert matched.read_array().dtype == np.float64
    ref_mean = float(reference.read_array().mean())
    assert float(matched.read_array().mean()) == pytest.approx(ref_mean, abs=0.02)
    # And it actually moved — the raw target mean is far from the reference.
    assert abs(float(target.read_array().mean()) - ref_mean) > 0.3


def test_match_histogram_shares_source_grid(scene_pair: tuple[Dataset, Dataset]):
    """The result carries the source geotransform and CRS, not the reference's."""
    reference, target = scene_pair
    matched = target.match_histogram(reference)
    assert matched.geotransform == target.geotransform
    assert matched.epsg == 4326


def test_match_histogram_facade_matches_engine(scene_pair: tuple[Dataset, Dataset]):
    """The Dataset facade and the engine method produce identical pixels."""
    reference, target = scene_pair
    np.testing.assert_array_equal(
        target.match_histogram(reference).read_array(),
        target.analysis.match_histogram(reference).read_array(),
    )


def test_match_histogram_reference_band_selection():
    """`reference_band` picks which reference band supplies the distribution."""
    rng = np.random.default_rng(5)
    source = Dataset.from_array(
        arr=(rng.random((1, 16, 16)) * 10).astype("float32"), geo_ref=GEO
    )
    # Two reference bands with very different levels; band 1 sits near 100.
    ref_arr = np.stack(
        [
            rng.random((16, 16)).astype("float32"),
            (rng.random((16, 16)) + 100).astype("float32"),
        ]
    )
    reference = Dataset.from_array(arr=ref_arr, geo_ref=GEO)
    matched = source.match_histogram(reference, reference_band=1)
    assert float(matched.read_array().mean()) > 90


def test_match_histogram_excludes_and_preserves_nodata():
    """No-data cells are excluded from both CDFs and kept as no-data in the output."""
    arr = np.linspace(0, 1, 64, dtype="float32").reshape(8, 8)
    arr[0, 0] = -9999.0
    target = Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=-9999.0)
    reference = Dataset.from_array(
        arr=np.linspace(5, 6, 64, dtype="float32").reshape(8, 8), geo_ref=GEO
    )
    matched = target.match_histogram(reference)
    out = matched.read_array()
    assert matched.no_data_value[0] == -9999.0
    assert out[0, 0] == -9999.0
    # Every other cell was remapped into the reference's 5..6 range.
    assert float(np.nanmin(out[out != -9999.0])) >= 5.0


def test_match_histogram_preserves_nan_nodata():
    """A NaN-sentinel source keeps its NaN gaps after matching."""
    arr = np.linspace(0, 1, 64, dtype="float32").reshape(8, 8)
    arr[1, 1] = np.nan
    target = Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=np.nan)
    reference = Dataset.from_array(
        arr=np.linspace(5, 6, 64, dtype="float32").reshape(8, 8), geo_ref=GEO
    )
    out = target.match_histogram(reference).read_array()
    assert np.isnan(out[1, 1])


def test_match_histogram_rejects_bad_bins(scene_pair: tuple[Dataset, Dataset]):
    """A bin count below one is rejected."""
    reference, target = scene_pair
    with pytest.raises(ValueError, match="bins must be >= 1"):
        target.match_histogram(reference, bins=0)


def test_match_histogram_rejects_out_of_range_band(scene_pair: tuple[Dataset, Dataset]):
    """Both the source and reference band indices are range-checked."""
    reference, target = scene_pair
    with pytest.raises((ValueError, IndexError)):
        target.match_histogram(reference, band=5)
    with pytest.raises((ValueError, IndexError)):
        target.match_histogram(reference, reference_band=5)


def test_match_histogram_rejects_degenerate_source(scene_pair: tuple[Dataset, Dataset]):
    """A constant source band has no range to remap and is rejected."""
    reference, _ = scene_pair
    const = Dataset.from_array(arr=np.full((8, 8), 7.0, dtype="float32"), geo_ref=GEO)
    with pytest.raises(ValueError, match="source band with a range"):
        const.match_histogram(reference)


def test_match_histogram_rejects_all_nodata_source():
    """A source with no finite samples cannot build a CDF."""
    arr = np.full((8, 8), -9999.0, dtype="float32")
    target = Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=-9999.0)
    reference = Dataset.from_array(
        arr=np.linspace(0, 1, 64, dtype="float32").reshape(8, 8), geo_ref=GEO
    )
    with pytest.raises(ValueError, match="finite samples"):
        target.match_histogram(reference)


def test_match_histogram_reaches_reference_extremes():
    """The matched output spans the reference's true min/max, not a compressed range."""
    rng = np.random.default_rng(3)
    target = Dataset.from_array(
        arr=rng.normal(0.5, 0.1, size=(64, 64)).astype("float32"), geo_ref=GEO
    )
    reference = Dataset.from_array(
        arr=np.linspace(5.0, 6.0, 64 * 64, dtype="float32").reshape(64, 64),
        geo_ref=GEO,
    )
    out = target.match_histogram(reference).read_array()
    assert float(out.min()) == pytest.approx(5.0, abs=1e-3)
    assert float(out.max()) == pytest.approx(6.0, abs=1e-3)


def test_match_histogram_discrete_source_maps_atoms_to_cdf_quantile():
    """A discrete source's min-atom maps to the reference quantile at its probability.

    For an integer source with a mass of identical minimum pixels, standard
    histogram matching maps that whole atom to one reference value — the
    reference quantile at the atom's cumulative probability — not the reference
    minimum. The output still reaches the reference maximum and stays monotone.
    """
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 50, size=(64, 64)).astype("float32")
    arr[arr < 10] = 0.0
    target = Dataset.from_array(arr=arr, geo_ref=GEO)
    reference = Dataset.from_array(
        arr=np.linspace(5.0, 6.0, 64 * 64, dtype="float32").reshape(64, 64),
        geo_ref=GEO,
    )
    out = target.match_histogram(reference).read_array()
    atom_prob = float((arr == 0.0).mean())
    expected_min = 5.0 + atom_prob
    assert float(out.min()) == pytest.approx(expected_min, abs=0.02)
    assert float(out.min()) > 5.0
    assert float(out.max()) == pytest.approx(6.0, abs=1e-3)


def test_match_histogram_single_bin(scene_pair: tuple[Dataset, Dataset]):
    """``bins=1`` degrades to a linear remap between the two ranges' endpoints."""
    reference, target = scene_pair
    matched = target.match_histogram(reference, bins=1).read_array()
    ref_arr = reference.read_array()
    assert float(matched.min()) == pytest.approx(float(ref_arr.min()), abs=1e-3)
    assert float(matched.max()) == pytest.approx(float(ref_arr.max()), abs=1e-3)


def test_percentile_stretch_default_percentiles():
    """The default 2nd/98th percentiles still span the clipped [0, 1] range."""
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=1.0, epsg=4326)
    )
    stretched = ds.percentile_stretch().read_array()
    assert float(stretched.min()) == 0.0
    assert float(stretched.max()) == 1.0


def test_percentile_stretch_full_range():
    """A 0..99 ramp stretched across its full range spans exactly [0, 1]."""
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=1.0, epsg=4326)
    )
    stretched = ds.percentile_stretch(0, 100).read_array()
    assert (float(stretched.min()), float(stretched.max())) == (0.0, 1.0)


def test_percentile_stretch_clips_tails():
    """A partial stretch clips the tails to the [0, 1] bounds."""
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=1.0, epsg=4326)
    )
    stretched = ds.percentile_stretch(10, 90).read_array()
    assert float(stretched.min()) == 0.0
    assert float(stretched.max()) == 1.0
    # The lowest raw values saturate at 0 after clipping.
    assert float(stretched.ravel()[0]) == 0.0


def test_percentile_stretch_preserves_nodata():
    """No-data cells are excluded from the percentiles and kept in the output."""
    arr = np.arange(64, dtype="float32").reshape(8, 8)
    arr[0, 0] = -9999.0
    ds = Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=-9999.0)
    out = ds.percentile_stretch(0, 100).read_array()
    assert out[0, 0] == -9999.0, "the no-data corner must be preserved"
    rest = out[out != -9999.0]
    assert float(rest.min()) >= 0.0, "stretched values must be >= 0"
    assert float(rest.max()) <= 1.0, "stretched values must be <= 1"


def test_percentile_stretch_rejects_bad_percentiles():
    """The percentiles must satisfy 0 <= lower < upper <= 100."""
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=(0, 10), cell_size=1.0, epsg=4326)
    )
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        ds.percentile_stretch(90, 10)


def test_percentile_stretch_rejects_degenerate_band():
    """A constant band has equal percentiles and cannot be stretched."""
    arr = np.full((8, 8), 7.0, dtype="float32")
    ds = Dataset.from_array(arr=arr, geo_ref=GEO)
    with pytest.raises(ValueError, match="degenerate stretch"):
        ds.percentile_stretch(2, 98)


def test_percentile_stretch_rejects_all_nodata():
    """A band with no finite samples cannot be stretched."""
    arr = np.full((8, 8), -9999.0, dtype="float32")
    ds = Dataset.from_array(arr=arr, geo_ref=GEO, no_data_value=-9999.0)
    with pytest.raises(ValueError, match="finite samples"):
        ds.percentile_stretch(2, 98)
