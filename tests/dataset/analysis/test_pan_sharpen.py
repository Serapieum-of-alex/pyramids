"""Tests for :meth:`Dataset.pan_sharpen` / ``Analysis.pan_sharpen``.

Covers the GDAL ``CreatePansharpenedVRT`` facade: fusing a coarse multispectral
raster with a finer panchromatic band, band selection and output order, weights,
resampling and no-data options, the facade/engine equivalence, and every guard
clause (bad algorithm/resampling, empty or out-of-range bands, weight-length
mismatch, CRS mismatch, disjoint extents). All fixtures are in-memory
``from_array`` rasters, so the suite needs no data files.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

TOP_LEFT = (31.0, 30.0)


@pytest.fixture
def multispectral() -> Dataset:
    """A 3-band 32x32 multispectral raster, top-left (31, 30), cell 0.01, EPSG:4326.

    Returns:
        Dataset: the coarse multispectral source.
    """
    rng = np.random.default_rng(0)
    return Dataset.from_array(
        arr=rng.integers(0, 4000, size=(3, 32, 32)).astype("uint16"),
        geo_ref=GeoReference(top_left_corner=TOP_LEFT, cell_size=0.01, epsg=4326),
        no_data_value=None,
    )


@pytest.fixture
def pan() -> Dataset:
    """A 1-band 128x128 panchromatic raster over the same extent at 4x the resolution.

    Returns:
        Dataset: the fine panchromatic source.
    """
    rng = np.random.default_rng(1)
    return Dataset.from_array(
        arr=rng.integers(0, 4000, size=(128, 128)).astype("uint16"),
        geo_ref=GeoReference(top_left_corner=TOP_LEFT, cell_size=0.0025, epsg=4326),
        no_data_value=None,
    )


def test_pan_sharpen_fuses_to_pan_resolution(multispectral: Dataset, pan: Dataset):
    """The fused raster is at the pan grid with one band per spectral band."""
    fused = multispectral.pan_sharpen(pan)
    assert (fused.rows, fused.columns) == (128, 128)
    assert fused.band_count == 3
    assert fused.epsg == 4326
    assert fused.cell_size == pytest.approx(0.0025)


def test_pan_sharpen_band_subset_controls_output_bands(
    multispectral: Dataset, pan: Dataset
):
    """`bands` selects which spectral bands to sharpen and their output order."""
    fused = multispectral.pan_sharpen(pan, bands=[0, 2])
    assert fused.band_count == 2


def test_pan_sharpen_accepts_weights_resampling_and_nodata(
    multispectral: Dataset, pan: Dataset
):
    """Weights, a resampling kernel, and a no-data value are all honoured."""
    fused = multispectral.pan_sharpen(
        pan, weights=[0.4, 0.3, 0.3], resampling="bilinear", nodata=0
    )
    assert fused.band_count == 3
    assert fused.no_data_value[0] == 0


@pytest.mark.parametrize(
    "resampling", ["nearest", "bilinear", "cubic", "cubicspline", "lanczos", "average"]
)
def test_pan_sharpen_resampling_kernels(
    multispectral: Dataset, pan: Dataset, resampling: str
):
    """Every documented resampling name is accepted by GDAL."""
    fused = multispectral.pan_sharpen(pan, resampling=resampling)
    assert (fused.rows, fused.columns, fused.band_count) == (128, 128, 3)


def test_pan_sharpen_weights_change_the_output(multispectral: Dataset, pan: Dataset):
    """Different Brovey weights produce different fused pixels (not silently ignored)."""
    red_heavy = multispectral.pan_sharpen(pan, weights=[0.9, 0.05, 0.05]).read_array()
    green_heavy = multispectral.pan_sharpen(pan, weights=[0.05, 0.9, 0.05]).read_array()
    assert not np.array_equal(red_heavy, green_heavy), "weights must affect the fusion"


def test_pan_sharpen_resampling_changes_the_output(
    multispectral: Dataset, pan: Dataset
):
    """A different resampling kernel produces different fused pixels."""
    nearest = multispectral.pan_sharpen(pan, resampling="nearest").read_array()
    cubic = multispectral.pan_sharpen(pan, resampling="cubic").read_array()
    assert not np.array_equal(nearest, cubic), "resampling must affect the fusion"


def test_pan_sharpen_facade_matches_engine(multispectral: Dataset, pan: Dataset):
    """The Dataset facade and the engine method produce identical pixels."""
    via_facade = multispectral.pan_sharpen(pan).read_array()
    via_engine = multispectral.analysis.pan_sharpen(pan).read_array()
    np.testing.assert_array_equal(via_facade, via_engine)


def test_pan_sharpen_rejects_unknown_algorithm(multispectral: Dataset, pan: Dataset):
    """Only the Brovey transform GDAL implements is accepted."""
    with pytest.raises(ValueError, match="algorithm must be 'brovey'"):
        multispectral.pan_sharpen(pan, algorithm="ihs")


def test_pan_sharpen_rejects_unknown_resampling(multispectral: Dataset, pan: Dataset):
    """An unrecognised resampling name is rejected with the valid set."""
    with pytest.raises(ValueError, match="resampling must be one of"):
        multispectral.pan_sharpen(pan, resampling="spline")


def test_pan_sharpen_non_string_option_raises_value_error(
    multispectral: Dataset, pan: Dataset
):
    """A non-string algorithm/resampling raises ValueError, not AttributeError."""
    with pytest.raises(ValueError, match="algorithm must be 'brovey'"):
        multispectral.pan_sharpen(pan, algorithm=42)
    with pytest.raises(ValueError, match="resampling must be one of"):
        multispectral.pan_sharpen(pan, resampling=42)


def test_pan_sharpen_rejects_empty_bands(multispectral: Dataset, pan: Dataset):
    """Selecting no spectral band is an error."""
    with pytest.raises(ValueError, match="at least one multispectral band"):
        multispectral.pan_sharpen(pan, bands=[])


def test_pan_sharpen_rejects_out_of_range_band(multispectral: Dataset, pan: Dataset):
    """A band index beyond the multispectral stack is rejected."""
    with pytest.raises((ValueError, IndexError)):
        multispectral.pan_sharpen(pan, bands=[0, 9])


def test_pan_sharpen_rejects_weight_length_mismatch(
    multispectral: Dataset, pan: Dataset
):
    """`weights` must carry one entry per selected band."""
    with pytest.raises(ValueError, match="one entry per selected band"):
        multispectral.pan_sharpen(pan, weights=[0.5, 0.5])


def test_pan_sharpen_rejects_crs_mismatch(multispectral: Dataset):
    """A pan band in a different CRS cannot be fused."""
    rng = np.random.default_rng(2)
    pan_3857 = Dataset.from_array(
        arr=rng.integers(0, 4000, size=(128, 128)).astype("uint16"),
        geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=100.0, epsg=3857),
        no_data_value=None,
    )
    with pytest.raises(AlignmentError, match="shared CRS"):
        multispectral.pan_sharpen(pan_3857)


def test_pan_sharpen_rejects_disjoint_extent(multispectral: Dataset):
    """A pan band that does not overlap the multispectral extent is rejected."""
    rng = np.random.default_rng(3)
    pan_far = Dataset.from_array(
        arr=rng.integers(0, 4000, size=(16, 16)).astype("uint16"),
        geo_ref=GeoReference(top_left_corner=(100.0, 10.0), cell_size=0.01, epsg=4326),
        no_data_value=None,
    )
    with pytest.raises(AlignmentError, match="overlap"):
        multispectral.pan_sharpen(pan_far)


def test_pan_sharpen_crs_fallback_compares_wkt(multispectral: Dataset, pan: Dataset):
    """When a raster has no EPSG, the CRS check falls back to comparing WKT."""
    multispectral.crs = ""
    pan.crs = ""
    assert multispectral.epsg is None and pan.epsg is None
    fused = multispectral.pan_sharpen(pan)
    assert (fused.rows, fused.columns, fused.band_count) == (128, 128, 3)


def test_pan_sharpen_rejects_one_sided_crs(multispectral: Dataset, pan: Dataset):
    """One raster declaring a CRS while the other does not is a mismatch."""
    pan.crs = ""
    assert multispectral.epsg == 4326 and pan.epsg is None
    with pytest.raises(AlignmentError, match="shared CRS"):
        multispectral.pan_sharpen(pan)


def test_pan_sharpen_single_band_multispectral(pan: Dataset):
    """A one-band multispectral input yields a one-band fused raster."""
    rng = np.random.default_rng(7)
    mono = Dataset.from_array(
        arr=rng.integers(0, 4000, size=(1, 32, 32)).astype("uint16"),
        geo_ref=GeoReference(top_left_corner=TOP_LEFT, cell_size=0.01, epsg=4326),
        no_data_value=None,
    )
    fused = mono.pan_sharpen(pan)
    assert (fused.rows, fused.columns, fused.band_count) == (128, 128, 1)
