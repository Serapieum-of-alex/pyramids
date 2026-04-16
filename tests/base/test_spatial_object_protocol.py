"""Tests for the :class:`pyramids.base.protocols.SpatialObject` protocol.

ARC-17: both :class:`pyramids.dataset.Dataset` and
:class:`pyramids.feature.FeatureCollection` implement the shared
``SpatialObject`` protocol, so callers can write generic code that
accepts either a raster or a vector.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.base.protocols import SpatialObject
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection


@pytest.fixture
def fc() -> FeatureCollection:
    """A simple FeatureCollection with one polygon in EPSG:32636."""
    gdf = gpd.GeoDataFrame(
        {"v": [1]},
        geometry=[box(500000.0, 3400000.0, 510000.0, 3410000.0)],
        crs="EPSG:32636",
    )
    return FeatureCollection(gdf)


@pytest.fixture
def ds() -> Dataset:
    """A small in-memory Dataset in EPSG:32636."""
    src = Dataset.create(
        cell_size=1000.0,
        rows=10,
        columns=10,
        dtype="float32",
        bands=1,
        top_left_corner=(500000.0, 3410000.0),
        epsg=32636,
        no_data_value=-9999.0,
    )
    src.raster.GetRasterBand(1).WriteArray(
        np.ones((10, 10), dtype=np.float32)
    )
    return src


class TestSpatialObjectProtocol:
    """Both Dataset and FeatureCollection satisfy SpatialObject."""

    def test_feature_collection_is_spatial_object(self, fc: FeatureCollection):
        """``isinstance(fc, SpatialObject)`` is True at runtime."""
        assert isinstance(fc, SpatialObject)

    def test_dataset_is_spatial_object(self, ds: Dataset):
        """``isinstance(ds, SpatialObject)`` is True at runtime."""
        assert isinstance(ds, SpatialObject)

    def test_both_expose_epsg(self, fc: FeatureCollection, ds: Dataset):
        """Both types expose ``epsg`` as the same kind of value."""
        assert fc.epsg == 32636
        assert ds.epsg == 32636

    def test_both_expose_total_bounds(self, fc: FeatureCollection, ds: Dataset):
        """Both types expose ``total_bounds`` as [minx, miny, maxx, maxy]."""
        fc_bounds = np.asarray(fc.total_bounds)
        ds_bounds = np.asarray(ds.total_bounds)
        assert fc_bounds.shape == (4,)
        assert ds_bounds.shape == (4,)
        # FC bounds match the box we built.
        assert np.allclose(
            fc_bounds, [500000.0, 3400000.0, 510000.0, 3410000.0]
        )
        # Dataset bbox reflects its geotransform (10 cols × 10 rows × 1000m).
        assert np.allclose(
            ds_bounds, [500000.0, 3400000.0, 510000.0, 3410000.0]
        )

    def test_both_expose_top_left_corner(
        self, fc: FeatureCollection, ds: Dataset
    ):
        """Both types expose ``top_left_corner`` as [minx, maxy]."""
        fc_tl = list(fc.top_left_corner)
        ds_tl = list(ds.top_left_corner)
        assert fc_tl == [500000.0, 3410000.0]
        assert ds_tl == [500000.0, 3410000.0]

    def test_generic_consumer_accepts_both(
        self, fc: FeatureCollection, ds: Dataset
    ):
        """A function typed against SpatialObject accepts both classes.

        This is the real value of the protocol — users can write
        cross-type utilities without importing both concrete classes.
        """

        def epsg_of(obj: SpatialObject) -> int | None:
            return obj.epsg

        assert epsg_of(fc) == 32636
        assert epsg_of(ds) == 32636


class TestSpatialObjectNegative:
    """Objects missing required attributes must NOT be SpatialObjects."""

    def test_plain_string_is_not_spatial_object(self):
        assert not isinstance("not spatial", SpatialObject)

    def test_dict_is_not_spatial_object(self):
        assert not isinstance({"epsg": 4326}, SpatialObject)

    def test_plain_geodataframe_is_spatial_object_only_when_attrs_present(self):
        """A plain GeoDataFrame does not have ``epsg`` / ``top_left_corner``.

        So it does NOT satisfy the protocol even though it has
        ``total_bounds`` / ``to_file`` / ``read_file`` / ``plot``.
        """
        plain = gpd.GeoDataFrame(
            {"v": [1]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        assert not isinstance(plain, SpatialObject)
