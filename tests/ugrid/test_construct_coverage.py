"""Guards and dropped-geometry paths of ``pyramids.netcdf.ugrid.construct``.

The mainline construction tests only feed ``from_structured`` / ``from_geodataframe`` well-formed
input, so the input guards (empty raster, wrong number of band names, a GeoDataFrame with nothing
usable in it) and the per-geometry drop rules are untested. These tests pin each guard's message,
pin that a non-polygon and a sub-triangular ring are dropped along with their attribute rows, and
pin that ``include_attributes=False`` really emits no data arrays.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf.ugrid.construct import from_geodataframe, from_structured

pytestmark = pytest.mark.core

UNIT_QUAD = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
"""A well-formed counter-clockwise quad, used as the one face that must survive each drop."""

SLIVER = shapely.from_wkt("POLYGON ((0 0, 1 1, 0 0))")
"""A polygon whose exterior ring holds only two distinct vertices, so it cannot form a face."""


class ZeroSizeRaster:
    """A stand-in raster with no rows and no columns.

    GDAL refuses to create a 0 x 0 dataset (``Invalid block dimension``), so the empty-grid guard
    in ``from_structured`` can only be reached through an object exposing the same public
    accessors ``from_structured`` reads off a ``Dataset``.
    """

    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    rows = 0
    columns = 0
    band_count = 1

    @staticmethod
    def read_array() -> np.ndarray:
        """Return the (empty) band array."""
        return np.zeros((0, 0), dtype=np.float64)


@pytest.fixture
def single_band_raster() -> Dataset:
    """A 2x2 north-up single-band raster with values 0..3."""
    array = np.arange(4, dtype="float32").reshape(2, 2)
    geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
    return Dataset.from_array(array, geo_ref=geo_ref)


@pytest.fixture
def quad_and_point() -> gpd.GeoDataFrame:
    """One usable quad plus a point, each with a ``value`` attribute."""
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[UNIT_QUAD, Point(5.0, 5.0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def quad_and_sliver() -> gpd.GeoDataFrame:
    """One usable quad plus a two-vertex sliver polygon, each with a ``value`` attribute."""
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[UNIT_QUAD, SLIVER],
        crs="EPSG:4326",
    )


class TestFromStructuredGuards:
    """Input validation in ``from_structured`` and its band reader."""

    def test_a_raster_with_no_rows_or_columns_is_rejected(self):
        """An empty grid cannot become a mesh, and the error quotes the raster's shape."""
        with pytest.raises(ValueError, match="Cannot build a mesh from a 0x0 raster"):
            from_structured(ZeroSizeRaster())

    def test_band_names_must_match_the_band_count(self, single_band_raster):
        """Two names for a single-band raster are rejected, reporting both counts."""
        with pytest.raises(
            ValueError, match=r"band_names has 2 entries but the raster has 1 band"
        ):
            from_structured(single_band_raster, band_names=["elevation", "mask"])

    def test_one_name_for_one_band_is_accepted(self, single_band_raster):
        """The matching-length case still names the single face variable as asked."""
        _, _, _, data = from_structured(single_band_raster, band_names=["elevation"])
        assert list(data) == ["elevation"]
        np.testing.assert_array_equal(data["elevation"], np.arange(4, dtype="float32"))


class TestFromGeoDataFrameDrops:
    """Which geometries ``from_geodataframe`` drops, and what it drops with them."""

    def test_a_non_polygon_geometry_is_dropped(self, quad_and_point):
        """A point has no exterior ring, so it yields no face and no attribute row."""
        _, _, face_node_connectivity, data = from_geodataframe(quad_and_point)
        assert face_node_connectivity.shape == (1, 4)
        assert data["value"].tolist() == [1.0]

    def test_a_two_vertex_ring_is_dropped(self, quad_and_sliver):
        """A ring with fewer than three distinct vertices yields no face and no attribute row."""
        node_x, _, face_node_connectivity, data = from_geodataframe(quad_and_sliver)
        assert face_node_connectivity.shape == (1, 4)
        assert node_x.shape == (4,)
        assert data["value"].tolist() == [1.0]

    def test_a_geodataframe_with_nothing_usable_is_rejected(self):
        """A frame holding only a sliver and a null geometry raises rather than returning no faces."""
        gdf = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]}, geometry=[SLIVER, None], crs="EPSG:4326"
        )
        with pytest.raises(ValueError, match="no valid polygon"):
            from_geodataframe(gdf)

    def test_attributes_can_be_skipped(self, quad_and_sliver):
        """``include_attributes=False`` emits the topology only, with no data arrays."""
        _, _, face_node_connectivity, data = from_geodataframe(
            quad_and_sliver, include_attributes=False
        )
        assert face_node_connectivity.shape == (1, 4)
        assert data == {}
