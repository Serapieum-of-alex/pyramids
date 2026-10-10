"""Tests for the standalone mesh-construction helpers in ``pyramids.netcdf.ugrid.construct``.

Covers flattening a regular raster grid into a quad mesh (:func:`from_structured` /
:func:`from_dataset`) and building a face mesh from polygon geometries
(:func:`from_geodataframe`), asserting node/face counts, counter-clockwise orientation, ragged
``-1`` fills, attribute carry-through, and round-tripped values.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from pyramids.dataset import Dataset, GeoReference
from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.construct import (
    from_dataset,
    from_geodataframe,
    from_structured,
)

pytestmark = pytest.mark.core


def _signed_areas(
    node_x: np.ndarray, node_y: np.ndarray, fnc: np.ndarray
) -> np.ndarray:
    """Signed shoelace area of every quad face (positive when counter-clockwise).

    Args:
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        fnc: ``(n_face, 4)`` face-node connectivity of quad faces (no fill).

    Returns:
        np.ndarray: Signed area per face.
    """
    x = node_x[fnc]
    y = node_y[fnc]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    return 0.5 * np.sum(x * y_next - x_next * y, axis=1)


@pytest.fixture
def north_up_dataset() -> Dataset:
    """A 3x3 north-up (descending-y) single-band raster with distinct values 0..8.

    Returns:
        Dataset: In-memory 3x3 raster, cell size 1, top-left at (0, 3), EPSG:4326.
    """
    arr = np.arange(9, dtype="float32").reshape(3, 3)
    geo_ref = GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326)
    return Dataset.from_array(arr, geo_ref=geo_ref)


class TestFromStructured:
    """Build a quad mesh from a structured raster grid."""

    def test_node_and_face_counts(self, north_up_dataset: Dataset):
        """A 3x3 raster yields a 4x4 node lattice and 9 quad faces."""
        node_x, node_y, fnc, data = from_structured(north_up_dataset)
        assert node_x.shape == (16,)
        assert node_y.shape == (16,)
        assert fnc.shape == (9, 4)
        assert list(data) == ["band_1"]
        assert data["band_1"].shape == (9,)

    def test_faces_are_counter_clockwise(self, north_up_dataset: Dataset):
        """Descending-y (north-up) rasters still produce positive signed areas (CCW)."""
        node_x, node_y, fnc, _ = from_structured(north_up_dataset)
        areas = _signed_areas(node_x, node_y, fnc)
        assert np.all(areas > 0), areas

    def test_positive_face_areas_via_mesh(self, north_up_dataset: Dataset):
        """The assembled UgridDataset reports positive (non-degenerate) face areas."""
        node_x, node_y, fnc, data = from_structured(north_up_dataset)
        uds = UgridDataset.from_arrays(
            node_x, node_y, fnc, data=data, epsg=north_up_dataset.epsg
        )
        assert uds.n_face == 9
        assert np.all(uds.mesh.face_areas > 0)

    def test_values_match_raster_row_major(self, north_up_dataset: Dataset):
        """Per-face values are the band raveled row-major, matching the face ordering."""
        _, _, _, data = from_structured(north_up_dataset)
        np.testing.assert_array_equal(data["band_1"], np.arange(9, dtype="float32"))

    def test_round_trip_value_at_cell_centroids(self, north_up_dataset: Dataset):
        """Each face centroid sits at its source cell center and carries that cell's value.

        Cell ``(r, c)`` of the top-left-(0, 3), unit-cell raster has center ``(c + 0.5,
        2.5 - r)`` and the face ordering is row-major, so face ``r * 3 + c`` must hold
        ``arr[r, c]`` at that centroid.
        """
        node_x, node_y, fnc, data = from_structured(north_up_dataset)
        uds = UgridDataset.from_arrays(
            node_x, node_y, fnc, data=data, epsg=north_up_dataset.epsg
        )
        cx, cy = uds.mesh.face_centroids
        source = north_up_dataset.read_array()
        for r in range(3):
            for c in range(3):
                face = r * 3 + c
                np.testing.assert_allclose([cx[face], cy[face]], [c + 0.5, 2.5 - r])
                assert data["band_1"][face] == source[r, c]

    def test_crs_carried_from_source(self, north_up_dataset: Dataset):
        """The CRS the facade carries comes from the source dataset EPSG."""
        node_x, node_y, fnc, data = from_structured(north_up_dataset)
        uds = UgridDataset.from_arrays(
            node_x, node_y, fnc, data=data, epsg=north_up_dataset.epsg
        )
        assert uds.epsg == 4326

    def test_multiband_one_variable_per_band(self):
        """A 2-band raster yields one face variable per band."""
        stack = np.stack(
            [
                np.arange(4, dtype="float32").reshape(2, 2),
                np.full((2, 2), 7.0, "float32"),
            ]
        )
        geo_ref = GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326)
        ds = Dataset.from_array(stack, geo_ref=geo_ref)
        _, _, fnc, data = from_structured(ds, band_names=["elevation", "mask"])
        assert fnc.shape == (4, 4)
        assert list(data) == ["elevation", "mask"]
        np.testing.assert_array_equal(data["elevation"], np.arange(4, dtype="float32"))
        np.testing.assert_array_equal(data["mask"], np.full(4, 7.0, "float32"))

    def test_from_dataset_alias(self, north_up_dataset: Dataset):
        """from_dataset is an alias producing identical arrays to from_structured."""
        a = from_structured(north_up_dataset)
        b = from_dataset(north_up_dataset)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])

    def test_source_not_mutated(self, north_up_dataset: Dataset):
        """Building a mesh leaves the source raster values untouched."""
        before = north_up_dataset.read_array().copy()
        from_structured(north_up_dataset)
        np.testing.assert_array_equal(north_up_dataset.read_array(), before)


class TestFromGeoDataFrame:
    """Build a face mesh from polygon geometries."""

    @staticmethod
    def _two_quads() -> "gpd.GeoDataFrame":  # noqa: F821 - string annotation only
        """Two unit quads sharing an edge, with a ``value`` attribute column."""
        import geopandas as gpd

        poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        return gpd.GeoDataFrame(
            {"value": [10.0, 20.0]}, geometry=[poly_a, poly_b], crs="EPSG:4326"
        )

    def test_node_and_face_counts_with_shared_edge(self):
        """Two edge-sharing quads dedup to 6 nodes and 2 faces."""
        gdf = self._two_quads()
        node_x, node_y, fnc, data = from_geodataframe(gdf)
        assert node_x.shape == (6,)
        assert fnc.shape == (2, 4)
        assert data["value"].tolist() == [10.0, 20.0]

    def test_ragged_triangle_and_quad(self):
        """A triangle plus a quad pad to max_nodes with a -1 fill."""
        import geopandas as gpd

        tri = Polygon([(0, 0), (1, 0), (0, 1)])
        quad = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        gdf = gpd.GeoDataFrame(geometry=[tri, quad], crs="EPSG:4326")
        _, _, fnc, _ = from_geodataframe(gdf)
        assert fnc.shape == (2, 4)
        assert (fnc[0] == -1).sum() == 1
        assert (fnc[1] == -1).sum() == 0

    def test_faces_are_counter_clockwise(self):
        """Clockwise input rings are reoriented to positive signed area."""
        import geopandas as gpd

        cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # clockwise ring
        gdf = gpd.GeoDataFrame(geometry=[cw], crs="EPSG:4326")
        node_x, node_y, fnc, _ = from_geodataframe(gdf)
        areas = _signed_areas(node_x, node_y, fnc)
        assert np.all(areas > 0), areas

    def test_round_trip_geometries(self):
        """from_geodataframe -> to_geodataframe recovers the polygon geometries."""
        gdf = self._two_quads()
        node_x, node_y, fnc, data = from_geodataframe(gdf)
        epsg = gdf.crs.to_epsg()
        uds = UgridDataset.from_arrays(node_x, node_y, fnc, data=data, epsg=epsg)
        recovered = uds.to_geodataframe()
        assert len(recovered) == 2
        for original, built in zip(gdf.geometry, recovered.geometry):
            assert original.equals(built)

    def test_crs_carried_from_gdf(self):
        """The EPSG flows from gdf.crs into the assembled dataset."""
        gdf = self._two_quads()
        node_x, node_y, fnc, data = from_geodataframe(gdf)
        uds = UgridDataset.from_arrays(
            node_x, node_y, fnc, data=data, epsg=gdf.crs.to_epsg()
        )
        assert uds.epsg == 4326

    def test_degenerate_polygon_dropped(self):
        """A null/degenerate geometry is dropped and its attribute row is not kept."""
        import geopandas as gpd

        good = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]}, geometry=[good, None], crs="EPSG:4326"
        )
        _, _, fnc, data = from_geodataframe(gdf)
        assert fnc.shape == (1, 4)
        assert data["value"].tolist() == [1.0]
