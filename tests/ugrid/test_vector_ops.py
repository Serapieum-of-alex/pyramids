"""Tests for the standalone mesh <-> vector operations in ``vector_ops``.

Covers :func:`~pyramids.netcdf.ugrid.vector_ops.polygonize` (F1),
:func:`~pyramids.netcdf.ugrid.vector_ops.sample_line` (F2), and
:func:`~pyramids.netcdf.ugrid.vector_ops.burn` (F3) on small meshes built with
``UgridDataset.from_arrays``.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.vector_ops import burn, polygonize, sample_line

pytestmark = pytest.mark.core


@pytest.fixture
def grid_2x2():
    """2x2 quad mesh over the unit-ish square, faces in row-major order.

    Nodes sit on a 3x3 lattice; faces cover ``[0,1]x[0,1]`` (f0), ``[1,2]x[0,1]`` (f1),
    ``[0,1]x[1,2]`` (f2) and ``[1,2]x[1,2]`` (f3).

    Returns:
        Mesh2d: The 4-face mesh.
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    dataset = UgridDataset.from_arrays(node_x, node_y, faces)
    return dataset.mesh


@pytest.fixture
def strip_1x4():
    """1x4 horizontal strip of quads, values increasing with x.

    Faces cover ``[0,1]``, ``[1,2]``, ``[2,3]`` and ``[3,4]`` in x over ``y in [0,1]``.

    Returns:
        Mesh2d: The 4-face strip mesh.
    """
    bottom_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    node_x = np.concatenate([bottom_x, bottom_x])
    node_y = np.concatenate([np.zeros(5), np.ones(5)])
    faces = np.array([[0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 8, 7], [3, 4, 9, 8]])
    dataset = UgridDataset.from_arrays(node_x, node_y, faces)
    return dataset.mesh


class TestPolygonize:
    """Dissolving faces by value into merged polygons (F1)."""

    def test_two_value_field_dissolves_into_two_regions(self, grid_2x2):
        """A 2-value field yields two rows whose areas and values are correct."""
        values = np.array([1, 1, 2, 2])
        result = polygonize(grid_2x2, values)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        by_value = dict(zip(result["value"], result.geometry.area))
        assert set(by_value) == {1, 2}
        assert by_value[1] == pytest.approx(2.0)
        assert by_value[2] == pytest.approx(2.0)

    def test_isolated_face_is_preserved(self, grid_2x2):
        """A lone-valued face survives as its own single-face row."""
        values = np.array([0, 1, 1, 1])
        result = polygonize(grid_2x2, values)

        areas = dict(zip(result["value"], result.geometry.area))
        assert set(areas) == {0, 1}
        assert areas[0] == pytest.approx(1.0)
        assert areas[1] == pytest.approx(3.0)

    def test_decimals_groups_approximately_equal_values(self, grid_2x2):
        """Rounding merges near-equal floats that exact equality keeps apart."""
        values = np.array([1.0, 1.0001, 2.0, 2.0])

        assert len(polygonize(grid_2x2, values)) == 3
        assert len(polygonize(grid_2x2, values, decimals=2)) == 2

    def test_nan_faces_are_dropped(self, grid_2x2):
        """Faces carrying NaN are excluded from the output."""
        values = np.array([1.0, np.nan, 2.0, np.nan])
        result = polygonize(grid_2x2, values)

        assert set(result["value"]) == {1.0, 2.0}
        assert result.geometry.area.sum() == pytest.approx(2.0)

    def test_crs_is_set_from_argument(self, grid_2x2):
        """An explicit CRS argument is attached to the result."""
        result = polygonize(grid_2x2, np.array([1, 1, 2, 2]), crs=4326)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_rejects_non_1d_values(self, grid_2x2):
        """A 2-D (temporal) value array is rejected with a clear message."""
        with pytest.raises(ValueError, match="1-D per-face"):
            polygonize(grid_2x2, np.zeros((2, 4)))

    def test_does_not_mutate_input(self, grid_2x2):
        """Polygonizing leaves the value array untouched."""
        values = np.array([1, 1, 2, 2])
        before = values.copy()
        polygonize(grid_2x2, values)
        np.testing.assert_array_equal(values, before)


class TestSampleLine:
    """Transect extraction along a line (F2)."""

    def test_linear_field_sampled_monotonically(self, strip_1x4):
        """A line across a left-to-right increasing field yields ordered samples."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        line = LineString([(0.5, 0.5), (3.5, 0.5)])

        distances, sampled, faces = sample_line(strip_1x4, values, line)

        np.testing.assert_array_equal(faces, [0, 1, 2, 3])
        np.testing.assert_array_equal(sampled, [10.0, 20.0, 30.0, 40.0])
        assert np.all(np.diff(distances) > 0)

    def test_accepts_coordinate_sequence(self, strip_1x4):
        """A plain coordinate sequence is coerced to a LineString."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        _, sampled, faces = sample_line(strip_1x4, values, [(0.5, 0.5), (3.5, 0.5)])

        np.testing.assert_array_equal(faces, [0, 1, 2, 3])
        np.testing.assert_array_equal(sampled, [10.0, 20.0, 30.0, 40.0])

    def test_off_mesh_line_returns_no_samples(self, strip_1x4):
        """A line that misses the mesh produces empty results."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        line = LineString([(0.5, -5.0), (3.5, -5.0)])

        distances, sampled, faces = sample_line(strip_1x4, values, line)

        assert distances.shape == (0,)
        assert faces.shape == (0,)
        assert sampled.shape == (0,)

    def test_preserves_leading_time_axis(self, strip_1x4):
        """A temporal field keeps its leading axis through the transect."""
        values = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
        line = LineString([(0.5, 0.5), (3.5, 0.5)])

        _, sampled, faces = sample_line(strip_1x4, values, line)

        assert sampled.shape == (2, 4)
        np.testing.assert_array_equal(faces, [0, 1, 2, 3])
        np.testing.assert_array_equal(sampled[0], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(sampled[1], [1.0, 2.0, 3.0, 4.0])

    def test_rejects_mismatched_values(self, strip_1x4):
        """A value array not matching n_face is rejected."""
        line = LineString([(0, 0), (1, 1)])
        with pytest.raises(ValueError, match="does not match n_face"):
            sample_line(strip_1x4, np.zeros(3), line)

    def test_does_not_mutate_input(self, strip_1x4):
        """Sampling leaves the value array untouched."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        before = values.copy()
        sample_line(strip_1x4, values, LineString([(0.5, 0.5), (3.5, 0.5)]))
        np.testing.assert_array_equal(values, before)


class TestBurn:
    """Rasterizing vector geometry onto faces (F3)."""

    def test_burn_point_sets_containing_face(self, grid_2x2):
        """A point burns only the face that contains it."""
        result = burn(grid_2x2, Point(0.5, 0.5), 7.0)

        assert result[0] == 7.0
        assert np.all(np.isnan(result[1:]))

    def test_point_off_mesh_leaves_all_fill(self, grid_2x2):
        """A point outside every face burns nothing."""
        result = burn(grid_2x2, Point(10.0, 10.0), 7.0)
        assert np.all(np.isnan(result))

    def test_line_burns_crossed_faces(self, grid_2x2):
        """A line burns each face it crosses and leaves the rest at fill."""
        line = LineString([(0.5, 0.5), (1.5, 0.5)])
        result = burn(grid_2x2, line, 5.0)

        assert result[0] == 5.0
        assert result[1] == 5.0
        assert np.all(np.isnan(result[2:]))

    def test_polygon_all_touched_true_selects_every_overlapped_face(self, grid_2x2):
        """``all_touched=True`` burns every face the polygon overlaps."""
        polygon = box(0.0, 0.0, 1.1, 1.1)
        result = burn(grid_2x2, polygon, 1.0, fill=0.0, all_touched=True)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0, 1.0])

    def test_polygon_all_touched_false_uses_centroids(self, grid_2x2):
        """``all_touched=False`` burns only faces whose centroid is inside."""
        polygon = box(0.0, 0.0, 1.1, 1.1)
        result = burn(grid_2x2, polygon, 1.0, fill=0.0, all_touched=False)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0, 0.0])

    def test_accepts_geodataframe(self, grid_2x2):
        """A GeoDataFrame of features is unioned and burned."""
        gdf = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(1.5, 1.5)])
        result = burn(grid_2x2, gdf, 3.0, fill=0.0)

        assert result[0] == 3.0
        assert result[3] == 3.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_output_length_matches_n_face(self, grid_2x2):
        """The burned array has one entry per face."""
        result = burn(grid_2x2, Point(0.5, 0.5), 1.0)
        assert result.shape == (grid_2x2.n_face,)

    def test_does_not_mutate_centroids_cache(self, grid_2x2):
        """Burning does not corrupt the mesh centroid cache."""
        cx_before, cy_before = grid_2x2.face_centroids
        burn(grid_2x2, box(0.0, 0.0, 1.1, 1.1), 1.0, all_touched=False)
        cx_after, cy_after = grid_2x2.face_centroids
        np.testing.assert_array_equal(cx_before, cx_after)
        np.testing.assert_array_equal(cy_before, cy_after)
