"""Tests for UgridDataset.extract and UgridDataset.sample."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyramids.netcdf.ugrid.analysis import extract_points, sample_random
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from tests.ugrid.conftest import make_two_triangle_ds as _make_ds

pytestmark = pytest.mark.core


class TestExtract:
    def test_at_face_centroids(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        cx, cy = ds.mesh.face_centroids
        result = ds.extract(cx, cy)
        assert result.shape == (2,)
        assert 10.0 in result or 20.0 in result

    def test_outside_mesh_returns_nan(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.extract(np.array([-100.0]), np.array([-100.0]))
        assert np.isnan(result[0])

    def test_single_point(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        cx, cy = ds.mesh.face_centroids
        result = ds.extract(cx[0:1], cy[0:1])
        assert result.shape == (1,)


class TestExtractNodeLocation:
    """Extract with location='node' — nearest-node lookup."""

    def test_returns_nearest_node_value(self):
        """Query near a known node returns that node's value."""
        fnc = Connectivity(
            data=np.array([[0, 1, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=fnc,
        )
        node_vals = np.array([100.0, 200.0, 300.0])
        var = MeshVariable(
            name="temp",
            location="node",
            mesh_name="mesh2d",
            shape=node_vals.shape,
            _data=node_vals,
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={"temp": var},
            global_attributes={},
        )
        result = extract_points(
            mesh,
            node_vals,
            np.array([0.01]),
            np.array([0.01]),
            location="node",
        )
        assert result[0] == 100.0, f"Expected node 0 value 100.0, got {result[0]}"


class TestExtractInvalidLocation:
    def test_raises_on_bad_location(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        with pytest.raises(ValueError, match="location must be"):
            extract_points(
                ds.mesh,
                np.array([10.0, 20.0]),
                np.array([0.5]),
                np.array([0.5]),
                location="edge",
            )


class TestExtractTemporal:
    """Extract with 2-D (temporal) values."""

    def test_temporal_values_shape(self):
        """Extracting from a (n_time, n_face) array returns (n_time, n_points).

        Test scenario:
            A 2-D values array with 3 time steps and 2 faces, queried
            at the first face centroid, returns shape (3, 1).
        """
        ds = _make_ds(np.array([10.0, 20.0]))
        temporal_vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cx, cy = ds.mesh.face_centroids
        result = extract_points(
            ds.mesh,
            temporal_vals,
            cx[0:1],
            cy[0:1],
            location="face",
        )
        assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"


class TestSampleRandomValidation:
    def test_negative_n_raises(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        with pytest.raises(ValueError, match="n.*must be >= 1"):
            ds.sample_random(-5)

    def test_zero_n_raises(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        with pytest.raises(ValueError, match="n.*must be >= 1"):
            ds.sample_random(0)


class TestSampleRandomDegenerate:
    """Test sample_random RuntimeError on a degenerate mesh."""

    def test_raises_on_degenerate_mesh(self):
        """A mesh where rejection sampling cannot find interior points.

        Test scenario:
            Mock MeshSpatialIndex.locate_faces to always return -1
            (outside), so the loop exceeds max_iters and raises
            RuntimeError.
        """
        ds = _make_ds(np.array([10.0, 20.0]))

        def _always_outside(self, x, y):
            return np.full(len(x), -1, dtype=np.intp)

        with patch("pyramids.netcdf.ugrid.analysis.MeshSpatialIndex") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.locate_faces = lambda x, y: np.full(len(x), -1, dtype=np.intp)
            with pytest.raises(RuntimeError, match="degenerate"):
                sample_random(ds.mesh, np.array([10.0, 20.0]), 5)


class TestSampleRandom:
    def test_returns_n_samples(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        x, y, vals = ds.sample_random(5, seed=42)
        assert len(x) == 5
        assert len(y) == 5
        assert len(vals) == 5

    def test_within_bounds(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        x, y, vals = ds.sample_random(10, seed=42)
        xmin, ymin, xmax, ymax = ds.bounds
        assert np.all(x >= xmin)
        assert np.all(x <= xmax)
        assert np.all(y >= ymin)
        assert np.all(y <= ymax)

    def test_reproducible_with_seed(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        x1, y1, v1 = ds.sample_random(5, seed=123)
        x2, y2, v2 = ds.sample_random(5, seed=123)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
