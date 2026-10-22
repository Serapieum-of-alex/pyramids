"""Tests for UgridDataset.extract and UgridDataset.sample."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _make_ds(
    values: np.ndarray | None = None,
) -> UgridDataset:
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=fnc,
    )
    if values is None:
        values = np.array([10.0, 20.0])
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=values.shape,
        _data=values,
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


class TestExtract:
    def test_at_face_centroids(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        cx, cy = ds.mesh.face_centroids
        result = ds.extract(cx, cy)
        assert result.shape == (2,)
        assert 10.0 in result or 20.0 in result

    def test_outside_mesh_returns_nan(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        result = ds.extract(
            np.array([-100.0]), np.array([-100.0])
        )
        assert np.isnan(result[0])

    def test_single_point(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        cx, cy = ds.mesh.face_centroids
        result = ds.extract(cx[0:1], cy[0:1])
        assert result.shape == (1,)


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
        assert np.all(x >= xmin) and np.all(x <= xmax)
        assert np.all(y >= ymin) and np.all(y <= ymax)

    def test_reproducible_with_seed(self):
        ds = _make_ds(np.array([10.0, 20.0]))
        x1, y1, v1 = ds.sample_random(5, seed=123)
        x2, y2, v2 = ds.sample_random(5, seed=123)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
