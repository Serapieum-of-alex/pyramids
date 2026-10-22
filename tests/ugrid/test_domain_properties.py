"""Tests for UgridDataset.domain_area and count_domain_cells."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _make_ds() -> UgridDataset:
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
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(2,),
        _data=np.array([10.0, 20.0]),
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


class TestDomainArea:
    def test_positive(self):
        ds = _make_ds()
        assert ds.domain_area() > 0.0

    def test_matches_face_areas_sum(self):
        ds = _make_ds()
        expected = float(ds.mesh.face_areas.sum())
        assert ds.domain_area() == pytest.approx(expected)


class TestCountDomainCells:
    def test_counts_all_faces(self):
        ds = _make_ds()
        assert ds.count_domain_cells() == 2

    def test_counts_active_faces(self):
        ds = _make_ds()
        ds["depth"].active = np.array([True, False])
        assert ds.count_domain_cells("depth") == 1

    def test_no_mask_counts_all(self):
        ds = _make_ds()
        assert ds.count_domain_cells("depth") == 2
