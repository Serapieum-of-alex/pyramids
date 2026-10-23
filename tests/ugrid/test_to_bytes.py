"""Tests for UgridDataset.to_bytes."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

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
    ds.set_crs(epsg=4326)
    return ds


class TestToBytes:
    def test_returns_bytes(self):
        ds = _make_ds()
        data = ds.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_roundtrip(self, tmp_path):
        ds = _make_ds()
        data = ds.to_bytes()
        out = tmp_path / "roundtrip.nc"
        out.write_bytes(data)
        loaded = UgridDataset.read_file(str(out))
        assert loaded.n_face == ds.n_face
        assert loaded.n_node == ds.n_node
