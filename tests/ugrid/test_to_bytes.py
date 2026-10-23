"""Tests for UgridDataset.to_bytes."""

from __future__ import annotations

import pytest

from pyramids.netcdf.ugrid.dataset import UgridDataset
from tests.ugrid.conftest import make_two_triangle_ds

pytestmark = pytest.mark.core


def _make_ds() -> UgridDataset:
    ds = make_two_triangle_ds()
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
