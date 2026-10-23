"""Tests for UgridDataset.set_crs and the Mesh2d CRS setter."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from tests.ugrid.conftest import make_single_triangle_ds as _crs_less_dataset

pytestmark = pytest.mark.core


class TestSetCrsFromEpsg:
    def test_epsg_is_assigned(self):
        ds = _crs_less_dataset()
        assert ds.epsg is None
        ds.set_crs(epsg=4326)
        assert ds.epsg == 4326

    def test_crs_wkt_is_populated(self):
        ds = _crs_less_dataset()
        assert ds.crs_wkt is None
        ds.set_crs(epsg=4326)
        assert ds.crs_wkt is not None
        assert "WGS" in ds.crs_wkt

    def test_crs_object_is_usable(self):
        ds = _crs_less_dataset()
        ds.set_crs(epsg=32631)
        assert isinstance(ds.crs, CRS)
        assert ds.crs.to_epsg() == 32631

    def test_mesh_crs_is_updated(self):
        ds = _crs_less_dataset()
        ds.set_crs(epsg=4326)
        assert ds.mesh.crs is not None


class TestSetCrsFromWkt:
    def test_wkt_is_assigned(self):
        wkt = CRS.from_epsg(4326).to_wkt()
        ds = _crs_less_dataset()
        ds.set_crs(crs=wkt)
        assert ds.epsg == 4326

    def test_projected_wkt(self):
        wkt = CRS.from_epsg(32631).to_wkt()
        ds = _crs_less_dataset()
        ds.set_crs(crs=wkt)
        assert ds.crs.to_epsg() == 32631


class TestSetCrsValidation:
    def test_neither_raises(self):
        ds = _crs_less_dataset()
        with pytest.raises(ValueError, match="Exactly one"):
            ds.set_crs()

    def test_both_raises(self):
        ds = _crs_less_dataset()
        wkt = CRS.from_epsg(4326).to_wkt()
        with pytest.raises(ValueError, match="Exactly one"):
            ds.set_crs(crs=wkt, epsg=4326)


class TestMeshCrsSetterValidation:
    def test_rejects_string(self):
        """Mesh2d.crs setter should reject non-CRS types."""
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
        with pytest.raises(TypeError, match="pyproj.CRS"):
            mesh.crs = "EPSG:4326"

    def test_rejects_integer(self):
        """Mesh2d.crs setter should reject plain integers."""
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
        with pytest.raises(TypeError, match="pyproj.CRS"):
            mesh.crs = 4326

    def test_accepts_none(self):
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
        mesh.crs = None
        assert mesh.crs is None


class TestSetCrsUnblocksToCrs:
    def test_to_crs_after_set_crs(self):
        ds = _crs_less_dataset()
        ds.set_crs(epsg=4326)
        reprojected = ds.to_crs(32631)
        assert reprojected.epsg == 32631

    def test_to_crs_without_set_crs_still_raises(self):
        ds = _crs_less_dataset()
        with pytest.raises(ValueError, match="source CRS is unknown"):
            ds.to_crs(32631)


class TestSetCrsPreservesData:
    def test_data_unchanged(self):
        ds = _crs_less_dataset(with_data=True)
        ds.set_crs(epsg=4326)
        assert ds["depth"].data[0] == pytest.approx(5.0)

    def test_coordinates_unchanged(self):
        ds = _crs_less_dataset()
        original_x = ds.mesh.node_x.copy()
        original_y = ds.mesh.node_y.copy()
        ds.set_crs(epsg=4326)
        np.testing.assert_array_equal(ds.mesh.node_x, original_x)
        np.testing.assert_array_equal(ds.mesh.node_y, original_y)


class TestSetCrsOverwrite:
    def test_overwrite_existing_crs(self):
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=4326,
        )
        assert ds.epsg == 4326
        ds.set_crs(epsg=32631)
        assert ds.epsg == 32631


class TestSetCrsWktNormalization:
    def test_wkt_and_epsg_produce_identical_datasets(self):
        """set_crs via WKT and via EPSG for the same CRS must produce
        identical datasets (H2: WKT normalization)."""
        ds_epsg = _crs_less_dataset()
        ds_epsg.set_crs(epsg=4326)
        ds_wkt = _crs_less_dataset()
        ds_wkt.set_crs(crs=CRS.from_epsg(4326).to_wkt())
        assert ds_epsg.identical(ds_wkt)


class TestMesh2dCrsSetter:
    def test_setter(self):
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
        assert mesh.crs is None
        crs_obj = CRS.from_epsg(4326)
        mesh.crs = crs_obj
        assert mesh.crs == crs_obj
