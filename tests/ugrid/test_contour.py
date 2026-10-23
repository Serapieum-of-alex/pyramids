"""Tests for UgridDataset.contour."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pyproj import CRS

os.environ.setdefault("MPLBACKEND", "Agg")

from pyramids.netcdf.ugrid.analysis import contour_lines
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = [pytest.mark.core, pytest.mark.plot]


def _make_ds() -> UgridDataset:
    fnc = Connectivity(
        data=np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.intp,
        ),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 0.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
        face_node_connectivity=fnc,
    )
    var = MeshVariable(
        name="elevation",
        location="face",
        mesh_name="mesh2d",
        shape=(4,),
        _data=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"elevation": var},
        global_attributes={},
    )
    return ds


class TestContour:
    def test_returns_geodataframe(self):
        ds = _make_ds()
        gdf = ds.contour(levels=3)
        assert hasattr(gdf, "geometry")
        assert "level" in gdf.columns

    def test_explicit_levels(self):
        ds = _make_ds()
        gdf = ds.contour(levels=[1.5, 2.5, 3.5])
        assert hasattr(gdf, "geometry")

    def test_default_levels(self):
        """Contour with levels=None falls back to 10 levels.

        Test scenario:
            Calling contour() without specifying levels uses the
            default (10), producing a valid GeoDataFrame.
        """
        ds = _make_ds()
        gdf = ds.contour()
        assert hasattr(gdf, "geometry"), (
            "Default-levels contour should return a GeoDataFrame"
        )

    def test_node_location_contour(self):
        """Contour with a node-located variable bypasses face-to-node averaging.

        Test scenario:
            A node-located variable passed to contour_lines with
            location='node' uses the values directly.
        """
        fnc = Connectivity(
            data=np.array(
                [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
                dtype=np.intp,
            ),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 0.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
            face_node_connectivity=fnc,
        )
        node_values = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
        gdf = contour_lines(mesh, node_values, levels=3, location="node")
        assert hasattr(gdf, "geometry"), (
            "Node-location contour should return a GeoDataFrame"
        )

    def test_crs_propagated_to_geodataframe(self):
        """Contour output GeoDataFrame carries the mesh CRS."""
        ds = _make_ds()
        ds.set_crs(epsg=4326)
        gdf = ds.contour(levels=3)
        assert gdf.crs is not None
        assert CRS(gdf.crs).to_epsg() == 4326

    def test_temporal_data_raises(self):
        """contour must reject temporal (2-D) data."""
        fnc = Connectivity(
            data=np.array(
                [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
                dtype=np.intp,
            ),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 1.0, 0.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
            face_node_connectivity=fnc,
        )
        temporal_values = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        )
        var = MeshVariable(
            name="elevation",
            location="face",
            mesh_name="mesh2d",
            shape=temporal_values.shape,
            _data=temporal_values,
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={"elevation": var},
            global_attributes={},
        )
        with pytest.raises(ValueError, match="temporal"):
            ds.contour(levels=3)
