"""Unit tests for pyramids.netcdf.ugrid.io.

Covers topology detection (parse_ugrid_topology), single topology
parsing, and CRS detection using the Western Scheldt test file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from osgeo import gdal

from pyramids.netcdf.ugrid.io import _parse_single_topology, parse_ugrid_topology


class TestParseUgridTopology:
    """Tests for parse_ugrid_topology() function."""

    @pytest.fixture
    def western_scheldt_rg(self, western_scheldt_path):
        """Open the Western Scheldt file and return the root group.

        Returns:
            GDAL root group for the UGRID test file.
        """
        ds = gdal.OpenEx(str(western_scheldt_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        return rg

    def test_detects_mesh_topology(self, western_scheldt_rg):
        """Test that topology detection finds the mesh2d topology.

        Test scenario:
            The Western Scheldt file has a single mesh2d topology
            that should be detected.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        assert len(topologies) >= 1, f"Expected at least 1 topology, got {len(topologies)}"
        topo = topologies[0]
        assert topo.mesh_name == "mesh2d", f"Expected mesh_name 'mesh2d', got '{topo.mesh_name}'"

    def test_topology_dimension(self, western_scheldt_rg):
        """Test that topology_dimension is 2 for the Western Scheldt mesh.

        Test scenario:
            The Western Scheldt is a 2D surface mesh.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert topo.topology_dimension == 2, (
            f"Expected topology_dimension 2, got {topo.topology_dimension}"
        )

    def test_node_coordinate_vars(self, western_scheldt_rg):
        """Test that node coordinate variable names are parsed.

        Test scenario:
            Western Scheldt should have mesh2d_node_x and mesh2d_node_y.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert topo.node_x_var == "mesh2d_node_x", (
            f"Expected node_x_var 'mesh2d_node_x', got '{topo.node_x_var}'"
        )
        assert topo.node_y_var == "mesh2d_node_y", (
            f"Expected node_y_var 'mesh2d_node_y', got '{topo.node_y_var}'"
        )

    def test_face_node_connectivity_var(self, western_scheldt_rg):
        """Test that face_node_connectivity variable name is parsed.

        Test scenario:
            Western Scheldt should have mesh2d_face_nodes.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert topo.face_node_var == "mesh2d_face_nodes", (
            f"Expected 'mesh2d_face_nodes', got '{topo.face_node_var}'"
        )

    def test_edge_node_connectivity_var(self, western_scheldt_rg):
        """Test that edge_node_connectivity variable name is parsed.

        Test scenario:
            Western Scheldt should have mesh2d_edge_nodes.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert topo.edge_node_var == "mesh2d_edge_nodes", (
            f"Expected 'mesh2d_edge_nodes', got '{topo.edge_node_var}'"
        )

    def test_data_variables_detected(self, western_scheldt_rg):
        """Test that data variables with mesh= attribute are detected.

        Test scenario:
            Western Scheldt should detect mesh2d_node_z (node)
            and mesh2d_edge_type (edge).
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert len(topo.data_variables) >= 2, (
            f"Expected at least 2 data variables, got {len(topo.data_variables)}"
        )
        assert "mesh2d_node_z" in topo.data_variables, (
            f"Expected 'mesh2d_node_z' in data_variables, got {list(topo.data_variables.keys())}"
        )
        assert topo.data_variables["mesh2d_node_z"] == "node", (
            f"Expected location 'node', got '{topo.data_variables['mesh2d_node_z']}'"
        )

    def test_face_coordinates_detected(self, western_scheldt_rg):
        """Test that face center coordinate variables are detected.

        Test scenario:
            Western Scheldt should have face center coordinates.
        """
        topologies = parse_ugrid_topology(western_scheldt_rg)
        topo = topologies[0]
        assert topo.face_x_var is not None, "Expected face_x_var to be set"
        assert topo.face_y_var is not None, "Expected face_y_var to be set"

    def test_no_topology_returns_empty(self):
        """Test that a non-UGRID structured NetCDF file returns empty list.

        Test scenario:
            The Noah precipitation file is a regular structured NetCDF
            without mesh_topology — should return [].
        """
        nc_path = "tests/data/netcdf/noah-precipitation-1979.nc"
        if not Path(nc_path).exists():
            pytest.skip("Noah NetCDF test file not available")

        ds = gdal.OpenEx(str(nc_path), gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            pytest.skip("Cannot open Noah NetCDF as MDIM")
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        assert topologies == [], f"Expected empty list, got {topologies}"


class TestTopologyParsingEdgeCases:
    """Tests for topology parsing edge cases (Issue #4)."""

    def test_optional_connectivity_detected(self, western_scheldt_path):
        """Test that optional connectivity variables are parsed.

        Test scenario:
            Western Scheldt has edge_node_connectivity. Verify it is
            detected and stored in the topology info.
        """
        ds = gdal.OpenEx(str(western_scheldt_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        topo = topologies[0]
        assert topo.edge_node_var is not None, (
            "Expected edge_node_connectivity to be detected"
        )

    def test_data_variable_locations(self, western_scheldt_path):
        """Test that data variable locations are correctly parsed.

        Test scenario:
            mesh2d_node_z should be 'node', mesh2d_edge_type should be 'edge'.
        """
        ds = gdal.OpenEx(str(western_scheldt_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        topologies = parse_ugrid_topology(rg)
        topo = topologies[0]
        assert topo.data_variables.get("mesh2d_node_z") == "node", (
            f"Expected 'node', got '{topo.data_variables.get('mesh2d_node_z')}'"
        )
        assert topo.data_variables.get("mesh2d_edge_type") == "edge", (
            f"Expected 'edge', got '{topo.data_variables.get('mesh2d_edge_type')}'"
        )

    def test_parse_single_topology_no_topo_dim_returns_none(
        self, western_scheldt_path
    ):
        """Test that a variable without topology_dimension returns None.

        Test scenario:
            Open a data variable (not topology) and try to parse it.
            Should return None.
        """

        ds = gdal.OpenEx(str(western_scheldt_path), gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray("mesh2d_node_x")
        result = _parse_single_topology(rg, "mesh2d_node_x", md_arr)
        assert result is None, f"Expected None for non-topology variable, got {result}"
