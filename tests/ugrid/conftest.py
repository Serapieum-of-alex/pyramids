"""Shared fixtures for UGRID tests.

Provides reusable mesh topologies, connectivity arrays, and
test file paths used across multiple test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid._connectivity import Connectivity
from pyramids.netcdf.ugrid._mesh import Mesh2d
from pyramids.netcdf.ugrid._models import MeshTopologyInfo, MeshVariable


@pytest.fixture(scope="session")
def western_scheldt_path():
    """Path to the Western Scheldt UGRID test file.

    Returns:
        Path: Absolute path to westernscheldt01_waqgeom.nc.
    """
    return Path("tests/mo/netcdf/westernscheldt01_waqgeom.nc")


@pytest.fixture
def triangle_mesh():
    """Minimal 2-face triangular mesh.

    Layout::

         2---3
        / \\ / \\
       0---1---4

    Returns:
        Mesh2d: 5 nodes, 2 triangular faces.
    """
    node_x = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    faces = np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp)
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces, fill_value=-1,
            cf_role="face_node_connectivity", original_start_index=0,
        ),
    )


@pytest.fixture
def mixed_mesh():
    """3-face mixed mesh: 1 quad + 2 triangles.

    Layout::

       3---4---5
       |   |  /
       0---1-2

    Returns:
        Mesh2d: 6 nodes, 3 faces (1 quad, 2 triangles).
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    faces = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, -1],
        [1, 5, 4, -1],
    ], dtype=np.intp)
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces, fill_value=-1,
            cf_role="face_node_connectivity", original_start_index=0,
        ),
    )


@pytest.fixture
def mesh_with_face_data(triangle_mesh):
    """Triangle mesh with face-centered data variable.

    Returns:
        Tuple of (Mesh2d, dict of MeshVariable).
    """
    data_vars = {
        "water_level": MeshVariable(
            name="water_level",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            attributes={"units": "m"},
            nodata=-999.0,
            units="m",
            standard_name="sea_surface_height",
            _data=np.array([1.5, 2.3]),
        ),
    }
    return triangle_mesh, data_vars


@pytest.fixture
def sample_topo_info():
    """Sample MeshTopologyInfo for a 2D mesh.

    Returns:
        MeshTopologyInfo: Minimal 2D mesh topology metadata.
    """
    return MeshTopologyInfo(
        mesh_name="mesh2d",
        topology_dimension=2,
        node_x_var="mesh2d_node_x",
        node_y_var="mesh2d_node_y",
        face_node_var="mesh2d_face_nodes",
        edge_node_var="mesh2d_edge_nodes",
        data_variables={"mesh2d_node_z": "node", "mesh2d_edge_type": "edge"},
    )
