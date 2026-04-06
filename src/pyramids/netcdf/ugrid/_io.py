"""UGRID topology detection and NetCDF I/O.

Handles reading UGRID mesh topology from NetCDF files using the
GDAL MDIM API, and writing UGRID-compliant NetCDF files.

Depends on:
    - cf.py: grid_mapping_to_srs() for CRS reconstruction
    - utils.py: _read_attributes() for reading GDAL attributes
"""

from __future__ import annotations

from typing import Any

import numpy as np
from osgeo import gdal

from pyramids.netcdf.cf import grid_mapping_to_srs
from pyramids.netcdf.ugrid._models import MeshTopologyInfo
from pyramids.netcdf.utils import _read_attributes


def parse_ugrid_topology(rg: gdal.Group) -> list[MeshTopologyInfo]:
    """Detect and parse all UGRID mesh topologies from a GDAL root group.

    Detection strategy (handles diverse real-world files):

    1. Primary: Scan all MDArrays for ``cf_role = "mesh_topology"`` attribute.
    2. Fallback: Scan for variables with ``topology_dimension`` AND
       ``node_coordinates`` attributes (older files without cf_role).
    3. Scalar check: GDAL may filter 0-dimensional arrays from
       ``GetMDArrayNames()``. Explicitly try ``OpenMDArray(name)`` for
       variable names found in other variables' ``mesh`` attributes.

    Args:
        rg: GDAL root group from a multidimensional NetCDF file.

    Returns:
        List of MeshTopologyInfo, one per mesh found in the file.
        Most files have exactly one mesh. Empty list if no UGRID topology.
    """
    topologies: list[MeshTopologyInfo] = []
    all_array_names = rg.GetMDArrayNames() or []

    mesh_var_names: list[str] = []
    for name in all_array_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        cf_role = attrs.get("cf_role", "")
        has_topo = "topology_dimension" in attrs and "node_coordinates" in attrs
        if cf_role == "mesh_topology" or has_topo:
            mesh_var_names.append(name)

    referenced_meshes: set[str] = set()
    for name in all_array_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        mesh_ref = attrs.get("mesh")
        if isinstance(mesh_ref, str) and mesh_ref not in all_array_names:
            referenced_meshes.add(mesh_ref)

    for mesh_name in referenced_meshes:
        if mesh_name not in mesh_var_names:
            md_arr = rg.OpenMDArray(mesh_name)
            if md_arr is not None:
                attrs = _read_attributes(md_arr)
                if "node_coordinates" in attrs:
                    mesh_var_names.append(mesh_name)

    for name in mesh_var_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        topo = _parse_single_topology(rg, name, md_arr)
        if topo is not None:
            topologies.append(topo)

    return topologies


def _parse_single_topology(
    rg: gdal.Group,
    mesh_name: str,
    md_arr: gdal.MDArray,
) -> MeshTopologyInfo | None:
    """Parse a single mesh topology variable into MeshTopologyInfo.

    Reads all UGRID-standard attributes from the topology variable
    and scans all other variables in the root group for data variables
    that reference this mesh.

    Args:
        rg: GDAL root group.
        mesh_name: Name of the topology variable.
        md_arr: The topology MDArray.

    Returns:
        MeshTopologyInfo or None if the variable lacks required attributes.
    """
    attrs = _read_attributes(md_arr)
    topo_dim = attrs.get("topology_dimension")
    if topo_dim is None:
        return None
    topo_dim = int(topo_dim)

    node_coords_str = attrs.get("node_coordinates", "")
    node_parts = node_coords_str.split() if isinstance(node_coords_str, str) else []
    node_x_var = node_parts[0] if len(node_parts) > 0 else None
    node_y_var = node_parts[1] if len(node_parts) > 1 else None

    if node_x_var is None or node_y_var is None:
        return None

    face_node_var = attrs.get("face_node_connectivity")
    edge_node_var = attrs.get("edge_node_connectivity")
    face_edge_var = attrs.get("face_edge_connectivity")
    face_face_var = attrs.get("face_face_connectivity")
    edge_face_var = attrs.get("edge_face_connectivity")
    boundary_node_var = attrs.get("boundary_node_connectivity")

    face_coords_str = attrs.get("face_coordinates", "")
    face_parts = face_coords_str.split() if isinstance(face_coords_str, str) else []
    face_x_var = face_parts[0] if len(face_parts) > 0 else None
    face_y_var = face_parts[1] if len(face_parts) > 1 else None

    edge_coords_str = attrs.get("edge_coordinates", "")
    edge_parts = edge_coords_str.split() if isinstance(edge_coords_str, str) else []
    edge_x_var = edge_parts[0] if len(edge_parts) > 0 else None
    edge_y_var = edge_parts[1] if len(edge_parts) > 1 else None

    data_variables: dict[str, str] = {}
    all_array_names = rg.GetMDArrayNames() or []
    for var_name in all_array_names:
        var_arr = rg.OpenMDArray(var_name)
        if var_arr is None:
            continue
        var_attrs = _read_attributes(var_arr)
        if var_attrs.get("mesh") == mesh_name:
            location = var_attrs.get("location", "unknown")
            if isinstance(location, str):
                data_variables[var_name] = location

    crs_wkt = _detect_crs(rg, node_x_var)

    result = MeshTopologyInfo(
        mesh_name=mesh_name,
        topology_dimension=topo_dim,
        node_x_var=node_x_var,
        node_y_var=node_y_var,
        face_node_var=face_node_var,
        edge_node_var=edge_node_var,
        face_edge_var=face_edge_var,
        face_face_var=face_face_var,
        edge_face_var=edge_face_var,
        boundary_node_var=boundary_node_var,
        face_x_var=face_x_var,
        face_y_var=face_y_var,
        edge_x_var=edge_x_var,
        edge_y_var=edge_y_var,
        data_variables=data_variables,
        crs_wkt=crs_wkt,
    )
    return result


def _detect_crs(rg: gdal.Group, node_x_var: str) -> str | None:
    """Detect CRS from node coordinate spatial reference or grid_mapping variable.

    Tries multiple strategies:
    1. Spatial reference on node coordinate variable.
    2. projected_coordinate_system / crs / spatial_ref variable with crs_wkt.
    3. Grid mapping variable with grid_mapping_name (uses cf.grid_mapping_to_srs).

    Args:
        rg: GDAL root group.
        node_x_var: Name of the node x-coordinate variable.

    Returns:
        CRS WKT string, or None if no CRS can be determined.
    """
    crs_wkt = None
    node_x_arr = rg.OpenMDArray(node_x_var)
    if node_x_arr is not None:
        srs = node_x_arr.GetSpatialRef()
        if srs is not None:
            crs_wkt = srs.ExportToWkt()

    if crs_wkt is not None:
        return crs_wkt

    for candidate in ("projected_coordinate_system", "crs", "spatial_ref"):
        crs_arr = rg.OpenMDArray(candidate)
        if crs_arr is None:
            continue
        crs_attrs = _read_attributes(crs_arr)
        wkt = crs_attrs.get("crs_wkt") or crs_attrs.get("spatial_ref")
        if isinstance(wkt, str):
            crs_wkt = wkt
            break
        gmn = crs_attrs.get("grid_mapping_name")
        if isinstance(gmn, str):
            try:
                srs = grid_mapping_to_srs(gmn, crs_attrs)
                crs_wkt = srs.ExportToWkt()
            except (ValueError, RuntimeError):
                pass
            break

    return crs_wkt
