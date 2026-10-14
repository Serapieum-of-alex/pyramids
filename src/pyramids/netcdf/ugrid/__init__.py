"""UGRID (Unstructured Grid) NetCDF support for pyramids.

This subpackage provides classes for reading, writing, and operating
on UGRID-compliant NetCDF files containing unstructured mesh data.

Beyond the core data model (:class:`UgridDataset`, :class:`Mesh2d`, :class:`Network1d`) it
carries the mesh operation modules: :mod:`~pyramids.netcdf.ugrid.graph` (topology and graph
operations), :mod:`~pyramids.netcdf.ugrid.fill` (spatial gap filling),
:mod:`~pyramids.netcdf.ugrid.vector_ops` (mesh↔vector), :mod:`~pyramids.netcdf.ugrid.regrid`
(sampling and regridding), :mod:`~pyramids.netcdf.ugrid.construct` (building a mesh from a
raster or polygons), :mod:`~pyramids.netcdf.ugrid.interop_xarray` (xarray / zarr hand-off) and
:mod:`~pyramids.netcdf.ugrid.network` (1-D networks).
Each is reachable through a :class:`UgridDataset` facade; the modules are exported too for
callers holding a bare :class:`Mesh2d`.
"""

from __future__ import annotations

from pyramids.netcdf.ugrid import (
    construct,
    fill,
    graph,
    interop_xarray,
    network,
    regrid,
    vector_ops,
)
from pyramids.netcdf.ugrid._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    MESH_DRIVER_API_VERSION,
    NETCDF_SUFFIXES,
    RESERVED_NAMES,
    MeshDriver,
    UnsupportedMeshFormatError,
    mesh_drivers,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable, UgridMetadata
from pyramids.netcdf.ugrid.network import Network1d
from pyramids.netcdf.ugrid.regrid import Regridder
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

__all__ = [
    "CAPABILITIES",
    "ENTRY_POINT_GROUP",
    "MESH_DRIVER_API_VERSION",
    "NETCDF_SUFFIXES",
    "RESERVED_NAMES",
    "Connectivity",
    "MeshDriver",
    "Mesh2d",
    "MeshSpatialIndex",
    "MeshTopologyInfo",
    "MeshVariable",
    "Network1d",
    "Regridder",
    "UgridDataset",
    "UgridMetadata",
    "UnsupportedMeshFormatError",
    "construct",
    "fill",
    "graph",
    "interop_xarray",
    "mesh_drivers",
    "network",
    "regrid",
    "vector_ops",
]
