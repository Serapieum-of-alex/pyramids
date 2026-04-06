"""UGRID (Unstructured Grid) NetCDF support for pyramids.

This subpackage provides classes for reading, writing, and operating
on UGRID-compliant NetCDF files containing unstructured mesh data.
"""

from __future__ import annotations

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable, UgridMetadata
from pyramids.netcdf.ugrid.spatial import MeshSpatialIndex

__all__ = [
    "Connectivity",
    "Mesh2d",
    "MeshSpatialIndex",
    "MeshTopologyInfo",
    "MeshVariable",
    "UgridDataset",
    "UgridMetadata",
]
