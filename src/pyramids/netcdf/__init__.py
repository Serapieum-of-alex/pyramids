"""NetCDF subpackage for pyramids."""

from __future__ import annotations

from pyramids.netcdf.metadata import from_json, get_metadata, to_dict, to_json
from pyramids.netcdf.models import (
    ArrayInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
)
from pyramids.netcdf.netcdf import NetCDF

__all__ = [
    "NetCDF",
    "NetCDFMetadata",
    "DimensionInfo",
    "ArrayInfo",
    "GroupInfo",
    "StructuralInfo",
    "get_metadata",
    "to_json",
    "from_json",
    "to_dict",
]
