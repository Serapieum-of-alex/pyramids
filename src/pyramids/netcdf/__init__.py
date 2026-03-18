"""NetCDF subpackage for pyramids."""

from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.models import NetCDFMetadata, DimensionInfo, ArrayInfo, GroupInfo
from pyramids.netcdf.metadata import get_metadata, to_json, from_json, to_dict

__all__ = [
    "NetCDF",
    "NetCDFMetadata",
    "DimensionInfo",
    "ArrayInfo",
    "GroupInfo",
    "get_metadata",
    "to_json",
    "from_json",
    "to_dict",
]
