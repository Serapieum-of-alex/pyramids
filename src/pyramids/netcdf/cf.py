"""CF (Climate and Forecast) conventions utilities.

Pure functions for detecting, reading, writing, and validating
CF convention attributes on NetCDF files. Used by both the
structured-grid NetCDF class and the unstructured-grid
UgridDataset class.
"""
from __future__ import annotations

from typing import Any

from osgeo import gdal


def write_attributes_to_md_array(
    md_arr: gdal.MDArray,
    attrs: dict[str, Any],
) -> None:
    """Write a dict of attributes to a GDAL MDArray.

    Handles str, int, float, and list values. Silently skips
    attributes that can't be written (GDAL limitation).

    Args:
        md_arr: The GDAL MDArray to write attributes to.
        attrs: Dict of attribute names to values.
    """
    for key, value in attrs.items():
        try:
            if isinstance(value, str):
                attr = md_arr.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
            elif isinstance(value, float):
                attr = md_arr.CreateAttribute(
                    key, [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                )
            elif isinstance(value, int):
                attr = md_arr.CreateAttribute(
                    key, [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Int32),
                )
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    attr = md_arr.CreateAttribute(
                        key,
                        [len(value)],
                        gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                    )
                else:
                    attr = md_arr.CreateAttribute(
                        key, [],
                        gdal.ExtendedDataType.CreateString(),
                    )
                    value = str(value)
            else:
                attr = md_arr.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
                value = str(value)
            attr.Write(value)
        except Exception:
            pass  # nosec B110


def build_coordinate_attrs(
    dim_name: str,
    is_geographic: bool = True,
) -> dict[str, str]:
    """Generate CF-compliant attributes for a coordinate variable.

    Maps dimension names to the appropriate CF ``axis``,
    ``standard_name``, ``long_name``, and ``units`` attributes
    based on whether the CRS is geographic or projected.

    Args:
        dim_name: Dimension name (e.g. ``"x"``, ``"y"``, ``"lat"``,
            ``"lon"``, ``"time"``).
        is_geographic: True if the CRS is geographic (lon/lat),
            False if projected (easting/northing in metres).

    Returns:
        Dict of CF attribute names to string values. Empty dict
        if the dimension name is not recognized.
    """
    name_lower = dim_name.lower()
    attrs: dict[str, str] = {}

    if name_lower in ("x", "lon", "longitude"):
        attrs["axis"] = "X"
        if is_geographic:
            attrs["standard_name"] = "longitude"
            attrs["long_name"] = "longitude"
            attrs["units"] = "degrees_east"
        else:
            attrs["standard_name"] = "projection_x_coordinate"
            attrs["long_name"] = "x coordinate of projection"
            attrs["units"] = "m"
    elif name_lower in ("y", "lat", "latitude"):
        attrs["axis"] = "Y"
        if is_geographic:
            attrs["standard_name"] = "latitude"
            attrs["long_name"] = "latitude"
            attrs["units"] = "degrees_north"
        else:
            attrs["standard_name"] = "projection_y_coordinate"
            attrs["long_name"] = "y coordinate of projection"
            attrs["units"] = "m"
    elif name_lower in ("time", "t"):
        attrs["axis"] = "T"
        attrs["standard_name"] = "time"
        attrs["long_name"] = "time"
    elif name_lower in ("z", "lev", "level", "depth", "height"):
        attrs["axis"] = "Z"
        attrs["long_name"] = dim_name

    return attrs


def write_global_attributes(
    rg: gdal.Group,
    attrs: dict[str, Any],
) -> None:
    """Write a dict of attributes to a GDAL root group.

    Args:
        rg: The GDAL root group to write attributes to.
        attrs: Dict of attribute names to values.
    """
    for key, value in attrs.items():
        try:
            if isinstance(value, str):
                attr = rg.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
            elif isinstance(value, float):
                attr = rg.CreateAttribute(
                    key, [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                )
            elif isinstance(value, int):
                attr = rg.CreateAttribute(
                    key, [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Int32),
                )
            else:
                attr = rg.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
                value = str(value)
            attr.Write(value)
        except Exception:
            pass  # nosec B110
