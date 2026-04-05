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
