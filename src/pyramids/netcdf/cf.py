"""CF (Climate and Forecast) conventions utilities.

Pure functions for detecting, reading, writing, and validating
CF convention attributes on NetCDF files. Used by both the
structured-grid NetCDF class and the unstructured-grid
UgridDataset class.
"""
from __future__ import annotations

from typing import Any

from osgeo import gdal, osr


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


_GDAL_PROJ_TO_CF: dict[str, str] = {
    "Transverse_Mercator": "transverse_mercator",
    "Lambert_Conformal_Conic_1SP": "lambert_conformal_conic",
    "Lambert_Conformal_Conic_2SP": "lambert_conformal_conic",
    "Albers_Conic_Equal_Area": "albers_conical_equal_area",
    "Mercator_1SP": "mercator",
    "Mercator_2SP": "mercator",
    "Polar_Stereographic": "polar_stereographic",
    "Stereographic": "stereographic",
    "Lambert_Azimuthal_Equal_Area": "lambert_azimuthal_equal_area",
    "Azimuthal_Equidistant": "azimuthal_equidistant",
    "Orthographic": "orthographic",
    "Geostationary_Satellite": "geostationary",
}


def srs_to_grid_mapping(
    srs: osr.SpatialReference,
) -> tuple[str, dict[str, Any]]:
    """Convert an OGR SpatialReference to CF grid_mapping name and params.

    Returns the CF ``grid_mapping_name`` and a dict of CF projection
    parameters (including ``crs_wkt`` for interoperability). For
    geographic CRS (no projection), returns ``"latitude_longitude"``
    with only ellipsoid parameters.

    Args:
        srs: An OGR SpatialReference object.

    Returns:
        Tuple of ``(grid_mapping_name, params_dict)``.
    """
    params: dict[str, Any] = {}
    params["crs_wkt"] = srs.ExportToWkt()
    params["semi_major_axis"] = srs.GetSemiMajor()
    inv_flat = srs.GetInvFlattening()
    if inv_flat > 0:
        params["inverse_flattening"] = inv_flat

    proj_name = srs.GetAttrValue("PROJECTION")
    if proj_name is None:
        grid_mapping_name = "latitude_longitude"
    elif proj_name in _GDAL_PROJ_TO_CF:
        grid_mapping_name = _GDAL_PROJ_TO_CF[proj_name]
        params.update(_extract_proj_params(srs, proj_name))
    else:
        grid_mapping_name = "unknown"

    return grid_mapping_name, params


def _extract_proj_params(
    srs: osr.SpatialReference, proj_name: str
) -> dict[str, Any]:
    """Extract CF projection parameters from an OGR SpatialReference.

    Args:
        srs: OGR SpatialReference with a defined projection.
        proj_name: GDAL projection name string.

    Returns:
        Dict of CF projection parameter names to values.
    """
    p: dict[str, Any] = {}
    fe = srs.GetProjParm(osr.SRS_PP_FALSE_EASTING, 0.0)
    fn = srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING, 0.0)
    if fe != 0.0:
        p["false_easting"] = fe
    if fn != 0.0:
        p["false_northing"] = fn

    if "Transverse_Mercator" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["scale_factor_at_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR, 1.0
        )
    elif "Lambert_Conformal_Conic" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        sp1 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        sp2 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2, 0.0)
        if sp1 == sp2:
            p["standard_parallel"] = sp1
        else:
            p["standard_parallel"] = [sp1, sp2]
    elif "Mercator" in proj_name:
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        sf = srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.0)
        sp = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        if sf != 0.0:
            p["scale_factor_at_projection_origin"] = sf
        if sp != 0.0:
            p["standard_parallel"] = sp
    elif "Polar_Stereographic" in proj_name:
        p["straight_vertical_longitude_from_pole"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        sf = srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.0)
        sp = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        if sf != 0.0:
            p["scale_factor_at_projection_origin"] = sf
        if sp != 0.0:
            p["standard_parallel"] = sp
    elif "Albers" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["standard_parallel"] = [
            srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0),
            srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2, 0.0),
        ]
    elif "Lambert_Azimuthal_Equal_Area" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif "Azimuthal_Equidistant" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif "Orthographic" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif proj_name == "Stereographic":
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["scale_factor_at_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR, 1.0
        )

    return p
