"""CRS / EPSG / reprojection helpers for :mod:`pyramids.feature`.

ARC-10 moved the CRS-related static helpers off
:class:`pyramids.feature.FeatureCollection` into this module so the
collection class can focus on per-feature behavior. The
FeatureCollection class keeps thin static-method delegates for
back-compat: callers that wrote ``FeatureCollection.get_epsg_from_prj(...)``
or ``FeatureCollection.reproject_points(...)`` continue to work
unchanged.
"""

from __future__ import annotations

import numpy as np
from osgeo import ogr, osr
from pyproj import Transformer


def create_sr_from_proj(
    prj: str, string_type: str | None = None
) -> osr.SpatialReference:
    """Create an :class:`osr.SpatialReference` from a projection string.

    Args:
        prj (str):
            The projection string (WKT, ESRI WKT, or Proj4).
        string_type (str | None):
            One of ``"WKT"``, ``"ESRI wkt"``, ``"PROj4"``, or ``None``
            for auto-detect (default). Auto-detect uses WKT import and
            falls back to ESRI WKT or Proj4 based on the prefix.

    Returns:
        osr.SpatialReference: The constructed spatial reference.
    """
    srs = osr.SpatialReference()
    if string_type is None:
        srs.ImportFromWkt(prj)
    elif prj.startswith("PROJCS") or prj.startswith("GEOGCS"):
        srs.ImportFromESRI([prj])
    else:
        srs.ImportFromProj4(prj)
    return srs


def get_epsg_from_prj(prj: str) -> int:
    """Return the EPSG code identified by a projection string.

    Auto-identifies the EPSG from a WKT / ESRI WKT / Proj4 string.

    ARC-7: an empty input string is no longer silently mapped to
    ``4326``. That legacy default masked real configuration errors.
    Callers that genuinely want a fallback should handle the
    ``ValueError`` themselves.

    Args:
        prj (str): Projection string.

    Returns:
        int: The resolved EPSG code.

    Raises:
        ValueError: If ``prj`` is an empty string.
    """
    if prj == "":
        raise ValueError(
            "get_epsg_from_prj received an empty projection string. "
            "An empty projection is ambiguous and is no longer "
            "silently defaulted to EPSG:4326 (ARC-7). If you want "
            "a fallback EPSG, catch ValueError and supply it at "
            "the call site."
        )
    srs = create_sr_from_proj(prj)
    try:
        response = srs.AutoIdentifyEPSG()
    except RuntimeError:
        response = 6

    if response == 0:
        epsg = int(srs.GetAuthorityCode(None))
    else:
        epsg = int(srs.GetAttrValue("AUTHORITY", 1))
    return epsg


def reproject_points(
    lat: list,
    lon: list,
    from_epsg: int = 4326,
    to_epsg: int = 3857,
    precision: int = 6,
) -> tuple[list[float], list[float]]:
    """Reproject point coordinates between EPSG codes.

    Uses :meth:`pyproj.Transformer.from_crs` (ARC-2: replaces the
    legacy ``Proj(init="epsg:…")`` API which pyproj deprecated in
    2.0).

    Args:
        lat (list): Y-coordinates in the source CRS.
        lon (list): X-coordinates in the source CRS.
        from_epsg (int): Source EPSG code. Default 4326.
        to_epsg (int): Target EPSG code. Default 3857.
        precision (int): Decimal places to round to.

    Returns:
        tuple[list[float], list[float]]: ``(y, x)`` lists in the target
        CRS. Kept in ``(y, x)`` order for back-compat;
        :func:`reproject_points_osr` returns ``(x, y)``.
    """
    transformer = Transformer.from_crs(
        f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True
    )
    x = np.full(len(lat), np.nan)
    y = np.full(len(lat), np.nan)
    for i in range(len(lat)):
        x[i], y[i] = np.round(
            transformer.transform(lon[i], lat[i]), precision
        )
    return y.tolist(), x.tolist()


def reproject_points_osr(
    lat: list, lng: list, from_epsg: int = 4326, to_epsg: int = 3857
) -> tuple[list[float], list[float]]:
    """Reproject point coordinates via :class:`osr.CoordinateTransformation`.

    Args:
        lat (list): Y-coordinates in the source CRS.
        lng (list): X-coordinates in the source CRS.
        from_epsg (int): Source EPSG code. Default 4326.
        to_epsg (int): Target EPSG code. Default 3857.

    Returns:
        tuple[list[float], list[float]]: ``(x, y)`` lists in the target CRS.
    """
    source = osr.SpatialReference()
    source.ImportFromEPSG(from_epsg)

    target = osr.SpatialReference()
    target.ImportFromEPSG(to_epsg)

    coord_transform = osr.CoordinateTransformation(source, target)
    x: list[float] = []
    y: list[float] = []
    for i in range(len(lat)):
        point = ogr.CreateGeometryFromWkt(
            "POINT (" + str(lng[i]) + " " + str(lat[i]) + ")"
        )
        point.Transform(coord_transform)
        x.append(point.GetPoints()[0][0])
        y.append(point.GetPoints()[0][1])
    return x, y
