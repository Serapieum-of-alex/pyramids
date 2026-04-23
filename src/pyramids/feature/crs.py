"""CRS / EPSG / reprojection helpers for :mod:`pyramids.feature`.

ARC-10 moved the CRS-related static helpers off
:class:`pyramids.feature.FeatureCollection` into this module so the
collection class can focus on per-feature behavior. The
FeatureCollection class keeps thin static-method delegates for
symbolic continuity.

ARC-14 collapsed the two previous reprojection helpers
(``reproject_points`` taking ``(lat, lon)`` and returning ``(y, x)``,
``reproject_points_osr``/``reproject_points2`` taking ``(lat, lng)``
and returning ``(x, y)``) into a single canonical function
:func:`reproject_coordinates` with consistent ``(x, y)`` order. The
old functions were deleted outright — no deprecation shims — per the
branch's refactor policy (the inconsistent axis order was a latent
foot-gun and keeping shims perpetuated the confusion).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from osgeo import osr
from pyproj import Transformer

from pyramids.base._errors import CRSError


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

    Examples:
        - Parse a standard EPSG:4326 WKT string and inspect the result:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(4326)
            >>> wkt = ref.ExportToWkt()
            >>> srs = create_sr_from_proj(wkt)
            >>> srs.IsGeographic()
            1
            >>> srs.GetName()
            'WGS 84'

            ```
        - Parse a Proj4 string by passing ``string_type="PROJ4"``:
            ```python
            >>> srs = create_sr_from_proj(
            ...     "+proj=longlat +datum=WGS84 +no_defs", string_type="PROJ4"
            ... )
            >>> srs.IsGeographic()
            1
            >>> srs.IsProjected()
            0

            ```
        - Parse an EPSG:3857 WKT and confirm the axis order is projected:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> srs = create_sr_from_proj(ref.ExportToWkt())
            >>> srs.IsProjected()
            1
            >>> srs.GetName()
            'WGS 84 / Pseudo-Mercator'

            ```
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

    Examples:
        - Resolve EPSG:4326 from its standard WKT representation:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(4326)
            >>> get_epsg_from_prj(ref.ExportToWkt())
            4326

            ```
        - Resolve EPSG:3857 (Web Mercator) from its WKT representation:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> get_epsg_from_prj(ref.ExportToWkt())
            3857

            ```
        - An empty projection string raises ``CRSError`` (a ``ValueError`` subclass):
            ```python
            >>> get_epsg_from_prj("")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.CRSError: get_epsg_from_prj received an empty projection string. ...

            ```
    """
    if prj == "":
        raise CRSError(
            "get_epsg_from_prj received an empty projection string. "
            "An empty projection is ambiguous and is no longer "
            "silently defaulted to EPSG:4326 (ARC-7). If you want "
            "a fallback EPSG, catch CRSError (also a ValueError) "
            "and supply it at the call site."
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


def reproject_coordinates(
    x: list[float],
    y: list[float],
    *,
    from_crs: Any = 4326,
    to_crs: Any = 3857,
    precision: int | None = 6,
) -> tuple[list[float], list[float]]:
    """Reproject parallel x / y coordinate lists between CRSes.

    ARC-14: canonical replacement for the two former helpers
    (``reproject_points`` and ``reproject_points_osr`` /
    ``reproject_points2``). Unambiguous ``(x, y)`` argument and return
    order throughout; accepts any CRS form
    :meth:`pyproj.Transformer.from_crs` understands (EPSG int, EPSG
    string, WKT, Proj4, :class:`pyproj.CRS`).

    Args:
        x (list[float]):
            X-coordinates in the source CRS (longitudes when
            ``from_crs`` is geographic).
        y (list[float]):
            Y-coordinates in the source CRS (latitudes when
            ``from_crs`` is geographic).
        from_crs:
            Source CRS. Accepts anything
            :meth:`pyproj.Transformer.from_crs` accepts: EPSG integer
            (``4326``), authority string (``"EPSG:4326"``), WKT, Proj4,
            or a :class:`pyproj.CRS` instance. Default ``4326``.
        to_crs:
            Target CRS, same forms as ``from_crs``. Default ``3857``.
        precision (int | None):
            Decimal places to round each returned coordinate to. Pass
            ``None`` to disable rounding. Default ``6``.

    Returns:
        tuple[list[float], list[float]]: ``(x, y)`` in the target CRS.

    Raises:
        ValueError: If ``len(x) != len(y)``.
        CRSError: If :meth:`pyproj.Transformer.from_crs` raises one
            of ``pyproj.exceptions.CRSError`` (malformed WKT / proj
            string), ``TypeError`` (input is not CRS-like — e.g. a
            bare ``object()``), or ``ValueError`` (out-of-range EPSG
            integer). The wrapper converts each into pyramids'
            :class:`pyramids.base._errors.CRSError` so callers do not
            need to import pyproj to catch bad-CRS failures, and the
            message names both CRSes plus the underlying explanation.
            Other exception types (``AttributeError``, ``ImportError``,
            …) propagate unchanged — they signal a real bug, not a bad
            user input (M1 narrowed the original blanket
            ``except Exception``).

    Examples:
        - Reproject two WGS84 points into Web Mercator:
            ```python
            >>> x, y = reproject_coordinates(
            ...     [31.0], [30.0], from_crs=4326, to_crs=3857
            ... )
            >>> round(x[0])
            3450904
            >>> round(y[0])
            3503550

            ```
        - Accepts WKT / authority strings / :class:`pyproj.CRS`:
            ```python
            >>> x, y = reproject_coordinates(
            ...     [31.0], [30.0], from_crs="EPSG:4326", to_crs="EPSG:3857"
            ... )
            >>> len(x) == 1 and len(y) == 1
            True

            ```
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have equal length; got len(x)={len(x)} "
            f"vs. len(y)={len(y)}."
        )
    # C23 / M1: ``pyproj.Transformer.from_crs`` raises
    # :class:`pyproj.exceptions.CRSError` on malformed CRS strings,
    # ``TypeError`` on unsupported input types (e.g. a dict), and
    # ``ValueError`` on e.g. out-of-range EPSG ints. Wrap those three
    # explicitly in :class:`pyramids.base._errors.CRSError` so callers
    # can catch pyramids' own typed exception without importing pyproj
    # — but do NOT swallow ``AttributeError`` / ``ImportError`` / etc.
    # which would mask real bugs in our own code.
    import pyproj.exceptions

    try:
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise CRSError(
            f"reproject_coordinates failed to parse CRS "
            f"(from_crs={from_crs!r}, to_crs={to_crs!r}): {exc}"
        ) from exc
    xs = np.full(len(x), np.nan)
    ys = np.full(len(x), np.nan)
    for i in range(len(x)):
        nx, ny = transformer.transform(x[i], y[i])
        if precision is not None:
            nx = round(nx, precision)
            ny = round(ny, precision)
        xs[i] = nx
        ys[i] = ny
    return xs.tolist(), ys.tolist()
