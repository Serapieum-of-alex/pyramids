"""CRS construction helpers shared across the pyramids package.

Single source of truth for ``osr.SpatialReference`` construction and
for WKT/Proj4 → EPSG resolution so raster, vector, and NetCDF code
paths cannot drift on the EPSG-import recipe.

Replaces three identical ``_create_sr_from_epsg`` definitions that
previously lived on :class:`AbstractDataset`,
:class:`~pyramids.dataset.Dataset`, and the ``Spatial`` mixin, plus
the hand-rolled ``osr.SpatialReference() + ImportFromEPSG`` call-sites
sprinkled across ``basemap``, ``ugrid``, and ``dataset.ops``.

Also hosts :func:`epsg_from_wkt`, the wrapper that absorbs the
``get_epsg_from_prj(wkt) if wkt else 4326`` idiom previously
duplicated in four places across the dataset stack. The underlying
``get_epsg_from_prj`` / ``create_sr_from_proj`` helpers were moved
here from :mod:`pyramids.feature.crs` so the dataset code path no
longer reaches up into ``feature/`` for a primitive that has nothing
vector-specific about it. ``pyramids.feature.crs`` re-exports both
names verbatim for backwards compatibility.
"""

from __future__ import annotations

from osgeo import osr

from pyramids.base._errors import CRSError


def sr_from_epsg(epsg: int) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from an EPSG code.

    Args:
        epsg: EPSG code; cast to ``int`` before being handed to
            :meth:`osr.SpatialReference.ImportFromEPSG`.

    Returns:
        osr.SpatialReference: The constructed SRS.

    Raises:
        ValueError: If GDAL cannot resolve the EPSG code (the
            non-zero return path from ``ImportFromEPSG`` — usually
            propagates as a GDAL exception when
            ``gdal.UseExceptions()`` is active, which pyramids
            installs at package import).
    """
    sr = osr.SpatialReference()
    err = sr.ImportFromEPSG(int(epsg))
    if err != 0:
        raise ValueError(
            f"Failed to create SRS from EPSG:{epsg} (osr returned error {err})."
        )
    return sr


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
    ``CRSError`` themselves, or use :func:`epsg_from_wkt` which
    accepts an explicit ``default``.

    Args:
        prj (str): Projection string.

    Returns:
        int: The resolved EPSG code.

    Raises:
        CRSError: If ``prj`` is an empty string.

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
            "and supply it at the call site, or call "
            "epsg_from_wkt(prj, default=...)."
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


def epsg_from_wkt(wkt: str, default: int = 4326) -> int:
    """Resolve an EPSG code from a WKT / Proj string with a fallback.

    Wraps :func:`get_epsg_from_prj` to absorb the
    ``get_epsg_from_prj(wkt) if wkt else default`` idiom that was
    previously open-coded in four places across the dataset stack.
    Returns ``default`` when ``wkt`` is empty (or ``None``); otherwise
    delegates to :func:`get_epsg_from_prj`.

    Use this in places where an empty projection should be treated as
    a soft "unknown CRS, assume WGS84" rather than a hard error — for
    example the ``Dataset.epsg`` property on a freshly-built
    in-memory raster that has no projection metadata yet. Use
    :func:`get_epsg_from_prj` directly when you want the strict
    behaviour where an empty projection raises.

    Args:
        wkt: Projection string (WKT, ESRI WKT, or Proj4). An empty
            string or ``None`` returns ``default``.
        default: EPSG code to return when ``wkt`` is empty / ``None``.
            Defaults to ``4326`` (the historical pyramids default).

    Returns:
        int: EPSG code resolved from ``wkt``, or ``default`` when
        ``wkt`` is empty.

    Examples:
        - Empty input falls back to the supplied default:
            ```python
            >>> from pyramids.base.crs import epsg_from_wkt
            >>> epsg_from_wkt("")
            4326
            >>> epsg_from_wkt("", default=3857)
            3857

            ```
        - Non-empty WKT delegates to :func:`get_epsg_from_prj`:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base.crs import epsg_from_wkt
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> epsg_from_wkt(ref.ExportToWkt())
            3857

            ```
    """
    if not wkt:
        return default
    return get_epsg_from_prj(wkt)


__all__ = [
    "create_sr_from_proj",
    "epsg_from_wkt",
    "get_epsg_from_prj",
    "sr_from_epsg",
]
