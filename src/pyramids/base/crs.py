"""CRS construction helpers shared across the pyramids package.

Single source of truth for ``osr.SpatialReference`` construction so
raster, vector, and NetCDF code paths cannot drift on the EPSG-import
recipe. Replaces three identical ``_create_sr_from_epsg`` definitions
that previously lived on :class:`AbstractDataset`,
:class:`~pyramids.dataset.Dataset`, and the ``Spatial`` mixin, plus
the hand-rolled ``osr.SpatialReference() + ImportFromEPSG`` call-sites
sprinkled across ``basemap``, ``ugrid``, and ``dataset.ops``.
"""

from __future__ import annotations

from osgeo import osr


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


__all__ = ["sr_from_epsg"]
