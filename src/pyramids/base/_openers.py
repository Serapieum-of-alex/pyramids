"""GDAL opener primitives used by :mod:`pyramids.base._file_manager`.

These small wrappers normalize the :mod:`osgeo.gdal` open API to a
uniform ``(path, access, **kwargs) -> gdal.Dataset`` signature that the
:class:`~pyramids.base._file_manager.FileManager` hierarchy can plug in
as the ``opener`` callable. They also pre-process the path through
:func:`pyramids.base.remote._to_vsi` so cloud URLs (``s3://``, ``gs://``,
``http://``, ...) are rewritten to GDAL's ``/vsi*`` form before each
open, which means a ``FileManager`` can be pickled to a worker and will
still transparently resolve cloud paths on the other side.
"""

from __future__ import annotations

from typing import Any

from osgeo import gdal, ogr

from pyramids.base.remote import _to_vsi


_ACCESS_FLAGS = {
    "read_only": gdal.GA_ReadOnly,
    "r": gdal.GA_ReadOnly,
    "write": gdal.GA_Update,
    "w": gdal.GA_Update,
    "update": gdal.GA_Update,
    "a": gdal.GA_Update,
}


def _resolve_access(access: str) -> int:
    """Normalize a pyramids ``access`` string to the matching GDAL flag.

    Args:
        access: One of ``"read_only"``, ``"r"``, ``"write"``, ``"w"``,
            ``"update"``, ``"a"``.

    Returns:
        int: The corresponding :data:`osgeo.gdal.GA_*` constant.

    Raises:
        ValueError: If ``access`` is not a recognized mode string.

    Examples:
        - Read-only aliases all resolve to ``GA_ReadOnly``:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids.base._openers import _resolve_access
            >>> _resolve_access("read_only") == gdal.GA_ReadOnly
            True
            >>> _resolve_access("r") == gdal.GA_ReadOnly
            True

            ```
        - Unknown access string raises a descriptive ValueError:
            ```python
            >>> from pyramids.base._openers import _resolve_access
            >>> _resolve_access("bogus")
            Traceback (most recent call last):
              ...
            ValueError: Unknown access mode 'bogus'; expected one of ['a', 'r', 'read_only', 'update', 'w', 'write']

            ```
    """
    try:
        return _ACCESS_FLAGS[access]
    except KeyError as exc:
        raise ValueError(
            f"Unknown access mode {access!r}; expected one of "
            f"{sorted(_ACCESS_FLAGS)}"
        ) from exc


def gdal_raster_open(path: str, access: str = "read_only", **_: Any) -> gdal.Dataset:
    """Open a classic-mode raster (GeoTIFF, COG, PNG, ...) via :func:`gdal.Open`.

    The ``path`` is rewritten through :func:`pyramids.base.remote._to_vsi`
    first, so callers can pass URL-scheme paths (``s3://bucket/file.tif``,
    ``https://example.com/file.tif``) directly.

    Args:
        path: File path or URL.
        access: Access mode string — see :func:`_resolve_access`.
        **_: Extra keyword arguments are accepted and ignored so that a
            single uniform opener signature can be used as a
            ``FileManager`` ``opener`` callable.

    Returns:
        osgeo.gdal.Dataset: The opened dataset handle.
    """
    return gdal.Open(_to_vsi(path), _resolve_access(access))


def gdal_mdarray_open(path: str, access: str = "read_only", **_: Any) -> gdal.Dataset:
    """Open a multidimensional raster (NetCDF, HDF5, Zarr) via :func:`gdal.OpenEx`.

    Equivalent to :func:`gdal_raster_open` but uses
    :data:`gdal.OF_MULTIDIM_RASTER`, which is required for group /
    :class:`gdal.MDArray` access on NetCDF and HDF5 files.

    Args:
        path: File path or URL.
        access: Access mode string — see :func:`_resolve_access`.
        **_: Extra keyword arguments accepted and ignored for signature
            uniformity.

    Returns:
        osgeo.gdal.Dataset: The opened MDIM dataset.
    """
    flags = gdal.OF_MULTIDIM_RASTER
    flags |= gdal.OF_UPDATE if access not in {"read_only", "r"} else gdal.OF_READONLY
    return gdal.OpenEx(_to_vsi(path), flags)


def ogr_open(path: str, access: str = "read_only", **_: Any) -> "ogr.DataSource":
    """Open a vector datasource via :func:`ogr.Open`.

    Args:
        path: File path or URL.
        access: ``"read_only"`` / ``"r"`` opens read-only; any other
            value opens for update.
        **_: Extra keyword arguments accepted and ignored for signature
            uniformity.

    Returns:
        osgeo.ogr.DataSource: The opened vector datasource.
    """
    update = 0 if access in {"read_only", "r"} else 1
    return ogr.Open(_to_vsi(path), update)
