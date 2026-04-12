"""COG validation wrapping ``osgeo_utils`` sample validator.

Provides :func:`validate` — a thin wrapper over
``osgeo_utils.samples.validate_cloud_optimized_geotiff.validate`` (GDAL
ships it as a "sample"; the signature has drifted between GDAL 3.4 /
3.6 / 3.8 / 3.12, so we defensively probe the return shape). If the
import fails entirely, a minimal in-house fallback checks that the file
is tiled and has overviews.

Returns a :class:`ValidationReport` — a frozen dataclass usable as a
:class:`bool` (``is_valid``) with ``errors``, ``warnings``, and
``details`` fields for richer reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osgeo import gdal


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating whether a file is a Cloud Optimized GeoTIFF.

    Attributes:
        is_valid: ``True`` iff :attr:`errors` is empty (and, under
            ``strict=True``, no warnings either).
        errors: Error messages (empty when valid).
        warnings: Non-fatal warnings (e.g., "no overviews").
        details: Structural metadata from the validator — typically
            ``ifd_offsets``, ``data_offsets``, and, in the fallback
            path, ``blocksize`` and ``overview_count``.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Truthy iff the file validates as a COG.

        Examples:
            >>> bool(ValidationReport(is_valid=True))
            True
            >>> bool(ValidationReport(is_valid=False, errors=["bad"]))
            False
        """
        return self.is_valid


def _osgeo_validate(
    path: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Invoke the osgeo_utils sample validator; return ``(errors, warnings, details)``.

    The sample validator's signature has drifted across GDAL versions.
    We probe defensively: GDAL 3.6+ returns
    ``(warnings, errors, details)`` while older builds may return just
    ``(warnings, errors)``.

    Args:
        path: File path or ``/vsi*`` path.

    Returns:
        Tuple of ``(errors, warnings, details)`` — errors listed first
        to match this module's public convention.

    Raises:
        ImportError: The ``osgeo_utils`` sample module is unavailable.
        FileNotFoundError: The underlying file cannot be opened
            (raised via ``ValidateCloudOptimizedGeoTIFFException``).
    """
    from osgeo_utils.samples import validate_cloud_optimized_geotiff as v

    try:
        result = v.validate(path, full_check=True)
    except v.ValidateCloudOptimizedGeoTIFFException as exc:
        # Translate "no such file" into FileNotFoundError for parity
        # with local path semantics; let other failures bubble up as
        # errors in the report rather than exceptions.
        if "No such file" in str(exc):
            raise FileNotFoundError(path) from exc
        return [str(exc)], [], {}
    except RuntimeError as exc:
        # gdal.Open inside the sample validator raises RuntimeError
        # when exceptions are enabled; translate "no such file" similarly.
        if "No such file" in str(exc) or "not recognized" in str(exc):
            raise FileNotFoundError(path) from exc
        return [str(exc)], [], {}

    errors: list[str]
    warnings: list[str]
    details: dict[str, Any]
    if len(result) == 3:
        warnings, errors, details = result
    else:  # pragma: no cover — defensive; older GDAL
        warnings, errors = result
        details = {}
    return list(errors), list(warnings), dict(details)


def _fallback_validate(
    path: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Minimal in-house validator used when the sample module is unavailable.

    Checks: file opens; image is tiled (block dimensions smaller than
    full extent); at least one overview present. Does NOT check the
    IFD-before-data layout; recommends upgrading GDAL if used.

    Args:
        path: File path or ``/vsi*`` path.

    Returns:
        ``(errors, warnings, details)`` — same convention as
        :func:`_osgeo_validate`.
    """
    errors: list[str] = []
    warnings: list[str] = ["using fallback validator; osgeo_utils sample unavailable"]
    details: dict[str, Any] = {}
    ds = gdal.Open(path)
    if ds is None:
        errors.append(f"cannot open {path}")
    else:
        band = ds.GetRasterBand(1)
        bx, by = band.GetBlockSize()
        details["blocksize"] = [bx, by]
        details["overview_count"] = band.GetOverviewCount()
        # Heuristic: GDAL returns (width, small_N) for stripped TIFFs
        # (e.g., (512, 4)) and (tile, tile) for tiled. A block with
        # by != bx and by much smaller than bx signals a stripped layout.
        is_stripped = by != bx and by * 4 < bx
        if is_stripped:
            errors.append("not tiled (stripped layout)")
        if band.GetOverviewCount() == 0:
            warnings.append("no overviews present")
        ds = None
    return errors, warnings, details


def validate(path: str | Path, strict: bool = False) -> ValidationReport:
    """Validate that the file at ``path`` is a valid Cloud Optimized GeoTIFF.

    Delegates to ``osgeo_utils.samples.validate_cloud_optimized_geotiff``
    when available (GDAL ≥ 3.4). Falls back to a minimal in-house check
    (tiled + overviews) when the import fails.

    Args:
        path: Local path or ``/vsi*`` VSI path.
        strict: If ``True``, warnings are promoted to errors.

    Returns:
        ValidationReport: Includes ``is_valid``, error/warning lists,
        and a ``details`` dict with structural metadata. Usable as a
        boolean (``bool(report) == report.is_valid``).

    Raises:
        FileNotFoundError: When a *local* ``path`` does not exist. VSI
            paths are passed through to GDAL, which reports the error
            through the normal validator surface.

    Examples:
        >>> from pyramids.dataset.cog import validate  # doctest: +SKIP
        >>> report = validate("scene.tif")  # doctest: +SKIP
        >>> bool(report)  # doctest: +SKIP
        True
    """
    p = str(path)
    if not p.startswith("/vsi") and not Path(p).exists():
        raise FileNotFoundError(p)

    try:
        errors, warnings, details = _osgeo_validate(p)
    except ImportError:  # pragma: no cover — osgeo_utils is a hard dep of GDAL
        errors, warnings, details = _fallback_validate(p)

    if strict:
        errors = list(errors) + list(warnings)
        warnings = []

    return ValidationReport(
        is_valid=not errors,
        errors=list(errors),
        warnings=list(warnings),
        details=dict(details),
    )
