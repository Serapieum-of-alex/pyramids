"""COG creation-option types, serialization, and validation.

Provides the :data:`CreationOptions` alias (a ``Mapping[str, Any]``) plus
three pure-Python helpers used by :mod:`pyramids.dataset.cog.write` and
:class:`pyramids.dataset.ops.cog.COGMixin`:

- :func:`to_gdal_options` — serialize a mapping into GDAL's ``['KEY=VALUE', ...]`` list form.
- :func:`merge_options` — merge defaults with user-supplied extras (dict or legacy ``list[str]``).
- :func:`validate_blocksize` — enforce the COG driver's power-of-2-in-[64, 4096] constraint.
- :func:`validate_option_keys` — gate unknown keys against :data:`COG_DRIVER_OPTIONS`.

The module has no GDAL dependency — all helpers operate on plain Python
values. GDAL is invoked only at the write call site.
"""

from __future__ import annotations

from typing import Any, Mapping


CreationOptions = Mapping[str, Any]
"""Alias for a mapping of GDAL creation-option names to Python values.

Keys are the GDAL option names (case-insensitive; normalized to upper case
during serialization). Values are scalars that stringify cleanly; booleans
are translated to ``"YES"``/``"NO"``; ``None`` entries are dropped.
"""


COG_DRIVER_OPTIONS: frozenset[str] = frozenset(
    {
        "COMPRESS",
        "LEVEL",
        "QUALITY",
        "NUM_THREADS",
        "BLOCKSIZE",
        "BIGTIFF",
        "RESAMPLING",
        "OVERVIEW_RESAMPLING",
        "OVERVIEW_COUNT",
        "OVERVIEW_COMPRESS",
        "OVERVIEW_QUALITY",
        "WARP_RESAMPLING",
        "OVERVIEW_PREDICTOR",
        "PREDICTOR",
        "NBITS",
        "TARGET_SRS",
        "RES",
        "EXTENT",
        "ALIGNED_LEVELS",
        "ADD_ALPHA",
        "TILING_SCHEME",
        "ZOOM_LEVEL",
        "ZOOM_LEVEL_STRATEGY",
        "MAX_Z_ERROR",
        "STATISTICS",
        "GEOTIFF_VERSION",
        "SPARSE_OK",
        "COPY_SRC_MDD",
        "SRC_MDD",
    }
)
"""Whitelist of GDAL COG driver option keys (uppercased)."""


_VALID_BLOCKSIZES: frozenset[int] = frozenset({64, 128, 256, 512, 1024, 2048, 4096})


def _stringify(value: Any) -> str:
    """Convert a Python value to the string form GDAL expects.

    Booleans become ``"YES"``/``"NO"`` (GDAL's convention); everything else
    falls back to :class:`str`.

    Args:
        value: Any Python value.

    Returns:
        The GDAL-style string form.
    """
    result: str
    if isinstance(value, bool):
        result = "YES" if value else "NO"
    else:
        result = str(value)
    return result


def to_gdal_options(opts: CreationOptions | None) -> list[str]:
    """Serialize a mapping into GDAL's ``['KEY=VALUE', ...]`` list form.

    Keys are uppercased; values are stringified via :func:`_stringify`
    (booleans become ``"YES"``/``"NO"``). ``None`` values are skipped so
    callers can pass optional kwargs through unchanged.

    Args:
        opts: Mapping of option names to values, or ``None``.

    Returns:
        List of ``"KEY=VALUE"`` strings. Empty list when ``opts`` is ``None``.

    Examples:
        >>> to_gdal_options({"COMPRESS": "DEFLATE", "LEVEL": 9})
        ['COMPRESS=DEFLATE', 'LEVEL=9']
        >>> to_gdal_options({"STATISTICS": True, "SPARSE_OK": False})
        ['STATISTICS=YES', 'SPARSE_OK=NO']
        >>> to_gdal_options({"COMPRESS": "LZW", "LEVEL": None})
        ['COMPRESS=LZW']
        >>> to_gdal_options(None)
        []
    """
    result: list[str]
    if opts is None:
        result = []
    else:
        result = [
            f"{str(k).upper()}={_stringify(v)}"
            for k, v in opts.items()
            if v is not None
        ]
    return result


def _parse_list_extra(items: list[str]) -> dict[str, Any]:
    """Parse ``['KEY=VALUE', ...]`` legacy list form back to a dict.

    Args:
        items: List of ``"KEY=VALUE"`` strings.

    Returns:
        Dict with uppercased keys and string values (split on first ``=``).

    Raises:
        ValueError: If any item lacks an ``=``.
    """
    parsed: dict[str, Any] = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError(f"creation_options entry missing '=': {entry!r}")
        k, _, v = entry.partition("=")
        parsed[k.upper()] = v
    return parsed


def merge_options(
    defaults: CreationOptions,
    extra: CreationOptions | list[str] | None,
) -> dict[str, Any]:
    """Merge default options with user-supplied extras; extras win.

    Accepts ``extra`` as either a mapping ``{'KEY': value}`` or the legacy
    list form ``['KEY=VALUE', ...]`` used by
    :meth:`pyramids.dataset.Dataset.to_file`. All keys in the returned
    dict are uppercased; ``None`` values from either source are dropped.

    Args:
        defaults: Baseline options (typically derived from kwargs in
            :meth:`~pyramids.dataset.ops.cog.COGMixin.to_cog`).
        extra: User-provided overrides as a mapping, ``list[str]``, or
            ``None``.

    Returns:
        New :class:`dict` with all keys uppercased and ``None`` values
        removed; ``extra`` entries override ``defaults`` on conflict.

    Raises:
        ValueError: When a legacy list-form entry lacks ``=``.

    Examples:
        >>> merge_options({"COMPRESS": "DEFLATE"}, {"COMPRESS": "ZSTD"})
        {'COMPRESS': 'ZSTD'}
        >>> merge_options({"COMPRESS": "DEFLATE"}, ["LEVEL=9"])
        {'COMPRESS': 'DEFLATE', 'LEVEL': '9'}
        >>> merge_options({"COMPRESS": "DEFLATE"}, None)
        {'COMPRESS': 'DEFLATE'}
    """
    merged: dict[str, Any] = {
        str(k).upper(): v for k, v in defaults.items() if v is not None
    }
    if extra is None:
        pass
    elif isinstance(extra, list):
        merged.update(_parse_list_extra(extra))
    else:
        merged.update(
            {str(k).upper(): v for k, v in extra.items() if v is not None}
        )
    return merged


def validate_blocksize(value: int) -> None:
    """Raise :class:`ValueError` if ``value`` is not a valid COG tile size.

    The GDAL COG driver requires ``BLOCKSIZE`` to be a power of 2 in
    the closed range [64, 4096].

    Args:
        value: Proposed blocksize.

    Raises:
        ValueError: If ``value`` is outside the allowed set.

    Examples:
        >>> validate_blocksize(512)  # no error
        >>> validate_blocksize(500)   # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: blocksize must be a power of 2 in [64, 4096]; got 500...
    """
    if value not in _VALID_BLOCKSIZES:
        raise ValueError(
            f"blocksize must be a power of 2 in [64, 4096]; got {value}. "
            f"Valid values: {sorted(_VALID_BLOCKSIZES)}"
        )


def validate_option_keys(opts: CreationOptions) -> None:
    """Raise :class:`ValueError` for any key not in :data:`COG_DRIVER_OPTIONS`.

    Keys are compared case-insensitively.

    Args:
        opts: Mapping of option names to values.

    Raises:
        ValueError: If any key is not a recognized COG driver option.

    Examples:
        >>> validate_option_keys({"COMPRESS": "DEFLATE"})  # no error
        >>> validate_option_keys({"NONSENSE": "x"})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Unknown COG driver option(s): ['NONSENSE']...
    """
    unknown = {str(k).upper() for k in opts.keys()} - COG_DRIVER_OPTIONS
    if unknown:
        raise ValueError(
            f"Unknown COG driver option(s): {sorted(unknown)}. "
            f"Valid options: {sorted(COG_DRIVER_OPTIONS)}"
        )
