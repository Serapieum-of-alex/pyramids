"""Read GRIB1/GRIB2 files (local or cloud) as pyramids datasets via GDAL.

GRIB is decoded through GDAL's native `GRIB` driver — no extra Python decoding
dependency. The driver ships in the conda-forge `libgdal-grib` plugin (a core
pyramids dependency, alongside `libgdal-netcdf` / `libgdal-hdf4`);
:func:`open_grib` raises a clear error if a GDAL build lacks it.

Each GRIB *message* becomes a band of the returned
:class:`~pyramids.dataset.Dataset`, carrying the per-message GRIB metadata
(element, level + type of level, reference / valid time, forecast horizon,
statistical step type, and ensemble member number). Use
:func:`grib_band_metadata` to introspect those bands and pick the band(s) you
want — by element, `level`, or ensemble `number`. Cloud URIs
(`s3://` / `gs://` / `https://` / `/vsi*`) are resolved through the same path as
:meth:`pyramids.dataset.Dataset.read_file`.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from osgeo import gdal

from pyramids.base._errors import DriverNotExistError
from pyramids.dataset.cog import BandSelection, Tiling
from pyramids.dataset.dataset import Dataset
from pyramids.dataset.engines._validate import validate_one_based_band

_GRIB_DRIVER = "GRIB"

_GRIB_FIELDS: dict[str, str] = {
    "element": "GRIB_ELEMENT",
    "short_name": "GRIB_SHORT_NAME",
    "comment": "GRIB_COMMENT",
    "unit": "GRIB_UNIT",
    "discipline": "GRIB_DISCIPLINE",
}

# GRIB2 product-definition templates (code table 4.0) that carry an individual
# ensemble member (4.1 at a level, 4.11 over a time interval).
_ENSEMBLE_PDTNS = frozenset({1, 11})
# Templates whose product is a statistic over a time interval (code table 4.0);
# the others are treated as instantaneous.
_INTERVAL_PDTNS = frozenset({8, 9, 10, 11, 12, 13, 42, 43, 46, 47})
# GRIB2 code table 4.10 (type of statistical processing) -> eccodes-style
# `stepType` label.
_STEP_TYPE_BY_STAT: dict[int, str] = {
    0: "avg",
    1: "accum",
    2: "max",
    3: "min",
    4: "diff",
    5: "rms",
    6: "sd",
    7: "cov",
    8: "ratio",
    9: "sum",
}
# Position of the type-of-statistical-processing value, counted from the end of
# the assembled PDS values, for a single time range (verified against GDAL's
# decoded PDT 4.8/4.11 output: the five trailing values are increment-type,
# unit, length, increment-unit, increment).
_STAT_TYPE_FROM_END = 6
# Position of the ensemble perturbation number within the assembled PDS values:
# the 15-value PDT 4.0 core, then type-of-ensemble-forecast (index 15), then the
# perturbation number (index 16) — before any time-interval block, so it holds
# for both PDT 4.1 and 4.11.
_ENSEMBLE_NUMBER_INDEX = 16


def _require_grib_driver() -> None:
    """Raise :class:`DriverNotExistError` when the GDAL GRIB driver is absent.

    Raises:
        DriverNotExistError: The active GDAL build has no `GRIB` driver.
    """
    if gdal.GetDriverByName(_GRIB_DRIVER) is None:
        raise DriverNotExistError(
            "GDAL build does not include the GRIB driver. Install the "
            "conda-forge `libgdal-grib` plugin (e.g. `pixi install -e dev`)."
        )


def _parse_grib_seconds(value: str | None) -> datetime | None:
    """Parse a GRIB `"<unix_seconds> sec UTC"` timestamp to an aware datetime.

    Args:
        value: A GRIB metadata value such as `"1700000000 sec UTC"`, or
            `None`.

    Returns:
        A timezone-aware UTC :class:`~datetime.datetime`, or `None` when the
        value is missing or not parseable.
    """
    result: datetime | None = None
    if value:
        token = value.split()[0]
        try:
            result = datetime.fromtimestamp(int(token), tz=UTC)
        except (ValueError, OverflowError, OSError):
            result = None
    return result


def _parse_leading_int(value: str | None) -> int | None:
    """Parse the leading integer of a GRIB metadata value (e.g. `"10800 s"`).

    Args:
        value: A GRIB metadata value whose first token is an integer, or
            `None`.

    Returns:
        The leading integer, or `None` when missing / not parseable.
    """
    result: int | None = None
    if value:
        token = value.split()[0]
        try:
            result = int(token)
        except ValueError:
            result = None
    return result


def _leading_signed_int(token: str | None) -> int | None:
    """Parse the leading signed integer of `token`, stopping at the first non-digit.

    Args:
        token: A string whose leading characters are an optional sign and digits
            (e.g. `"85000"`, `"0-180"`), or `None`.

    Returns:
        The leading integer, or `None` when the token has no leading digits.
    """
    result: int | None = None
    if token:
        end = 1 if token[:1] in "+-" else 0
        while end < len(token) and token[end].isdigit():
            end += 1
        head = token[:end]
        if head not in ("", "+", "-"):
            result = int(head)
    return result


def _parse_discipline_code(value: str | None) -> int | None:
    """Parse the numeric discipline from a `GRIB_DISCIPLINE` value.

    Args:
        value: A value such as `"0(Meteorological)"`, a bare `"0"`, or `None`.

    Returns:
        The GRIB2 discipline number, or `None` when absent / not parseable.
    """
    result: int | None = None
    if value:
        head = value.split("(", 1)[0].strip()
        try:
            result = int(head)
        except ValueError:
            result = None
    return result


def _parse_short_name_level(short_name: str | None) -> tuple[int | None, str | None]:
    """Split a `GRIB_SHORT_NAME` into its level value and fixed-surface type.

    GDAL formats the short name as `"<level>-<TYPE>"` (e.g. `"2-HTGL"`,
    `"85000-ISBL"`, `"0-SFC"`), where `TYPE` is the GDAL/NCEP abbreviation of the
    first fixed surface. A layer between two surfaces reads `"<v1>-<v2>-<TYPE>"`;
    its first surface value is returned.

    Args:
        short_name: The `GRIB_SHORT_NAME` value, or `None`.

    Returns:
        `(level, type_of_level)` — the first fixed-surface value as an `int` (or
        `None` when not numeric) and the surface-type abbreviation (or `None`).
    """
    level: int | None = None
    type_of_level: str | None = None
    if short_name and "-" in short_name:
        value_part, type_of_level = short_name.rsplit("-", 1)
        level = _leading_signed_int(value_part)
    elif short_name:
        type_of_level = short_name
    return level, type_of_level


def _assembled_values(band_md: dict[str, str]) -> list[int] | None:
    """Parse `GRIB_PDS_TEMPLATE_ASSEMBLED_VALUES` into a list of integers.

    Args:
        band_md: A GRIB band's GDAL metadata dict.

    Returns:
        The assembled PDS values as a list of `int`, or `None` when the key is
        absent or any value is not an integer.
    """
    raw = band_md.get("GRIB_PDS_TEMPLATE_ASSEMBLED_VALUES")
    result: list[int] | None = None
    if raw:
        try:
            result = [int(token) for token in raw.split()]
        except ValueError:
            result = None
    return result


def _ensemble_number(pdtn: int | None, assembled: list[int] | None) -> int | None:
    """Return the ensemble perturbation number, or `None` when not an ensemble.

    Args:
        pdtn: The product-definition template number.
        assembled: The assembled PDS values from :func:`_assembled_values`.

    Returns:
        The perturbation number for an ensemble template (PDT 4.1 / 4.11), else
        `None`.
    """
    result: int | None = None
    if (
        pdtn in _ENSEMBLE_PDTNS
        and assembled is not None
        and len(assembled) > _ENSEMBLE_NUMBER_INDEX
    ):
        result = assembled[_ENSEMBLE_NUMBER_INDEX]
    return result


def _step_type(pdtn: int | None, assembled: list[int] | None) -> str | None:
    """Classify the forecast step as instantaneous or a time-interval statistic.

    Args:
        pdtn: The product-definition template number.
        assembled: The assembled PDS values from :func:`_assembled_values`.

    Returns:
        `"instant"` for a non-interval template; the code-table-4.10 label
        (`"accum"`, `"avg"`, `"max"`, `"min"`, …) for a single-range interval
        template; `"interval"` when the statistic type cannot be read; or `None`
        when there is no template number.
    """
    result: str | None = None
    if pdtn is not None:
        if pdtn not in _INTERVAL_PDTNS:
            result = "instant"
        elif assembled is not None and len(assembled) >= _STAT_TYPE_FROM_END:
            result = _STEP_TYPE_BY_STAT.get(assembled[-_STAT_TYPE_FROM_END], "interval")
        else:
            result = "interval"
    return result


def open_grib(path: str | Path, *, vsi: str | None = None) -> Dataset:
    """Open a GRIB1/GRIB2 file as a pyramids :class:`~pyramids.dataset.Dataset`.

    Decodes via GDAL's native `GRIB` driver; every GRIB message is a band.
    Local paths, `/vsi*` paths, and `http(s)://` / `s3://` / `gs://` /
    `az://` URIs are all accepted (resolved by
    :meth:`~pyramids.dataset.Dataset.read_file`).

    Args:
        path: Path or URI to a GRIB1/GRIB2 file.
        vsi: Optional explicit archive kind forwarded to
            :meth:`~pyramids.dataset.Dataset.read_file` (e.g. for a GRIB inside
            a `.zip`); `None` opens `path` directly.

    Returns:
        A :class:`~pyramids.dataset.Dataset` with one band per GRIB message.
        Use :func:`grib_band_metadata` to identify the bands.

    Raises:
        DriverNotExistError: The GDAL build lacks the GRIB driver (install the
            `libgdal-grib` plugin).

    Examples:
        - Open a local GRIB2 file and inspect its bands (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import open_grib, grib_band_metadata  # doctest: +SKIP
            >>> ds = open_grib("gfs.t00z.pgrb2.0p25.f000.grib2")  # doctest: +SKIP
            >>> ds.band_count  # doctest: +SKIP
            5
            >>> grib_band_metadata(ds)[0]["element"]  # doctest: +SKIP
            'TMP'

            ```
        - Open a GRIB file from cloud storage:
            ```python
            >>> ds = open_grib(  # doctest: +SKIP
            ...     "s3://noaa-gfs-bdp-pds/gfs.20260518/00/atmos/"
            ...     "gfs.t00z.pgrb2.0p25.f000"
            ... )

            ```
    """
    _require_grib_driver()
    return Dataset.read_file(path, vsi=vsi)


def grib_band_metadata(dataset: Dataset) -> list[dict[str, Any]]:
    """Return per-band GRIB metadata for a dataset opened from a GRIB file.

    One dict per band exposes the common GRIB message fields, with reference /
    valid times decoded to timezone-aware UTC datetimes and the forecast
    horizon as an integer number of seconds. Missing fields are `None`.

    Args:
        dataset: A :class:`~pyramids.dataset.Dataset` opened from a GRIB file
            (e.g. via :func:`open_grib`).

    Returns:
        A list (one entry per band, in band order) of dicts with keys:
        `band` (1-based index), `element`, `short_name`, `comment`,
        `unit`, `discipline`, `ref_time` / `valid_time`
        (:class:`~datetime.datetime` or `None`), and `forecast_seconds`
        (:class:`int` or `None`), plus the NWP keys decoded from the
        product-definition section: `pdtn` (template number), `discipline_code`,
        `parameter_category`, `parameter_number` (the message-native parameter
        identity — the ECMWF integer `paramId` is an eccodes-database concept not
        present in the message and is not derived here), `level` + `type_of_level`
        (first fixed surface and its GDAL/NCEP abbreviation), `step_type`
        (`"instant"` or a code-table-4.10 statistic such as `"accum"`), and
        `number` (ensemble perturbation number, or `None`). Keys are `None` when
        the message does not carry them.

    Examples:
        - Find the band index carrying 2-metre temperature (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import open_grib, grib_band_metadata  # doctest: +SKIP
            >>> ds = open_grib("gfs.t00z.pgrb2.0p25.f000.grib2")  # doctest: +SKIP
            >>> meta = grib_band_metadata(ds)  # doctest: +SKIP
            >>> [m["band"] for m in meta if m["element"] == "TMP"]  # doctest: +SKIP
            [1]

            ```
        - Read the valid time of the first band:
            ```python
            >>> meta = grib_band_metadata(open_grib("f000.grib2"))  # doctest: +SKIP
            >>> meta[0]["valid_time"].year  # doctest: +SKIP
            2026

            ```
    """
    raster = dataset.raster
    metadata: list[dict[str, Any]] = []
    for index in range(1, raster.RasterCount + 1):
        band_md = raster.GetRasterBand(index).GetMetadata()
        entry: dict[str, Any] = {"band": index}
        for key, grib_key in _GRIB_FIELDS.items():
            entry[key] = band_md.get(grib_key)
        entry["ref_time"] = _parse_grib_seconds(band_md.get("GRIB_REF_TIME"))
        entry["valid_time"] = _parse_grib_seconds(band_md.get("GRIB_VALID_TIME"))
        entry["forecast_seconds"] = _parse_leading_int(
            band_md.get("GRIB_FORECAST_SECONDS")
        )
        pdtn = _parse_leading_int(band_md.get("GRIB_PDS_PDTN"))
        assembled = _assembled_values(band_md)
        level, type_of_level = _parse_short_name_level(band_md.get("GRIB_SHORT_NAME"))
        entry["pdtn"] = pdtn
        entry["discipline_code"] = _parse_discipline_code(
            band_md.get("GRIB_DISCIPLINE")
        )
        entry["parameter_category"] = assembled[0] if assembled else None
        entry["parameter_number"] = (
            assembled[1] if assembled is not None and len(assembled) > 1 else None
        )
        entry["level"] = level
        entry["type_of_level"] = type_of_level
        entry["step_type"] = _step_type(pdtn, assembled)
        entry["number"] = _ensemble_number(pdtn, assembled)
        metadata.append(entry)
    return metadata


def _select_grib_band(
    metadata: list[dict[str, Any]],
    variable: str | int | None,
    *,
    level: int | None = None,
    number: int | None = None,
) -> int:
    """Resolve the 0-based band index for `variable`, optionally filtered.

    Args:
        metadata: Per-band metadata from :func:`grib_band_metadata`.
        variable: How to pick the message. An `int` is a 1-based band number
            (the unambiguous escape hatch for files that repeat an element
            across levels/horizons). A non-empty `str` matches the GRIB element
            (case-insensitive, e.g. `"TMP"`); when several messages share the
            element the first is used and a warning is emitted. `None` selects
            the sole band of a single-message file, or — with `level`/`number`
            — the sole message matching those alone.
        level: When given, keep only messages whose `level` equals it (e.g.
            `850` for 850 hPa). Combined with `variable` to pick "TMP at 850".
        number: When given, keep only messages whose ensemble `number` equals it.

    Returns:
        The 0-based band index to keep.

    Raises:
        ValueError: `variable` is not a `str`, `int`, or `None` (a `bool` is
            rejected too); an `int` band number is out of range; `variable` is an
            empty string; `variable` is `None` with no `level`/`number` but the
            file holds more than one message; or no message matches the
            element/`level`/`number` criteria.
    """
    band_count = len(metadata)
    # `variable`, `band` and the user's band number are all 1-based; the returned
    # index (and `to_cog(bands=BandSelection(indexes=))`) is 0-based, hence the `- 1` below.
    if isinstance(variable, bool) or not isinstance(variable, (str, int, type(None))):
        raise ValueError(
            f"variable must be a band number, element name, or None; got {variable!r}."
        )
    if isinstance(variable, int):
        # An explicit 1-based band number is unambiguous; level/number are ignored.
        validate_one_based_band(variable, band_count, name="variable")
        selected = variable - 1
    elif level is None and number is None:
        if variable is None:
            if band_count > 1:
                elements = [m.get("element") for m in metadata]
                raise ValueError(
                    f"GRIB file has {band_count} messages; "
                    f"pass variable= to pick one of {elements}."
                )
            selected = 0
        else:
            selected = _match_grib_element(metadata, variable)
    else:
        selected = _match_grib_filtered(metadata, variable, level, number)
    return selected


def _match_grib_filtered(
    metadata: list[dict[str, Any]],
    variable: str | None,
    level: int | None,
    number: int | None,
) -> int:
    """Resolve a band by GRIB element and/or `level` / ensemble `number`.

    Args:
        metadata: Per-band metadata from :func:`grib_band_metadata`.
        variable: An optional GRIB element to match first (case-insensitive); a
            blank string is rejected. `None` matches on `level`/`number` alone.
        level: Keep only messages whose `level` equals this, when not `None`.
        number: Keep only messages whose ensemble `number` equals this, when not
            `None`.

    Returns:
        The 0-based index of the first message matching every given criterion.

    Raises:
        ValueError: `variable` is a blank string, or no message matches the
            combined element / `level` / `number` criteria.
    """
    candidates = metadata
    if variable is not None:
        wanted = variable.strip().upper()
        if not wanted:
            raise ValueError(
                "variable must be a non-empty element name or band number."
            )
        candidates = [
            m for m in candidates if (m.get("element") or "").strip().upper() == wanted
        ]
    if level is not None:
        candidates = [m for m in candidates if m.get("level") == level]
    if number is not None:
        candidates = [m for m in candidates if m.get("number") == number]
    criteria = ", ".join(
        f"{name}={value!r}"
        for name, value in (
            ("element", variable),
            ("level", level),
            ("number", number),
        )
        if value is not None
    )
    if not candidates:
        available = [
            {
                "band": m["band"],
                "element": m.get("element"),
                "level": m.get("level"),
                "number": m.get("number"),
            }
            for m in metadata
        ]
        raise ValueError(f"No GRIB message matches {criteria}; available: {available}.")
    if len(candidates) > 1:
        warnings.warn(
            f"{len(candidates)} GRIB messages match {criteria}; "
            f"using the first (band {candidates[0]['band']}).",
            stacklevel=4,
        )
    return cast(int, candidates[0]["band"]) - 1


def _match_grib_element(metadata: list[dict[str, Any]], variable: str) -> int:
    """Resolve the 0-based band index of the first message matching a GRIB element.

    Args:
        metadata: Per-band metadata from :func:`grib_band_metadata`.
        variable: A non-empty GRIB element name (case-insensitive).

    Returns:
        The 0-based index of the first matching message.

    Raises:
        ValueError: `variable` is blank, or no message carries the element.
    """
    wanted = variable.strip().upper()
    if not wanted:
        raise ValueError("variable must be a non-empty element name or band number.")
    matches = [
        m for m in metadata if (m.get("element") or "").strip().upper() == wanted
    ]
    if not matches:
        elements = sorted({m["element"] for m in metadata if m.get("element")})
        raise ValueError(
            f"No GRIB message with element {variable!r}; "
            f"available elements: {elements}."
        )
    if len(matches) > 1:
        warnings.warn(
            f"{len(matches)} GRIB messages match element {variable!r}; "
            f"using the first (band {matches[0]['band']}).",
            stacklevel=4,
        )
    return cast(int, matches[0]["band"]) - 1


def grib_to_cog(
    grib_path: str | Path,
    *,
    output: str | Path,
    variable: str | int | None = None,
    level: int | None = None,
    number: int | None = None,
    target_crs: int | str | None = None,
    cog_profile: str = "deflate",
    vsi: str | None = None,
) -> Path:
    """Convert a single GRIB message to a Cloud-Optimized GeoTIFF in one call.

    Chains :func:`open_grib` → select the `variable` band → write with
    :meth:`~pyramids.dataset.Dataset.to_cog`. Everything is in-repo (GDAL's GRIB
    driver + pyramids' COG writer) — no third-party GRIB or COG libraries. The
    written COG carries the GRIB band's CRS and its `GRIB_*` band metadata, and
    passes the COG validator (:func:`pyramids.dataset.cog.cog_info` `.is_cog`).

    Args:
        grib_path: Path or URI to a GRIB1/GRIB2 file (local, `/vsi*`, or a cloud
            `s3://` / `gs://` / `https://` URI — resolved like
            :func:`open_grib`).
        output: Destination COG path. Its parent directory must already exist.
        variable: Which GRIB message to convert. A `str` matches the GRIB element
            (case-insensitive, e.g. `"TMP"`); when several messages share the
            element the first is used and a warning is emitted. Pass an `int`
            1-based band number to select a message unambiguously (e.g. one
            element repeated across pressure levels). `None` is allowed only when
            the file holds a single message, or when `level`/`number` narrow it to
            one.
        level: Optional `level` to narrow the selection to (e.g. `850` for the
            850 hPa message); combined with `variable` as "that element at that
            level".
        number: Optional ensemble `number` to narrow the selection to (the
            perturbation number of a single member).
        target_crs: Optional CRS to reproject to before the COG is written — an
            EPSG `int` or a WKT `str` (forwarded as `to_cog(tiling=Tiling(target_srs=...))`).
            `None` keeps the GRIB's native CRS.
        cog_profile: Named COG compression preset forwarded to
            :meth:`~pyramids.dataset.Dataset.to_cog` as `compression=`, e.g.
            `"deflate"`, `"zstd"`, `"lzw"`.
        vsi: Optional explicit archive kind forwarded to :func:`open_grib` (e.g.
            for a GRIB inside a `.zip`).

    Returns:
        The `output` path as a :class:`~pathlib.Path`.

    Raises:
        DriverNotExistError: The GDAL build lacks the GRIB driver.
        FileNotFoundError: `grib_path` does not exist (raised by :func:`open_grib`).
        ValueError: `variable` cannot be resolved (`None` with no `level`/`number`
            on a multi-message file, an out-of-range band number, an empty string,
            an unknown element, or no message matching the element/`level`/`number`
            criteria), or `cog_profile` is not a recognised COG profile (raised by
            `to_cog`).

    Examples:
        - Convert the 2-metre temperature message to a COG (requires libgdal-grib):
            ```python
            >>> from pyramids.grib import grib_to_cog  # doctest: +SKIP
            >>> from pyramids.dataset.cog import cog_info  # doctest: +SKIP
            >>> out = grib_to_cog(  # doctest: +SKIP
            ...     "tmp2m.grib2", variable="TMP", output="tmp2m_cog.tif"
            ... )
            >>> cog_info(out).is_cog  # doctest: +SKIP
            True

            ```
    """
    with open_grib(grib_path, vsi=vsi) as dataset:
        band_index = _select_grib_band(
            grib_band_metadata(dataset), variable, level=level, number=number
        )
        result = cast(
            Path,
            dataset.to_cog(
                output,
                compression=cog_profile,
                bands=BandSelection(indexes=[band_index]),
                tiling=Tiling(target_srs=target_crs),
            ),
        )
    return result
