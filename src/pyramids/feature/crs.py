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

D-2 / L-1 progress: :func:`create_sr_from_proj` and
:func:`get_epsg_from_prj` were moved to
:mod:`pyramids.base.crs` so that ``base.crs`` can host the new
:func:`pyramids.base.crs.epsg_from_wkt` wrapper without inverting
the layering. Both names are re-exported here verbatim so all
existing imports — including
:meth:`pyramids.feature.FeatureCollection.get_epsg_from_prj` —
keep working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj.exceptions
from pyproj import Transformer

from pyramids.base._errors import CRSError
from pyramids.base.crs import create_sr_from_proj, get_epsg_from_prj


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


__all__ = [
    "create_sr_from_proj",
    "get_epsg_from_prj",
    "reproject_coordinates",
]
