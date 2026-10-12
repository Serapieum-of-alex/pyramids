"""Shared dispatcher for dask-array nan-aware reductions.

Used by :meth:`DatasetCollection._reduce` (time-axis reductions)
and :func:`_grouped_reduce` (per-label grouped reductions) so both
call sites agree on the supported ops and the nan-aware naming
convention.

The ``"mean" / "sum" / "min" / "max" / "std" / "var"`` ops are plain
associative :mod:`dask.array` reductions resolved by
:func:`resolve_dask_op`. :meth:`DatasetCollection.reduce_time` additionally
offers the *holistic* ops ``"median"``, ``"quantile"`` / ``"percentile"``,
``"mode"`` and ``"trend"`` / ``"sens_slope"``; :func:`build_group_reduction`
dispatches every op to a single lazy reduction so the collection keeps its
single-pass :func:`dask.compute` batching.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from pyramids.base._reductions import reduce_axis
from pyramids.base._utils import extra_hint

_SUPPORTED_OPS = ("mean", "sum", "min", "max", "std", "var")
_NAN_TABLE = {op: f"nan{op}" for op in _SUPPORTED_OPS}

#: Holistic ops :meth:`DatasetCollection.reduce_time` adds on top of the six
#: associative ones; none is a simple :mod:`dask.array` reduction.
_HOLISTIC_OPS = ("median", "quantile", "percentile", "mode", "trend", "sens_slope")

#: Every op :meth:`DatasetCollection.reduce_time` accepts.
TIME_REDUCE_OPS = _SUPPORTED_OPS + _HOLISTIC_OPS

#: Ops that require a level ``q`` (and reject it for every other op).
_Q_OPS = ("quantile", "percentile")

#: Per-pixel slope ops that need the window's numeric timestamps.
_TREND_OPS = ("trend", "sens_slope")


def resolve_dask_op(op_name: str, *, skipna: bool) -> Callable[..., Any]:
    """Return the :mod:`dask.array` callable for a reduction.

    Centralises the ``nan{op} if skipna else op`` name mangling and the
    ``getattr(dask.array, name)`` resolution so every call site agrees on
    the supported ops and the nan-aware naming convention.

    Args:
        op_name: One of ``mean``, ``sum``, ``min``, ``max``, ``std``, ``var``.
        skipna: When True, return the nan-aware variant
            (e.g. ``nanmean``); otherwise the plain op.

    Returns:
        The resolved :mod:`dask.array` callable (e.g. ``dask.array.nanmean``).

    Raises:
        ValueError: If ``op_name`` is outside the supported set.
        ImportError: If the optional ``dask`` dependency is not
            installed.
    """
    if op_name not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported reduction {op_name!r}; supported: {_SUPPORTED_OPS}"
        )
    try:
        import dask.array as da
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            extra_hint(
                "DatasetCollection reductions require the optional 'dask' dependency.",
                "lazy",
            )
        ) from exc
    name = _NAN_TABLE[op_name] if skipna else op_name
    # getattr on a module is untyped; every name in _NAN_TABLE / _SUPPORTED_OPS
    # names a real dask.array reduction function.
    return cast("Callable[..., Any]", getattr(da, name))


def validate_time_op(op: str, q: float | None) -> None:
    """Validate a :meth:`DatasetCollection.reduce_time` ``op`` / ``q`` pair.

    Args:
        op: The reduction name; must be one of :data:`TIME_REDUCE_OPS`.
        q: The level for ``"percentile"`` (``0 < q <= 100``) or ``"quantile"``
            (``0 < q <= 1``). Required for those two ops, and refused (must be
            ``None``) for every other op.

    Raises:
        ValueError: ``op`` is unsupported; ``q`` is missing for a
            percentile/quantile op or out of range; or ``q`` is given for an
            op that takes none.
    """
    if op not in TIME_REDUCE_OPS:
        raise ValueError(f"op must be one of {TIME_REDUCE_OPS}, got {op!r}.")
    if op in _Q_OPS:
        high = 100.0 if op == "percentile" else 1.0
        if q is None:
            raise ValueError(f"op={op!r} requires a level q in (0, {high:g}].")
        if not 0 < q <= high:
            raise ValueError(
                f"q must satisfy 0 < q <= {high:g} for op={op!r}, got {q!r}."
            )
    elif q is not None:
        raise ValueError(
            f"q is only valid for 'percentile' / 'quantile', not op={op!r}."
        )


def _mode_axis0(block: Any) -> Any:
    """Per-pixel modal value over axis 0, NaN-aware, smallest value on ties.

    Sorts each pixel column ascending (NaN sorts to the end), then walks the
    time axis once accumulating equal-value run lengths. Because equal values
    are adjacent after the sort, the longest run is the most frequent value;
    the strict ``>`` update keeps the first (hence smallest) value when two
    values tie on count. An all-gap column yields ``NaN``.

    Args:
        block: A ``(n_t, ...)`` array; axis 0 is the time axis to reduce.

    Returns:
        The ``(...)`` per-pixel modal array (float64).
    """
    arr = np.asarray(block, dtype="float64")
    n_t = arr.shape[0]
    flat = arr.reshape(n_t, -1)
    srt = np.sort(flat, axis=0)
    best_val = np.full(flat.shape[1], np.nan)
    best_len = np.zeros(flat.shape[1], dtype=np.int64)
    cur_val = srt[0].copy()
    cur_len = np.where(np.isnan(srt[0]), 0, 1).astype(np.int64)
    better = cur_len > best_len
    best_len = np.where(better, cur_len, best_len)
    best_val = np.where(better, cur_val, best_val)
    for i in range(1, n_t):
        row = srt[i]
        same = row == cur_val  # NaN compares False, so a gap breaks the run
        cur_len = np.where(same, cur_len + 1, np.where(np.isnan(row), 0, 1))
        cur_val = np.where(same, cur_val, row)
        better = cur_len > best_len  # strict: the earliest (smallest) run wins a tie
        best_len = np.where(better, cur_len, best_len)
        best_val = np.where(better, cur_val, best_val)
    return best_val.reshape(arr.shape[1:])


def _ols_slope(y: Any, x: Any, *, skipna: bool) -> Any:
    """Per-pixel ordinary-least-squares slope of ``y`` against ``x`` over axis 0."""
    valid = ~np.isnan(y)
    if not skipna:
        valid = valid & valid.all(axis=0)  # any gap invalidates the whole pixel
    count = valid.sum(axis=0)
    xm = np.where(valid, x, np.nan)
    ym = np.where(valid, y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_dev = xm - np.nanmean(xm, axis=0)
        y_dev = ym - np.nanmean(ym, axis=0)
        num = np.nansum(np.where(valid, x_dev * y_dev, 0.0), axis=0)
        den = np.nansum(np.where(valid, x_dev * x_dev, 0.0), axis=0)
        slope = np.where((count >= 2) & (den > 0), num / den, np.nan)
    return slope


def _theil_sen_slope(y: Any, x: Any, *, skipna: bool) -> Any:
    """Per-pixel Theil–Sen slope: the median of pairwise slopes over axis 0."""
    n_t = y.shape[0]
    lower, upper = np.triu_indices(n_t, k=1)
    dx = x[upper] - x[lower]
    dy = y[upper] - y[lower]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pair_slopes = np.where(dx == 0, np.nan, dy / dx)
        slope = np.nanmedian(pair_slopes, axis=0)
    if not skipna:
        slope = np.where(np.isnan(y).any(axis=0), np.nan, slope)
    return slope


def _trend_block(block: Any, *, x: Any, method: str, skipna: bool) -> Any:
    """Reduce axis 0 of ``block`` to a per-pixel slope against timestamps ``x``.

    Args:
        block: A ``(n_t, ...)`` array; axis 0 is the time axis.
        x: The ``(n_t,)`` numeric timestamps (day offsets) for this window.
        method: ``"trend"`` for least-squares, ``"sens_slope"`` for Theil–Sen.
        skipna: When True, fit on each pixel's non-gap timesteps (needs >= 2);
            when False, any gap makes that pixel ``NaN``.

    Returns:
        The ``(...)`` per-pixel slope array (float64), in value-units per day.
    """
    arr = np.asarray(block, dtype="float64")
    n_t = arr.shape[0]
    y = arr.reshape(n_t, -1)
    x_col = np.asarray(x, dtype="float64").reshape(n_t, 1)
    if method == "sens_slope":
        slope = _theil_sen_slope(y, x_col, skipna=skipna)
    else:
        slope = _ols_slope(y, x_col, skipna=skipna)
    return slope.reshape(arr.shape[1:])


def build_group_reduction(
    group: Any, op: str, *, skipna: bool, q: float | None = None, x: Any = None
) -> Any:
    """Build the lazy :mod:`dask.array` reduction of ``group`` over axis 0.

    One dispatcher so :meth:`DatasetCollection.reduce_time` can batch every
    window's reduction into a single :func:`dask.compute`, whatever the op.
    The six associative ops go through :func:`resolve_dask_op`; ``median`` /
    ``quantile`` / ``percentile`` reuse the shared :func:`reduce_axis`
    (NaN / no-data-aware, percentile mapped onto the ``[0, 1]`` quantile);
    ``mode`` and ``trend`` / ``sens_slope`` are numpy kernels mapped over the
    time axis after rechunking it into a single block.

    Args:
        group: The window's lazy ``(n_t, band, rows, cols)`` dask slice.
        op: One of :data:`TIME_REDUCE_OPS`.
        skipna: Ignore the no-data value / NaN in the reduction.
        q: The level for ``percentile`` (``0 < q <= 100``) / ``quantile``
            (``0 < q <= 1``); ``None`` otherwise.
        x: The window's ``(n_t,)`` numeric day offsets, for the trend ops only.

    Returns:
        The lazy ``(band, rows, cols)`` dask reduction for this window.

    Raises:
        ImportError: If the optional ``dask`` dependency is not installed.
    """
    try:
        import dask.array as da
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            extra_hint(
                "DatasetCollection reductions require the optional 'dask' dependency.",
                "lazy",
            )
        ) from exc

    if op in _SUPPORTED_OPS:
        result = resolve_dask_op(op, skipna=skipna)(group, axis=0)
    elif op in ("median", "quantile", "percentile"):
        if op == "median":
            frac = None
        else:
            # validate_time_op guarantees q is set for quantile / percentile.
            assert q is not None  # nosec B101
            frac = q / 100.0 if op == "percentile" else q
        how = "median" if op == "median" else "quantile"
        result = reduce_axis(group, 0, how, skipna, None, frac)
    elif op == "mode":
        result = da.map_blocks(
            _mode_axis0, group.rechunk({0: -1}), drop_axis=0, dtype="float64"
        )
    else:  # trend / sens_slope
        result = da.map_blocks(
            _trend_block,
            group.rechunk({0: -1}),
            drop_axis=0,
            dtype="float64",
            x=np.asarray(x, dtype="float64"),
            method=op,
            skipna=skipna,
        )
    return result
