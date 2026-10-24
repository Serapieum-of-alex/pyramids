"""Per-element along-axis fitting, calculus and ranking kernels (issue #1377).

Pure numpy / scipy kernels operating along one axis of an array, with no geometry and no GDAL
— the same design contract as :mod:`pyramids.base._reductions` and
:mod:`pyramids.base._array_ops`, so the raster and mesh sides can both import them safely.

Every kernel takes the values, the axis to work along, and (where the operation depends on
spacing) the coordinate along that axis. Passing the coordinate explicitly is what makes these
correct on an **unevenly spaced** axis, which a simulation's output interval often is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "coarsen_axis",
    "cumulative_integrate_axis",
    "curvefit_axis",
    "differentiate_axis",
    "integrate_axis",
    "interp_axis",
    "numeric_coordinate",
    "polyfit_axis",
    "rank_axis",
]


def numeric_coordinate(values: Any) -> np.typing.NDArray:
    """A float coordinate from anything an axis coordinate may be.

    Fitting and calculus need numbers, but a decoded time axis carries datetimes (issue
    #1381). A datetime-like coordinate is converted to **seconds elapsed from its first
    stamp**, so a derivative comes out per second and an integral in value-seconds, both of
    which are the physically meaningful units for a simulation. A numeric coordinate is
    returned as float unchanged.

    Args:
        values: The coordinate values — datetimes, numbers, or anything numpy can read as one.

    Returns:
        np.ndarray: A 1-D float64 coordinate.

    Raises:
        ValueError: The coordinate cannot be read as numbers or datetimes.
    """
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        deltas = array - array[0]
        return np.asarray(
            deltas.astype("timedelta64[ns]").astype("float64") / 1e9, dtype="float64"
        )
    if array.dtype == object:
        # A list of datetimes or cftime objects arrives as object dtype; let numpy try the
        # datetime reading first, then fall back to plain numbers.
        try:
            return numeric_coordinate(np.asarray(array, dtype="datetime64[ns]"))
        except (TypeError, ValueError):
            try:
                return np.asarray(array, dtype="float64")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "coordinate: cannot read the axis coordinate as numbers or datetimes; "
                    "fitting and calculus need one or the other."
                ) from error
    return np.asarray(array, dtype="float64")


def polyfit_axis(
    values: np.ndarray, coord: Any, deg: int, axis: int = 0
) -> np.typing.NDArray:
    """Least-squares polynomial coefficients per element, along ``axis``.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``, length ``values.shape[axis]``.
        deg: Polynomial degree.
        axis: The axis to fit along.

    Returns:
        np.ndarray: Coefficients with ``axis`` replaced by one of length ``deg + 1``, highest
        power first — the order :func:`numpy.polyfit` returns.

    Raises:
        ValueError: ``deg`` is negative, or the axis is too short for the degree.
    """
    if deg < 0:
        raise ValueError(f"polyfit: deg must be non-negative, got {deg}.")
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(coord)
    if array.shape[0] < deg + 1:
        raise ValueError(
            f"polyfit: a degree-{deg} fit needs at least {deg + 1} steps, got "
            f"{array.shape[0]}."
        )
    flat = array.reshape(array.shape[0], -1)
    coeffs = np.polyfit(x, flat, deg)
    return np.moveaxis(coeffs.reshape((deg + 1, *array.shape[1:])), 0, axis)


def curvefit_axis(
    values: np.ndarray,
    coord: Any,
    func: Any,
    p0: Any = None,
    axis: int = 0,
) -> np.typing.NDArray:
    """Fit an arbitrary model per element along ``axis``.

    Unlike :func:`polyfit_axis` there is no closed form, so each element is fitted in turn.
    An element whose fit does not converge yields ``NaN`` coefficients rather than failing the
    whole call — one bad cell should not lose the other thousands.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        func: The model ``f(x, *params)``.
        p0: Initial parameter guess, or ``None``.
        axis: The axis to fit along.

    Returns:
        np.ndarray: Fitted parameters with ``axis`` replaced by one of length ``n_params``.

    Raises:
        ValueError: The number of parameters cannot be determined from ``func`` without ``p0``.
    """
    from scipy.optimize import curve_fit

    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(coord)
    flat = array.reshape(array.shape[0], -1)
    if p0 is None:
        import inspect

        params = list(inspect.signature(func).parameters)
        n_params = max(len(params) - 1, 1)
    else:
        n_params = len(np.atleast_1d(p0))
    out = np.full((n_params, flat.shape[1]), np.nan, dtype="float64")
    for column in range(flat.shape[1]):
        series = flat[:, column]
        finite = np.isfinite(series)
        if finite.sum() < n_params:
            continue
        try:
            popt, _ = curve_fit(func, x[finite], series[finite], p0=p0)
        except (RuntimeError, ValueError, TypeError):
            # Non-convergence (or a model the data cannot support) is per element: leave this
            # one NaN and carry on.
            continue
        out[:, column] = popt
    return np.moveaxis(out.reshape((n_params, *array.shape[1:])), 0, axis)


def differentiate_axis(
    values: np.ndarray, coord: Any, axis: int = 0
) -> np.typing.NDArray:
    """The derivative along ``axis``, honouring the coordinate's own spacing.

    Uses :func:`numpy.gradient` with the coordinate, so an unevenly spaced axis gives the right
    rate rather than one that assumes a uniform step.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to differentiate along.

    Returns:
        np.ndarray: The derivative, the same shape as ``values``.

    Raises:
        ValueError: The axis has fewer than two steps, so no difference is defined.
    """
    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"differentiate: needs at least two steps along the axis, got "
            f"{array.shape[axis]}."
        )
    return np.asarray(np.gradient(array, x, axis=axis), dtype="float64")


def integrate_axis(values: np.ndarray, coord: Any, axis: int = 0) -> np.typing.NDArray:
    """The definite integral along ``axis``, which the integral consumes.

    Trapezoidal over the coordinate's own spacing.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to integrate along.

    Returns:
        np.ndarray: The integral, with ``axis`` removed.

    Raises:
        ValueError: The axis has fewer than two steps.
    """
    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"integrate: needs at least two steps along the axis, got {array.shape[axis]}."
        )
    return np.asarray(np.trapezoid(array, x=x, axis=axis), dtype="float64")


def cumulative_integrate_axis(
    values: np.ndarray, coord: Any, axis: int = 0
) -> np.typing.NDArray:
    """The running integral along ``axis``, which keeps the axis.

    The first step is ``0`` — nothing has been accumulated yet — matching
    ``scipy.integrate.cumulative_trapezoid(initial=0)``.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to integrate along.

    Returns:
        np.ndarray: The running integral, the same shape as ``values``.

    Raises:
        ValueError: The axis has fewer than two steps.
    """
    from scipy.integrate import cumulative_trapezoid

    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"cumulative_integrate: needs at least two steps along the axis, got "
            f"{array.shape[axis]}."
        )
    return np.asarray(
        cumulative_trapezoid(array, x=x, axis=axis, initial=0.0), dtype="float64"
    )


def rank_axis(values: np.ndarray, axis: int = 0) -> np.typing.NDArray:
    """The rank of each element's values along ``axis``, ties averaged.

    Matches the raster ``rank``: ranks run from 1, and tied values share their mean rank. NaN
    is ranked last by :func:`scipy.stats.rankdata`'s ``omit`` policy, which leaves it NaN.

    Args:
        values: The data array.
        axis: The axis to rank along.

    Returns:
        np.ndarray: Ranks, the same shape as ``values``.
    """
    from scipy.stats import rankdata

    array = np.asarray(values, dtype="float64")
    return np.asarray(rankdata(array, axis=axis, nan_policy="omit"), dtype="float64")


def coarsen_axis(
    values: np.ndarray, window: int, how: str = "mean", axis: int = 0
) -> np.typing.NDArray:
    """Block-aggregate ``axis`` into windows of ``window`` steps.

    Bins by a **count of steps**, needing no coordinate at all — which is what makes it usable
    where ``resample`` is not, since that one bins by a calendar frequency and so requires
    date-like coordinates (issue #1377).

    A trailing remainder shorter than a full window is **dropped**, so every reported bin is a
    full one rather than a short bin that would read as a dip in the series.

    Args:
        values: The data array.
        window: Steps per block, at least 1.
        how: ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"`` or ``"std"``.
        axis: The axis to aggregate along.

    Returns:
        np.ndarray: The aggregate, with ``axis`` shortened to ``n // window``.

    Raises:
        ValueError: ``window`` is below 1, ``how`` is unknown, or the axis is shorter than one
            window, which would leave nothing to report.
    """
    reducers: dict[str, Any] = {
        "mean": np.nanmean,
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
        "std": np.nanstd,
    }
    if window < 1:
        raise ValueError(f"coarsen: window must be at least 1, got {window}.")
    if how not in reducers:
        raise ValueError(
            f"coarsen: how must be one of {sorted(reducers)}, got {how!r}."
        )
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    blocks = array.shape[0] // window
    if blocks == 0:
        raise ValueError(
            f"coarsen: the axis has {array.shape[0]} steps, fewer than one window of "
            f"{window}; there would be nothing to report."
        )
    trimmed = array[: blocks * window]
    reshaped = trimmed.reshape((blocks, window, *array.shape[1:]))
    reduced = reducers[how](reshaped, axis=1)
    return np.moveaxis(np.asarray(reduced, dtype="float64"), 0, axis)


def interp_axis(
    values: np.ndarray,
    source: Any,
    target: Any,
    axis: int = 0,
) -> np.typing.NDArray:
    """Interpolate ``values`` from the ``source`` coordinate onto ``target``.

    Linear, and deliberately **not** extrapolating: a target position outside the source span
    comes back NaN — a gap — which is the contract the raster ``interp`` already has.

    Args:
        values: The data array.
        source: The coordinate ``values`` is on, along ``axis``.
        target: The coordinate to interpolate onto.
        axis: The axis to interpolate along.

    Returns:
        np.ndarray: The interpolated values, with ``axis`` resized to ``len(target)``.

    Raises:
        ValueError: The source axis has fewer than two steps, so there is nothing to
            interpolate between.
    """
    from scipy.interpolate import interp1d

    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(source)
    # The target is put on the *source's* scale: both are converted the same way, so two
    # datetime axes line up even when their own units differ.
    x_new = numeric_coordinate(
        np.concatenate([np.asarray(source)[:1], np.asarray(target)])
    )[1:]
    if array.shape[0] < 2:
        raise ValueError(
            f"interp: needs at least two steps to interpolate between, got {array.shape[0]}."
        )
    interpolator: Any = interp1d(
        x, array, axis=0, bounds_error=False, fill_value=np.nan, assume_sorted=False
    )
    return np.moveaxis(np.asarray(interpolator(x_new), dtype="float64"), 0, axis)
