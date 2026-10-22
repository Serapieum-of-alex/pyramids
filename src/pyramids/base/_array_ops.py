"""Element-wise array operations shared by raster and mesh pipelines.

Every function takes a plain numpy array (plus metadata like a no-data
sentinel) and returns a plain numpy array.  No geometry, no GDAL, no
mesh — the caller (a Dataset engine or a UgridFacades method) owns the
spatial context.

Same design contract as :mod:`pyramids.base._reductions`: pure numpy
kernels that both sides of the class tree can import safely.
"""

from __future__ import annotations

import numpy as np


def nodata_mask(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask — ``True`` where *values* is missing.

    For floating types NaN positions are always missing regardless of
    the *nodata* sentinel.  For integer types only exact matches of
    *nodata* count.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None`` for none.

    Returns:
        Boolean array, same shape as *values*.
    """
    is_float = np.issubdtype(values.dtype, np.floating)
    if is_float:
        mask = np.isnan(values)
    else:
        mask = np.zeros(values.shape, dtype=bool)
    if nodata is not None:
        if is_float and np.isnan(nodata):
            pass
        else:
            mask = mask | (values == nodata)
    return mask


def fillna(
    values: np.ndarray, fill_value, nodata=None
) -> np.ndarray:
    """Replace missing values with *fill_value*.

    Args:
        values: The data array.
        fill_value: The replacement value.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        A copy of *values* with missing positions replaced.
    """
    mask = nodata_mask(values, nodata)
    result = values.copy()
    result[mask] = fill_value
    return result


def isnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return nodata_mask(values, nodata)


def notnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of non-missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return ~nodata_mask(values, nodata)


def clip_values(
    values: np.ndarray, lower, upper
) -> np.ndarray:
    """Clamp values to ``[lower, upper]``.

    Args:
        values: The data array.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        A new array with values clamped.
    """
    return np.clip(values, lower, upper)


def round_values(
    values: np.ndarray, decimals: int = 0
) -> np.ndarray:
    """Round values to *decimals* decimal places.

    Args:
        values: The data array.
        decimals: Number of decimal places.

    Returns:
        A new array with rounded values.
    """
    return np.round(values, decimals)


def astype_values(values: np.ndarray, dtype) -> np.ndarray:
    """Cast to a different dtype.

    Args:
        values: The data array.
        dtype: Target numpy dtype.

    Returns:
        A new array with the requested dtype.
    """
    return values.astype(dtype)


def isin_values(
    values: np.ndarray, test_values
) -> np.ndarray:
    """Boolean mask where *values* is in *test_values*.

    Args:
        values: The data array.
        test_values: Set of values to test membership against.

    Returns:
        Boolean array, same shape as *values*.
    """
    return np.isin(values, test_values)


def array_equals(
    a: np.ndarray,
    b: np.ndarray,
    nodata_a=None,
    nodata_b=None,
) -> bool:
    """Value equality — same values at every non-missing position.

    Two arrays are *equal* when they have the same shape, the same set
    of missing positions, and identical values everywhere else.

    Args:
        a: First array.
        b: Second array.
        nodata_a: No-data sentinel for *a*.
        nodata_b: No-data sentinel for *b*.

    Returns:
        ``True`` when the arrays are value-equal.
    """
    if a.shape != b.shape:
        return False
    mask_a = nodata_mask(a, nodata_a)
    mask_b = nodata_mask(b, nodata_b)
    if not np.array_equal(mask_a, mask_b):
        return False
    valid = ~mask_a
    if not np.any(valid):
        return True
    return bool(np.array_equal(a[valid], b[valid]))


def array_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict equality — same dtype, same values including NaN.

    Args:
        a: First array.
        b: Second array.

    Returns:
        ``True`` when the arrays are bitwise-identical.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))
