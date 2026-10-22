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

    Examples:
        - Float array — NaN is always missing:

            >>> import numpy as np
            >>> nodata_mask(np.array([1.0, np.nan, 3.0]))
            array([False,  True, False])

        - Integer array with a sentinel value:

            >>> nodata_mask(np.array([1, -9999, 3]), nodata=-9999)
            array([False,  True, False])

    See Also:
        isnull: Identical behaviour, provided as a named alias.
        notnull: Inverted mask (``True`` where valid).
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

    Examples:
        - Replace NaN in a float array:

            >>> import numpy as np
            >>> fillna(np.array([1.0, np.nan, 3.0]), 0.0)
            array([1., 0., 3.])

        - Replace a sentinel value in an integer array:

            >>> fillna(np.array([1, -9999, 3]), 0, nodata=-9999)
            array([1, 0, 3])

    See Also:
        nodata_mask: The underlying mask logic used to locate
            missing positions.
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

    Examples:
        >>> import numpy as np
        >>> clip_values(np.array([1.0, 5.0, 10.0]), 2.0, 8.0)
        array([2., 5., 8.])

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

    Examples:
        - NaN positions must match but are otherwise ignored:

            >>> import numpy as np
            >>> a = np.array([1.0, np.nan, 3.0])
            >>> b = np.array([1.0, np.nan, 3.0])
            >>> array_equals(a, b)
            True

        - Different shapes are never equal:

            >>> array_equals(np.array([1.0]), np.array([1.0, 2.0]))
            False

    See Also:
        array_identical: Strict comparison including dtype and NaN
            bit-patterns.
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

    Examples:
        - Arrays must share dtype to be identical:

            >>> import numpy as np
            >>> a = np.array([1, 2], dtype=np.int32)
            >>> b = np.array([1, 2], dtype=np.int64)
            >>> array_identical(a, b)
            False

    See Also:
        array_equals: Looser comparison that ignores nodata positions.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))
