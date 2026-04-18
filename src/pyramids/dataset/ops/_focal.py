"""Focal (neighborhood) operations on a :class:`Dataset`.

DASK-26: per-pixel neighborhood filters that read a small halo of
surrounding cells. Two backends:

* Eager (default): SciPy :mod:`scipy.ndimage` filter applied to the
  full numpy array.
* Lazy (``chunks=<spec>``): wrap the same kernel in
  :func:`dask.array.map_overlap` with ``depth=radius``,
  ``boundary='reflect'``. dask-image's universal primitive.

Supported ops:

* ``focal_mean(radius)`` — uniform box average.
* ``focal_std(radius)`` — standard deviation.
* ``focal_apply(func, radius)`` — user-supplied kernel.
* ``slope()``, ``aspect()``, ``hillshade(az, alt)`` — classic DEM
  derivatives via centered-difference gradient.

``scipy`` is already a core pyramids dep, so the eager path has
zero import cost. Dask is imported only when ``chunks`` is given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from pyramids.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "chunks= requires the optional 'dask' dependency. "
    "Install with: pip install 'pyramids-gis[lazy]'"
)


def _apply_eager_or_lazy(
    func: Callable,
    ds: "Dataset",
    radius: int,
    chunks: Any,
    band: int,
    dtype: Any,
) -> Any:
    """Run ``func`` on the band eagerly or wrap with ``dask.map_overlap``.

    ``func`` must accept a 2-D numpy array and return a 2-D numpy
    array of the same shape.
    """
    if chunks is None:
        arr = np.asarray(ds.read_array(band=band), dtype=dtype)
        return func(arr)
    try:
        import dask.array as da
    except ImportError as exc:
        raise ImportError(_LAZY_IMPORT_ERROR) from exc
    lazy = ds.read_array(band=band, chunks=chunks)
    if not hasattr(lazy, "dask"):
        lazy = da.from_array(np.asarray(lazy), chunks="auto")
    lazy = lazy.astype(dtype)
    return lazy.map_overlap(
        func, depth=radius, boundary="reflect", trim=True, dtype=dtype,
    )


def focal_mean(
    ds: "Dataset",
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Uniform box mean over a ``(2*radius+1)``-side window.

    Args:
        ds: Source :class:`~pyramids.dataset.Dataset`.
        radius: Half-window in pixels. Default 1 (→ 3×3 window).
        chunks: If given, switch to the lazy path via
            :func:`dask.array.map_overlap`.
        band: Zero-based band index. Default 0.
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(arr, size=size, mode="reflect")

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def focal_std(
    ds: "Dataset",
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Standard deviation over a ``(2*radius+1)``-side window.

    L4: uses the two-pass formulation ``sqrt(mean((x - local_mean)²))``
    rather than the unstable ``sqrt(E[x²] - E[x]²)``. The cancellation
    error in the latter blows up for large magnitudes with small
    variance (a common DEM case — elevations in metres where the
    local deviation is centimetres). The two-pass variant does one
    extra uniform_filter; the cost is linear in pixels and negligible
    compared to the I/O.
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        local_mean = ndimage.uniform_filter(arr, size=size, mode="reflect")
        deviations = (arr - local_mean) ** 2
        var = ndimage.uniform_filter(deviations, size=size, mode="reflect")
        return np.sqrt(np.clip(var, 0.0, None))

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def focal_apply(
    ds: "Dataset",
    func: Callable[[np.ndarray], float],
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Apply a user-supplied aggregation over a ``(2*radius+1)`` window.

    ``func`` receives a flat 1-D array of window values and returns
    one scalar per window. Wrapped with
    :func:`scipy.ndimage.generic_filter`.
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        return ndimage.generic_filter(arr, func, size=size, mode="reflect")

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def _gradient(arr: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered-difference gradient (dz/dx, dz/dy) at each cell."""
    dz_dy, dz_dx = np.gradient(arr, cell_size)
    return dz_dx, dz_dy


def slope(
    ds: "Dataset",
    *,
    chunks: Any = None,
    band: int = 0,
    units: str = "degrees",
) -> Any:
    """Slope of a DEM in degrees (default) or radians.

    Computed via :func:`numpy.gradient` centered differences.
    """
    cell_size = float(ds.cell_size)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        magnitude = np.hypot(dz_dx, dz_dy)
        radians = np.arctan(magnitude)
        return np.degrees(radians) if units == "degrees" else radians

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


def aspect(
    ds: "Dataset",
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Aspect (degrees clockwise from north) of a DEM."""
    cell_size = float(ds.cell_size)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        angle = np.degrees(np.arctan2(dz_dy, -dz_dx))
        return np.mod(450.0 - angle, 360.0)

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


def hillshade(
    ds: "Dataset",
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Shaded-relief map in 0..255 given sun azimuth + altitude (degrees)."""
    cell_size = float(ds.cell_size)
    az_rad = np.deg2rad(360.0 - azimuth + 90.0)
    alt_rad = np.deg2rad(altitude)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        shaded = (
            np.sin(alt_rad) * np.cos(slope_rad)
            + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
        )
        return np.clip(shaded * 255.0, 0.0, 255.0)

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


__all__ = [
    "focal_mean", "focal_std", "focal_apply",
    "slope", "aspect", "hillshade",
]
