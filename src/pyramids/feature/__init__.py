"""Feature subpackage.

ARC-10 split the former single ``feature.py`` into a subpackage:

- :mod:`pyramids.feature.collection` — the :class:`FeatureCollection`
  class (a :class:`geopandas.GeoDataFrame` subclass).
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors.
- :mod:`pyramids.feature.crs` — CRS / EPSG / reprojection helpers.
- :mod:`pyramids.feature._ogr` — private OGR bridge (internal only).
"""

from pyramids.feature.collection import FeatureCollection

# L3: LazyFeatureCollection is only available when the [parquet-lazy] extra
# is installed. Two design goals:
#
# 1. Eager import on ``from pyramids.feature import LazyFeatureCollection``
#    should give the user the real class (when available) or a clear
#    ImportError with install instructions — NOT a silent ``None`` sentinel
#    that later breaks ``isinstance`` with a confusing ``TypeError``.
# 2. ``import pyramids.feature`` on a minimal install must not fail; the
#    :class:`FeatureCollection` path has to keep working.
#
# The PEP-562 ``__getattr__`` hook satisfies both: attribute lookups for
# ``LazyFeatureCollection`` try the real import, and raise a branded
# :class:`ImportError` when dask-geopandas is absent. ``hasattr(module,
# "LazyFeatureCollection")`` returns False on minimal installs (because
# ``__getattr__`` raised), so library authors can guard with a clean
# ``hasattr`` check instead of an ``isinstance`` against ``None``.
try:
    import dask_geopandas as _dask_geopandas  # noqa: F401
except ImportError:  # pragma: no cover - minimal install path
    _HAS_DASK_GEOPANDAS = False
else:
    _HAS_DASK_GEOPANDAS = True
    from pyramids.feature._lazy_collection import LazyFeatureCollection  # noqa: F401


_LAZY_FC_INSTALL_HINT = (
    "LazyFeatureCollection requires the optional 'dask-geopandas' "
    "dependency. Install with: pip install 'pyramids-gis[parquet-lazy]'"
)


def __getattr__(name: str) -> object:
    """PEP-562 hook — raise a clean ImportError for missing optional classes.

    Only ``LazyFeatureCollection`` is handled here. Any other unknown
    attribute falls through to the default :class:`AttributeError`.
    """
    if name == "LazyFeatureCollection":
        if _HAS_DASK_GEOPANDAS:
            from pyramids.feature._lazy_collection import LazyFeatureCollection
            return LazyFeatureCollection
        raise ImportError(_LAZY_FC_INSTALL_HINT)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FeatureCollection", "LazyFeatureCollection"]
