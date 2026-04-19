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

# DASK-40b / DASK-40c: LazyFeatureCollection is only importable when the
# [parquet-lazy] extra is installed. Narrow the guard to the import of
# dask_geopandas itself — a broad ``except ImportError`` around the whole
# ``_lazy_collection`` import would also swallow a SyntaxError or a missing
# pyproj and silently set the class to None, which is worse than failing
# loudly. On minimal installs, consumer code using ``isinstance`` must guard:
# ``if LazyFeatureCollection is not None and isinstance(x, LazyFeatureCollection): ...``.
try:
    import dask_geopandas  # noqa: F401
except ImportError:  # pragma: no cover - minimal install without dask-geopandas
    LazyFeatureCollection = None  # type: ignore[assignment,misc]
else:
    from pyramids.feature._lazy_collection import LazyFeatureCollection

__all__ = ["FeatureCollection", "LazyFeatureCollection"]
