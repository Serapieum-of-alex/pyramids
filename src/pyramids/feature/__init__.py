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

__all__ = ["FeatureCollection"]
