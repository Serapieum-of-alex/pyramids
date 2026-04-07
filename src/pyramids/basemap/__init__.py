"""Basemap support for pyramids plots.

Adds web tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. Tiles are fetched, stitched, and
optionally warped to the data's CRS using GDAL.

Requires optional dependencies: mercantile, xyzservices, Pillow.
Install with: ``pip install pyramids-gis[viz]``

Examples
--------
>>> from pyramids.basemap import add_basemap
>>> from pyramids.dataset import Dataset
>>> ds = Dataset.read_file("dem.tif")
>>> glyph = ds.plot(band=0)
>>> add_basemap(glyph.ax, crs=ds.epsg)
"""


def add_basemap(*args, **kwargs):
    """Add a basemap to a matplotlib Axes. See basemap.add_basemap."""
    from pyramids.basemap.basemap import add_basemap as _impl

    return _impl(*args, **kwargs)


def get_provider(*args, **kwargs):
    """Resolve a tile provider by name. See basemap.get_provider."""
    from pyramids.basemap.basemap import get_provider as _impl

    return _impl(*args, **kwargs)


__all__ = ["add_basemap", "get_provider"]
