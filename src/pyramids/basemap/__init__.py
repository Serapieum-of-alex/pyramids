"""Basemap support for pyramids plots.

Adds web tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. Tiles are fetched, stitched, and
optionally warped to the data's CRS using GDAL.

Requires optional dependencies: mercantile, xyzservices, Pillow.
Install with: ``pyramids-gis[viz]``

Lazy imports are used so that importing this package does not fail
when the optional dependencies are not installed.
"""


def add_basemap(*args, **kwargs):
    """Add a basemap to a matplotlib Axes.

    Thin wrapper that lazily imports the real implementation from
    ``pyramids.basemap.basemap`` to avoid import errors when
    optional dependencies are not installed.

    See Also:
        pyramids.basemap.basemap.add_basemap: Full implementation
            with all parameters documented.
    """
    from pyramids.basemap.basemap import add_basemap as _impl

    return _impl(*args, **kwargs)


def get_provider(*args, **kwargs):
    """Resolve a tile provider by name.

    Thin wrapper that lazily imports the real implementation from
    ``pyramids.basemap.basemap``.

    See Also:
        pyramids.basemap.basemap.get_provider: Full implementation
            with all parameters documented.
    """
    from pyramids.basemap.basemap import get_provider as _impl

    return _impl(*args, **kwargs)


__all__ = ["add_basemap", "get_provider"]
