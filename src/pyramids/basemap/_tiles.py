"""Tile math, fetching, and stitching.

This module handles XYZ web tile operations: computing zoom levels from
geographic bounds, fetching tile PNG images over HTTP in parallel, and
stitching them into a single composite image. No GIS dependencies are
used here -- only mercantile (tile math), urllib (HTTP), and Pillow
(image processing).
"""

from __future__ import annotations

import io
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

USER_AGENT = "pyramids-gis/Python"
MAX_TILES = 256


def _auto_zoom(bounds_4326: tuple[float, float, float, float]) -> int:
    """Compute an appropriate zoom level from bounds in EPSG:4326.

    Uses the formula ``zoom = ceil(log2(360 / max(lon_extent, lat_extent)))``
    clamped to the range 0--19.

    Parameters
    ----------
    bounds_4326 : tuple[float, float, float, float]
        ``(west, south, east, north)`` in EPSG:4326 degrees.

    Returns
    -------
    int
        Zoom level between 0 and 19.

    Examples
    --------
    >>> _auto_zoom((-180, -85, 180, 85))
    0
    >>> _auto_zoom((13.0, 52.4, 13.6, 52.6))
    8
    """
    west, south, east, north = bounds_4326
    lon_extent = abs(east - west)
    lat_extent = abs(north - south)
    max_extent = max(lon_extent, lat_extent, 1e-10)
    zoom = math.ceil(math.log2(360.0 / max_extent))
    result = max(0, min(zoom, 19))
    return result


def _fetch_single_tile(
    tile: Any,
    provider: Any,
    timeout: int,
    retries: int,
) -> tuple[Any, bytes]:
    """Fetch a single tile with retries.

    Parameters
    ----------
    tile : mercantile.Tile
        Tile to fetch (has x, y, z attributes).
    provider : xyzservices.TileProvider
        Tile provider with url template.
    timeout : int
        HTTP request timeout in seconds.
    retries : int
        Number of retry attempts.

    Returns
    -------
    tuple[mercantile.Tile, bytes]
        The tile and its PNG image bytes.

    Raises
    ------
    ConnectionError
        If the tile cannot be fetched after all retries.
    """
    url = provider.build_url(x=tile.x, y=tile.y, z=tile.z)
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            png_bytes = response.read()
            return tile, png_bytes
        except (OSError, urllib.error.URLError, ConnectionError) as e:
            last_error = e
            logger.debug(
                "Tile fetch attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                url,
                e,
            )
    raise ConnectionError(
        f"Failed to fetch tile z={tile.z}/x={tile.x}/y={tile.y} "
        f"after {retries + 1} attempts: {last_error}"
    )


def _fetch_tiles(
    tiles: list,
    provider: Any,
    max_workers: int = 8,
    timeout: int = 10,
    retries: int = 2,
) -> dict:
    """Fetch tile PNG images over HTTP in parallel.

    Uses ``concurrent.futures.ThreadPoolExecutor`` for parallel downloads.
    Each tile URL is constructed from the provider's URL template by
    substituting ``{x}``, ``{y}``, ``{z}`` placeholders. A ``User-Agent``
    header (``pyramids-gis/Python``) is set on all requests to comply with
    tile provider requirements.

    Parameters
    ----------
    tiles : list[mercantile.Tile]
        Tiles to fetch (each has x, y, z attributes).
    provider : xyzservices.TileProvider
        Tile provider with a url template and optional subdomains.
    max_workers : int, optional
        Maximum concurrent HTTP connections. Default is 8.
    timeout : int, optional
        HTTP request timeout in seconds per tile. Default is 10.
    retries : int, optional
        Number of retry attempts per failed tile. Default is 2.

    Returns
    -------
    dict[mercantile.Tile, bytes]
        Mapping of Tile to PNG bytes.

    Raises
    ------
    ConnectionError
        If any tile cannot be fetched after all retries.
    """
    tile_data: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_single_tile, tile, provider, timeout, retries
            ): tile
            for tile in tiles
        }
        for future in as_completed(futures):
            tile_obj, png_bytes = future.result()
            tile_data[tile_obj] = png_bytes
    return tile_data


def _stitch_tiles(
    tile_data: dict,
    tiles: list,
    zoom: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch tile images into a single RGBA array.

    Arranges tiles in a grid based on their x, y positions. The tile
    size is read from the first fetched image (typically 256 or 512px).
    The output array has shape ``(height, width, 3 or 4)`` with dtype
    ``uint8``.

    Also computes the geographic extent of the stitched image in
    EPSG:3857 coordinates using ``mercantile.xy_bounds()`` on the corner
    tiles.

    Parameters
    ----------
    tile_data : dict[mercantile.Tile, bytes]
        Mapping of Tile to PNG bytes (from ``_fetch_tiles``).
    tiles : list[mercantile.Tile]
        All tiles in the grid (defines the grid dimensions).
    zoom : int
        Zoom level of the tiles.

    Returns
    -------
    image : numpy.ndarray
        Stitched image, shape ``(H, W, 3 or 4)``, dtype ``uint8``.
    extent_3857 : tuple[float, float, float, float]
        ``(west, south, east, north)`` in EPSG:3857 meters.
    """
    from PIL import Image

    import mercantile

    first_img = Image.open(io.BytesIO(next(iter(tile_data.values()))))
    tile_size = first_img.width

    x_indices = sorted(set(t.x for t in tiles))
    y_indices = sorted(set(t.y for t in tiles))
    width = len(x_indices) * tile_size
    height = len(y_indices) * tile_size

    merged = Image.new("RGBA", (width, height))
    for tile, png_bytes in tile_data.items():
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        x_offset = (tile.x - x_indices[0]) * tile_size
        y_offset = (tile.y - y_indices[0]) * tile_size
        merged.paste(img, (x_offset, y_offset))

    image = np.array(merged)

    tl = mercantile.xy_bounds(
        mercantile.Tile(x_indices[0], y_indices[0], zoom)
    )
    br = mercantile.xy_bounds(
        mercantile.Tile(x_indices[-1], y_indices[-1], zoom)
    )
    extent_3857 = (tl.left, br.bottom, br.right, tl.top)

    return image, extent_3857
