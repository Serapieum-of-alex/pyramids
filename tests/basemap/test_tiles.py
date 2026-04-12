"""Tests for pyramids.basemap.tiles module.

Covers auto_zoom, fetch_single_tile, fetch_tiles, and stitch_tiles
with mocked HTTP and synthetic tile images (no real network calls).
"""

from __future__ import annotations

import io
from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.plot

pytest.importorskip("PIL", reason="Pillow not installed (viz extra)")
from PIL import Image

from pyramids.basemap.tiles import (
    MAX_TILES,
    USER_AGENT,
    auto_zoom,
    fetch_single_tile,
    fetch_tiles,
    stitch_tiles,
)

Tile = namedtuple("Tile", ["x", "y", "z"])


def _make_tile_png(
    size: int = 256,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> bytes:
    """Create a solid-color PNG tile image as bytes.

    Parameters
    ----------
    size : int
        Width and height of the square tile in pixels.
    color : tuple[int, int, int, int]
        RGBA color tuple for the solid fill.

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    img = Image.new("RGBA", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgb_tile_png(
    size: int = 256,
    color: tuple[int, int, int] = (0, 128, 255),
) -> bytes:
    """Create a solid-color RGB (no alpha) PNG tile image as bytes.

    Parameters
    ----------
    size : int
        Width and height of the square tile in pixels.
    color : tuple[int, int, int]
        RGB color tuple.

    Returns
    -------
    bytes
        PNG-encoded image bytes (RGB mode, no alpha channel).
    """
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


VALID_PNG = _make_tile_png(size=8)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock xyzservices.TileProvider.

    Returns:
        MagicMock: Provider mock with build_url that returns a
        predictable URL based on tile coordinates.
    """
    provider = MagicMock()
    provider.build_url = MagicMock(
        side_effect=lambda x, y, z: f"https://tile.example.com/{z}/{x}/{y}.png"
    )
    return provider


class TestAutoZoom:
    """Tests for auto_zoom function."""

    @pytest.mark.parametrize(
        "bounds, expected_zoom",
        [
            ((-180.0, -85.0, 180.0, 85.0), 0),
            ((-90.0, -45.0, 90.0, 45.0), 1),
            ((0.0, 0.0, 1.0, 1.0), 9),
            ((13.0, 52.4, 13.6, 52.6), 10),
            ((0.0, 0.0, 0.01, 0.01), 16),
        ],
        ids=[
            "global_extent",
            "hemisphere_extent",
            "one_degree_extent",
            "city_scale_berlin",
            "small_extent",
        ],
    )
    def test_zoom_for_known_extents(
        self,
        bounds: tuple[float, float, float, float],
        expected_zoom: int,
    ):
        """Test zoom computation for well-known geographic extents.

        Test scenario:
            Each extent maps to a known zoom level via the formula
            ceil(log2(360 / max_extent)). The result is verified to
            match the expected zoom.
        """
        result = auto_zoom(bounds)
        assert (
            result == expected_zoom
        ), f"Expected zoom {expected_zoom} for bounds {bounds}, got {result}"

    def test_zoom_clamped_to_zero_for_oversized_extent(self):
        """Test that zoom never goes below 0.

        Test scenario:
            An extent wider than 360 degrees would compute a negative
            zoom. The function should clamp the result to 0.
        """
        bounds = (-360.0, -90.0, 360.0, 90.0)
        result = auto_zoom(bounds)
        assert result == 0, f"Expected zoom 0 for oversized extent, got {result}"

    def test_zoom_clamped_to_19_for_tiny_extent(self):
        """Test that zoom never exceeds 19.

        Test scenario:
            An extremely small extent (sub-meter) would compute zoom
            >19. The function should clamp to 19.
        """
        bounds = (0.0, 0.0, 1e-8, 1e-8)
        result = auto_zoom(bounds)
        assert result == 19, f"Expected zoom 19 for tiny extent, got {result}"

    def test_zoom_with_zero_extent_does_not_crash(self):
        """Test that a zero-width extent does not cause division by zero.

        Test scenario:
            When east == west and north == south, the max_extent falls
            back to 1e-10 via the epsilon guard, preventing log2(inf).
        """
        bounds = (10.0, 50.0, 10.0, 50.0)
        result = auto_zoom(bounds)
        assert (
            0 <= result <= 19
        ), f"Expected zoom in [0, 19] for zero extent, got {result}"

    def test_return_type_is_int(self):
        """Test that zoom is always an int.

        Test scenario:
            The function should return a plain int, not float.
        """
        result = auto_zoom((0.0, 0.0, 10.0, 10.0))
        assert isinstance(result, int), f"Expected int return type, got {type(result)}"

    def test_lat_dominant_extent(self):
        """Test zoom when latitude extent exceeds longitude extent.

        Test scenario:
            Tall narrow bounding box: the latitude extent (10 deg)
            dominates the zoom calculation.
        """
        bounds = (10.0, 40.0, 10.5, 50.0)
        result = auto_zoom(bounds)
        expected = 6
        assert (
            result == expected
        ), f"Expected zoom {expected} for lat-dominant extent, got {result}"


class TestFetchSingleTile:
    """Tests for fetch_single_tile function."""

    def test_successful_fetch_on_first_attempt(self, mock_provider: MagicMock):
        """Test that a successful HTTP response returns tile data.

        Test scenario:
            Mock urlopen returns 256 bytes on the first call. The
            function should return the tile and its bytes without
            retrying.
        """
        tile = Tile(x=1, y=2, z=3)
        expected_bytes = _make_tile_png(size=256)

        mock_response = MagicMock()
        mock_response.read.return_value = expected_bytes

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen", return_value=mock_response
        ):
            result_tile, result_bytes = fetch_single_tile(
                tile, mock_provider, timeout=5, retries=2
            )

        assert result_tile == tile, f"Expected tile {tile}, got {result_tile}"
        assert (
            result_bytes == expected_bytes
        ), f"Expected {expected_bytes!r}, got {result_bytes!r}"

    def test_user_agent_header_is_set(self, mock_provider: MagicMock):
        """Test that requests include the pyramids User-Agent header.

        Test scenario:
            The Request object passed to urlopen must have the
            User-Agent header set to USER_AGENT constant.
        """
        tile = Tile(x=0, y=0, z=0)
        mock_response = MagicMock()
        mock_response.read.return_value = VALID_PNG

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            return_value=mock_response,
        ) as mock_urlopen:
            fetch_single_tile(tile, mock_provider, timeout=5, retries=0)

        called_request = mock_urlopen.call_args[0][0]
        actual_ua = called_request.get_header("User-agent")
        assert (
            actual_ua == USER_AGENT
        ), f"Expected User-Agent '{USER_AGENT}', got '{actual_ua}'"

    def test_retries_on_failure_then_succeeds(self, mock_provider: MagicMock):
        """Test retry logic: fail twice, succeed on third attempt.

        Test scenario:
            urlopen raises on the first two calls and returns data on
            the third. With retries=2, all three attempts are made and
            the function succeeds.
        """
        tile = Tile(x=5, y=5, z=10)
        expected = VALID_PNG
        mock_response = MagicMock()
        mock_response.read.return_value = expected

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            side_effect=[
                ConnectionError("timeout"),
                ConnectionError("refused"),
                mock_response,
            ],
        ) as mock_urlopen:
            result_tile, result_bytes = fetch_single_tile(
                tile, mock_provider, timeout=5, retries=2
            )

        assert (
            result_bytes == expected
        ), f"Expected tile data after retries, got {result_bytes!r}"
        assert (
            mock_urlopen.call_count == 3
        ), f"Expected 3 attempts (1 + 2 retries), got {mock_urlopen.call_count}"

    def test_raises_connection_error_after_all_retries_exhausted(
        self, mock_provider: MagicMock
    ):
        """Test that ConnectionError is raised when all retries fail.

        Test scenario:
            All attempts (1 initial + 2 retries) fail. The function
            should raise ConnectionError with tile coordinates and
            attempt count in the message.
        """
        tile = Tile(x=1, y=1, z=5)

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            side_effect=ConnectionError("network down"),
        ):
            with pytest.raises(ConnectionError, match=r"z=5/x=1/y=1") as exc_info:
                fetch_single_tile(tile, mock_provider, timeout=5, retries=2)

        assert "3 attempts" in str(
            exc_info.value
        ), f"Error message should mention attempt count: {exc_info.value}"

    def test_zero_retries_means_single_attempt(self, mock_provider: MagicMock):
        """Test that retries=0 makes exactly one attempt.

        Test scenario:
            With retries=0, the function tries once and raises on
            failure without any retry.
        """
        tile = Tile(x=0, y=0, z=0)

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            side_effect=ConnectionError("fail"),
        ) as mock_urlopen:
            with pytest.raises(ConnectionError, match=r"1 attempts"):
                fetch_single_tile(tile, mock_provider, timeout=5, retries=0)

        assert (
            mock_urlopen.call_count == 1
        ), f"Expected exactly 1 attempt, got {mock_urlopen.call_count}"

    def test_provider_build_url_called_with_tile_coords(self, mock_provider: MagicMock):
        """Test that provider.build_url receives the tile coordinates.

        Test scenario:
            The tile x, y, z values should be passed as keyword
            arguments to provider.build_url.
        """
        tile = Tile(x=42, y=99, z=15)
        mock_response = MagicMock()
        mock_response.read.return_value = VALID_PNG

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            return_value=mock_response,
        ):
            fetch_single_tile(tile, mock_provider, timeout=5, retries=0)

        mock_provider.build_url.assert_called_once_with(x=42, y=99, z=15)


class TestFetchTiles:
    """Tests for fetch_tiles function."""

    def test_fetch_multiple_tiles_in_parallel(self, mock_provider: MagicMock):
        """Test parallel fetching of multiple tiles.

        Test scenario:
            Three tiles are submitted to the thread pool. Each returns
            unique data. The result dict should contain all three.
        """
        tiles = [Tile(x=0, y=0, z=1), Tile(x=1, y=0, z=1), Tile(x=0, y=1, z=1)]
        responses = {
            "https://tile.example.com/1/0/0.png": VALID_PNG,
            "https://tile.example.com/1/1/0.png": VALID_PNG,
            "https://tile.example.com/1/0/1.png": VALID_PNG,
        }

        def mock_urlopen(request, timeout=None):
            url = request.full_url
            resp = MagicMock()
            resp.read.return_value = responses[url]
            return resp

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            side_effect=mock_urlopen,
        ):
            result = fetch_tiles(
                tiles, mock_provider, max_workers=2, timeout=5, retries=0
            )

        assert len(result) == 3, f"Expected 3 tiles, got {len(result)}"
        for tile in tiles:
            assert tile in result, f"Tile {tile} missing from result"

    def testfetch_single_tile_list(self, mock_provider: MagicMock):
        """Test fetching a list with a single tile.

        Test scenario:
            A list with one tile should produce a dict with one entry.
        """
        tiles = [Tile(x=0, y=0, z=0)]
        mock_response = MagicMock()
        mock_response.read.return_value = VALID_PNG

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            return_value=mock_response,
        ):
            result = fetch_tiles(
                tiles, mock_provider, max_workers=1, timeout=5, retries=0
            )

        assert len(result) == 1, f"Expected 1 tile, got {len(result)}"

    def test_propagates_connection_error(self, mock_provider: MagicMock):
        """Test that ConnectionError from fetch_single_tile propagates.

        Test scenario:
            When a tile fetch fails after all retries, the
            ConnectionError should bubble up from fetch_tiles.
        """
        tiles = [Tile(x=0, y=0, z=0)]

        with patch(
            "pyramids.basemap.tiles.urllib.request.urlopen",
            side_effect=ConnectionError("fail"),
        ):
            with pytest.raises(ConnectionError):
                fetch_tiles(tiles, mock_provider, max_workers=1, timeout=5, retries=0)


class TestStitchTiles:
    """Tests for stitch_tiles function."""

    def test_single_tile_produces_correct_shape(self):
        """Test stitching a single 256x256 tile.

        Test scenario:
            One tile should produce an image with shape (256, 256, 4)
            since tiles are converted to RGBA.
        """
        tile = Tile(x=0, y=0, z=0)
        png = _make_tile_png(size=256)
        tile_data = {tile: png}

        image, extent = stitch_tiles(tile_data, [tile], zoom=0)

        assert image.shape == (
            256,
            256,
            4,
        ), f"Expected shape (256, 256, 4), got {image.shape}"
        assert image.dtype == np.uint8, f"Expected dtype uint8, got {image.dtype}"

    def test_2x2_grid_produces_correct_shape(self):
        """Test stitching a 2x2 grid of 256px tiles.

        Test scenario:
            Four tiles in a 2x2 grid should produce an image with
            shape (512, 512, 4).
        """
        tiles = [
            Tile(x=0, y=0, z=1),
            Tile(x=1, y=0, z=1),
            Tile(x=0, y=1, z=1),
            Tile(x=1, y=1, z=1),
        ]
        tile_data = {t: _make_tile_png(size=256) for t in tiles}

        image, extent = stitch_tiles(tile_data, tiles, zoom=1)

        assert image.shape == (
            512,
            512,
            4,
        ), f"Expected shape (512, 512, 4), got {image.shape}"

    def test_512px_tiles_produce_correct_shape(self):
        """Test stitching tiles with non-standard 512px size.

        Test scenario:
            Two tiles of 512px each in a 2x1 grid should produce
            an image with shape (512, 1024, 4).
        """
        tiles = [Tile(x=0, y=0, z=1), Tile(x=1, y=0, z=1)]
        tile_data = {t: _make_tile_png(size=512) for t in tiles}

        image, extent = stitch_tiles(tile_data, tiles, zoom=1)

        assert image.shape == (
            512,
            1024,
            4,
        ), f"Expected shape (512, 1024, 4), got {image.shape}"

    def test_rgb_tiles_converted_to_rgba(self):
        """Test that RGB (3-channel) tiles are converted to RGBA.

        Test scenario:
            An RGB-only PNG (no alpha) is converted to RGBA during
            stitching, so the output always has 4 channels.
        """
        tile = Tile(x=0, y=0, z=0)
        png = _make_rgb_tile_png(size=256)
        tile_data = {tile: png}

        image, extent = stitch_tiles(tile_data, [tile], zoom=0)

        assert image.shape[2] == 4, f"Expected 4 channels (RGBA), got {image.shape[2]}"

    def test_extent_is_in_epsg_3857(self):
        """Test that the returned extent is in EPSG:3857 meters.

        Test scenario:
            For zoom=0, tile (0,0), the extent should span the full
            Web Mercator range (~-20M to +20M meters).
        """
        tile = Tile(x=0, y=0, z=0)
        png = _make_tile_png(size=256)
        tile_data = {tile: png}

        image, extent = stitch_tiles(tile_data, [tile], zoom=0)
        west, south, east, north = extent

        assert west < east, f"West ({west}) should be < East ({east})"
        assert south < north, f"South ({south}) should be < North ({north})"
        assert west < 0, f"West should be negative for z=0, got {west}"
        assert east > 0, f"East should be positive for z=0, got {east}"

    def test_2x2_extent_covers_more_than_single_tile(self):
        """Test that 2x2 grid extent is larger than single tile extent.

        Test scenario:
            A 2x2 grid of tiles at zoom=1 should cover a larger
            geographic area than a single tile at the same zoom.
        """
        single_tile = Tile(x=0, y=0, z=1)
        single_data = {single_tile: _make_tile_png(size=256)}
        _, single_extent = stitch_tiles(single_data, [single_tile], zoom=1)

        grid_tiles = [
            Tile(x=0, y=0, z=1),
            Tile(x=1, y=0, z=1),
            Tile(x=0, y=1, z=1),
            Tile(x=1, y=1, z=1),
        ]
        grid_data = {t: _make_tile_png(size=256) for t in grid_tiles}
        _, grid_extent = stitch_tiles(grid_data, grid_tiles, zoom=1)

        single_width = single_extent[2] - single_extent[0]
        grid_width = grid_extent[2] - grid_extent[0]
        assert grid_width > single_width, (
            f"2x2 grid width ({grid_width}) should exceed "
            f"single tile width ({single_width})"
        )

    def test_tile_placement_preserves_color(self):
        """Test that each tile's pixels are placed in the correct position.

        Test scenario:
            Two tiles with distinct colors are stitched side by side.
            The left half of the output should match the first tile's
            color, and the right half should match the second.
        """
        red = (255, 0, 0, 255)
        blue = (0, 0, 255, 255)
        tile_left = Tile(x=0, y=0, z=1)
        tile_right = Tile(x=1, y=0, z=1)
        tile_data = {
            tile_left: _make_tile_png(size=256, color=red),
            tile_right: _make_tile_png(size=256, color=blue),
        }

        image, _ = stitch_tiles(tile_data, [tile_left, tile_right], zoom=1)

        left_pixel = tuple(image[128, 128])
        right_pixel = tuple(image[128, 384])
        assert (
            left_pixel == red
        ), f"Left tile pixel should be red {red}, got {left_pixel}"
        assert (
            right_pixel == blue
        ), f"Right tile pixel should be blue {blue}, got {right_pixel}"


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_user_agent_is_string(self):
        """Test that USER_AGENT is a non-empty string.

        Test scenario:
            The constant must be a descriptive string suitable for
            HTTP headers.
        """
        assert isinstance(
            USER_AGENT, str
        ), f"USER_AGENT should be str, got {type(USER_AGENT)}"
        assert len(USER_AGENT) > 0, "USER_AGENT should not be empty"

    def test_max_tiles_is_positive_int(self):
        """Test that MAX_TILES is a positive integer.

        Test scenario:
            MAX_TILES guards against excessive tile downloads.
        """
        assert isinstance(
            MAX_TILES, int
        ), f"MAX_TILES should be int, got {type(MAX_TILES)}"
        assert MAX_TILES > 0, f"MAX_TILES should be positive, got {MAX_TILES}"
