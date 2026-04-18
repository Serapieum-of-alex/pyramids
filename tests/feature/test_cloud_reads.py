"""ARC-22: cloud / virtual-filesystem reads in FeatureCollection.read_file.

After ARC-23 wired ``pyramids._io._parse_path`` into
``FeatureCollection.read_file``, URL-scheme paths (``s3://``, ``gs://``,
``az://``, ``http(s)://``, ``file://``) are rewritten to GDAL
``/vsi*`` form before the file is opened. These tests cover that
code path end-to-end without requiring external cloud credentials:

* A local ``http.server`` serves a directory containing a GeoJSON
  (and a zipped GeoJSON). Reading the HTTP URL exercises the
  ``/vsicurl/`` path in GDAL's VFS.
* ``file://`` paths are exercised against a regular tmp_path file.
* Actual cloud-service tests (``s3://`` etc.) are marked ``vfs`` and
  skipped by default — they run only when real credentials are in
  the environment. They document the intended surface.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that stays quiet under tests."""

    def log_message(self, *args, **kwargs):  # noqa: D401, N802
        """Suppress the default stderr access log."""
        return


@pytest.fixture(scope="module")
def http_vector_dir(tmp_path_factory) -> Path:
    """Directory containing a GeoJSON + a zipped GeoJSON to serve over HTTP."""
    root = tmp_path_factory.mktemp("http_vector_dir")

    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "name": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    plain = root / "points.geojson"
    gdf.to_file(plain, driver="GeoJSON")

    zipped = root / "points.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.write(plain, arcname="points.geojson")

    return root


@pytest.fixture(scope="module")
def http_server(http_vector_dir: Path):
    """Local HTTP server serving ``http_vector_dir``; yield the base URL."""
    handler = functools.partial(_QuietHandler, directory=str(http_vector_dir))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.vfs
class TestHttpRead:
    """Read vectors via http:// URLs (routed through /vsicurl/).

    Marked ``vfs`` because GDAL's ``/vsicurl/`` issues blocking HTTP
    requests (HEAD + Range GETs) that can hang indefinitely on some
    firewall-restricted or sandboxed environments. Skipped by default;
    run explicitly with ``pytest -m vfs``.
    """

    def test_read_geojson_over_http(self, http_server: str):
        """FeatureCollection.read_file accepts http:// URLs directly."""
        url = f"{http_server}/points.geojson"
        fc = FeatureCollection.read_file(url)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 3
        assert fc.epsg == 4326
        assert sorted(fc["name"].tolist()) == ["a", "b", "c"]

    def test_http_url_is_rewritten_not_passed_through(
        self, http_server: str, caplog
    ):
        """The rewrite from http:// to /vsicurl/ fires on the code path.

        Checks the ``pyramids.base.remote`` log message rather than
        asserting feature count again — the rewrite is the ARC-22
        behavior under test.
        """
        import logging

        url = f"{http_server}/points.geojson"
        with caplog.at_level(logging.INFO, logger="pyramids.base.remote"):
            FeatureCollection.read_file(url)
        assert any(
            "rewritten" in rec.getMessage() and "/vsicurl/" in rec.getMessage()
            for rec in caplog.records
        ), f"expected a /vsicurl/ rewrite log; got: {[r.getMessage() for r in caplog.records]}"


class TestFileUrlRead:
    """``file://`` URLs are rewritten to plain local paths."""

    def test_read_file_url(self, tmp_path: Path):
        gdf = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
        p = tmp_path / "one.geojson"
        gdf.to_file(p, driver="GeoJSON")

        # ``file://`` URLs require an absolute path. as_uri() handles
        # Windows/Posix differences correctly.
        fc = FeatureCollection.read_file(p.resolve().as_uri())
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 1


@pytest.mark.vfs
class TestCloudReads:
    """Cloud-service reads (s3:// etc.) — gated behind the ``vfs`` marker.

    Skipped by default. Run with ``pytest -m vfs`` when you have real
    AWS / GS / Azure credentials available in the environment. Each
    test opens a well-known public or user-controlled object; replace
    the URLs as needed for your setup.
    """

    def test_s3_read_requires_credentials(self):
        """Placeholder: s3:// URL round-trip.

        Fill in with a bucket/object you control. The default is
        skipped unless the AWS_* env vars are set AND the ``vfs``
        marker is selected via ``-m vfs``. This test documents the
        intended public surface; it is not expected to run in CI.
        """
        import os

        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            pytest.skip("AWS credentials not in env; skipping s3:// test")
        # Example:
        # fc = FeatureCollection.read_file("s3://my-bucket/rivers.geojson")
        # assert len(fc) > 0
        pytest.skip("no concrete s3:// fixture; this is a documentation stub")
