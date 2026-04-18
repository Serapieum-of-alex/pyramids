"""Private OGR bridge for :mod:`pyramids.feature`.

This module is **not part of the public API**. It exists solely so that
``FeatureCollection`` methods that internally need a GDAL/OGR object
(for example :func:`osgeo.gdal.Rasterize` or :func:`osgeo.gdal.Warp` with
``cutlineDSName``) can obtain one without leaking ``ogr.DataSource`` or
``gdal.Dataset`` into the package's public surface.

All helpers in this module are context-managed where possible so that the
backing ``/vsimem/`` file and the OGR handle are deterministically
released when the ``with`` block exits.

Do not import this module from user code; its signatures are unstable.
"""

from __future__ import annotations

import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import geopandas as gpd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr

from pyramids.base._errors import VectorDriverError


def _new_vsimem_path() -> str:
    """Return a fresh unique ``/vsimem/`` path for a GeoJSON serialization.

    The path is unique per call (UUID4) so that concurrent internal
    conversions cannot clobber each other's backing files.

    Returns:
        str: A ``/vsimem/<uuid>.geojson`` path.

    Examples:
        - Check the shape of a freshly generated path:
            ```python
            >>> from pyramids.feature._ogr import _new_vsimem_path
            >>> path = _new_vsimem_path()
            >>> path.startswith("/vsimem/")
            True
            >>> path.endswith(".geojson")
            True

            ```
        - Two successive calls return distinct paths (UUID4 collision is
          astronomically unlikely) so concurrent conversions cannot
          clobber each other:
            ```python
            >>> from pyramids.feature._ogr import _new_vsimem_path
            >>> p1 = _new_vsimem_path()
            >>> p2 = _new_vsimem_path()
            >>> p1 != p2
            True

            ```
    """
    return f"/vsimem/{uuid.uuid4()}.geojson"


@contextmanager
def as_datasource(
    gdf: GeoDataFrame, *, gdal_dataset: bool = False
) -> Iterator[ogr.DataSource | gdal.Dataset]:
    """Yield a short-lived OGR ``DataSource`` for a GeoDataFrame.

    The DataSource (or ``gdal.Dataset`` with OGR contents when
    ``gdal_dataset=True``) is backed by an in-memory ``/vsimem/`` GeoJSON
    file that is unlinked when the context exits. The caller must use the
    yielded object only inside the ``with`` block — storing it past the
    block is a use-after-free.

    Args:
        gdf (GeoDataFrame):
            The GeoDataFrame to expose as an OGR DataSource. Any subclass
            of :class:`geopandas.GeoDataFrame` is accepted — in particular
            ``pyramids.feature.FeatureCollection`` works unchanged since
            it IS a ``GeoDataFrame``.
        gdal_dataset (bool):
            When ``True``, yield a ``gdal.Dataset`` (opened via
            :func:`gdal.OpenEx`) instead of an ``ogr.DataSource``. This
            is the form required by :func:`gdal.Rasterize` when the
            vector argument is a Python GDAL object rather than a path.

    Yields:
        ogr.DataSource | gdal.Dataset: A short-lived handle to the vector
        data. Do not store past the ``with`` block.

    Examples:
        - Open a GeoDataFrame as an OGR DataSource inside a ``with``
          block, and confirm the yielded handle exposes one layer with
          the feature count the GDF had:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogr import as_datasource
            >>> gdf = gpd.GeoDataFrame(
            ...     {"v": [1, 2]},
            ...     geometry=[Point(0, 0), Point(1, 1)],
            ...     crs="EPSG:4326",
            ... )
            >>> with as_datasource(gdf) as ds:
            ...     layer = ds.GetLayer(0)
            ...     n_features = layer.GetFeatureCount()
            >>> n_features
            2

            ```
        - Request a ``gdal.Dataset`` instead of an ``ogr.DataSource``
          (the form :func:`gdal.Rasterize` expects when its vector
          argument is a Python GDAL object rather than a file path):
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from osgeo import gdal
            >>> from pyramids.feature._ogr import as_datasource
            >>> gdf = gpd.GeoDataFrame(
            ...     {"v": [10]}, geometry=[Point(0, 0)], crs="EPSG:4326"
            ... )
            >>> with as_datasource(gdf, gdal_dataset=True) as ds:
            ...     kind = isinstance(ds, gdal.Dataset)
            >>> kind
            True

            ```
    """
    mem_path = _new_vsimem_path()
    # We must write into osgeo.gdal's own /vsimem/ — geopandas' default
    # pyogrio engine uses its own bundled GDAL with a separate VFS, so a
    # ``gdf.to_file("/vsimem/…")`` would write to *that* engine's memory
    # store and ``osgeo.gdal`` would never see it. Round-tripping through
    # the GeoJSON serialization + ``gdal.FileFromMemBuffer`` guarantees
    # the file lands in the GDAL VFS we can open.
    geojson_bytes = gdf.to_json().encode("utf-8")
    gdal.FileFromMemBuffer(mem_path, geojson_bytes)
    ds: ogr.DataSource | gdal.Dataset | None = None
    try:
        ds = gdal.OpenEx(mem_path) if gdal_dataset else ogr.Open(mem_path)
        yield ds
    finally:
        ds = None
        gdal.Unlink(mem_path)


@contextmanager
def as_vsimem_path(gdf: GeoDataFrame) -> Iterator[str]:
    """Yield a ``/vsimem/`` path to a GeoJSON serialization of ``gdf``.

    Useful where a GDAL API needs a *path string* (e.g. the
    ``cutlineDSName`` option of :func:`gdal.Warp`) rather than a Python
    GDAL object. The path is unlinked on exit.

    Args:
        gdf (GeoDataFrame):
            The GeoDataFrame to serialize.

    Yields:
        str: A ``/vsimem/<uuid>.geojson`` path valid only inside the
        ``with`` block.

    Examples:
        - Confirm the yielded path has the expected shape and that the
          backing GeoJSON file exists for the duration of the ``with``
          block (opened via :func:`osgeo.ogr.Open`):
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from osgeo import ogr
            >>> from pyramids.feature._ogr import as_vsimem_path
            >>> gdf = gpd.GeoDataFrame(
            ...     {"id": [7]}, geometry=[Point(0, 0)], crs="EPSG:4326"
            ... )
            >>> with as_vsimem_path(gdf) as path:
            ...     prefix_ok = path.startswith("/vsimem/")
            ...     ds = ogr.Open(path)
            ...     n = ds.GetLayer(0).GetFeatureCount()
            ...     ds = None
            >>> prefix_ok, n
            (True, 1)

            ```
    """
    mem_path = _new_vsimem_path()
    # See the note in ``as_datasource`` for why we use
    # ``gdal.FileFromMemBuffer`` instead of ``gdf.to_file``.
    geojson_bytes = gdf.to_json().encode("utf-8")
    gdal.FileFromMemBuffer(mem_path, geojson_bytes)
    try:
        yield mem_path
    finally:
        gdal.Unlink(mem_path)


def datasource_to_gdf(ds: ogr.DataSource | gdal.Dataset) -> GeoDataFrame:
    """Materialize an OGR ``DataSource`` into a ``GeoDataFrame``.

    Used by internal operations (for example :func:`gdal.Polygonize`) that
    begin by allocating an OGR DataSource and need to hand a
    ``GeoDataFrame`` back to the public layer. The conversion goes via a
    ``/vsimem/`` GeoJSON round-trip using :func:`gdal.VectorTranslate`.

    Args:
        ds (ogr.DataSource | gdal.Dataset):
            The source DataSource to materialize. Not consumed; the
            caller retains ownership.

    Returns:
        GeoDataFrame: A plain ``GeoDataFrame`` (never a
        ``FeatureCollection``) containing the layer's features. Callers
        that want a ``FeatureCollection`` should wrap the result:
        ``FeatureCollection(datasource_to_gdf(ds))``.

    Raises:
        RuntimeError: If :func:`gdal.VectorTranslate` fails to write the
            intermediate GeoJSON.

    Examples:
        - Round-trip a GeoDataFrame through the OGR bridge: first open
          it as an in-memory OGR ``DataSource`` via :func:`as_datasource`,
          then materialize it back to a ``GeoDataFrame`` via
          :func:`datasource_to_gdf`. Attribute and row counts survive:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature._ogr import (
            ...     as_datasource,
            ...     datasource_to_gdf,
            ... )
            >>> gdf = gpd.GeoDataFrame(
            ...     {"score": [10, 20, 30]},
            ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            ...     crs="EPSG:4326",
            ... )
            >>> with as_datasource(gdf) as ds:
            ...     back = datasource_to_gdf(ds)
            >>> len(back)
            3
            >>> sorted(back["score"].tolist())
            [10, 20, 30]

            ```
    """
    # geopandas' default pyogrio engine reads from its own bundled GDAL
    # VFS, not osgeo.gdal's /vsimem/. Round-trip through a real temp file
    # so pyogrio can open what we wrote.
    tmp = Path(tempfile.gettempdir()) / f"pyramids_ogr_{uuid.uuid4()}.geojson"
    try:
        result = gdal.VectorTranslate(str(tmp), ds, format="GeoJSON")
        if result is None:
            raise VectorDriverError(
                "gdal.VectorTranslate failed to materialize the DataSource "
                "to GeoJSON."
            )
        result = None  # flush + close the translation output
        gdf = gpd.read_file(tmp)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return gdf
