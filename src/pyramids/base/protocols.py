"""Structural-typing protocols shared across the pyramids package.

ARC-17 defines :class:`SpatialObject`, a runtime-checkable protocol
covering the surface that both :class:`pyramids.dataset.Dataset` and
:class:`pyramids.feature.FeatureCollection` implement. User code can
type-annotate ``def describe(obj: SpatialObject) -> str: ...`` and
accept either a raster or a vector without importing both concrete
classes (and without creating an import cycle between them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpatialObject(Protocol):
    """Minimum surface shared by pyramids raster and vector objects.

    Both :class:`pyramids.dataset.Dataset` (raster) and
    :class:`pyramids.feature.FeatureCollection` (vector) implement
    this protocol, so callers can write generic geospatial utilities
    that accept either.

    Attributes / properties:
        epsg (int | None):
            EPSG code of the CRS; ``None`` when the CRS is unset.
        total_bounds:
            Array-like ``[minx, miny, maxx, maxy]`` in the object's
            CRS. FeatureCollection inherits this from
            :class:`geopandas.GeoDataFrame`; Dataset exposes the
            same shape via the same attribute.
        top_left_corner:
            Sequence ``[minx, maxy]`` — the NW corner of the
            bounding box.

    Methods:
        read_file(path) (classmethod):
            Construct an instance from a file path.
        to_file(path, ...):
            Serialize the object to ``path``.
        plot(...):
            Render a matplotlib view of the object.

    Because this is :func:`typing.runtime_checkable`, you can use it
    with :func:`isinstance`:

    >>> from pyramids.base.protocols import SpatialObject
    >>> def describe(obj: SpatialObject) -> int | None:
    ...     return obj.epsg

    Runtime isinstance checks verify method/attribute presence only
    (PEP 544 — they do not verify signatures or return types).
    """

    epsg: int | None
    total_bounds: Any
    top_left_corner: Any

    @classmethod
    def read_file(cls, path: str | Path, *args: Any, **kwargs: Any) -> "SpatialObject":
        """Read an on-disk representation into an instance."""
        ...

    def to_file(self, path: str | Path, *args: Any, **kwargs: Any) -> None:
        """Serialize this object to ``path``."""
        ...

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Render a matplotlib view of this object."""
        ...
