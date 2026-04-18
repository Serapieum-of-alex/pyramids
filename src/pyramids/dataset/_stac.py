"""STAC ItemCollection → :class:`DatasetCollection`.

DASK-19 first cut: given a sequence of :class:`pystac.Item`
objects (or anything that iterates as such), extract the chosen
asset's ``href`` from each item and delegate to
:meth:`DatasetCollection.from_files`. Full odc-stac-style features
(geobox-tiled graph, bbox filtering, auto-geobox derivation,
``fuse_func``, ``errors_as_nodata``) are deliberately out of
scope — those users are better served by the odc-stac or
stackstac packages directly.

pystac is an optional dependency behind the ``[stac]`` extra. This
module imports :mod:`pystac` lazily inside the entry function so
importing pyramids does not pull in pystac.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from pyramids.dataset.collection import DatasetCollection


_STAC_IMPORT_ERROR = (
    "from_stac requires the optional 'pystac' dependency. "
    "Install it with: pip install 'pyramids-gis[stac]'"
)


def _require_pystac() -> Any:
    """Lazy-import :mod:`pystac` with a pyramids-friendly error."""
    try:
        import pystac
    except ImportError as exc:
        raise ImportError(_STAC_IMPORT_ERROR) from exc
    return pystac


def _iter_items(items: Any) -> list[Any]:
    """Normalise ``items`` to a list of STAC Items.

    Accepts a :class:`pystac.ItemCollection`, a list, or any iterable
    yielding STAC items.
    """
    if hasattr(items, "__iter__"):
        return list(items)
    raise TypeError(
        f"items must be iterable (ItemCollection or list), got {type(items).__name__}"
    )


def _resolve_asset_href(item: Any, asset_key: str) -> str:
    """Return the href of a named asset on a STAC item.

    Args:
        item: A :class:`pystac.Item`.
        asset_key: Asset name (``"B04"``, ``"visual"``, ...).

    Returns:
        str: The asset's href.

    Raises:
        KeyError: When ``asset_key`` is not present on the item.
    """
    assets = getattr(item, "assets", None)
    if assets is None or asset_key not in assets:
        available = list(assets or [])
        raise KeyError(
            f"asset {asset_key!r} not found on STAC item "
            f"{getattr(item, 'id', '?')}; available: {available}"
        )
    asset = assets[asset_key]
    return str(asset.href)


def _item_intersects_bbox(
    item: Any, bbox: tuple[float, float, float, float],
) -> bool:
    """Return True if ``item.bbox`` overlaps ``bbox`` (lon/lat box)."""
    item_bbox = getattr(item, "bbox", None)
    if item_bbox is None:
        result = True
    else:
        minx, miny, maxx, maxy = bbox
        i_minx, i_miny, i_maxx, i_maxy = item_bbox
        result = not (
            i_maxx < minx or i_minx > maxx or i_maxy < miny or i_miny > maxy
        )
    return result


def from_stac(
    items: Any,
    asset: str,
    *,
    patch_url: Callable[[str], str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_items: int | None = None,
) -> "DatasetCollection":
    """Build a :class:`DatasetCollection` from a STAC ItemCollection.

    Extracts one named asset's href from each item, optionally runs
    ``patch_url`` on each href (typical use: sign a Planetary Computer
    URL), and forwards to :meth:`DatasetCollection.from_files`.

    Args:
        items: Iterable of :class:`pystac.Item`, or a
            :class:`pystac.ItemCollection`.
        asset: Asset key (e.g. ``"B04"``, ``"visual"``) whose
            ``href`` on each item becomes a timestep in the
            resulting collection.
        patch_url: Optional callable applied to each href — use for
            signing requester-pays URLs
            (``planetary_computer.sign``, etc.).

    Returns:
        DatasetCollection: A file-backed collection whose
        ``time_length`` equals ``len(items)`` and whose per-timestep
        backing file is the resolved asset URL.

    Raises:
        ImportError: When pystac is not installed.
        KeyError: When any item is missing the requested asset.
        ValueError: When ``items`` is empty.
    """
    _require_pystac()
    item_list = _iter_items(items)
    if bbox is not None:
        item_list = [i for i in item_list if _item_intersects_bbox(i, bbox)]
    if max_items is not None:
        item_list = item_list[:max_items]
    hrefs = []
    for item in item_list:
        href = _resolve_asset_href(item, asset)
        if patch_url is not None:
            href = patch_url(href)
        hrefs.append(href)

    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(hrefs)


__all__ = ["from_stac"]
