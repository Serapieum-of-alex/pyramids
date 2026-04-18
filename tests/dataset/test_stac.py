"""Tests for :meth:`DatasetCollection.from_stac`.

DASK-19: thin STAC loader — takes a sequence of STAC items (a
pystac.ItemCollection or list), extracts a named asset's href from
each, and builds a ``DatasetCollection``. pystac is optional; tests
skip when absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection


try:
    import pystac

    HAS_PYSTAC = True
except ImportError:  # pragma: no cover
    HAS_PYSTAC = False


requires_pystac = pytest.mark.skipif(
    not HAS_PYSTAC, reason="pystac not installed"
)


@pytest.fixture
def three_tifs(tmp_path):
    """Three small GeoTIFFs, each with a different fill value."""
    paths = []
    for i in range(3):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr, top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
        )
        p = str(tmp_path / f"tile_{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.fixture
def stac_items(three_tifs):
    """Wrap three local GeoTIFFs as pystac.Item objects."""
    if not HAS_PYSTAC:  # pragma: no cover
        pytest.skip("pystac required")
    items = []
    for i, path in enumerate(three_tifs):
        item = pystac.Item(
            id=f"item-{i}",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            bbox=[0.0, 0.0, 1.0, 1.0],
            datetime=None,
            properties={
                "start_datetime": "2024-01-01T00:00:00Z",
                "end_datetime": "2024-01-02T00:00:00Z",
            },
        )
        item.add_asset(
            "data", pystac.Asset(href=path, media_type=pystac.MediaType.GEOTIFF),
        )
        items.append(item)
    return items


class TestFromStac:
    """Happy-path: ItemCollection → DatasetCollection."""

    @requires_pystac
    def test_returns_dataset_collection(self, stac_items):
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        assert isinstance(collection, DatasetCollection)
        assert collection.time_length == 3

    @requires_pystac
    def test_files_match_asset_hrefs(self, stac_items, three_tifs):
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        assert collection.files == three_tifs

    @requires_pystac
    def test_lazy_data_computes(self, stac_items):
        try:
            import dask.array  # noqa: F401
        except ImportError:
            pytest.skip("dask not installed")
        collection = DatasetCollection.from_stac(stac_items, asset="data")
        arr = collection.data.compute()
        assert arr.shape[0] == 3
        for i in range(3):
            assert (arr[i] == i + 1).all()


class TestPatchUrl:
    """patch_url rewrites every href before it becomes a file path."""

    @requires_pystac
    def test_patch_url_called_per_href(self, stac_items):
        seen: list[str] = []

        def patch(href: str) -> str:
            seen.append(href)
            return href

        DatasetCollection.from_stac(stac_items, asset="data", patch_url=patch)
        assert len(seen) == 3


class TestAssetMissing:
    """Missing asset keys raise KeyError with available assets listed."""

    @requires_pystac
    def test_unknown_asset_raises(self, stac_items):
        with pytest.raises(KeyError, match="not found"):
            DatasetCollection.from_stac(stac_items, asset="doesnotexist")


class TestImportError:
    def test_raises_without_pystac(self, three_tifs, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pystac":
                raise ImportError("no pystac")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[stac\\]"):
            DatasetCollection.from_stac([], asset="data")
