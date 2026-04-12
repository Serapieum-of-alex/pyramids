"""Tests for DatasetCollection.to_cog_stack (Task 9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection


@pytest.fixture(scope="module")
def small_collection(rasters_folder_path: str) -> DatasetCollection:
    """A 6-slice DatasetCollection read from the existing test fixture dir."""
    dc = DatasetCollection.read_multiple_files(
        rasters_folder_path, with_order=False
    )
    dc.open_multi_dataset(band=0)
    return dc


class TestToCogStackBasics:
    def test_writes_one_file_per_slice(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out)
        assert len(paths) == small_collection.time_length

    def test_all_outputs_exist(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out)
        for p in paths:
            assert p.exists()

    def test_returns_path_objects(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out)
        for p in paths:
            assert isinstance(p, Path)

    def test_default_filename_pattern(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out)
        assert paths[0].name == "slice_0000.tif"
        assert paths[1].name == "slice_0001.tif"

    def test_creates_missing_directory(self, small_collection, tmp_path):
        out = tmp_path / "deeply" / "nested" / "out"
        paths = small_collection.to_cog_stack(out)
        assert out.exists()
        assert len(paths) > 0

    def test_outputs_are_valid_cogs(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out)
        for p in paths:
            reopened = Dataset.read_file(p)
            assert reopened.is_cog is True
            reopened.close()


class TestToCogStackPattern:
    def test_custom_pattern_and_name(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(
            out, pattern="B04_{i:03d}.tif", name="B04"
        )
        assert paths[0].name == "B04_000.tif"
        assert paths[1].name == "B04_001.tif"

    def test_time_placeholder_raises(self, small_collection, tmp_path):
        with pytest.raises(ValueError, match=r"\{t\}"):
            small_collection.to_cog_stack(tmp_path, pattern="x_{t}.tif")


class TestToCogStackOverwrite:
    def test_overwrite_false_raises_on_existing(self, small_collection, tmp_path):
        # First write creates the files
        small_collection.to_cog_stack(tmp_path)
        # Second should fail without overwrite
        with pytest.raises(FileExistsError):
            small_collection.to_cog_stack(tmp_path)

    def test_overwrite_true_replaces(self, small_collection, tmp_path):
        small_collection.to_cog_stack(tmp_path)
        # Should not raise
        paths = small_collection.to_cog_stack(tmp_path, overwrite=True)
        assert len(paths) == small_collection.time_length


class TestToCogStackKwargs:
    def test_kwargs_forwarded_compress(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out, compress="LZW")
        info = gdal.Info(str(paths[0]))
        assert "COMPRESSION=LZW" in info

    def test_kwargs_forwarded_blocksize(self, small_collection, tmp_path):
        out = tmp_path / "cog_stack"
        paths = small_collection.to_cog_stack(out, blocksize=128)
        reopened = gdal.Open(str(paths[0]))
        bx, by = reopened.GetRasterBand(1).GetBlockSize()
        assert bx == 128
        reopened = None
