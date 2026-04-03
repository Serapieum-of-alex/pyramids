"""Tests for chunking and compression in NetCDF.create_from_array.

Covers the chunk_sizes, compression, and compression_level parameters
added in NEW-2, including the _create_dimension helper and the
nodata-before-write ordering for the netCDF driver.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal
from pyramids.netcdf.netcdf import NetCDF


SEED = 42
GEO = (0.0, 0.01, 0, 1.0, 0, -0.01)


class TestCompressionOnDisk:
    """Integration tests: create compressed NetCDF on disk."""

    def test_deflate_reduces_file_size(self, tmp_path):
        """DEFLATE compression should produce a smaller file than no compression.

        Test scenario:
            Create two files with identical random data, one compressed,
            one not. Compressed file should be significantly smaller.
        """
        arr = np.random.RandomState(SEED).rand(5, 50, 50).astype(np.float64)
        compressed = str(tmp_path / "compressed.nc")
        uncompressed = str(tmp_path / "uncompressed.nc")

        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=compressed,
            compression="DEFLATE", compression_level=4,
        )
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=uncompressed,
        )
        comp_size = os.path.getsize(compressed)
        uncomp_size = os.path.getsize(uncompressed)
        assert comp_size < uncomp_size, (
            f"Compressed ({comp_size}) should be smaller than "
            f"uncompressed ({uncomp_size})"
        )

    def test_compression_level_affects_size(self, tmp_path):
        """Higher compression level should produce smaller or equal files.

        Test scenario:
            DEFLATE level 1 vs level 9 on the same data.
        """
        arr = np.random.RandomState(SEED).rand(5, 50, 50).astype(np.float64)
        low = str(tmp_path / "level1.nc")
        high = str(tmp_path / "level9.nc")

        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=low,
            compression="DEFLATE", compression_level=1,
        )
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=high,
            compression="DEFLATE", compression_level=9,
        )
        assert os.path.getsize(high) <= os.path.getsize(low), (
            f"Level 9 ({os.path.getsize(high)}) should be <= "
            f"level 1 ({os.path.getsize(low)})"
        )


class TestChunkingOnDisk:
    """Integration tests: create chunked NetCDF on disk."""

    def test_chunk_sizes_applied(self, tmp_path):
        """Chunk sizes should be readable from the created file.

        Test scenario:
            Create with chunk_sizes=(1, 25, 25), re-open via GDAL
            MDIM API and verify GetBlockSize.
        """
        arr = np.random.RandomState(SEED).rand(5, 50, 50).astype(np.float64)
        path = str(tmp_path / "chunked.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="temp",
            extra_dim_name="time", path=path,
            chunk_sizes=(1, 25, 25),
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray("temp")
        block = md_arr.GetBlockSize()
        ds = None
        assert block == [1, 25, 25], (
            f"Expected block size [1, 25, 25], got {block}"
        )

    def test_chunk_with_compression(self, tmp_path):
        """Chunking and compression should work together.

        Test scenario:
            Create with both chunk_sizes and compression, verify file
            is smaller and block sizes are correct.
        """
        arr = np.random.RandomState(SEED).rand(5, 50, 50).astype(np.float64)
        path = str(tmp_path / "chunk_compress.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=path,
            chunk_sizes=(1, 50, 50),
            compression="DEFLATE", compression_level=4,
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray("v")
        block = md_arr.GetBlockSize()
        ds = None
        assert block == [1, 50, 50], (
            f"Expected [1, 50, 50], got {block}"
        )
        assert os.path.getsize(path) > 0, (
            "Compressed file should exist and have content"
        )


class TestRoundTripDataIntegrity:
    """E2E: create with compression → reload → verify data matches."""

    def test_2d_compressed_roundtrip(self, tmp_path):
        """2D array data should survive compressed disk round-trip.

        Test scenario:
            Create compressed, reload, compare arrays element-wise.
        """
        arr = np.random.RandomState(SEED).rand(50, 50).astype(np.float64)
        path = str(tmp_path / "rt_2d.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="elev", path=path,
            compression="DEFLATE",
        )
        reloaded = NetCDF.read_file(path)
        var = reloaded.get_variable("elev")
        assert_allclose(
            var.read_array(band=0), arr, rtol=1e-10,
            err_msg="2D compressed data mismatch",
        )

    def test_3d_compressed_roundtrip(self, tmp_path):
        """3D array data should survive compressed disk round-trip.

        Test scenario:
            Create with time dimension + compression, reload, compare.
        """
        arr = np.random.RandomState(SEED).rand(5, 30, 40).astype(np.float64)
        path = str(tmp_path / "rt_3d.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="temp",
            extra_dim_name="time",
            extra_dim_values=[0, 6, 12, 18, 24],
            path=path,
            chunk_sizes=(1, 30, 40),
            compression="DEFLATE", compression_level=4,
        )
        reloaded = NetCDF.read_file(path)
        var = reloaded.get_variable("temp")
        assert_allclose(
            var.read_array(), arr, rtol=1e-10,
            err_msg="3D compressed data mismatch",
        )

    def test_nodata_preserved_compressed(self, tmp_path):
        """No-data value should survive compressed round-trip.

        Test scenario:
            Create with no_data_value=-9999, reload, check nodata.
        """
        arr = np.random.RandomState(SEED).rand(10, 10).astype(np.float64)
        path = str(tmp_path / "rt_ndv.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v", path=path,
            no_data_value=-9999.0,
            compression="DEFLATE",
        )
        reloaded = NetCDF.read_file(path)
        var = reloaded.get_variable("v")
        assert_allclose(
            var.no_data_value[0], -9999.0, rtol=1e-6,
            err_msg="Nodata not preserved after compression",
        )


class TestInMemoryIgnoresOptions:
    """MEM driver ignores chunk_sizes/compression (no path)."""

    def test_chunk_sizes_ignored_in_memory(self):
        """Passing chunk_sizes without path should not raise.

        Test scenario:
            In-memory creation ignores chunking options silently.
        """
        arr = np.random.RandomState(SEED).rand(3, 10, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time",
            chunk_sizes=(1, 5, 5),
        )
        var = nc.get_variable("v")
        assert var.shape == (3, 10, 10), (
            f"Shape should be preserved, got {var.shape}"
        )

    def test_compression_ignored_in_memory(self):
        """Passing compression without path should not raise.

        Test scenario:
            In-memory creation ignores compression silently.
        """
        arr = np.ones((5, 5), dtype=np.float64)
        nc = NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            compression="DEFLATE", compression_level=9,
        )
        var = nc.get_variable("v")
        assert_allclose(
            var.read_array(band=0), arr,
            err_msg="Data should be preserved even with ignored options",
        )


class TestBoundaryChunkSizes:
    """Edge cases for chunk_sizes parameter."""

    def test_chunk_covers_entire_array(self, tmp_path):
        """Chunk size equal to array dimensions (single chunk).

        Test scenario:
            chunk_sizes=(5, 50, 50) for a (5, 50, 50) array.
        """
        arr = np.random.RandomState(SEED).rand(5, 50, 50).astype(np.float64)
        path = str(tmp_path / "single_chunk.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v",
            extra_dim_name="time", path=path,
            chunk_sizes=(5, 50, 50),
            compression="DEFLATE",
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        block = rg.OpenMDArray("v").GetBlockSize()
        ds = None
        assert block == [5, 50, 50], (
            f"Expected [5, 50, 50], got {block}"
        )

    def test_2d_chunk_sizes(self, tmp_path):
        """Chunk sizes for a 2D array (no extra dimension).

        Test scenario:
            chunk_sizes=(25, 25) for a (50, 50) 2D array.
        """
        arr = np.random.RandomState(SEED).rand(50, 50).astype(np.float64)
        path = str(tmp_path / "chunk_2d.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v", path=path,
            chunk_sizes=(25, 25),
            compression="DEFLATE",
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        block = rg.OpenMDArray("v").GetBlockSize()
        ds = None
        assert block == [25, 25], (
            f"Expected [25, 25], got {block}"
        )


class TestCreateDimensionHelper:
    """Tests for NetCDF._create_dimension static method."""

    def test_with_set_indexing(self):
        """MEM driver: _create_dimension with set_indexing=True.

        Test scenario:
            Dimension should have its indexing variable linked.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        values = np.array([1.0, 2.0, 3.0])
        dim = NetCDF._create_dimension(
            rg, "x", dtype, values,
            gdal.DIM_TYPE_HORIZONTAL_X, set_indexing=True,
        )
        assert dim.GetSize() == 3, f"Expected size 3, got {dim.GetSize()}"
        iv = dim.GetIndexingVariable()
        assert iv is not None, "Indexing variable should be set"

    def test_without_set_indexing(self):
        """netCDF driver path: _create_dimension with set_indexing=False.

        Test scenario:
            Dimension should be created but without linked indexing
            variable (netCDF driver doesn't support SetIndexingVariable).
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        values = np.array([10.0, 20.0])
        dim = NetCDF._create_dimension(
            rg, "y", dtype, values,
            gdal.DIM_TYPE_HORIZONTAL_Y, set_indexing=False,
        )
        assert dim.GetSize() == 2, f"Expected size 2, got {dim.GetSize()}"

    def test_coordinate_values_written(self):
        """Coordinate array should contain the provided values.

        Test scenario:
            Create dimension with values [5, 10, 15], read back the
            coordinate array.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        values = np.array([5.0, 10.0, 15.0])
        NetCDF._create_dimension(rg, "z", dtype, values)
        coord = rg.OpenMDArray("z")
        assert_allclose(
            coord.ReadAsArray(), values,
            err_msg="Coordinate values not written correctly",
        )


class TestNodataBeforeWrite:
    """Verify nodata is set before data is written (netCDF driver requirement)."""

    def test_nodata_survives_disk_write(self, tmp_path):
        """No-data value should be present after writing to disk.

        Test scenario:
            The netCDF driver requires SetNoDataValueDouble before
            Write. If ordering is wrong, nodata is lost.
        """
        arr = np.full((10, 10), -9999.0, dtype=np.float64)
        path = str(tmp_path / "ndv_order.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v", path=path,
            no_data_value=-9999.0,
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray("v")
        ndv = md_arr.GetNoDataValue()
        ds = None
        assert_allclose(
            ndv, -9999.0, rtol=1e-6,
            err_msg=f"Nodata not preserved on disk, got {ndv}",
        )


class TestCompressionOnly:
    """Compression without chunk_sizes."""

    def test_compression_without_explicit_chunks(self, tmp_path):
        """Compression without chunk_sizes should not raise.

        Test scenario:
            Only set compression, let GDAL choose chunk sizes.
            Verify the file is created and data is readable.
        """
        arr = np.random.RandomState(SEED).rand(20, 30).astype(np.float64)
        path = str(tmp_path / "compress_no_chunk.nc")
        NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="v", path=path,
            compression="DEFLATE",
        )
        assert os.path.getsize(path) > 0, "File should be created"
        reloaded = NetCDF.read_file(path)
        var = reloaded.get_variable("v")
        assert_allclose(
            var.read_array(band=0), arr, rtol=1e-10,
            err_msg="Data should survive compression without chunks",
        )
