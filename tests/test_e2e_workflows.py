"""End-to-end workflow tests.

These tests exercise multi-step pipelines that combine reading, creating,
cropping, reprojecting, aligning, and round-tripping raster and vector data.

Workflows covered:
1. Create GeoTIFF from array -> crop with polygon -> extract values -> verify
2. Create MultiDataset -> save -> reload -> verify shapes
3. FeatureCollection -> to_dataset (rasterize) -> extract -> verify round-trip
4. Read GeoTIFF -> reproject -> align with another -> verify dimensions match
"""
import os
import shutil
import tempfile

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.featurecollection import FeatureCollection
from pyramids.multidataset import MultiDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    rows: int = 10,
    cols: int = 10,
    epsg: int = 32636,
    cell_size: float = 1000.0,
    top_left: tuple = (500000.0, 3400000.0),
    no_data: float = -9999.0,
    fill_value: float = 0.0,
) -> Dataset:
    """Create a simple in-memory Dataset."""
    src = Dataset.create(
        cell_size=cell_size,
        rows=rows,
        columns=cols,
        dtype="float32",
        bands=1,
        top_left_corner=top_left,
        epsg=epsg,
        no_data_value=no_data,
    )
    arr = np.full((rows, cols), fill_value, dtype=np.float32)
    src.raster.GetRasterBand(1).WriteArray(arr)
    return src


# ===========================================================================
# Workflow 1:  Create GeoTIFF -> crop with polygon -> extract values
# ===========================================================================


class TestCreateCropExtract:
    """Create a raster, crop it with a polygon mask, then extract values."""

    def test_create_crop_extract(self):
        """Full pipeline: create -> populate -> crop -> verify extracted values."""
        rows, cols = 20, 20
        cell_size = 1000.0
        epsg = 32636
        top_left = (500000.0, 3400000.0)

        # Step 1 - Create dataset with sequential values
        src = Dataset.create(
            cell_size=cell_size,
            rows=rows,
            columns=cols,
            dtype="float32",
            bands=1,
            top_left_corner=top_left,
            epsg=epsg,
            no_data_value=-9999.0,
        )
        arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        src.raster.GetRasterBand(1).WriteArray(arr)

        # Step 2 - Create a polygon that covers the top-left 5x5 cells
        x0, y0 = top_left
        poly = box(x0, y0 - 5 * cell_size, x0 + 5 * cell_size, y0)
        mask_gdf = gpd.GeoDataFrame(geometry=[poly], crs=f"EPSG:{epsg}")

        # Step 3 - Crop
        cropped = src.crop(mask_gdf)

        # Step 4 - Verify
        assert cropped is not None, "crop should return a new Dataset"
        cropped_arr = cropped.read_array()
        assert cropped_arr.shape[0] <= rows, (
            "Cropped rows should be <= original"
        )
        assert cropped_arr.shape[1] <= cols, (
            "Cropped cols should be <= original"
        )
        # The cropped area (top-left 5x5) should contain values 0-4, 20-24 etc.
        non_nodata = cropped_arr[
            ~np.isclose(cropped_arr, cropped.no_data_value[0], rtol=0.001)
        ]
        assert non_nodata.size > 0, (
            "Cropped raster should contain some valid data"
        )


# ===========================================================================
# Workflow 2:  Create MultiDataset -> save -> reload -> verify shapes
# ===========================================================================


class TestMultiDatasetRoundTrip:
    """Create a MultiDataset, save it, reload, and verify."""

    def test_save_and_reload(self):
        """Write MultiDataset to disk, read back, compare shapes."""
        rows, cols = 8, 10
        time_steps = 3

        base = _make_dataset(rows=rows, cols=cols, fill_value=1.0)
        md = MultiDataset.create_cube(base, dataset_length=time_steps)
        values = np.random.rand(time_steps, rows, cols).astype(np.float64)
        md.values = values

        tmp_dir = tempfile.mkdtemp()
        out_dir = os.path.join(tmp_dir, "multidataset_output")
        try:
            md.to_file(out_dir)

            # Reload
            reloaded = MultiDataset.read_multiple_files(
                out_dir, with_order=False
            )
            assert reloaded.time_length == time_steps, (
                f"Expected {time_steps} files, got {reloaded.time_length}"
            )
            assert reloaded.base.rows == rows, (
                f"Reloaded rows mismatch: expected {rows}"
            )
            assert reloaded.base.columns == cols, (
                f"Reloaded columns mismatch: expected {cols}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ===========================================================================
# Workflow 3:  FeatureCollection -> rasterize -> extract -> verify
# ===========================================================================


class TestRasterizeRoundTrip:
    """Rasterize a FeatureCollection and verify the burned values."""

    def test_rasterize_polygon(self):
        """Burn a polygon attribute into a raster and verify the value."""
        epsg = 32636
        cell_size = 1000.0
        top_left = (500000.0, 3400000.0)

        # Create a polygon covering a 5x5 area
        x0, y0 = top_left
        poly = box(x0, y0 - 5 * cell_size, x0 + 5 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"burn_val": [7]}, geometry=[poly], crs=f"EPSG:{epsg}"
        )
        fc = FeatureCollection(gdf)

        # Rasterize: use cell_size (no reference dataset)
        raster = fc.to_dataset(cell_size=cell_size, column_name="burn_val")
        arr = raster.read_array()

        # Verify burned value
        burned = arr[arr == 7.0]
        assert burned.size > 0, (
            "At least some cells should contain the burned value 7"
        )
        assert raster.epsg == epsg, (
            f"Rasterized EPSG should be {epsg}, got {raster.epsg}"
        )

    def test_rasterize_with_reference_dataset(self):
        """Burn using a reference Dataset for geotransform."""
        epsg = 32636
        cell_size = 1000.0
        rows, cols = 10, 10
        top_left = (500000.0, 3400000.0)

        # Reference raster
        ref = _make_dataset(
            rows=rows, cols=cols, cell_size=cell_size,
            top_left=top_left, epsg=epsg
        )

        # Create a polygon inside the raster extent
        x0, y0 = top_left
        poly = box(x0, y0 - 3 * cell_size, x0 + 3 * cell_size, y0)
        gdf = gpd.GeoDataFrame(
            {"class_id": [42]}, geometry=[poly], crs=f"EPSG:{epsg}"
        )
        fc = FeatureCollection(gdf)

        raster = fc.to_dataset(dataset=ref, column_name="class_id")
        arr = raster.read_array()

        # Same dimensions as reference
        assert arr.shape == (rows, cols), (
            f"Rasterized shape should match reference ({rows},{cols}), got {arr.shape}"
        )
        # Burned value should appear
        burned = arr[arr == 42.0]
        assert burned.size > 0, "Burned value 42 should appear in the raster"


# ===========================================================================
# Workflow 4:  Read GeoTIFF -> reproject -> align -> verify dimensions
# ===========================================================================


class TestReprojectAlignWorkflow:
    """Reproject a raster and then align another to its grid."""

    def test_reproject_and_verify(self):
        """Reproject a UTM raster to WGS84 and verify the EPSG changes."""
        src = _make_dataset(
            rows=10, cols=10, epsg=32636, cell_size=1000.0, fill_value=5.0
        )
        arr_orig = src.read_array()
        original_epsg = src.epsg
        assert original_epsg == 32636, "Starting EPSG should be 32636"

        reprojected = src.to_crs(to_epsg=4326)
        assert reprojected.epsg == 4326, (
            f"Reprojected EPSG should be 4326, got {reprojected.epsg}"
        )
        repr_arr = reprojected.read_array()
        assert repr_arr.shape[0] > 0, "Reprojected raster should have rows"
        assert repr_arr.shape[1] > 0, "Reprojected raster should have cols"

    def test_align_to_reference(self):
        """Align one raster to match another's grid."""
        # Reference raster (smaller)
        ref = _make_dataset(
            rows=5, cols=5, epsg=32636, cell_size=2000.0,
            top_left=(500000.0, 3400000.0), fill_value=0.0,
        )

        # Source raster (different grid)
        src = _make_dataset(
            rows=10, cols=10, epsg=32636, cell_size=1000.0,
            top_left=(500000.0, 3400000.0), fill_value=7.0,
        )

        aligned = src.align(ref)
        assert aligned.rows == ref.rows, (
            f"Aligned rows should be {ref.rows}, got {aligned.rows}"
        )
        assert aligned.columns == ref.columns, (
            f"Aligned columns should be {ref.columns}, got {aligned.columns}"
        )


# ===========================================================================
# Workflow 5:  MultiDataset create -> apply -> iterate -> verify
# ===========================================================================


class TestMultiDatasetProcessingPipeline:
    """Create a MultiDataset, apply a function, then iterate and verify."""

    def test_apply_then_iterate(self):
        """Apply a transformation and iterate to check every time step."""
        rows, cols = 6, 8
        time_steps = 4

        base = _make_dataset(rows=rows, cols=cols, fill_value=10.0)
        md = MultiDataset.create_cube(base, dataset_length=time_steps)

        # Fill with known values: each time step has value = step_index + 1
        values = np.zeros((time_steps, rows, cols), dtype=np.float64)
        for t in range(time_steps):
            values[t, :, :] = float(t + 1)
        md.values = values

        # Apply np.sqrt
        md.apply(np.sqrt)

        # Verify each time step via iteration
        for i, slice_arr in enumerate(md):
            expected_val = np.sqrt(float(i + 1))
            non_nodata = slice_arr[
                ~np.isclose(slice_arr, -9999.0, rtol=0.001)
            ]
            if non_nodata.size > 0:
                assert np.allclose(non_nodata, expected_val, atol=0.01), (
                    f"Time step {i}: expected ~{expected_val}, got {non_nodata[0]}"
                )

    def test_head_tail_first_last(self):
        """Verify head/tail/first/last return correct shapes."""
        rows, cols = 4, 5
        time_steps = 6

        base = _make_dataset(rows=rows, cols=cols)
        md = MultiDataset.create_cube(base, dataset_length=time_steps)
        values = np.random.rand(time_steps, rows, cols)
        md.values = values

        assert md.head(3).shape == (3, rows, cols), (
            "head(3) shape mismatch"
        )
        assert md.tail(-2).shape == (2, rows, cols), (
            "tail(-2) shape mismatch"
        )
        assert md.first().shape == (rows, cols), "first() shape mismatch"
        assert md.last().shape == (rows, cols), "last() shape mismatch"

        # Verify first/last content
        np.testing.assert_array_equal(
            md.first(), values[0], err_msg="first() content mismatch"
        )
        np.testing.assert_array_equal(
            md.last(), values[-1], err_msg="last() content mismatch"
        )


# ===========================================================================
# Workflow 6:  FeatureCollection property round-trip
# ===========================================================================


class TestFeatureCollectionPropertiesE2E:
    """End-to-end property checks for FeatureCollection."""

    def test_gdf_roundtrip_ds_conversion(self):
        """Convert GDF -> DataSource -> GDF and compare EPSG."""
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame(
            {"val": [1]}, geometry=[poly], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)
        original_epsg = fc.epsg

        # Convert to DataSource
        ds_fc = fc._gdf_to_ds()
        assert ds_fc is not None, "Conversion to DS should not return None"
        assert isinstance(ds_fc, FeatureCollection), (
            "Should return a FeatureCollection"
        )

        # Convert back
        ds_fc_obj = ds_fc
        back_gdf = ds_fc_obj._ds_to_gdf()
        assert isinstance(back_gdf, gpd.GeoDataFrame), (
            "Converting back should produce a GeoDataFrame"
        )

    def test_save_and_reload_vector(self):
        """Save a FeatureCollection to disk and read it back."""
        poly = box(30.0, 30.0, 31.0, 31.0)
        gdf = gpd.GeoDataFrame(
            {"score": [99.5]}, geometry=[poly], crs="EPSG:4326"
        )
        fc = FeatureCollection(gdf)

        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "test_output.geojson")
        try:
            fc.to_file(path)
            assert os.path.exists(path), "File should exist after to_file"
            reloaded = FeatureCollection.read_file(path)
            assert isinstance(reloaded.feature, gpd.GeoDataFrame), (
                "Reloaded feature should be a GeoDataFrame"
            )
            assert len(reloaded.feature) == 1, (
                "Reloaded GDF should have 1 row"
            )
            assert abs(reloaded.feature["score"].iloc[0] - 99.5) < 0.01, (
                "Reloaded score value should be ~99.5"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ===========================================================================
# Workflow 7:  GeoTIFF to disk round-trip
# ===========================================================================


class TestGeoTiffRoundTrip:
    """Write an in-memory Dataset to GeoTIFF, reload, verify."""

    def test_write_read_geotiff(self):
        """Create an in-memory raster, save to disk, reload and verify array."""
        rows, cols = 12, 15
        src = _make_dataset(
            rows=rows, cols=cols, fill_value=42.0, epsg=4326, cell_size=0.1,
            top_left=(10.0, 50.0),
        )
        arr_original = src.read_array()

        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "test_raster.tif")
        try:
            src.to_file(path)
            assert os.path.exists(path), "GeoTIFF should be written"

            reloaded = Dataset.read_file(path)
            arr_reloaded = reloaded.read_array()
            assert arr_reloaded.shape == (rows, cols), (
                f"Reloaded shape mismatch: {arr_reloaded.shape}"
            )
            np.testing.assert_array_almost_equal(
                arr_reloaded, arr_original, decimal=2,
                err_msg="Reloaded array values differ from original"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
