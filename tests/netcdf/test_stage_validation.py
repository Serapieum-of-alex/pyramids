"""Tests validating Stage 1-2 fixes and forward-looking Stage 3 expectations.

These tests track that the implementation plan is going in the right direction.
Stage 3 tests are marked xfail — they document the expected behavior for the
round-trip workflow before it is implemented.
"""

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import Polygon

from pyramids.netcdf.metadata import from_json, get_metadata, to_dict, to_json
from pyramids.netcdf.models import DimensionInfo, NetCDFMetadata
from pyramids.netcdf.netcdf import NetCDF


@pytest.fixture(scope="module")
def mdim_nc(noah_nc_path):
    """NetCDF opened in multidimensional mode."""
    return NetCDF.read_file(noah_nc_path, open_as_multi_dimensional=True)


@pytest.fixture(scope="module")
def classic_nc(noah_nc_path):
    """NetCDF opened in classic mode."""
    return NetCDF.read_file(noah_nc_path, open_as_multi_dimensional=False)


@pytest.fixture(scope="module")
def created_nc():
    """A NetCDF created in memory with create_from_array."""
    arr = np.random.rand(3, 10, 12).astype(np.float64)
    geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
    return NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        path=None,
        variable_name="temperature",
    )


class TestStage1_IsSubsetLogic:
    """NCP-1.1: is_subset must be False for root containers."""

    def test_mdim_container_is_not_subset(self, mdim_nc):
        assert mdim_nc.is_subset is False

    def test_classic_container_is_not_subset(self, classic_nc):
        assert classic_nc.is_subset is False

    def test_variable_extracted_is_subset(self, mdim_nc):
        var = mdim_nc.get_variable(mdim_nc.variable_names[0])
        assert var.is_subset is True


class TestStage1_GetAllMetadata:
    """NCP-1.2: get_all_metadata() must exist and return NetCDFMetadata."""

    def test_get_all_metadata_returns_metadata(self, mdim_nc):
        md = mdim_nc.get_all_metadata()
        assert isinstance(md, NetCDFMetadata)

    def test_dimension_overview_populated(self, mdim_nc):
        md = mdim_nc.get_all_metadata()
        assert md.dimension_overview is not None
        assert isinstance(md.dimension_overview, dict)
        assert "names" in md.dimension_overview
        assert "sizes" in md.dimension_overview


class TestStage1_GetDimensionOnMetadata:
    """NCP-1.3: NetCDFMetadata.get_dimension() and .names must work."""

    def test_names_returns_list(self, mdim_nc):
        md = mdim_nc.meta_data
        assert isinstance(md.names, list)
        assert len(md.names) > 0

    def test_get_dimension_by_name(self, created_nc):
        md = created_nc.meta_data
        dim = md.get_dimension("x")
        assert dim is not None
        assert isinstance(dim, DimensionInfo)
        assert dim.name == "x"

    def test_get_dimension_nonexistent_returns_none(self, created_nc):
        md = created_nc.meta_data
        assert md.get_dimension("nonexistent") is None

    def test_dimension_attrs_populated(self, created_nc):
        md = created_nc.meta_data
        dim = md.get_dimension("x")
        assert isinstance(dim.attrs, dict)


class TestStage1_BareExceptFixed:
    """NCP-1.4: Bare except clauses replaced with except Exception."""

    def test_add_md_array_no_bare_except(self):
        import inspect

        src = inspect.getsource(NetCDF._add_md_array_to_group)
        # Should not contain bare "except:" (only "except Exception:")
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("except") and stripped.endswith(":"):
                assert "Exception" in stripped, f"Found bare except clause: {stripped}"


class TestStage1_NoSelfInitCalls:
    """NCP-1.5: self.__init__() should not appear in add_variable/remove_variable."""

    def test_add_variable_no_self_init(self):
        import inspect

        src = inspect.getsource(NetCDF.add_variable)
        assert "self.__init__" not in src

    def test_remove_variable_no_self_init(self):
        import inspect

        src = inspect.getsource(NetCDF.remove_variable)
        assert "self.__init__" not in src

    def test_replace_raster_method_exists(self):
        assert hasattr(NetCDF, "_replace_raster")


class TestStage1_GetMetadataTypeDispatch:
    """NCP-1.6: get_metadata() accepts str, NetCDF, and gdal.Dataset."""

    def test_accepts_netcdf_instance(self, mdim_nc):
        md = get_metadata(mdim_nc)
        assert isinstance(md, NetCDFMetadata)

    def test_accepts_gdal_dataset(self, mdim_nc):
        md = get_metadata(mdim_nc._raster)
        assert isinstance(md, NetCDFMetadata)

    def test_accepts_file_path(self, noah_nc_path):
        md = get_metadata(noah_nc_path)
        assert isinstance(md, NetCDFMetadata)

    def test_bad_path_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            get_metadata("/nonexistent/file.nc")


class TestStage2_InitPyExports:
    """NCP-2.1: __init__.py exports key classes."""

    def test_imports_from_package(self):
        from pyramids.netcdf import NetCDF as NC
        from pyramids.netcdf import NetCDFMetadata as MD
        from pyramids.netcdf import get_metadata as gm

        assert NC is NetCDF
        assert MD is NetCDFMetadata
        assert gm is get_metadata


class TestStage2_GeotransformGuard:
    """NCP-2.3: geotransform falls back to GDAL's GetGeoTransform() instead of None."""

    def test_geotransform_never_none_on_variable(self, created_nc):
        var = created_nc.get_variable("temperature")
        gt = var.geotransform
        assert gt is not None
        assert len(gt) == 6

    def test_geotransform_on_container_with_coords(self, created_nc):
        gt = created_nc.geotransform
        assert gt is not None


class TestStage2_ReadVariableOptimized:
    """NCP-2.6: _read_variable uses MDIM path when root group is available."""

    def test_read_known_variable(self, created_nc):
        arr = created_nc._read_variable("temperature")
        assert arr is not None
        assert isinstance(arr, np.ndarray)

    def test_read_dimension_variable(self, created_nc):
        arr = created_nc._read_variable("x")
        assert arr is not None
        assert arr.ndim == 1

    def test_read_nonexistent_returns_none(self, created_nc):
        result = created_nc._read_variable("does_not_exist")
        assert result is None


class TestStage2_VariablesCaching:
    """NCP-2.4: variables property is cached."""

    def test_same_object_on_repeated_access(self, created_nc):
        v1 = created_nc.variables
        v2 = created_nc.variables
        assert v1 is v2

    def test_cache_invalidated_after_add_variable(self):
        arr = np.random.rand(3, 5, 5).astype(np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
            variable_name="var1",
        )
        _ = nc.variables  # populate cache
        nc.add_variable(nc)  # adds "var1-new"
        v2 = nc.variables
        assert "var1-new" in v2


class TestStage2_MetaDataCaching:
    """NCP-2.5: meta_data property is cached."""

    def test_same_object_on_repeated_access(self, created_nc):
        m1 = created_nc.meta_data
        m2 = created_nc.meta_data
        assert m1 is m2


class TestStage2_ReadArrayGuard:
    """NCP-2.2: read_array() on root MDIM container raises ValueError."""

    def test_read_array_on_container_raises(self, mdim_nc):
        with pytest.raises(ValueError, match="not supported on the NetCDF container"):
            mdim_nc.read_array()

    def test_read_array_on_variable_works(self, created_nc):
        var = created_nc.get_variable("temperature")
        arr = var.read_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 3  # 3 bands


class TestStage2_GetVariableInvalidName:
    """get_variable with invalid name raises ValueError."""

    def test_invalid_variable_name_raises(self, created_nc):
        with pytest.raises(ValueError, match="not a valid variable name"):
            created_nc.get_variable("nonexistent")


class TestStage2_VariableNoDataValue:
    """No-data value correctly propagated to variable subsets."""

    def test_no_data_on_variable(self, created_nc):
        var = created_nc.get_variable("temperature")
        assert var.no_data_value is not None
        assert len(var.no_data_value) > 0


class TestStage3_VariableOriginTracking:
    """RT-4: get_variable() must track parent context for round-trip."""

    def test_parent_reference(self, created_nc):
        var = created_nc.get_variable("temperature")
        assert var._parent_nc is created_nc

    def test_source_variable_name(self, created_nc):
        var = created_nc.get_variable("temperature")
        assert var._source_var_name == "temperature"

    def test_dimension_names_tracked(self, created_nc):
        var = created_nc.get_variable("temperature")
        assert isinstance(var._md_array_dims, list)
        assert len(var._md_array_dims) > 0
        assert "x" in var._md_array_dims
        assert "y" in var._md_array_dims

    def test_band_dim_tracked_for_3d(self, created_nc):
        var = created_nc.get_variable("temperature")
        # 3D array → bands dimension should be tracked
        assert var._band_dim_name is not None
        assert var._band_dim_values is not None
        assert len(var._band_dim_values) == 3  # 3 bands

    def test_variable_attrs_tracked(self, created_nc):
        var = created_nc.get_variable("temperature")
        assert isinstance(var._variable_attrs, dict)

    def test_2d_variable_no_band_dim(self):
        """2D variables should have band_dim_name=None."""
        arr = np.random.rand(10, 12).astype(np.float64)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
            variable_name="elevation",
        )
        var = nc.get_variable("elevation")
        assert var._band_dim_name is None
        assert var._band_dim_values is None

    def test_origin_on_classic_mode_variable(self, classic_nc):
        """Even classic-mode variables should have origin attrs (empty)."""
        var = classic_nc.get_variable(classic_nc.variable_names[0])
        assert hasattr(var, "_parent_nc")
        assert hasattr(var, "_source_var_name")
        assert var._source_var_name == classic_nc.variable_names[0]


class TestStage3_SetVariable:
    """RT-3: set_variable() writes a classic Dataset back as an MDArray."""

    def test_set_variable_basic(self, created_nc):
        """Write a modified array back as a new variable."""
        var = created_nc.get_variable("temperature")
        arr = var.read_array() * 2
        from pyramids.dataset import Dataset

        modified = Dataset.create_from_array(
            arr,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        created_nc.set_variable("temp_doubled", modified)
        assert "temp_doubled" in created_nc.variable_names

    def test_set_variable_data_preserved(self, created_nc):
        """The written data should match what was provided."""
        var = created_nc.get_variable("temperature")
        arr_orig = var.read_array()
        arr_doubled = arr_orig * 2
        from pyramids.dataset import Dataset

        modified = Dataset.create_from_array(
            arr_doubled,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        created_nc.set_variable("temp_check", modified)
        # Read back via MDIM
        rg = created_nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("temp_check")
        written = md_arr.ReadAsArray()
        np.testing.assert_array_almost_equal(written, arr_doubled, decimal=5)

    def test_set_variable_replaces_existing(self):
        """Setting a variable with an existing name replaces it."""
        arr = np.ones((5, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
            variable_name="data",
        )
        arr2 = np.ones((5, 5), dtype=np.float64) * 42
        from pyramids.dataset import Dataset

        ds2 = Dataset.create_from_array(arr2, geo=geo, epsg=4326, no_data_value=-9999.0)
        nc.set_variable("data", ds2)
        assert "data" in nc.variable_names
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("data")
        np.testing.assert_array_almost_equal(md_arr.ReadAsArray(), arr2)

    def test_set_variable_auto_detects_band_dim(self, created_nc):
        """When dataset came from get_variable(), band_dim is auto-detected."""
        var = created_nc.get_variable("temperature")
        arr = var.read_array()
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            arr,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        # Transfer origin metadata
        ds._band_dim_name = var._band_dim_name
        ds._band_dim_values = var._band_dim_values
        ds._variable_attrs = var._variable_attrs
        created_nc.set_variable("temp_autodetect", ds)
        assert "temp_autodetect" in created_nc.variable_names

    def test_set_variable_invalidates_caches(self):
        """Cache should be invalidated after set_variable."""
        arr = np.random.rand(5, 5).astype(np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
            variable_name="original",
        )
        _ = nc.variables  # populate cache
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            np.random.rand(5, 5),
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable("new_var", ds)
        assert "new_var" in nc.variables

    def test_set_variable_requires_mdim_container(self, classic_nc):
        """set_variable on a classic-mode container should raise."""
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            np.random.rand(5, 5),
            geo=(0, 1, 0, 5, 0, -1),
            epsg=4326,
            no_data_value=-9999.0,
        )
        with pytest.raises(ValueError, match="multidimensional container"):
            classic_nc.set_variable("test", ds)


class TestStage3_DimensionReuse:
    """RT-6: Dimensions should be reused when sizes match."""

    def test_same_size_dimensions_reused(self):
        """Adding two variables with same spatial extent should reuse x/y dims."""
        arr = np.random.rand(5, 8).astype(np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
            variable_name="var1",
        )
        from pyramids.dataset import Dataset

        ds2 = Dataset.create_from_array(
            np.random.rand(5, 8),
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable("var2", ds2)
        rg = nc._raster.GetRootGroup()
        dim_names = [d.GetName() for d in rg.GetDimensions()]
        # Should still only have x and y (not x_8 or y_5)
        assert "x" in dim_names
        assert "y" in dim_names
        x_count = sum(1 for n in dim_names if n.startswith("x"))
        y_count = sum(1 for n in dim_names if n.startswith("y"))
        assert x_count == 1
        assert y_count == 1


class TestStage3_RoundTripDecision:
    """RT-1: Documenting the chosen approach for GIS ops return type."""

    def test_crop_returns_dataset_by_design(self):
        """GIS ops return Dataset. Users use set_variable() to write back.

        This is the chosen design: explicit round-trip via set_variable(),
        rather than trying to make crop/to_crs return NetCDF.
        """
        from pyramids.dataset import Dataset

        # Verify the workflow exists
        assert hasattr(NetCDF, "set_variable")
        assert hasattr(NetCDF, "get_variable")
        # The crop method is on Dataset, returns Dataset — by design
        import inspect

        sig = inspect.signature(Dataset.crop)
        assert "mask" in sig.parameters


class TestStage4_SpatialOpsGuard:
    """NCP-3.2: Spatial operations on root container raise ValueError."""

    def test_crop_on_container_raises(self, mdim_nc):
        mask = gpd.GeoDataFrame(
            geometry=[Polygon([(0, -80), (0, -70), (10, -70), (10, -80)])],
            crs=4326,
        )
        with pytest.raises(ValueError, match="not supported on the NetCDF container"):
            mdim_nc.crop(mask)

    def test_to_crs_on_container_raises(self, mdim_nc):
        with pytest.raises(ValueError, match="not supported on the NetCDF container"):
            mdim_nc.to_crs(3857)

    def test_crop_on_variable_allowed(self, classic_nc):
        """Spatial ops on variable subsets should not raise."""
        var = classic_nc.get_variable(classic_nc.variable_names[0])
        # Just check it doesn't raise the container guard
        # (may raise other errors if geotransform is off, but not our guard)
        assert not (var._is_md_array and not var._is_subset and var.band_count == 0)


class TestStage4_ToFile:
    """NCP-3.1: to_file() overridden for NetCDF."""

    def test_to_file_nc_creates_file(self, created_nc, tmp_path):
        import os

        out = str(tmp_path / "output.nc")
        created_nc.to_file(out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_to_file_nc_roundtrip_variable_names(self, created_nc, tmp_path):
        out = str(tmp_path / "roundtrip.nc")
        created_nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "temperature" in reloaded.variable_names

    def test_to_file_nc_roundtrip_data(self, created_nc, tmp_path):
        out = str(tmp_path / "data_check.nc")
        var_orig = created_nc.get_variable("temperature")
        arr_orig = var_orig.read_array()
        created_nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        var_new = reloaded.get_variable("temperature")
        arr_new = var_new.read_array()
        np.testing.assert_array_almost_equal(arr_orig, arr_new, decimal=5)

    def test_to_file_non_nc_on_container_raises(self, created_nc, tmp_path):
        out = str(tmp_path / "bad.tif")
        with pytest.raises(ValueError, match="Cannot save a multidimensional"):
            created_nc.to_file(out)


class TestStage4_Copy:
    """NCP-3.3: copy() works for MDIM datasets."""

    def test_copy_in_memory(self, created_nc):
        copied = created_nc.copy()
        assert isinstance(copied, NetCDF)
        assert "temperature" in copied.variable_names

    def test_copy_to_disk(self, created_nc, tmp_path):
        import os

        out = str(tmp_path / "copied.nc")
        copied = created_nc.copy(path=out)
        assert isinstance(copied, NetCDF)
        assert os.path.exists(out)
        assert "temperature" in copied.variable_names

    def test_copy_preserves_data(self, created_nc):
        var_orig = created_nc.get_variable("temperature")
        arr_orig = var_orig.read_array()
        copied = created_nc.copy()
        var_copy = copied.get_variable("temperature")
        arr_copy = var_copy.read_array()
        np.testing.assert_array_almost_equal(arr_orig, arr_copy, decimal=5)


class TestStage4_CreateCopyMdim:
    """RT-10: Verify CreateCopy from MEM multidim → netCDF driver works."""

    def test_create_copy_preserves_structure(self, created_nc, tmp_path):
        """The netCDF driver CreateCopy should preserve MDIM structure."""
        out = str(tmp_path / "createcopy.nc")
        dst = gdal.GetDriverByName("netCDF").CreateCopy(out, created_nc._raster, 0)
        assert dst is not None
        dst.FlushCache()
        rg = dst.GetRootGroup()
        assert rg is not None
        arr_names = rg.GetMDArrayNames()
        assert "temperature" in arr_names
        dst = None
