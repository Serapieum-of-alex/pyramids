"""Tests validating Stage 1-2 fixes and forward-looking Stage 3 expectations.

These tests track that the implementation plan is going in the right direction.
Stage 3 tests are marked xfail — they document the expected behavior for the
round-trip workflow before it is implemented.
"""

import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import Polygon
import geopandas as gpd

from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.models import NetCDFMetadata, DimensionInfo
from pyramids.netcdf.metadata import get_metadata, to_json, from_json, to_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        arr=arr, geo=geo, epsg=4326, no_data_value=-9999.0,
        driver_type="netcdf", path=None,
        variable_name="temperature",
    )


# ===========================================================================
# Stage 1 validation — Bug fixes
# ===========================================================================

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
                assert "Exception" in stripped, (
                    f"Found bare except clause: {stripped}"
                )


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


# ===========================================================================
# Stage 2 validation — Core API
# ===========================================================================

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
            arr=arr, geo=geo, epsg=4326, no_data_value=-9999.0,
            driver_type="netcdf", path=None,
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
        with pytest.raises(ValueError, match="Cannot read array from NetCDF container"):
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


# ===========================================================================
# Stage 3 forward-looking tests — Round-trip workflow
# These are xfail until Stage 3 is implemented.
# ===========================================================================

class TestStage3_RoundTrip:
    """Forward-looking tests for the MDArray <-> Classic Dataset round-trip.

    These verify expected behavior that Stage 3 must implement.
    They don't call GDAL operations (which can hang on MEM views on Windows),
    instead checking for API surface and code patterns.
    """

    def test_set_variable_not_yet_implemented(self):
        """RT-3: set_variable() should be added in Stage 3."""
        has_it = hasattr(NetCDF, "set_variable")
        if not has_it:
            pytest.xfail("RT-3: set_variable() not implemented yet")

    def test_variable_origin_not_yet_tracked(self, mdim_nc):
        """RT-4: get_variable() should track parent context."""
        var = mdim_nc.get_variable(mdim_nc.variable_names[0])
        has_parent = hasattr(var, "_parent_nc") and var._parent_nc is not None
        if not has_parent:
            pytest.xfail("RT-4: variable subsets don't track parent context yet")

    def test_crop_returns_dataset_not_netcdf(self):
        """RT-1: GIS ops currently return Dataset — Stage 3 should fix this."""
        from pyramids.dataset import Dataset
        import inspect
        src = inspect.getsource(Dataset._crop_with_polygon_warp)
        preserves_type = "type(self)" in src or "NetCDF" in src
        if not preserves_type:
            pytest.xfail("RT-1: crop returns Dataset, not NetCDF")
