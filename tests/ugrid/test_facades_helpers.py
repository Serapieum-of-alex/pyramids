"""Tests for _facades.py private helpers and facade edge cases.

Covers the ``_representable``, ``_facade_variable``, and
``count_domain_cells`` branches not reached by the happy-path
facade tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyramids.netcdf.ugrid as ugrid_pkg
from pyramids.netcdf.ugrid._facades import _representable
from pyramids.netcdf.ugrid.analysis import (
    contour_lines,
    extract_points,
    sample_random,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from tests.ugrid.conftest import make_two_triangle_ds

pytestmark = pytest.mark.core


def _two_var_ds() -> UgridDataset:
    """Build a UgridDataset with two data variables."""
    fnc = Connectivity(
        data=np.array([[0, 1, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5]),
        node_y=np.array([0.0, 0.0, 1.0]),
        face_node_connectivity=fnc,
    )
    var_a = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(1,),
        _data=np.array([5.0]),
    )
    var_b = MeshVariable(
        name="temp",
        location="face",
        mesh_name="mesh2d",
        shape=(1,),
        _data=np.array([15.0]),
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var_a, "temp": var_b},
        global_attributes={},
    )
    return ds


def _no_data_var_ds() -> UgridDataset:
    """Build a UgridDataset whose only variable has no loaded data."""
    fnc = Connectivity(
        data=np.array([[0, 1, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5]),
        node_y=np.array([0.0, 0.0, 1.0]),
        face_node_connectivity=fnc,
    )
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(1,),
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


class TestRepresentable:
    """Tests for ``_representable`` — the nodata/dtype guard in ``_rewrap``."""

    def test_none_nodata_always_safe(self):
        """None declares nothing, so it is representable in every dtype.

        Test scenario:
            nodata=None with any dtype returns True.
        """
        assert _representable(None, np.dtype("float64")) is True, (
            "None should be representable"
        )

    def test_bool_dtype_refuses_everything(self):
        """A two-valued domain has no spare sentinel.

        Test scenario:
            Any nodata value with bool dtype returns False.
        """
        assert _representable(0.0, np.dtype("bool")) is False, (
            "bool dtype should refuse all nodata values"
        )

    def test_non_number_nodata_not_representable(self):
        """A string nodata is not a number and cannot round-trip.

        Test scenario:
            nodata='missing' with float64 dtype returns False.
        """
        assert _representable("missing", np.dtype("float64")) is False, (
            "Non-numeric nodata should not be representable"
        )

    def test_nan_representable_in_float(self):
        """NaN is a real fill value for a float array.

        Test scenario:
            nodata=NaN with float64 dtype returns True.
        """
        assert _representable(float("nan"), np.dtype("float64")) is True, (
            "NaN should be representable in float64"
        )

    def test_nan_not_representable_in_integer(self):
        """NaN is meaningless in an integer array.

        Test scenario:
            nodata=NaN with int32 dtype returns False.
        """
        assert _representable(float("nan"), np.dtype("int32")) is False, (
            "NaN should not be representable in int32"
        )

    def test_integer_nodata_in_range(self):
        """An integer sentinel within the dtype range passes.

        Test scenario:
            nodata=-9999 with int32 dtype returns True.
        """
        assert _representable(-9999, np.dtype("int32")) is True, (
            "-9999 should be representable in int32"
        )

    def test_integer_nodata_out_of_range(self):
        """An integer sentinel outside the dtype range fails.

        Test scenario:
            nodata=99999 with int8 (max 127) returns False.
        """
        assert _representable(99999, np.dtype("int8")) is False, (
            "99999 should not be representable in int8"
        )

    def test_float_nodata_non_integer_in_int_dtype(self):
        """A fractional float cannot survive an int cast.

        Test scenario:
            nodata=1.5 with int32 dtype returns False.
        """
        assert _representable(1.5, np.dtype("int32")) is False, (
            "1.5 should not be representable in int32"
        )

    def test_float_nodata_integer_valued_in_int_dtype(self):
        """An integer-valued float within range survives the int cast.

        Test scenario:
            nodata=42.0 with int32 dtype returns True.
        """
        assert _representable(42.0, np.dtype("int32")) is True, (
            "42.0 should be representable in int32"
        )

    def test_float_roundtrip_fail(self):
        """A very large float does not survive a float16 round-trip.

        Test scenario:
            nodata=1e30 with float16 dtype returns False.
        """
        assert _representable(1e30, np.dtype("float16")) is False, (
            "1e30 should not survive float16 round-trip"
        )

    def test_type_error_returns_false(self):
        """An un-castable object triggers TypeError inside, caught as False.

        Test scenario:
            nodata=object() with float64 dtype returns False because
            object is not a numeric dtype.
        """
        assert _representable(object(), np.dtype("float64")) is False, (
            "Un-castable object should not be representable"
        )

    def test_complex_nodata_triggers_except(self):
        """Complex nodata triggers TypeError in float() cast, caught as False.

        Test scenario:
            nodata=complex(1, 2) with int32 dtype: np.asarray succeeds
            (complex128 is numeric), but float(complex) raises TypeError
            which the except clause catches, returning False.
        """
        assert _representable(complex(1, 2), np.dtype("int32")) is False, (
            "Complex nodata should not be representable in int32"
        )


class TestFacadeVariable:
    """Tests for ``_facade_variable`` error paths."""

    def test_ambiguous_variable_raises(self):
        """Two variables without naming one raises ValueError.

        Test scenario:
            A dataset with 'depth' and 'temp' called with
            variable_name=None raises ValueError listing both.
        """
        ds = _two_var_ds()
        with pytest.raises(ValueError, match="2 data variable"):
            ds.fillna(0.0)

    def test_missing_variable_raises(self):
        """Naming a non-existent variable raises KeyError.

        Test scenario:
            variable_name='nonexistent' raises KeyError.
        """
        ds = _two_var_ds()
        with pytest.raises(KeyError, match="nonexistent"):
            ds.fillna(0.0, variable_name="nonexistent")

    def test_no_loaded_data_raises(self):
        """A variable with no data raises ValueError.

        Test scenario:
            A variable created without data or loader raises
            ValueError mentioning 'no loaded data'.
        """
        ds = _no_data_var_ds()
        with pytest.raises(ValueError, match="no loaded data"):
            ds.fillna(0.0)


class TestCountDomainCellsTwoDimMask:
    """Test count_domain_cells with a 2-D (temporal) active mask."""

    def test_uses_last_step_of_2d_mask(self):
        """A 2-D active mask is reduced to the last time step.

        Test scenario:
            A (2, n_face) active mask where the last row has 1 active
            face returns 1 from count_domain_cells.
        """
        ds = make_two_triangle_ds()
        ds["depth"].active = np.array(
            [[True, True], [True, False]]
        )
        count = ds.count_domain_cells("depth")
        assert count == 1, (
            f"Expected 1 active face from last time step, got {count}"
        )


class TestRewrapStripsBoolNodata:
    """Test that _rewrap drops the nodata sentinel for boolean results."""

    def test_isnull_drops_float_nodata(self):
        """isnull produces a bool array; -9999.0 nodata is stripped.

        Test scenario:
            A variable with nodata=-9999.0 passed through isnull()
            should produce a dataset whose variable has nodata=None
            (because bool has no spare sentinel).
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(1,),
            _data=np.array([-9999.0]),
            nodata=-9999.0,
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var},
            global_attributes={},
        )
        result = ds.isnull()
        assert result["depth"].nodata is None, (
            "Boolean result should have nodata=None, "
            f"got {result['depth'].nodata}"
        )


class TestUgridAnalysisReExport:
    """Verify analysis module is accessible via the ugrid package."""

    def test_analysis_in_dir(self):
        """analysis should appear in dir(pyramids.netcdf.ugrid).

        Test scenario:
            After the M2 fix, ``analysis`` is in the import block
            and visible to static analysis tools.
        """
        assert hasattr(ugrid_pkg, "analysis"), (
            "analysis module not exported from ugrid __init__"
        )

    def test_analysis_functions_accessible(self):
        """Key analysis functions are importable through the package.

        Test scenario:
            extract_points and contour_lines are reachable via
            pyramids.netcdf.ugrid.analysis.
        """
        assert callable(extract_points), (
            "extract_points should be callable"
        )
        assert callable(contour_lines), (
            "contour_lines should be callable"
        )
        assert callable(sample_random), (
            "sample_random should be callable"
        )
