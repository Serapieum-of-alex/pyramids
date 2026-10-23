"""Tests for UgridDataset.equals and UgridDataset.identical."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _make_ds(
    values: np.ndarray | None = None,
) -> UgridDataset:
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 0.5, 1.5]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=fnc,
    )
    if values is None:
        values = np.array([10.0, 20.0])
    var = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=values.shape,
        _data=values,
    )
    ds = UgridDataset(
        mesh=mesh,
        data_variables={"depth": var},
        global_attributes={},
    )
    return ds


class TestEquals:
    def test_identical_datasets_are_equal(self):
        assert _make_ds().equals(_make_ds())

    def test_different_values_not_equal(self):
        ds1 = _make_ds(np.array([1.0, 2.0]))
        ds2 = _make_ds(np.array([1.0, 3.0]))
        assert not ds1.equals(ds2)

    def test_nan_positions_must_match(self):
        ds1 = _make_ds(np.array([1.0, np.nan]))
        ds2 = _make_ds(np.array([1.0, np.nan]))
        assert ds1.equals(ds2)

    def test_different_nan_positions_not_equal(self):
        ds1 = _make_ds(np.array([np.nan, 2.0]))
        ds2 = _make_ds(np.array([1.0, np.nan]))
        assert not ds1.equals(ds2)

    def test_non_ugrid_returns_false(self):
        assert not _make_ds().equals("not a dataset")


class TestEqualsDifferentVarNames:
    """Equals when datasets have different variable names."""

    def test_different_variable_names_not_equal(self):
        """Datasets with different variable name sets are not equal.

        Test scenario:
            One dataset has 'depth', the other has 'temp'; equals
            returns False.
        """
        ds1 = _make_ds()
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="temp",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"temp": var},
            global_attributes={},
        )
        assert not ds1.equals(ds2), (
            "Datasets with different variable names should not be equal"
        )


class TestEqualsDataNone:
    """Equals when one variable has data=None."""

    def test_one_data_none_not_equal(self):
        """One variable loaded, the other not — not equal.

        Test scenario:
            ds1 has loaded data, ds2 has data=None on the same variable
            name; equals returns False.
        """
        ds1 = _make_ds()
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var},
            global_attributes={},
        )
        assert not ds1.equals(ds2), (
            "Dataset with loaded data should not equal one without"
        )

    def test_both_data_none_are_equal(self):
        """Both variables have data=None — they are equal.

        Test scenario:
            Two datasets whose only variable both have data=None
            should be equal.
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var1 = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        var2 = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        ds1 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var1},
            global_attributes={},
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var2},
            global_attributes={},
        )
        assert ds1.equals(ds2), (
            "Both-None-data datasets should be equal"
        )


class TestIdentical:
    def test_identical_datasets(self):
        assert _make_ds().identical(_make_ds())

    def test_different_dtype_not_identical(self):
        ds1 = _make_ds(np.array([1.0, 2.0]))
        ds2 = _make_ds(np.array([1, 2]))
        assert not ds1.identical(ds2)

    def test_nan_nodata_identical(self):
        """Two datasets with NaN nodata sentinels should be identical."""
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var_a = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
            nodata=float("nan"),
        )
        var_b = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
            nodata=float("nan"),
        )
        ds_a = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_a},
            global_attributes={},
        )
        ds_b = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_b},
            global_attributes={},
        )
        assert ds_a.identical(ds_b)

    def test_different_global_attributes_not_identical(self):
        """Different global attributes make datasets non-identical.

        Test scenario:
            ds1 has global_attributes={}, ds2 has {'source': 'test'};
            identical returns False.
        """
        ds1 = _make_ds()
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var},
            global_attributes={"source": "test"},
        )
        assert not ds1.identical(ds2), (
            "Different global_attributes should be non-identical"
        )

    def test_different_crs_wkt_not_identical(self):
        """Different CRS WKT makes datasets non-identical.

        Test scenario:
            ds1 has EPSG 4326, ds2 has no CRS; identical returns False.
        """
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        assert not ds1.identical(ds2), (
            "Different CRS WKT should be non-identical"
        )

    def test_different_nodata_not_identical(self):
        """Different nodata sentinels make datasets non-identical.

        Test scenario:
            ds1 has nodata=-9999.0, ds2 has nodata=-1.0; identical
            returns False.
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var_a = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
            nodata=-9999.0,
        )
        var_b = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10.0, 20.0]),
            nodata=-1.0,
        )
        ds1 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_a},
            global_attributes={},
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_b},
            global_attributes={},
        )
        assert not ds1.identical(ds2), (
            "Different nodata sentinels should be non-identical"
        )

    def test_identical_both_data_none(self):
        """Both variables having data=None are identical.

        Test scenario:
            Two datasets whose only variable has data=None should
            be identical (the variable loop hits continue).
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var1 = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        var2 = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        ds1 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var1},
            global_attributes={},
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var2},
            global_attributes={},
        )
        assert ds1.identical(ds2), (
            "Both-None-data datasets should be identical"
        )

    def test_integer_nodata_mismatch_not_identical(self):
        """Non-float nodata comparison takes the else branch.

        Test scenario:
            ds1 has nodata=-9999 (int), ds2 has nodata=-1 (int);
            identical returns False via the else branch (line 655).
        """
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var_a = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10, 20], dtype=np.int32),
            nodata=-9999,
        )
        var_b = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
            _data=np.array([10, 20], dtype=np.int32),
            nodata=-1,
        )
        ds1 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_a},
            global_attributes={},
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var_b},
            global_attributes={},
        )
        assert not ds1.identical(ds2), (
            "Integer nodata mismatch should be non-identical"
        )

    def test_identical_one_data_none_not_identical(self):
        """One variable loaded, the other not — not identical.

        Test scenario:
            ds1 has data, ds2 has data=None; identical returns False.
        """
        ds1 = _make_ds()
        fnc = Connectivity(
            data=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=fnc,
        )
        var = MeshVariable(
            name="depth",
            location="face",
            mesh_name="mesh2d",
            shape=(2,),
        )
        ds2 = UgridDataset(
            mesh=mesh,
            data_variables={"depth": var},
            global_attributes={},
        )
        assert not ds1.identical(ds2), (
            "One-None-data should be non-identical"
        )

    def test_same_crs_both_set_identical(self):
        """Two datasets with the same CRS set via EPSG are identical.

        Test scenario:
            Both datasets set CRS via epsg=4326; identical() uses
            semantic CRS comparison (not WKT string).
        """
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        ds2.set_crs(epsg=4326)
        assert ds1.identical(ds2), (
            "Same EPSG should be identical"
        )

    def test_different_crs_both_set_not_identical(self):
        """Two datasets with different CRS are not identical.

        Test scenario:
            ds1 has EPSG 4326, ds2 has EPSG 32631; identical()
            returns False via semantic CRS comparison.
        """
        ds1 = _make_ds()
        ds1.set_crs(epsg=4326)
        ds2 = _make_ds()
        ds2.set_crs(epsg=32631)
        assert not ds1.identical(ds2), (
            "Different EPSG should not be identical"
        )

    def test_both_crs_none_identical(self):
        """Two datasets with no CRS are identical.

        Test scenario:
            Neither dataset has a CRS; identical() falls back to
            string comparison of _crs_wkt (both None).
        """
        ds1 = _make_ds()
        ds2 = _make_ds()
        assert ds1.identical(ds2), (
            "Both None CRS should be identical"
        )

    def test_non_ugrid_returns_false(self):
        assert not _make_ds().identical(42)
