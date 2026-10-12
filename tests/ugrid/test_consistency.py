"""Consistency details: subclass preservation, the public CRS, 0-d export, module exports.

Small contracts that are easy to break silently and that nothing else pins.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyramids.netcdf.ugrid as ugrid
from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.interop_xarray import _variable_dims
from pyramids.netcdf.ugrid.mesh import Mesh2d

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


class _Derived(UgridDataset):
    """A subclass, to check derivations do not downgrade it."""


@pytest.fixture
def dataset() -> UgridDataset:
    """A two-triangle mesh with one face variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array([1.0, 2.0])},
    )


@pytest.fixture
def derived(dataset) -> _Derived:
    """The same dataset as a subclass instance, carrying a file name."""
    return _Derived(
        mesh=dataset.mesh,
        data_variables=dataset.data_vars,
        global_attributes={},
        topology_info=None,
        crs_wkt=None,
        file_name="source.nc",
    )


class TestDerivationsKeepTheSubclass:
    def test_copy_keeps_it(self, derived):
        assert type(derived.copy()) is _Derived

    def test_set_global_attribute_keeps_it(self, derived):
        assert type(derived.set_global_attribute("a", 1)) is _Derived

    def test_delete_global_attribute_keeps_it(self, derived):
        with_attr = derived.set_global_attribute("a", 1)
        assert type(with_attr.delete_global_attribute("a")) is _Derived

    def test_the_shared_rebuild_keeps_it(self, derived):
        assert type(derived.to_location("node")) is _Derived

    def test_the_rebuild_carries_the_file_name(self, derived):
        assert derived.to_location("node")._file_name == "source.nc"


class TestPublicCrs:
    def test_mesh2d_exposes_crs(self):
        mesh = Mesh2d(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=Connectivity(
                data=TRIANGLES,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            crs="EPSG:4326",
        )
        assert mesh.crs == "EPSG:4326"

    def test_it_is_none_when_absent(self, dataset):
        assert dataset.mesh.crs is None


class TestZeroDimensionalExport:
    def test_a_scalar_variable_gets_no_dimension_names(self, dataset):
        scalar = dataset["v"].with_data(np.asarray(3.0))
        assert _variable_dims(scalar, "mesh2d") == ()

    def test_a_one_d_variable_still_gets_the_element_dimension(self, dataset):
        assert _variable_dims(dataset["v"], "mesh2d") == ("mesh2d_nFaces",)


class TestModuleExports:
    @pytest.mark.parametrize(
        "name",
        [
            "construct",
            "fill",
            "graph",
            "interop_xarray",
            "network",
            "regrid",
            "vector_ops",
        ],
    )
    def test_each_capability_module_is_exported(self, name):
        assert name in ugrid.__all__
        assert hasattr(ugrid, name)


class TestEveryDerivationKeepsTheSubclassAndFileName:
    """`to_crs`, `sel_time` and `sel_time_range` were the siblings the conversion missed.

    The constructor's docstring says it is "what every derivation calls through
    `type(self)(...)`, which is how a subclass survives them". `_rebuild`, `copy`,
    `set_global_attribute`, `delete_global_attribute` and `_wrap_subset` all do. These three
    hard-coded `UgridDataset(`, so each silently downgraded a subclass -- and they were also
    the only derivations still dropping `file_name`, which `_wrap_subset` was changed to carry.
    """

    class Derived(UgridDataset):
        """A subclass that must survive every derivation."""

    @pytest.fixture
    def derived(self, tmp_path):
        """A `Derived` instance with a time axis, a CRS and a `file_name`."""
        dataset = self.Derived.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"v": np.array([[1.0, 2.0], [3.0, 4.0]])},
            data_locations={"v": "face"},
            epsg=4326,
        )
        dataset._file_name = str(tmp_path / "source.nc")
        return dataset

    def test_to_crs_keeps_the_subclass(self, derived):
        """Reprojection is a derivation like any other."""
        assert type(derived.to_crs(3857)) is self.Derived

    def test_sel_time_keeps_the_subclass(self, derived):
        """Selecting a single step keeps the type."""
        assert type(derived.sel_time(0)) is self.Derived

    def test_sel_time_range_keeps_the_subclass(self, derived):
        """So does selecting a range."""
        assert type(derived.sel_time_range(0, 2)) is self.Derived

    @pytest.mark.parametrize(
        "derive",
        [
            lambda ds: ds.to_crs(3857),
            lambda ds: ds.sel_time(0),
            lambda ds: ds.sel_time_range(0, 2),
        ],
        ids=["to_crs", "sel_time", "sel_time_range"],
    )
    def test_the_file_name_is_carried(self, derive, derived):
        """The path the dataset came from survives, as it does through `isel`.

        Args:
            derive: The derivation to apply.
            derived: The fixture dataset.

        Test scenario:
            `file_name` is what `to_file` and `repr` name, so losing it on one derivation and
            not another makes the attribute unreliable rather than absent.
        """
        assert derive(derived)._file_name == derived._file_name, (
            "the source path should survive the derivation"
        )
