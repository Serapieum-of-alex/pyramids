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
