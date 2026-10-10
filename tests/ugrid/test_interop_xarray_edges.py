"""Edge cases of the UGRID-CF xarray bridge: edges, face coords, CRS, dim fallbacks.

`test_interop_xarray.py` covers the happy round trip and the connectivity encoding. This file
covers the optional parts of the encoding, which are the branches a mesh built the long way
(with edge connectivity, face centres or a CRS) actually exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.interop_xarray import from_xarray, read_crs_wkt, to_xarray

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop


def _square_mesh(epsg: int = 4326) -> UgridDataset:
    """A unit square split into two triangles, carrying one face variable."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"depth": np.array([2.0, 4.0])},
        epsg=epsg,
    )


class TestEdgeConnectivity:
    def test_derived_edges_are_exported_and_read_back(self):
        mesh = _square_mesh()
        mesh.mesh.build_edge_connectivity()
        assert mesh.mesh.edge_node_connectivity is not None
        exported = to_xarray(mesh.mesh, mesh.data_vars)
        assert any("edge_nodes" in name for name in exported.data_vars)
        back, _ = from_xarray(exported)
        assert back.edge_node_connectivity is not None
        np.testing.assert_array_equal(
            np.asarray(back.edge_node_connectivity.data),
            np.asarray(mesh.mesh.edge_node_connectivity.data),
        )

    def test_topology_attrs_name_the_edge_connectivity(self):
        mesh = _square_mesh()
        mesh.mesh.build_edge_connectivity()
        exported = to_xarray(mesh.mesh, mesh.data_vars)
        topology = [
            var
            for var in exported.variables.values()
            if var.attrs.get("cf_role") == "mesh_topology"
        ][0]
        assert "edge_node_connectivity" in topology.attrs

    def test_a_mesh_without_edges_omits_them(self):
        mesh = _square_mesh()
        exported = to_xarray(mesh.mesh, mesh.data_vars)
        assert not any("edge_nodes" in name for name in exported.data_vars)


class TestCrsRoundTrip:
    def test_crs_variable_is_written_and_read(self):
        mesh = _square_mesh(epsg=3857)
        exported = to_xarray(mesh.mesh, mesh.data_vars, crs_wkt=mesh.crs_wkt)
        assert "crs" in exported.variables
        assert read_crs_wkt(exported) is not None
        back, _ = from_xarray(exported)
        assert back.n_face == 2

    def test_values_survive_a_crs_carrying_round_trip(self):
        mesh = _square_mesh(epsg=3857)
        _, variables = from_xarray(
            to_xarray(mesh.mesh, mesh.data_vars, crs_wkt=mesh.crs_wkt)
        )
        assert np.asarray(variables["depth"].data).tolist() == [2.0, 4.0]

    def test_the_dataset_facade_round_trips_the_epsg(self):
        # A UgridDataset keeps its CRS on the dataset, not on the Mesh2d, so the export has to
        # be handed the dataset's WKT or the CRS is silently lost.
        mesh = _square_mesh(epsg=3857)
        back = UgridDataset.from_xarray(mesh.to_xarray())
        assert back.epsg == 3857

    def test_a_mesh_without_a_crs_exports_none(self):
        bare = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
        )
        exported = to_xarray(bare.mesh, bare.data_vars, crs_wkt=None)
        assert "crs" not in exported.variables


class TestTopologyName:
    def test_the_mesh_name_is_recovered(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([1.0, 2.0])},
            mesh_name="my_mesh",
        )
        exported = to_xarray(mesh.mesh, mesh.data_vars, mesh_name="my_mesh")
        names = [
            name
            for name, var in exported.variables.items()
            if var.attrs.get("cf_role") == "mesh_topology"
        ]
        assert names == ["my_mesh"]

    def test_a_dataset_without_a_topology_variable_is_refused_clearly(self):
        # Dropping the dummy topology variable makes the dataset non-UGRID. The reader must
        # say so, rather than surfacing a bare KeyError naming an internal variable name.
        mesh = _square_mesh()
        exported = to_xarray(mesh.mesh, mesh.data_vars)
        stripped = exported.copy()
        for name, var in list(stripped.variables.items()):
            if var.attrs.get("cf_role") == "mesh_topology":
                stripped = stripped.drop_vars(name)
        with pytest.raises(ValueError, match="not a UGRID dataset"):
            from_xarray(stripped)


class TestDimensionNaming:
    def test_a_temporal_variable_names_its_time_axis(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([[1.0, 2.0], [3.0, 4.0]])},
            data_locations={"d": "face"},
        )
        exported = to_xarray(mesh.mesh, mesh.data_vars)
        assert "time" in exported["d"].dims

    def test_a_variable_with_no_declared_dimensions_still_exports(self):
        mesh = _square_mesh()
        variables = dict(mesh.data_vars)
        variables["depth"].dimensions = ()
        exported = to_xarray(mesh.mesh, variables)
        assert exported["depth"].shape == (2,)
        back, read = from_xarray(exported)
        assert np.asarray(read["depth"].data).tolist() == [2.0, 4.0]


class TestInputIsNotMutated:
    def test_to_xarray_leaves_the_mesh_and_variables_alone(self):
        mesh = _square_mesh()
        before_nodes = np.array(mesh.mesh.node_x, copy=True)
        before_values = np.array(mesh["depth"].data, copy=True)
        before_conn = np.array(mesh.mesh.face_node_connectivity.data, copy=True)
        to_xarray(mesh.mesh, mesh.data_vars)
        np.testing.assert_array_equal(mesh.mesh.node_x, before_nodes)
        np.testing.assert_array_equal(mesh["depth"].data, before_values)
        np.testing.assert_array_equal(
            mesh.mesh.face_node_connectivity.data, before_conn
        )
