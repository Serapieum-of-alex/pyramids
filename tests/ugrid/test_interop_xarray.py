"""Tests for the standalone UGRID xarray / Zarr / dask interop module.

Exercises :mod:`pyramids.netcdf.ugrid.interop_xarray`: the ``to_xarray`` / ``from_xarray`` round trip
(including the connectivity ``start_index`` / ``_FillValue`` encoding trap), the Zarr round trip, and
the dask ``chunked_variables`` lazy path.

``xarray`` is pulled at module scope via ``pytest.importorskip`` and the module carries the
``interop`` marker, so the whole file skips when the optional interop stack is absent (guard
:mod:`tests.guards.test_marker_hygiene`). The Zarr / dask tests add ``@pytest.mark.lazy`` and an
inner ``importorskip`` so they also skip without the ``[lazy]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.errors import OptionalPackageDoesNotExist
from pyramids.netcdf.ugrid import interop_xarray
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

_SAMPLE_NC = Path(__file__).parents[1] / "data" / "netcdf" / "ugrid" / "ugrid.nc"


def _mesh_with_two_variables() -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Build a 2-triangle mesh plus a static and a temporal face variable.

    The temporal variable carries explicit ``dimensions=("time", ...)`` so it is detected as temporal
    without relying on the leading-axis fallback (the ``from_arrays`` trap in S.3).

    Returns:
        A ``(mesh, data_variables)`` pair with a static ``depth`` and a temporal ``level``.
    """
    node_x = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    faces = np.array([[0, 1, 2], [1, 4, 3]], dtype=np.intp)
    mesh = Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    static = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(2,),
        units="m",
        nodata=-999.0,
        dimensions=("mesh2d_nFaces",),
        _data=np.array([1.5, 2.5]),
    )
    temporal = MeshVariable(
        name="level",
        location="face",
        mesh_name="mesh2d",
        shape=(3, 2),
        units="m",
        dimensions=("time", "mesh2d_nFaces"),
        _data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    return mesh, {"depth": static, "level": temporal}


class TestToFromXarray:
    """The ``to_xarray`` / ``from_xarray`` round trip."""

    def test_round_trips_nodes_faces_and_values(self):
        """``from_xarray(to_xarray(...))`` reproduces nodes, faces and both variables exactly."""
        mesh, data_vars = _mesh_with_two_variables()

        ds = interop_xarray.to_xarray(mesh, data_vars)
        assert isinstance(ds, xr.Dataset)

        mesh2, data_vars2 = interop_xarray.from_xarray(ds)

        np.testing.assert_array_equal(mesh2.node_x, mesh.node_x)
        np.testing.assert_array_equal(mesh2.node_y, mesh.node_y)
        np.testing.assert_array_equal(
            mesh2.face_node_connectivity.data, mesh.face_node_connectivity.data
        )
        assert mesh2.n_node == mesh.n_node
        assert mesh2.n_face == mesh.n_face

        assert set(data_vars2) == {"depth", "level"}
        np.testing.assert_array_equal(data_vars2["depth"].data, data_vars["depth"].data)
        np.testing.assert_array_equal(data_vars2["level"].data, data_vars["level"].data)

    def test_preserves_variable_metadata_and_time(self):
        """Location, units, nodata and the temporal flag survive the round trip."""
        mesh, data_vars = _mesh_with_two_variables()

        mesh2, data_vars2 = interop_xarray.from_xarray(
            interop_xarray.to_xarray(mesh, data_vars)
        )

        depth = data_vars2["depth"]
        level = data_vars2["level"]
        assert depth.location == "face"
        assert depth.units == "m"
        assert depth.nodata == -999.0
        assert depth.has_time is False
        assert level.has_time is True
        assert level.time_index == 0
        # Element axis stays last.
        assert level.data.shape[-1] == mesh.n_face

    def test_topology_variable_carries_cf_role(self):
        """The exported dataset holds a dummy ``mesh_topology`` variable with UGRID attributes."""
        mesh, data_vars = _mesh_with_two_variables()

        ds = interop_xarray.to_xarray(mesh, data_vars)

        assert ds["mesh2d"].attrs["cf_role"] == "mesh_topology"
        assert ds["mesh2d"].attrs["topology_dimension"] == 2
        assert ds["mesh2d"].attrs["node_coordinates"] == "mesh2d_node_x mesh2d_node_y"
        assert ds["mesh2d"].attrs["face_node_connectivity"] == "mesh2d_face_nodes"

    def test_does_not_mutate_inputs(self):
        """Exporting leaves the source mesh connectivity and variable arrays untouched."""
        mesh, data_vars = _mesh_with_two_variables()
        original_conn = mesh.face_node_connectivity.data.copy()
        original_depth = data_vars["depth"].data.copy()

        interop_xarray.to_xarray(mesh, data_vars)

        np.testing.assert_array_equal(mesh.face_node_connectivity.data, original_conn)
        np.testing.assert_array_equal(data_vars["depth"].data, original_depth)

    def test_from_xarray_rejects_non_dataset(self):
        """A non-``xarray.Dataset`` argument raises ``TypeError``."""
        with pytest.raises(TypeError, match="xarray.Dataset"):
            interop_xarray.from_xarray({"not": "a dataset"})


class TestConnectivityEncoding:
    """The connectivity ``start_index`` / ``_FillValue`` round-trip trap."""

    def test_start_index_and_fill_restored_on_disk(self):
        """A 1-indexed, filled connectivity is emitted with on-disk ``start_index`` / ``_FillValue``."""
        node_x = np.arange(6, dtype=np.float64)
        node_y = np.zeros(6, dtype=np.float64)
        # Internal form: 0-indexed with -1 fill (a quad and a triangle).
        internal = np.array([[0, 1, 4, 3], [1, 2, 5, -1]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=internal,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=1,
            ),
        )

        ds = interop_xarray.to_xarray(mesh, {})
        face_nodes = ds["mesh2d_face_nodes"]

        assert face_nodes.attrs["start_index"] == 1
        assert face_nodes.attrs["_FillValue"] == -999
        # Valid entries shifted by +1; the -1 fill became the file fill -999.
        expected = np.array([[1, 2, 5, 4], [2, 3, 6, -999]])
        np.testing.assert_array_equal(face_nodes.values, expected)

        # And the inverse restores the internal 0-indexed / -1-fill form exactly.
        mesh2, _ = interop_xarray.from_xarray(ds)
        np.testing.assert_array_equal(mesh2.face_node_connectivity.data, internal)
        assert mesh2.face_node_connectivity.original_start_index == 1
        assert mesh2.face_node_connectivity.fill_value == -1


class TestMissingXarray:
    """Guarded import wiring when xarray is unavailable."""

    def test_to_xarray_raises_optional_package(self, monkeypatch):
        """A missing xarray surfaces as ``OptionalPackageDoesNotExist`` from ``to_xarray``."""

        def _boom(_message):
            raise OptionalPackageDoesNotExist("xarray missing")

        monkeypatch.setattr(interop_xarray, "import_xarray", _boom)
        mesh, data_vars = _mesh_with_two_variables()
        with pytest.raises(OptionalPackageDoesNotExist):
            interop_xarray.to_xarray(mesh, data_vars)


@pytest.mark.lazy
class TestZarrRoundTrip:
    """The Zarr round trip, skipped without the ``[lazy]`` extra."""

    def test_to_zarr_from_zarr(self, tmp_path):
        """A mesh and its variables round-trip through a Zarr store on disk."""
        pytest.importorskip("zarr")
        mesh, data_vars = _mesh_with_two_variables()
        store = str(tmp_path / "mesh.zarr")

        interop_xarray.to_zarr(mesh, data_vars, store)
        mesh2, data_vars2 = interop_xarray.from_zarr(store)

        np.testing.assert_array_equal(mesh2.node_x, mesh.node_x)
        np.testing.assert_array_equal(
            mesh2.face_node_connectivity.data, mesh.face_node_connectivity.data
        )
        np.testing.assert_array_equal(data_vars2["depth"].data, data_vars["depth"].data)
        np.testing.assert_array_equal(data_vars2["level"].data, data_vars["level"].data)
        assert data_vars2["level"].has_time is True


@pytest.mark.lazy
class TestChunkedVariables:
    """The dask ``chunked_variables`` lazy path, skipped without the ``[lazy]`` extra."""

    def test_chunked_read_computes_matching_reduction(self):
        """A chunked read of the sample file yields dask arrays whose reduction matches eager."""
        da = pytest.importorskip("dask.array")
        ug = UgridDataset.read_file(str(_SAMPLE_NC))
        data_vars = ug.data_vars

        chunked = interop_xarray.chunked_variables(data_vars, chunks="auto")

        assert chunked, "expected at least one chunked variable from the sample file"
        name = next(iter(chunked))
        lazy = chunked[name]
        assert isinstance(lazy, da.Array)

        eager = np.asarray(ug.get_data(name).data, dtype="float64")
        np.testing.assert_allclose(
            float(lazy.astype("float64").mean().compute()), float(np.nanmean(eager))
        )

    def test_in_memory_variable_is_wrapped_as_dask(self):
        """An in-memory variable (no source path) is wrapped via ``dask.array.from_array``."""
        da = pytest.importorskip("dask.array")
        _, data_vars = _mesh_with_two_variables()

        chunked = interop_xarray.chunked_variables(data_vars, chunks=1)

        assert isinstance(chunked["level"], da.Array)
        np.testing.assert_array_equal(
            chunked["level"].compute(), data_vars["level"].data
        )
