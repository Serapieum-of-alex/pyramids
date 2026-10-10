"""`to_xarray` output must survive being written to a real NetCDF file and read back.

The in-memory `to_xarray` -> `from_xarray` round trip was covered, but the export had never been
persisted and re-read. Two things broke when it was:

- the dummy topology and grid-mapping variables were true 0-d scalars, and GDAL's
  multidimensional reader does not expose a dimensionless variable as an array, so the file read
  back as "no UGRID mesh topology found";
- every data variable carried the mesh name the *variable* remembered rather than the topology
  being written, so exporting under another `mesh_name` left them pointing at a topology the file
  does not contain and the reader skipped them all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.interop_xarray import to_xarray

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def dataset() -> UgridDataset:
    """A two-triangle mesh with a CRS and one face variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array([1.0, 2.0])},
        epsg=3857,
    )


def _written(dataset: UgridDataset, path: Path, mesh_name: str) -> Path:
    """Export through xarray to a NetCDF file on disk."""
    exported = to_xarray(
        dataset.mesh, dataset.data_vars, mesh_name=mesh_name, crs_wkt=dataset.crs_wkt
    )
    exported.to_netcdf(path)
    return path


def _two_topologies(path: Path) -> Path:
    """A single file declaring two independent mesh topologies."""
    parts = []
    for name, offset in (("mesh_a", 0.0), ("mesh_b", 10.0)):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X + offset,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={f"v_{name}": np.array([1.0, 2.0])},
        )
        parts.append(to_xarray(dataset.mesh, dataset.data_vars, mesh_name=name))
    xr.merge(parts).to_netcdf(path)
    return path


class TestFileRoundTrip:
    def test_the_topology_is_found(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "mesh.nc", "mesh_a")
        assert UgridDataset.mesh_names(path) == ["mesh_a"]

    def test_the_geometry_comes_back(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "mesh.nc", "mesh_a")
        back = UgridDataset.read_file(path)
        assert (back.n_face, back.n_node) == (2, 4)

    def test_the_data_variable_comes_back(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "mesh.nc", "mesh_a")
        back = UgridDataset.read_file(path)
        assert list(back) == ["v"]

    def test_the_values_come_back(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "mesh.nc", "mesh_a")
        back = UgridDataset.read_file(path)
        assert np.asarray(back["v"].data).tolist() == [1.0, 2.0]

    def test_the_crs_comes_back(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "mesh.nc", "mesh_a")
        assert UgridDataset.read_file(path).epsg == 3857

    def test_the_default_mesh_name_also_round_trips(self, dataset, tmp_path):
        path = _written(dataset, tmp_path / "default.nc", "mesh2d")
        back = UgridDataset.read_file(path)
        assert list(back) == ["v"]
        assert back.n_face == 2

    def test_the_data_variable_names_the_written_topology(self, dataset, tmp_path):
        exported = to_xarray(dataset.mesh, dataset.data_vars, mesh_name="mesh_a")
        assert exported["v"].attrs["mesh"] == "mesh_a"


class TestMultipleTopologies:
    def test_mesh_names_lists_both(self, tmp_path):
        path = _two_topologies(tmp_path / "two.nc")
        assert UgridDataset.mesh_names(path) == ["mesh_a", "mesh_b"]

    def test_read_all_returns_one_dataset_each(self, tmp_path):
        path = _two_topologies(tmp_path / "two.nc")
        everything = UgridDataset.read_all(path)
        assert sorted(everything) == ["mesh_a", "mesh_b"]

    def test_each_dataset_carries_only_its_own_variable(self, tmp_path):
        path = _two_topologies(tmp_path / "two.nc")
        everything = UgridDataset.read_all(path)
        assert list(everything["mesh_a"]) == ["v_mesh_a"]
        assert list(everything["mesh_b"]) == ["v_mesh_b"]

    def test_reading_one_by_name_picks_that_topology(self, tmp_path):
        path = _two_topologies(tmp_path / "two.nc")
        assert list(UgridDataset.read_file(path, mesh_name="mesh_b")) == ["v_mesh_b"]
