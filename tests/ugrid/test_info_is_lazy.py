"""`info()` must describe a dataset without reading it.

A lazily-read variable loads its array on first touch of `.data`. `info()` only prints a summary,
so touching `.data` to learn the shape pulled every variable's full array off disk — the opposite
of what a summary is for. The declared `shape` and `has_data_source` answer the same questions
without a read.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.models import MeshVariable

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


class _CountingLoader:
    """A lazy loader that records how many times it was asked for the data."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.calls = 0

    def __call__(self) -> np.ndarray:
        self.calls += 1
        return self.values


def _with_variables(variables: dict[str, MeshVariable]) -> UgridDataset:
    """A dataset carrying exactly the given variables, bypassing from_arrays."""
    template = UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"seed": np.array([1.0, 2.0])},
    )
    return UgridDataset(
        mesh=template.mesh,
        data_variables=variables,
        global_attributes={},
        topology_info=None,
        crs_wkt=None,
        file_name=None,
    )


@pytest.fixture
def lazy_dataset() -> tuple[UgridDataset, _CountingLoader]:
    """A dataset whose single variable is read through a counting loader."""
    loader = _CountingLoader(np.array([1.0, 2.0]))
    lazy = MeshVariable(
        name="v",
        location="face",
        mesh_name="mesh2d",
        shape=(2,),
        _loader=loader,
    )
    return _with_variables({"v": lazy}), loader


class TestInfoDoesNotLoad:
    def test_the_loader_is_never_called(self, lazy_dataset):
        dataset, loader = lazy_dataset
        dataset.info(buf=io.StringIO())
        assert loader.calls == 0

    def test_the_declared_shape_is_still_printed(self, lazy_dataset):
        dataset, _ = lazy_dataset
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        assert "shape=(2,)" in buffer.getvalue()

    def test_the_variable_is_still_listed(self, lazy_dataset):
        dataset, _ = lazy_dataset
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        assert "v: location=face" in buffer.getvalue()

    def test_a_variable_with_no_source_is_marked(self):
        empty = MeshVariable(name="w", location="face", mesh_name="mesh2d", shape=(2,))
        buffer = io.StringIO()
        _with_variables({"w": empty}).info(buf=buffer)
        assert "unloaded" in buffer.getvalue()

    def test_an_eager_variable_prints_its_shape(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([1.0, 2.0])},
        )
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        assert "shape=(2,)" in buffer.getvalue()

    def test_the_header_still_names_the_mesh(self, lazy_dataset):
        dataset, _ = lazy_dataset
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        assert "UgridDataset: mesh2d" in buffer.getvalue()
