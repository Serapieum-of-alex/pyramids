"""Remapping a variable between mesh facets.

Two things are pinned here. A remap has to move the variable's *dimension name* as well as its
`location`, because a file-read variable carries real dimension names and the xarray bridge
prefers them — leaving the old facet's name on an axis that now has the new facet's length makes
the export fail on conflicting dimension sizes. And a remap onto the facet a variable already
occupies is trivially satisfied, so it must be a no-op rather than an error from the averaging
kernel, which only accepts the two *other* facets as a source.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _dataset(values: list[float], location: str) -> UgridDataset:
    """A two-triangle mesh with one variable on `location`."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array(values)},
        data_locations={"v": location},
    )


@pytest.fixture
def face_variable() -> UgridDataset:
    """A face variable carrying the dimension names a file read would give it."""
    dataset = _dataset([10.0, 20.0], "face")
    dataset["v"].dimensions = ("mesh2d_nFaces",)
    return dataset


class TestDimensionNamesFollowTheLocation:
    def test_the_element_dimension_is_renamed(self, face_variable):
        moved = face_variable.to_location("node")
        assert moved["v"].dimensions == ("mesh2d_nNodes",)

    def test_the_export_no_longer_conflicts(self, face_variable):
        pytest.importorskip("xarray")
        exported = face_variable.to_location("node").to_xarray()
        assert exported["v"].shape == (4,)

    def test_remapping_to_edges_renames_too(self, face_variable):
        moved = face_variable.to_location("edge")
        assert moved["v"].dimensions == ("mesh2d_nEdges",)

    def test_a_variable_without_dimensions_is_left_alone(self):
        moved = _dataset([10.0, 20.0], "face").to_location("node")
        assert moved["v"].dimensions == ()

    def test_the_values_are_the_element_averages(self, face_variable):
        moved = face_variable.to_location("node")
        # Node 0 and node 2 touch both triangles; nodes 1 and 3 touch one each.
        assert np.asarray(moved["v"].data)[1] == pytest.approx(10.0)
        assert np.asarray(moved["v"].data)[3] == pytest.approx(20.0)


class TestIdentityRemapIsANoOp:
    def test_face_to_face_keeps_the_values(self):
        dataset = _dataset([10.0, 20.0], "face")
        same = dataset.to_location("face")
        assert np.asarray(same["v"].data).tolist() == [10.0, 20.0]
        assert same["v"].location == "face"

    def test_node_to_node_keeps_the_values(self):
        dataset = _dataset([1.0, 2.0, 3.0, 4.0], "node")
        same = dataset.to_location("node")
        assert np.asarray(same["v"].data).tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_the_result_is_a_new_dataset(self):
        dataset = _dataset([10.0, 20.0], "face")
        assert dataset.to_location("face") is not dataset

    def test_an_unknown_location_is_still_refused(self):
        dataset = _dataset([10.0, 20.0], "face")
        with pytest.raises(ValueError, match="location must be one of"):
            dataset.to_location("vertex")
