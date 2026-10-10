"""The tidy frame on the variable shapes that are not "one value per face".

Two gaps are pinned. A node- or edge-located variable has one value per *node* or *edge*, so
addressing its rows with face centroids gave every row NaN coordinates. And a layered variable
was flattened with `ravel()`, which invented element indices running past the element count and
dropped the layer identity entirely — `(3 layers, 2 faces)` became six rows numbered 0..5 with no
way to tell which layer a row came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _dataset(data: dict, locations: dict) -> UgridDataset:
    """A two-triangle mesh carrying the given variables."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data=data,
        data_locations=locations,
    )


@pytest.fixture
def node_variable() -> UgridDataset:
    """One value per node."""
    return _dataset({"h": np.array([1.0, 2.0, 3.0, 4.0])}, {"h": "node"})


@pytest.fixture
def layered() -> UgridDataset:
    """Three layers over two faces, with named dimensions."""
    dataset = _dataset({"salinity": np.arange(6.0).reshape(3, 2)}, {"salinity": "face"})
    dataset["salinity"].dimensions = ("mesh2d_nLayers", "mesh2d_nFaces")
    return dataset


class TestNodeLocatedVariables:
    def test_the_coordinates_are_the_node_coordinates(self, node_variable):
        frame = node_variable.to_dataframe()
        assert frame["x"].tolist() == NODE_X.tolist()
        assert frame["y"].tolist() == NODE_Y.tolist()

    def test_there_is_one_row_per_node(self, node_variable):
        assert len(node_variable.to_dataframe()) == 4

    def test_the_element_index_runs_over_the_nodes(self, node_variable):
        assert node_variable.to_dataframe()["element"].tolist() == [0, 1, 2, 3]

    def test_a_face_variable_still_uses_centroids(self):
        dataset = _dataset({"v": np.array([10.0, 20.0])}, {"v": "face"})
        frame = dataset.to_dataframe()
        centre_x, centre_y = dataset.mesh.face_centroids
        assert frame["x"].tolist() == pytest.approx(list(centre_x))
        assert frame["y"].tolist() == pytest.approx(list(centre_y))


class TestLayeredVariables:
    def test_the_element_index_stays_within_the_faces(self, layered):
        frame = layered.to_dataframe()
        assert sorted(set(frame["element"].tolist())) == [0, 1]

    def test_a_layer_column_identifies_the_layer(self, layered):
        frame = layered.to_dataframe()
        assert sorted(set(frame["layer"].tolist())) == [0, 1, 2]

    def test_every_layer_element_pair_appears_once(self, layered):
        frame = layered.to_dataframe()
        pairs = sorted(zip(frame["layer"].tolist(), frame["element"].tolist()))
        assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_the_values_follow_their_layer(self, layered):
        frame = layered.to_dataframe()
        row = frame[(frame["layer"] == 1) & (frame["element"] == 0)]
        assert row["value"].tolist() == [2.0]

    def test_the_coordinates_repeat_per_layer(self, layered):
        frame = layered.to_dataframe()
        assert frame["x"].notna().all()

    def test_an_unlayered_variable_has_no_layer_label(self):
        dataset = _dataset({"v": np.array([10.0, 20.0])}, {"v": "face"})
        frame = dataset.to_dataframe()
        assert frame["layer"].isna().all()


class TestTemporalAndLayered:
    def test_both_axes_are_labelled(self):
        dataset = _dataset({"v": np.zeros((2, 3, 2))}, {"v": "face"})
        dataset["v"].dimensions = ("time", "mesh2d_nLayers", "mesh2d_nFaces")
        frame = dataset.to_dataframe()
        assert len(frame) == 12
        assert sorted(set(frame["layer"].tolist())) == [0, 1, 2]
        assert sorted(set(frame["time"].tolist())) == [0, 1]

    def test_dropna_still_drops(self):
        dataset = _dataset({"v": np.array([np.nan, 20.0])}, {"v": "face"})
        assert len(dataset.to_dataframe(dropna=True)) == 1
