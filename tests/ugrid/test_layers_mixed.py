"""Layer handling on a mesh that carries both layer centres and layer interfaces.

Canonical 3-D model output puts some variables on the n layer centres and others on the n+1
layer interfaces. Those counts legitimately differ, so `n_layers` must not treat the pair as a
contradiction — it previously raised, which took `sel_layer` down with it while `isel_layer`,
which indexes each variable on its own layer axis, kept working.

Genuinely inconsistent layer counts are still an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _layered(counts: dict[str, int]) -> UgridDataset:
    """A mesh whose variables sit on the given layer counts, with explicit dimension names."""
    data = {
        name: np.arange(float(count * 2)).reshape(count, 2)
        for name, count in counts.items()
    }
    dataset = UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data=data,
        data_locations=dict.fromkeys(counts, "face"),
    )
    for name in counts:
        dataset[name].dimensions = ("mesh2d_nLayers", "mesh2d_nFaces")
    return dataset


@pytest.fixture
def centres_and_interfaces() -> UgridDataset:
    """Three layer centres plus four layer interfaces, as a 3-D model writes them."""
    return _layered({"salinity": 3, "w": 4})


class TestMixedLayerCounts:
    def test_n_layers_reports_the_centre_count(self, centres_and_interfaces):
        assert centres_and_interfaces.n_layers == 3

    def test_sel_layer_works(self, centres_and_interfaces):
        picked = centres_and_interfaces.sel_layer(1)
        assert np.asarray(picked["salinity"].data).shape == (2,)

    def test_isel_layer_still_indexes_each_variable(self, centres_and_interfaces):
        picked = centres_and_interfaces.isel_layer(0)
        assert np.asarray(picked["salinity"].data).shape == (2,)
        assert np.asarray(picked["w"].data).shape == (2,)

    def test_a_slice_trims_both_variables(self, centres_and_interfaces):
        trimmed = centres_and_interfaces.isel_layer(slice(0, 2))
        assert np.asarray(trimmed["salinity"].data).shape == (2, 2)
        assert np.asarray(trimmed["w"].data).shape == (2, 2)

    def test_genuinely_inconsistent_counts_still_raise(self):
        with pytest.raises(ValueError, match="disagree on their layer count"):
            _ = _layered({"a": 3, "b": 7}).n_layers

    def test_a_single_layer_count_is_unchanged(self):
        assert _layered({"a": 5}).n_layers == 5


class TestLayerAxisDetection:
    def test_a_temporal_layered_variable_finds_the_middle_axis(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.zeros((2, 3, 2))},
            data_locations={"v": "face"},
        )
        dataset["v"].dimensions = ("time", "mesh2d_nLayers", "mesh2d_nFaces")
        assert dataset.n_layers == 3

    def test_a_named_non_temporal_two_d_variable_is_layered(self):
        assert _layered({"a": 4}).n_layers == 4

    def test_a_variable_whose_time_dimension_is_last_has_no_layer_axis(self):
        # Time in the element slot breaks the [time?, layer?, element] layout, so no axis can
        # be claimed as the layer axis — reporting axis 0 would name the element axis.
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.zeros((2, 3))},
            data_locations={"v": "face"},
        )
        dataset["v"].dimensions = ("mesh2d_nFaces", "time")
        assert dataset.n_layers is None
