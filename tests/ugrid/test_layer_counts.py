"""`n_layers` on a 3-D output, where two counts differing by one is the normal shape.

A 3-D mesh output puts scalars on the `n` layer **centres** and vertical fluxes on the `n + 1`
layer **interfaces**, so a file carrying both 20 and 21 is correct rather than contradictory.
`n_layers` reports 20, the layer count proper -- which is right, and was also the whole of the
information available: nothing said a variable in the dataset had 21 layers, so
`range(ds.n_layers)` over an interface variable silently skipped its last interface.
`n_layer_interfaces` makes that visible instead of leaving it to be inferred.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _layered(counts):
    """A mesh whose variables carry the layer counts in `counts` (name -> n_layers)."""
    dataset = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"anchor": np.array([1.0, 2.0])},
        data_locations={"anchor": "face"},
    )
    for name, n_layers in counts.items():
        dataset = dataset.with_variable(name, np.zeros((n_layers, 2)), location="face")
        dataset[name].dimensions = ("nLayers", "mesh2d_nFaces")
    return dataset


class TestLayerCentreCount:
    """What `n_layers` answers."""

    def test_no_layered_variable_answers_none(self):
        """A dataset with only per-element variables has no layer count."""
        dataset = _layered({})

        assert dataset.n_layers is None, f"expected None, got {dataset.n_layers!r}"

    def test_a_single_count_is_that_count(self):
        """Centres only: the obvious case."""
        assert _layered({"temp": 20}).n_layers == 20

    def test_the_centre_interface_pair_reports_the_centre_count(self):
        """20 centres and 21 interfaces is a 20-layer mesh.

        Test scenario:
            This is the shape every 3-D unstructured output has -- scalars on the centres,
            vertical velocity and fluxes on the interfaces. Raising on it (the original
            behaviour) made `n_layers` unusable on a real file; reporting 21 would name a
            layer count the mesh does not have.
        """
        assert _layered({"temp": 20, "w": 21}).n_layers == 20

    def test_the_pair_is_order_independent(self):
        """The interface variable coming first changes nothing."""
        assert _layered({"w": 21, "temp": 20}).n_layers == 20

    def test_a_real_disagreement_still_raises(self):
        """Two counts that are not a centre/interface pair are an inconsistency.

        Test scenario:
            The tolerance must stay narrow: `{20, 25}` cannot be a centre/interface pair, so
            there is no count to report and guessing one would be worse than refusing.
        """
        with pytest.raises(ValueError, match="disagree on their layer count"):
            _ = _layered({"temp": 20, "other": 25}).n_layers

    def test_three_consecutive_counts_also_raise(self):
        """`{20, 21, 22}` is not a pair either, though each neighbour differs by one.

        Test scenario:
            The guard is `counts - {smallest, smallest + 1}`, not "every gap is 1". A mesh
            cannot have both 21 and 22 interfaces for 20 centres, so this is a real
            inconsistency and the narrow test is what catches it.
        """
        with pytest.raises(ValueError, match="disagree on their layer count"):
            _ = _layered({"a": 20, "b": 21, "c": 22}).n_layers


class TestLayerInterfaceCount:
    """`n_layer_interfaces`, so the pair is discoverable rather than inferred."""

    def test_it_reports_the_interface_count_when_one_exists(self):
        """20 centres and 21 interfaces: both counts are now readable."""
        dataset = _layered({"temp": 20, "w": 21})

        assert dataset.n_layers == 20
        assert dataset.n_layer_interfaces == 21, (
            f"the interface count should be visible, got {dataset.n_layer_interfaces!r}"
        )

    def test_it_is_none_when_nothing_sits_on_interfaces(self):
        """Centres only: there is no interface count to report.

        Test scenario:
            `n_layers + 1` would always be *computable*, but reporting it would assert that
            something in the dataset sits on the interfaces. Only a variable that actually
            carries that count makes it true.
        """
        dataset = _layered({"temp": 20})

        assert dataset.n_layers == 20
        assert dataset.n_layer_interfaces is None, (
            f"nothing sits on interfaces, so expected None, got "
            f"{dataset.n_layer_interfaces!r}"
        )

    def test_it_is_none_with_no_layered_variable(self):
        """No layers at all means no interfaces either."""
        assert _layered({}).n_layer_interfaces is None

    def test_it_raises_on_the_same_inconsistency_as_n_layers(self):
        """The two properties agree about what is readable.

        Test scenario:
            If `n_layers` cannot name a count, neither can this -- answering a number from an
            inconsistent set would let a caller past the guard the other property applies.
        """
        with pytest.raises(ValueError, match="disagree on their layer count"):
            _ = _layered({"temp": 20, "other": 25}).n_layer_interfaces
