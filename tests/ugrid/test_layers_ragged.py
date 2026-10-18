"""Ragged (per-column) vertical layers for a genuinely 3-D mesh (issue #1330).

A 3-D hydrodynamic mesh routinely has a different number of active layers per column — a deep
channel column with ten layers, a shallow one with two. That data is ragged, a flat
``(time?, n_volume)`` array indexed through CSR offsets, not a rectangular ``(layer, face)``
block. These tests pin the :class:`VolumeTopology` value object, the ragged-aware accessors
(``isel_layer`` making absent layers explicit, ``to_location`` collapsing to 2-D), and the
NetCDF round-trip. The uniform-layer path must stay untouched when no volume topology is set.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, VolumeTopology
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _mesh() -> Mesh2d:
    """A 3-face strip mesh with face centres, the columns of the ragged topology."""
    node_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    node_y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    fnc = Connectivity(
        data=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.intp),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=fnc,
        face_x=np.array([1.0, 2.0, 3.0]),
        face_y=np.array([0.3, 0.6, 0.3]),
    )


def _topology() -> VolumeTopology:
    """Columns with 3, 1 and 2 layers — six volumes."""
    return VolumeTopology(
        layer_count=np.array([3, 1, 2]),
        volume_offset=np.array([0, 3, 4, 6]),
    )


def _volume_dataset(data=None, topology=None) -> UgridDataset:
    topology = topology or _topology()
    data = np.arange(6.0) if data is None else data
    dims = ("mesh2d_nVolumes",) if data.ndim == 1 else ("time", "mesh2d_nVolumes")
    var = MeshVariable(
        name="sal",
        location="volume",
        mesh_name="mesh2d",
        shape=data.shape,
        dimensions=dims,
        _data=data,
    )
    return UgridDataset(_mesh(), {"sal": var}, {}, volume_topology=topology)


class TestVolumeTopologyValidation:
    """The CSR invariants are enforced at build so a bad topology fails early."""

    def test_valid_topology_exposes_its_counts(self):
        topology = _topology()
        assert topology.n_face == 3
        assert topology.n_volume == 6
        assert topology.max_layers == 3

    def test_offset_length_must_be_n_face_plus_one(self):
        with pytest.raises(ValueError, match="n_face \\+ 1"):
            VolumeTopology(
                layer_count=np.array([3, 1, 2]), volume_offset=np.array([0, 3, 6])
            )

    def test_offset_must_start_at_zero(self):
        with pytest.raises(ValueError, match="start at 0"):
            VolumeTopology(
                layer_count=np.array([3, 1, 2]), volume_offset=np.array([1, 4, 5, 7])
            )

    def test_offset_must_be_non_decreasing(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            VolumeTopology(
                layer_count=np.array([3, -1, 2]), volume_offset=np.array([0, 3, 2, 4])
            )

    def test_layer_count_must_match_offset_gaps(self):
        with pytest.raises(ValueError, match="layer_count must equal"):
            VolumeTopology(
                layer_count=np.array([3, 2, 2]), volume_offset=np.array([0, 3, 4, 6])
            )

    def test_arrays_must_be_one_dimensional(self):
        with pytest.raises(ValueError, match="1-D arrays"):
            VolumeTopology(
                layer_count=np.array([[3, 1, 2]]), volume_offset=np.array([0, 3, 4, 6])
            )


class TestVolumeTopologyMaps:
    """The face <-> volume maps the accessors build on."""

    def test_volumes_of_face(self):
        topology = _topology()
        assert topology.volumes_of_face(0) == slice(0, 3)
        assert topology.volumes_of_face(2) == slice(4, 6)

    def test_face_and_layer_of_volume(self):
        topology = _topology()
        assert topology.face_of_volume().tolist() == [0, 0, 0, 1, 2, 2]
        assert topology.layer_of_volume().tolist() == [0, 1, 2, 0, 0, 1]

    def test_volume_index_present_and_absent(self):
        topology = _topology()
        assert topology.volume_index(0, 2) == 2
        assert topology.volume_index(2, 0) == 4
        assert topology.volume_index(1, 1) is None  # column 1 has one layer


class TestDatasetAccessors:
    """What a dataset exposes once a volume topology is present."""

    def test_volume_topology_and_n_volume(self):
        ds = _volume_dataset()
        assert ds.volume_topology is not None
        assert ds.n_volume == 6

    def test_n_layers_reports_the_max(self):
        assert _volume_dataset().n_layers == 3

    def test_a_2d_dataset_has_no_volume_topology(self):
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"depth": np.array([2.0, 4.0])},
        )
        assert ds.volume_topology is None
        assert ds.n_volume is None


class TestIselLayerRagged:
    """``isel_layer`` selects per column and makes absent layers explicit."""

    def test_layer_present_in_every_column(self):
        out = _volume_dataset().isel_layer(0)["sal"]
        assert out.location == "face"
        assert out.data.tolist() == [0.0, 3.0, 4.0]

    def test_absent_layers_become_nan_not_padded(self):
        out = _volume_dataset().isel_layer(2)["sal"]
        # Only column 0 (3 layers) has a layer 2; columns 1 and 2 are NaN.
        assert out.data[0] == 2.0
        assert np.isnan(out.data[1]) and np.isnan(out.data[2])

    def test_result_drops_the_volume_topology(self):
        out = _volume_dataset().isel_layer(0)
        assert out.volume_topology is None  # it is a plain 2-D dataset now

    def test_temporal_volume_variable(self):
        data = np.stack([np.arange(6.0), np.arange(6.0) + 10])  # (time=2, volume=6)
        out = _volume_dataset(data=data).isel_layer(0)["sal"]
        assert out.data.shape == (2, 3)
        assert out.data[1].tolist() == [10.0, 13.0, 14.0]

    def test_non_int_index_is_refused(self):
        with pytest.raises(ValueError, match="int layer position"):
            _volume_dataset().isel_layer(slice(0, 2))

    def test_negative_layer_is_refused(self):
        """Regression (L1): a negative layer must raise, not wrap into another column's
        volumes the way a bare ``offset + layer`` index would."""
        with pytest.raises(ValueError, match="non-negative"):
            _volume_dataset().isel_layer(-1)

    def test_sel_layer_by_position(self):
        out = _volume_dataset().sel_layer(1)["sal"]
        assert out.data[0] == 1.0 and np.isnan(out.data[1]) and out.data[2] == 5.0


class TestCollapseLayer:
    """The explicit 2-D collapse."""

    def test_face_at_a_layer(self):
        out = _volume_dataset().collapse_layer(0, location="face")["sal"]
        assert out.location == "face"
        assert out.data.tolist() == [0.0, 3.0, 4.0]

    def test_default_location_is_face(self):
        out = _volume_dataset().collapse_layer(0)["sal"]
        assert out.data.tolist() == [0.0, 3.0, 4.0]

    def test_requires_a_volume_topology(self):
        ds = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"depth": np.array([2.0, 4.0])},
        )
        with pytest.raises(ValueError, match="needs a ragged volume topology"):
            ds.collapse_layer(0)

    def test_rejects_non_face_location(self):
        with pytest.raises(ValueError, match="only 'face'"):
            _volume_dataset().collapse_layer(0, location="node")


class TestToDataFrame:
    """The tidy frame lays a ragged variable out one row per volume, no fabrication."""

    def test_one_row_per_volume_with_column_and_layer(self):
        df = _volume_dataset().to_dataframe()
        assert len(df) == 6
        assert df["element"].tolist() == [0, 0, 0, 1, 2, 2]
        assert df["layer"].tolist() == [0, 1, 2, 0, 0, 1]
        # face centres are carried onto every volume row of that column
        assert df.loc[df["element"] == 1, "x"].iloc[0] == 2.0


class TestSpatialSubsetRefused:
    """A face subset of a ragged volume mesh errors clearly instead of breaking silently."""

    def test_isel_element_is_refused(self):
        """Regression (M3): isel(element=) on a ragged mesh must raise, not return a
        dataset whose mesh, volume variable and (dropped) topology disagree."""
        with pytest.raises(ValueError, match="ragged"):
            _volume_dataset().isel(element=[0, 1])

    def test_subset_by_bounds_is_refused(self):
        """Regression (M3): subset_by_bounds on a ragged mesh must raise too."""
        with pytest.raises(ValueError, match="ragged"):
            _volume_dataset().subset_by_bounds(0.0, 0.0, 2.5, 1.0)


class TestNetcdfRoundTrip:
    """The ragged topology and a volume variable survive write -> read."""

    def test_topology_and_variable_round_trip(self, tmp_path):
        ds = _volume_dataset()
        path = tmp_path / "ragged.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.volume_topology is not None
        assert back.volume_topology.layer_count.tolist() == [3, 1, 2]
        assert back.volume_topology.volume_offset.tolist() == [0, 3, 4, 6]
        assert back.isel_layer(0)["sal"].data.tolist() == [0.0, 3.0, 4.0]

    def test_vertical_coordinates_round_trip(self, tmp_path):
        topology = VolumeTopology(
            layer_count=np.array([3, 1, 2]),
            volume_offset=np.array([0, 3, 4, 6]),
            layer_z=np.array([-2.5, -1.5, -0.5, -0.5, -1.0, -0.3]),
            interface_z=np.arange(9.0),  # n_volume (6) + n_face (3)
        )
        ds = _volume_dataset(topology=topology)
        path = tmp_path / "ragged_z.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        np.testing.assert_allclose(back.volume_topology.layer_z, topology.layer_z)
        np.testing.assert_allclose(
            back.volume_topology.interface_z, topology.interface_z
        )
