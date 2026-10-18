"""First-class per-timestep active (wet/dry) mask on ``MeshVariable`` (issue #1331).

A hydrodynamic result carries a per-step wet/dry status per element: a cell can be wet at
``t=5`` and dry at ``t=6``. Folding "dry" into ``NaN`` is lossy — a consumer cannot then tell a
dry cell from a missing one. These tests pin the mask as first-class and kept separate from
``nodata``: it rides through the shape-preserving selections, it is applied to the values only
when a caller opts in, and it round-trips to disk distinct from the fill value.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


def _mesh() -> Mesh2d:
    """A tiny 3-face strip mesh (5 nodes, 3 triangles) with face centres."""
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


def _temporal_var(active=None) -> MeshVariable:
    """A ``(time=2, face=3)`` face variable, optionally carrying ``active``."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return MeshVariable(
        name="wl",
        location="face",
        mesh_name="mesh2d",
        shape=data.shape,
        dimensions=("time", "mesh2d_nFaces"),
        active=active,
        _data=data,
    )


def _dataset(var: MeshVariable) -> UgridDataset:
    return UgridDataset(_mesh(), {var.name: var}, {})


class TestMeshVariableMask:
    """The mask field and its shape-aware helpers on ``MeshVariable``."""

    def test_defaults_to_none(self):
        """A variable built without a mask has ``active is None`` — additive, nothing changes."""
        assert _temporal_var().active is None

    def test_active_has_time_distinguishes_static_from_per_step(self):
        """A 1-D mask is static; a 2-D mask carries its own time axis."""
        assert _temporal_var(np.array([True, False, True])).active_has_time is False
        per_step = np.array([[True, True, False], [True, False, False]])
        assert _temporal_var(per_step).active_has_time is True
        assert _temporal_var(None).active_has_time is False

    def test_time_sliced_active_static_is_unchanged(self):
        """A static mask applies to every step, so a time slice leaves it as-is."""
        var = _temporal_var(np.array([True, False, True]))
        np.testing.assert_array_equal(
            var._time_sliced_active(1, None), [True, False, True]
        )

    def test_time_sliced_active_per_step_single_and_range(self):
        """A per-step mask is indexed on its own leading time axis."""
        mask = np.array([[True, True, False], [True, False, False]])
        var = _temporal_var(mask)
        np.testing.assert_array_equal(
            var._time_sliced_active(1, None), [True, False, False]
        )
        np.testing.assert_array_equal(var._time_sliced_active(0, 1), mask[0:1])

    def test_time_indexed_active(self):
        """``_time_indexed_active`` gathers arbitrary steps of a per-step mask."""
        mask = np.array([[True, True, False], [True, False, False]])
        var = _temporal_var(mask)
        np.testing.assert_array_equal(
            var._time_indexed_active(np.array([1, 0])), mask[[1, 0]]
        )
        static = np.array([True, False, True])
        np.testing.assert_array_equal(
            _temporal_var(static)._time_indexed_active(np.array([0])), static
        )
        assert _temporal_var(None)._time_indexed_active(np.array([0])) is None

    def test_element_sliced_active(self):
        """``_element_sliced_active`` takes the mask's last (element) axis."""
        mask = np.array([[True, True, False], [True, False, False]])
        var = _temporal_var(mask)
        np.testing.assert_array_equal(
            var._element_sliced_active(np.array([0, 2])), mask[:, [0, 2]]
        )
        assert _temporal_var(None)._element_sliced_active(np.array([0])) is None

    def test_with_data_keeps_replaces_and_clears(self):
        """``with_data`` keeps the mask implicitly, replaces it explicitly, or clears it."""
        mask = np.array([True, False, True])
        var = _temporal_var(mask)
        kept = var.with_data(np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]))
        np.testing.assert_array_equal(kept.active, mask)
        replaced = var.with_data(var.data, active=np.array([False, False, False]))
        np.testing.assert_array_equal(replaced.active, [False, False, False])
        assert var.with_data(var.data, active=None).active is None

    def test_with_data_drops_an_inconsistent_implicit_mask(self):
        """An implicit carry is dropped when the new shape no longer fits the mask."""
        per_step = np.array([[True, True, False], [True, False, False]])
        var = _temporal_var(per_step)
        # Element count changes (3 -> 2): the element-indexed mask can no longer apply.
        shrunk = var.with_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert shrunk.active is None
        # Time length changes (2 -> 1): a per-step mask no longer lines up.
        one_step = var.with_data(np.array([[1.0, 2.0, 3.0]]))
        assert one_step.active is None


class TestTimeSelectionCarry:
    """The mask rides through the shape-preserving time selections."""

    def test_sel_time_slices_per_step_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        assert ds.sel_time(1)["wl"].active.tolist() == [True, False, False]

    def test_sel_time_keeps_static_mask(self):
        ds = _dataset(_temporal_var(np.array([True, False, True])))
        assert ds.sel_time(0)["wl"].active.tolist() == [True, False, True]

    def test_sel_time_range_slices_per_step_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        out = ds.sel_time_range(0, 1)["wl"]
        assert out.active.shape == (1, 3)
        np.testing.assert_array_equal(out.active, mask[0:1])

    def test_isel_time_slice_and_list(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        assert ds.isel(time=slice(1, 2))["wl"].active.shape == (1, 3)
        np.testing.assert_array_equal(ds.isel(time=[1, 0])["wl"].active, mask[[1, 0]])

    def test_isel_time_int_collapses_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        collapsed = ds.isel(time=1)["wl"]
        assert collapsed.active.tolist() == [True, False, False]


class TestElementSelectionCarry:
    """A face selection slices the element-indexed mask to match the kept faces."""

    def test_isel_element_slices_face_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        sub = ds.isel(element=[0, 2])["wl"]
        assert sub.active.shape == (2, 2)
        np.testing.assert_array_equal(sub.active, mask[:, [0, 2]])

    def test_static_node_mask_follows_node_renumbering(self):
        """A node-located variable's static mask is sliced on the surviving nodes."""
        node_mask = np.array([True, False, True, False, True])
        node_var = MeshVariable(
            name="nd",
            location="node",
            mesh_name="mesh2d",
            shape=(5,),
            dimensions=("mesh2d_nNodes",),
            active=node_mask,
            _data=np.arange(5.0),
        )
        ds = _dataset(node_var)
        # Keep only face 0 (nodes 0,1,2): the node mask must shrink to those nodes.
        sub = ds.isel(element=0)["nd"]
        assert sub.active.tolist() == [True, False, True]


class TestAccessors:
    """The opt-in accessors — ``active_mask`` reads, ``masked_values`` applies."""

    def test_active_mask_returns_the_mask_or_none(self):
        mask = np.array([True, False, True])
        assert _dataset(_temporal_var(mask)).active_mask("wl").tolist() == mask.tolist()
        assert _dataset(_temporal_var()).active_mask("wl") is None

    def test_masked_values_blanks_dry_to_nan_per_step(self):
        mask = np.array([[True, True, False], [True, False, False]])
        mv = _dataset(_temporal_var(mask)).masked_values("wl")
        assert np.isnan(mv[0, 2]) and np.isnan(mv[1, 1]) and np.isnan(mv[1, 2])
        assert mv[0, 0] == 1.0 and mv[1, 0] == 4.0

    def test_masked_values_static_mask_broadcasts_over_time(self):
        mv = _dataset(_temporal_var(np.array([True, False, True]))).masked_values("wl")
        assert np.isnan(mv[:, 1]).all()
        assert mv[:, 0].tolist() == [1.0, 4.0]

    def test_masked_values_no_mask_is_just_nodata_blanking(self):
        var = _temporal_var()
        var.nodata = 5.0
        mv = _dataset(var).masked_values("wl")
        assert np.isnan(mv[1, 1])  # the 5.0 value
        assert mv[0, 0] == 1.0

    def test_masked_values_requires_loaded_data(self):
        var = MeshVariable(
            name="empty", location="face", mesh_name="mesh2d", shape=(3,)
        )
        with pytest.raises(ValueError, match="no loaded data"):
            _dataset(var).masked_values("empty")

    def test_masked_values_trailing_time_layout(self):
        """Regression (M2): a trailing-time ``(n_face, time)`` variable must mask correctly.

        The element axis is 0 and time is last here — a layout the rest of the module supports
        via ``time_index``. The mask is still the canonical ``(n_time, n_element)``.
        """
        base = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"wl": np.zeros(2)},
        )
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (face=2, time=3)
        mask = np.array(
            [[True, False], [False, True], [True, True]]
        )  # (time=3, face=2)
        var = MeshVariable(
            name="wl",
            location="face",
            mesh_name="mesh2d",
            shape=data.shape,
            dimensions=("mesh2d_nFaces", "time"),
            active=mask,
            _data=data,
        )
        ds = UgridDataset(base.mesh, {"wl": var}, {})
        assert var.time_index == 1, "this variable stores time trailing"
        out = ds.masked_values("wl")
        np.testing.assert_array_equal(
            np.isnan(out), [[False, True, False], [True, False, False]]
        )
        np.testing.assert_array_equal(out[~np.isnan(out)], [1.0, 3.0, 5.0, 6.0])

    def test_masked_values_spans_a_layer_axis(self):
        """A per-step ``(time, face)`` mask applies across an intervening layer axis."""
        data = np.arange(2 * 2 * 3, dtype="float64").reshape(
            2, 2, 3
        )  # (time, layer, face)
        mask = np.array([[True, False, True], [True, True, False]])  # (time, face)
        var = MeshVariable(
            name="v",
            location="face",
            mesh_name="mesh2d",
            shape=data.shape,
            dimensions=("time", "nLayers", "mesh2d_nFaces"),
            active=mask,
            _data=data,
        )
        mv = _dataset(var).masked_values("v")
        # Face 1 is dry at t=0 across both layers.
        assert np.isnan(mv[0, :, 1]).all()
        assert not np.isnan(mv[0, :, 0]).any()


class TestElementValuesOptIn:
    """``_element_values`` leaves the mask off unless asked (used by sample/weighted)."""

    def test_default_does_not_apply_the_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        _, values = _dataset(_temporal_var(mask))._element_values("wl", 0)
        assert values.tolist() == [1.0, 2.0, 3.0]  # dry face 2 NOT blanked

    def test_apply_active_blanks_dry_for_the_step(self):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        _, values = ds._element_values("wl", 1, apply_active=True)
        assert values[0] == 4.0 and np.isnan(values[1]) and np.isnan(values[2])


class TestReductionDropsPerStepMask:
    """A time reduction cannot keep a per-step mask; it drops it, keeps a static one."""

    def test_mean_drops_per_step_mask(self):
        mask = np.array([[True, True, False], [True, False, False]])
        reduced = _dataset(_temporal_var(mask)).mean()
        assert reduced["wl"].active is None

    def test_mean_keeps_static_mask(self):
        static = np.array([True, False, True])
        reduced = _dataset(_temporal_var(static)).mean()
        np.testing.assert_array_equal(reduced["wl"].active, static)

    def test_mean_drops_per_step_mask_when_square(self):
        """Regression (M1): a square ``(n_time, n_element)`` mask (n_time == n_element) must
        be dropped by a time reduction, not kept on the 1-D reduced data where the element
        count coincides with the old time length."""
        mask = np.array(
            [[True, True, False], [True, False, False], [False, True, True]]
        )
        data = np.arange(9.0).reshape(3, 3)  # (time=3, face=3) -> square
        var = MeshVariable(
            name="wl",
            location="face",
            mesh_name="mesh2d",
            shape=data.shape,
            dimensions=("time", "mesh2d_nFaces"),
            active=mask,
            _data=data,
        )
        reduced = _dataset(var).mean()
        assert reduced["wl"].active is None, "square per-step mask must be dropped"
        # Must not raise: with the bug the (3, 3) mask survived onto (3,) data.
        assert reduced.masked_values("wl").shape == (3,)


class TestNetcdfRoundTrip:
    """The mask survives write -> read, distinct from ``nodata``."""

    def test_per_step_mask_round_trips(self, tmp_path):
        mask = np.array([[True, True, False], [True, False, False]])
        ds = _dataset(_temporal_var(mask))
        path = tmp_path / "per_step.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.active_mask("wl") is not None
        np.testing.assert_array_equal(back.active_mask("wl"), mask)

    def test_static_mask_round_trips_and_is_distinct_from_nodata(self, tmp_path):
        var = _temporal_var(np.array([True, False, True]))
        var.nodata = -999.0
        ds = _dataset(var)
        path = tmp_path / "static.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        np.testing.assert_array_equal(back.active_mask("wl"), [True, False, True])
        assert back["wl"].nodata == -999.0  # mask did not become the fill value

    def test_a_mesh_with_no_mask_reads_exactly_as_before(self, tmp_path):
        ds = _dataset(_temporal_var())
        path = tmp_path / "plain.nc"
        ds.to_file(path)
        back = UgridDataset.read_file(path)
        assert back.active_mask("wl") is None
