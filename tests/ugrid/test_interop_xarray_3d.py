"""xarray round-trip of the 3-D results-fidelity features (issues #1331 and #1330).

The active mask and the ragged volume topology must survive the ``to_xarray`` -> ``from_xarray``
bridge as they do the NetCDF one — the mask as an ``<var>_active`` companion, the topology as
its ``layer_count`` / ``volume_offset`` (and optional vertical) companion arrays.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, VolumeTopology
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop


def _mesh() -> Mesh2d:
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        node_y=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
        face_x=np.array([1.0, 2.0, 3.0]),
        face_y=np.array([0.3, 0.6, 0.3]),
    )


def test_active_mask_round_trips_through_xarray():
    mask = np.array([[True, True, False], [True, False, False]])
    var = MeshVariable(
        name="wl",
        location="face",
        mesh_name="mesh2d",
        shape=(2, 3),
        dimensions=("time", "mesh2d_nFaces"),
        active=mask,
        _data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    ds = UgridDataset(_mesh(), {"wl": var}, {})
    exported = ds.to_xarray()
    assert "wl_active" in exported.data_vars
    back = UgridDataset.from_xarray(exported)
    np.testing.assert_array_equal(back.active_mask("wl"), mask)


def test_volume_topology_round_trips_through_xarray():
    topology = VolumeTopology(
        layer_count=np.array([3, 1, 2]),
        volume_offset=np.array([0, 3, 4, 6]),
        layer_z=np.array([-2.5, -1.5, -0.5, -0.5, -1.0, -0.3]),
    )
    var = MeshVariable(
        name="sal",
        location="volume",
        mesh_name="mesh2d",
        shape=(6,),
        dimensions=("mesh2d_nVolumes",),
        _data=np.arange(6.0),
    )
    ds = UgridDataset(_mesh(), {"sal": var}, {}, volume_topology=topology)
    exported = ds.to_xarray()
    assert "mesh2d_layer_count" in exported.data_vars
    back = UgridDataset.from_xarray(exported)
    assert back.volume_topology is not None
    assert back.volume_topology.layer_count.tolist() == [3, 1, 2]
    np.testing.assert_allclose(back.volume_topology.layer_z, topology.layer_z)
    assert back.isel_layer(0)["sal"].data.tolist() == [0.0, 3.0, 4.0]
