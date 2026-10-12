"""What the topology parser does with a declared `topology_dimension` other than 2.

`network.py` explained the absence of a 1-D reader by saying the parser "recognises
`topology_dimension == 2` meshes". It has no such check -- it records whatever dimension the
file declares and builds a `Mesh2d` regardless. The absence is real, but it is `UgridDataset`
always carrying a `Mesh2d` and there being no `Network1d` reader, not a parser filter.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.interop


@pytest.fixture
def one_dimensional_file(tmp_path):
    """A UGRID file whose topology variable declares `topology_dimension = 1`."""
    mesh = UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.array([1.0, 2.0])},
        data_locations={"d": "face"},
    )
    dataset = mesh.to_xarray()
    dataset[mesh.mesh_name].attrs["topology_dimension"] = np.int32(1)
    path = tmp_path / "one_d.nc"
    dataset.to_netcdf(path)
    return path


class TestDeclaredTopologyDimension:
    """A file declaring a dimension other than 2."""

    def test_it_is_read_rather_than_refused(self, one_dimensional_file):
        """The parser accepts it; there is no `topology_dimension == 2` filter.

        Test scenario:
            The topology check is `topo_dim is not None and node_x_var is not None and
            node_y_var is not None` -- the value is cast to int and recorded, never compared
            with 2. So the stated reason a network cannot be read from a file was not the
            operative one.
        """
        read_back = UgridDataset.read_file(str(one_dimensional_file))

        assert read_back.n_node == 4, (
            f"expected the file's 4 nodes, got {read_back.n_node}"
        )

    def test_the_declared_dimension_is_recorded_as_given(self, one_dimensional_file):
        """`topology_dimension` round trips as 1, not normalised to 2."""
        read_back = UgridDataset.read_file(str(one_dimensional_file))
        topologies = read_back.metadata.mesh_topologies

        assert [topo.topology_dimension for topo in topologies] == [1], (
            f"the declared dimension should survive, got "
            f"{[topo.topology_dimension for topo in topologies]}"
        )

    def test_it_still_comes_back_as_a_mesh2d(self, one_dimensional_file):
        """This is the real limitation: the reader has only one mesh type to build.

        Test scenario:
            `UgridDataset` carries a `Mesh2d`, and nothing maps a declared 1-D topology onto a
            `Network1d`. A genuinely 1-D file -- edges and no faces -- would come back as a
            face-less `Mesh2d` rather than a network, which is what makes `Network1d`
            in-memory only.
        """
        read_back = UgridDataset.read_file(str(one_dimensional_file))

        assert isinstance(read_back.mesh, Mesh2d), (
            f"the reader builds a Mesh2d whatever the declared dimension, got "
            f"{type(read_back.mesh).__name__}"
        )
