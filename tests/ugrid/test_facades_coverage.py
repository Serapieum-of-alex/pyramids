"""The capability facades `test_facades.py` leaves out, and the guard they share.

`test_facades.py` pins the wiring of the graph / fill / vector / regrid facades. Four of them are
still only exercised through their kernels, so the facade itself — the part that resolves the
variable, hands the kernel the mesh and wraps the answer — is untested: `circumcenters`, `voronoi`,
`rasterize_like` and `interp_time`'s carry-through of a non-temporal variable.

The shared `_facade_variable` guard has a third refusal beyond "name one of several" and "no such
variable": a variable that is present but carries no readable array. A file-read dataset whose
variable has neither eager data nor a loader hits it, and every facade must report that by name
rather than failing inside a kernel on `None`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.models import MeshVariable

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def triangles() -> UgridDataset:
    """The unit square as two right triangles carrying one face variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"depth": np.array([2.0, 4.0])},
    )


@pytest.fixture
def grid() -> UgridDataset:
    """A 2x2 quad grid carrying one face variable, for the raster facades."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        ),
        data={"v": np.array([1.0, 2.0, 3.0, 4.0])},
    )


@pytest.fixture
def unloaded(triangles) -> UgridDataset:
    """The same mesh carrying a variable with neither eager data nor a loader."""
    return UgridDataset(
        mesh=triangles.mesh,
        data_variables={
            "v": MeshVariable(
                name="v",
                location="face",
                mesh_name=triangles.mesh_name,
                shape=(2,),
            )
        },
        global_attributes={},
        crs_wkt=triangles.crs_wkt,
    )


class TestDualGeometryFacades:
    def test_circumcenters_sit_on_each_hypotenuse(self, triangles):
        """Both right triangles of the unit square circumscribe the diagonal's midpoint."""
        centres = triangles.circumcenters()
        assert centres.shape == (2, 2)
        assert centres[0].tolist() == pytest.approx([0.5, 0.5])

    def test_voronoi_tessellates_the_nodes(self, triangles):
        """The dual is built over the node coordinates, so it carries one point per node."""
        assert triangles.voronoi().points.shape == (4, 2)


class TestRasterFacade:
    def test_rasterize_like_reproduces_the_reference_grid(self, grid):
        """Snapping onto a grid the mesh itself produced returns that grid's own values."""
        reference = grid.to_dataset("v", cell_size=0.5, method="nearest")
        out = grid.rasterize_like(reference, "v")
        assert tuple(out.geotransform) == tuple(reference.geotransform)
        np.testing.assert_array_equal(out.read_array(), reference.read_array())


class TestInterpTimeCarriesStaticVariables:
    def test_a_variable_without_time_is_passed_through_untouched(self, triangles):
        """`interp_time` relabels the temporal variables and leaves a static one alone."""
        both = triangles.with_variable(
            "d", np.array([[0.0, 0.0], [10.0, 10.0]]), location="face"
        )
        both["d"].attributes = {**both["d"].attributes, "time_values": [0.0, 10.0]}
        out = both.interp_time([5.0])
        assert out["d"].data[0].tolist() == pytest.approx([5.0, 5.0])
        assert out["depth"].data.tolist() == [2.0, 4.0]
        assert "time_values" not in out["depth"].attributes


class TestUnloadedVariableIsRefusedByName:
    def test_the_facade_names_the_operation_and_the_variable(self, unloaded):
        """A variable with no readable array is refused before any kernel sees it."""
        with pytest.raises(
            ValueError, match="fill_na_spatial: variable 'v' has no loaded data"
        ):
            unloaded.fill_na_spatial()

    def test_the_refusal_is_shared_by_every_facade(self, unloaded):
        """The guard lives in `_facade_variable`, so a second facade reports it the same way."""
        with pytest.raises(
            ValueError, match="to_location: variable 'v' has no loaded data"
        ):
            unloaded.to_location("node")
