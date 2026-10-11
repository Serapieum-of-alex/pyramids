"""The no-data value a facade's result declares, when the facade changes the dtype.

The graph facades hand back a boolean mask. `_rewrap` rebuilt the variable with
`var.with_data(...)`, which keeps every other field of the source variable -- including a
no-data value belonging to the source's dtype. A boolean mask declaring `_FillValue = -9999`
is meaningless: no element can ever equal it, and writing it stamps a sentinel on a variable
that cannot carry one.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core


@pytest.fixture
def mask_with_nodata():
    """A two-face mesh holding a boolean mask that declares a float no-data value."""
    mesh = UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
        data={"m": np.array([True, False])},
        data_locations={"m": "face"},
    )
    mesh["m"].nodata = -9999.0
    return mesh


class TestMorphologyResultNodata:
    """What `binary_dilation` / `binary_erosion` declare on the mask they return."""

    def test_dilation_drops_an_unrepresentable_nodata(self, mask_with_nodata):
        """A boolean result cannot hold -9999, so it declares no no-data value.

        Test scenario:
            The source mask declares -9999.0 because it was read from a file that did. The
            dilated mask is boolean, and `bool(-9999.0)` is `True` -- so keeping the
            declaration would mark every set element as missing if anything ever compared
            against it, and writes a `_FillValue` no element can equal.
        """
        grown = mask_with_nodata.binary_dilation("m")

        assert np.asarray(grown["m"].data).dtype == np.bool_, (
            f"the result is a boolean mask, got {np.asarray(grown['m'].data).dtype}"
        )
        assert grown["m"].nodata is None, (
            f"a boolean mask has no representable no-data value, yet it declares "
            f"{grown['m'].nodata!r}"
        )

    def test_erosion_drops_it_too(self, mask_with_nodata):
        """The sibling facade behaves the same way."""
        shrunk = mask_with_nodata.binary_erosion("m")

        assert shrunk["m"].nodata is None, (
            f"a boolean mask has no representable no-data value, yet it declares "
            f"{shrunk['m'].nodata!r}"
        )

    def test_a_representable_nodata_is_kept(self):
        """A no-data value the result dtype can hold is left alone.

        Test scenario:
            The guard must only drop what cannot be represented. A float variable keeps its
            NaN through a facade that returns floats, so nothing that already worked changes.
        """
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
            data={"v": np.array([1.0, np.nan])},
            data_locations={"v": "face"},
        )
        mesh["v"].nodata = float("nan")

        filled = mesh.fill_na_spatial("v")

        assert np.isnan(filled["v"].nodata), (
            f"a float result keeps its NaN no-data value, got {filled['v'].nodata!r}"
        )

    def test_the_mask_values_are_unaffected(self, mask_with_nodata):
        """Dropping the declaration does not touch the data.

        Test scenario:
            Face 0 is set and face 1 shares an edge with it, so one dilation pass sets both.
        """
        grown = mask_with_nodata.binary_dilation("m")

        assert np.asarray(grown["m"].data).tolist() == [True, True], (
            f"both faces should be set after one pass, got "
            f"{np.asarray(grown['m'].data).tolist()}"
        )
