"""The no-data value a facade's result declares, when the facade changes the dtype.

The graph facades hand back a boolean mask. `_rewrap` rebuilt the variable with
`var.with_data(...)`, which keeps every other field of the source variable -- including a
no-data value belonging to the source's dtype. A boolean mask declaring `_FillValue = -9999`
is meaningless: no element can ever equal it, and writing it stamps a sentinel on a variable
that cannot carry one.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

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


class TestBurnFillDeclaration:
    """Which `fill` values `burn` declares as the variable's no-data value."""

    @pytest.fixture
    def strip(self):
        """A 1 x 3 strip of unit quads along x."""
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]),
            data={"anchor": np.array([1.0, 2.0, 3.0])},
            data_locations={"anchor": "face"},
        )

    def test_an_explicit_numeric_fill_is_not_declared_missing(self, strip):
        """`burn(geom, 1.0, fill=0.0)` is the mask idiom; its zeros are data.

        Test scenario:
            Declaring the fill unconditionally made `0.0` the no-data value, so every unburned
            face read as missing: `dropna` dropped them and every `skipna` reduction ignored
            them. The zeros are the mask's "not here", which is a value, not an absence.
        """
        square = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        out = strip.burn(square, 1.0, fill=0.0, all_touched=False, name="mask")

        assert np.asarray(out["mask"].data).tolist() == [1.0, 0.0, 0.0]
        assert out["mask"].nodata is None, (
            f"an explicit numeric fill is data, not a sentinel; got {out['mask'].nodata!r}"
        )

    def test_the_default_nan_fill_is_still_declared(self, strip):
        """A NaN fill does mean "untouched", so naming it is what makes it readable.

        Test scenario:
            The contrast case that keeps the original L2 fix intact -- without the
            declaration, `to_file` writes those NaNs with no `_FillValue` and a reader cannot
            tell them from data.
        """
        square = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        out = strip.burn(square, 1.0, all_touched=False, name="nanmask")
        values = np.asarray(out["nanmask"].data)

        assert values[0] == 1.0
        assert np.isnan(values[1]) and np.isnan(values[2])
        assert out["nanmask"].nodata is not None and np.isnan(out["nanmask"].nodata), (
            f"a NaN fill stays declared, got {out['nanmask'].nodata!r}"
        )


class TestInterpTimeDocumentedValues:
    """Every value `interp_time`'s docstring names must work, and no other."""

    @pytest.fixture
    def timed(self):
        """A two-face mesh with two numeric time steps."""
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"v": np.array([[2.0, 3.0], [4.0, 5.0]])},
            data_locations={"v": "face"},
        )
        mesh["v"].attributes["time_values"] = [0.0, 1.0]
        return mesh

    def test_bounds_nan_leaves_an_outside_label_empty(self, timed):
        """`"nan"`, the default, answers NaN past the ends."""
        out = timed.interp_time([0.5, 9.0], bounds="nan")

        values = np.asarray(out["v"].data)
        np.testing.assert_allclose(values[0], [3.0, 4.0])
        assert np.isnan(values[1]).all(), (
            f"9.0 is past the axis, got {values[1].tolist()}"
        )

    def test_bounds_clamp_holds_the_last_step(self, timed):
        """`"clamp"` is the working non-default value, and it was undocumented.

        Test scenario:
            The docstring named `"extrapolate"`, which raises, and omitted `"clamp"` -- so the
            only non-default value a reader could find was the invalid one.
        """
        out = timed.interp_time([0.5, 9.0], bounds="clamp")

        values = np.asarray(out["v"].data)
        np.testing.assert_allclose(values[1], [4.0, 5.0])

    def test_the_value_the_docstring_used_to_name_is_refused(self, timed):
        """`"extrapolate"` raises, which is why it no longer appears in the Args block."""
        with pytest.raises(ValueError, match="'nan' or 'clamp'"):
            timed.interp_time([0.5], bounds="extrapolate")

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_both_documented_methods_work(self, method, timed):
        """The two kinds the kernel accepts.

        Args:
            method: The interpolation kind.
            timed: The fixture mesh.
        """
        out = timed.interp_time([0.5], method=method)

        assert np.asarray(out["v"].data).shape == (1, 2)

    def test_another_scipy_kind_is_refused(self, timed):
        """The docstring said "a `scipy.interpolate.interp1d` kind", which overstates it."""
        with pytest.raises(ValueError, match="'linear' or 'nearest'"):
            timed.interp_time([0.5], method="cubic")
