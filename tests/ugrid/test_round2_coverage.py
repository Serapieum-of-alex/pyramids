"""The branches round 2's fixes added that no test had reached yet.

Each fix below was verified by executing it when it was written, but three of them left a
branch the suite never ran: the CRS-without-an-EPSG path in `from_geodataframe`, the
empty-source guard in `_interpolate_idw`, and the unrepresentable-value arm of the no-data
check the facades use. A branch only a scratch script ever ran is a branch that can rot.
"""

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid._facades import _representable
from pyramids.netcdf.ugrid.network import Network1d
from pyramids.netcdf.ugrid.regrid import _interpolate_idw

pytestmark = pytest.mark.core

#: A Lambert azimuthal equal-area CRS given as proj4: a real projected CRS in metres that
#: carries no EPSG code, which is the case `from_geodataframe` used to flatten to 4326.
_NO_EPSG_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=52 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


class TestFromGeoDataFrameWithoutAnEpsgCode:
    """A frame whose CRS is projected but has no EPSG code."""

    def test_the_crs_is_carried_as_wkt(self):
        """The mesh keeps the frame's own CRS rather than a 4326 placeholder.

        Test scenario:
            `from_geodataframe` used to read `gdf.crs.to_epsg()` and fall back to 4326, which
            stamped degrees on a frame in metres. The guard added for M1 raises for a *missing*
            CRS; this is the other half -- a CRS that exists but cannot be reduced to an EPSG
            code, which is kept verbatim as WKT.
        """
        assert _NO_EPSG_CRS.to_epsg() is None, (
            "the fixture must be a CRS with no EPSG code for this test to mean anything"
        )
        frame = gpd.GeoDataFrame(
            {"v": [1.0]},
            geometry=[Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
            crs=_NO_EPSG_CRS,
        )

        mesh = UgridDataset.from_geodataframe(frame)

        assert mesh.crs_wkt is not None, "the mesh should carry a CRS"
        assert CRS.from_wkt(mesh.crs_wkt).equals(_NO_EPSG_CRS), (
            "the mesh CRS should be the frame's own, not a 4326 placeholder"
        )

    def test_the_units_survive(self):
        """The point of carrying the WKT: the mesh is still in metres.

        Test scenario:
            The 4326 fallback was not merely imprecise -- it changed the unit of every
            coordinate from metres to degrees, silently, while the numbers stayed put.
        """
        frame = gpd.GeoDataFrame(
            {"v": [1.0]},
            geometry=[Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
            crs=_NO_EPSG_CRS,
        )

        mesh = UgridDataset.from_geodataframe(frame)
        axis = CRS.from_wkt(mesh.crs_wkt).axis_info[0]

        assert axis.unit_name == "metre", (
            f"the mesh should still be in metres, got {axis.unit_name!r}"
        )


class TestIdwWithNoSourcePoints:
    """`_interpolate_idw` when the source carries nothing to interpolate from."""

    def test_every_target_is_nodata(self):
        """An empty source answers the declared no-data value for every target.

        Test scenario:
            `cKDTree.query(k=0)` raises a zero-size reduction error, so the guard answers the
            `(M,)` array the contract promises instead. The shape matters as much as the
            values: a caller reshapes the result against the target grid.
        """
        out = _interpolate_idw(
            np.empty((0, 2)),
            np.empty(0),
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            k=3,
            power=2.0,
            nodata=-9999.0,
            max_distance=float("inf"),
        )

        assert out.shape == (3,), f"expected one value per target, got {out.shape}"
        assert out.tolist() == [-9999.0, -9999.0, -9999.0], (
            f"every target should be no-data, got {out.tolist()}"
        )

    def test_it_does_not_raise(self):
        """The guard exists because the unguarded path raised, so pin that it no longer does."""
        out = _interpolate_idw(
            np.empty((0, 2)),
            np.empty(0),
            np.empty((0, 2)),
            k=1,
            power=2.0,
            nodata=np.nan,
            max_distance=float("inf"),
        )

        assert out.shape == (0,), (
            f"an empty target gives an empty answer, got {out.shape}"
        )


class TestRepresentableNodata:
    """The helper `_rewrap` uses to decide whether a no-data value still means something."""

    @pytest.mark.parametrize(
        "nodata,dtype,expected",
        [
            (None, np.dtype("float64"), True),
            (-9999.0, np.dtype("float64"), True),
            (-9999.0, np.dtype("int32"), True),
            (-9999.0, np.dtype("bool"), False),
            (1.0, np.dtype("bool"), False),
            (0.0, np.dtype("bool"), False),
            (float("nan"), np.dtype("float64"), True),
            (float("nan"), np.dtype("int32"), False),
            (1e300, np.dtype("float32"), False),
            (np.iinfo("int64").max, np.dtype("int64"), True),
            (np.iinfo("int64").min, np.dtype("int64"), True),
            (2**63, np.dtype("int64"), False),
        ],
        ids=[
            "none-is-always-safe",
            "float-sentinel-in-float",
            "float-sentinel-in-int",
            "float-sentinel-in-bool",
            "one-in-bool",
            "zero-in-bool",
            "nan-in-float",
            "nan-in-int",
            "overflowing-float-in-float32",
            "int64-max-in-int64",
            "int64-min-in-int64",
            "past-int64-max-in-int64",
        ],
    )
    def test_representability(self, nodata, dtype, expected):
        """A no-data value is representable only when it survives a round trip through `dtype`.

        Args:
            nodata: The declared no-data value.
            dtype: The dtype of the data it would be declared on.
            expected: Whether the declaration still means anything.

        Test scenario:
            `bool` is the case that prompted the helper, and it needs **every** value refused,
            not just the out-of-range ones: `0.0` and `1.0` survive any round trip through
            `bool`, so a round-trip test kept them -- and a boolean mask declaring `1.0` as
            missing marks every `True` face missing, the exact outcome the helper exists to
            prevent. A two-valued domain has no spare sentinel.

            `int64.max` is the mirror case: it *is* representable in `int64`, but comparing
            through `float64` could not show that, because a double's 53-bit mantissa cannot
            hold it. `2**63` is one past the range and must still be refused.
        """
        assert _representable(nodata, dtype) is expected, (
            f"_representable({nodata!r}, {dtype}) should be {expected}"
        )

    @pytest.mark.parametrize("value", ["not a number", "-9999", ""])
    def test_a_non_numeric_value_answers_false_rather_than_raising(self, value):
        """A value that is not a number answers False instead of propagating an error.

        Args:
            value: The non-numeric no-data value.

        Test scenario:
            Both readers coerce a file's `_FillValue` with `float(...)`, so a string does not
            reach here in practice -- the original rationale for this case ("a `_FillValue`
            read from a file attribute can arrive as a string") was simply wrong. What matters
            is that the helper answers the question instead of raising out of a public facade,
            and the *numeric* string is the case that did: `np.isnan("-9999")` raised
            `TypeError: ufunc 'isnan' not supported for the input types`.
        """
        assert _representable(value, np.dtype("int64")) is False, (
            f"{value!r} is not a number, so it is not representable"
        )

    def test_a_string_no_data_does_not_raise_through_a_facade(self):
        """The end-to-end form of the case above.

        Test scenario:
            A variable carrying a string no-data value went through `binary_dilation` and out
            as `TypeError` from inside numpy, because the `np.isnan` call sat outside the
            helper's `try`.
        """
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
            data={"m": np.array([True, False])},
            data_locations={"m": "face"},
        )
        mesh["m"].nodata = "-9999"

        grown = mesh.binary_dilation("m")

        assert grown["m"].nodata is None, (
            f"an unusable declaration is dropped, got {grown['m'].nodata!r}"
        )


class TestNetworkEdgeAccessors:
    """The per-edge accessors on `Network1d`."""

    @pytest.fixture
    def chain(self):
        """Three nodes in a line joined by two edges."""
        return Network1d.from_arrays(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([[0, 1], [1, 2]]),
        )

    def test_get_edge_nodes_names_both_endpoints(self, chain):
        """`get_edge_nodes` answers the two node indices of one edge."""
        assert chain.get_edge_nodes(1).tolist() == [1, 2], (
            f"edge 1 joins nodes 1 and 2, got {chain.get_edge_nodes(1).tolist()}"
        )

    def test_get_edge_coords_matches_those_nodes(self, chain):
        """`get_edge_coords` answers the same edge's endpoint coordinates."""
        coords = np.asarray(chain.get_edge_coords(1))

        assert coords.shape == (2, 2), (
            f"two endpoints with two coordinates, got {coords.shape}"
        )
        np.testing.assert_allclose(coords, np.array([[1.0, 0.0], [2.0, 0.0]]))
