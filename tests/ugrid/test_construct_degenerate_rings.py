"""Which geometries `from_geodataframe` drops, and whether it drops what it promises.

The docstring says a polygon with "fewer than three distinct vertices" is dropped, and calls
the check sub-triangular. The test is on the raw coordinate count, so a ring of three
identical points passes it and becomes a face whose three node slots hold one node -- a
zero-area face with a duplicated vertex. A MultiPolygon is dropped too, which nothing said.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from pyramids.netcdf.ugrid.construct import from_geodataframe

pytestmark = pytest.mark.core

_UNIT = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _frame(geometries, values=None):
    """A GeoDataFrame of `geometries` in EPSG:4326, with an optional `v` attribute column."""
    columns = {} if values is None else {"v": values}
    return gpd.GeoDataFrame(columns, geometry=list(geometries), crs=4326)


class TestDegenerateRings:
    """Rings that carry three or more coordinates but fewer than three distinct points."""

    def test_an_all_identical_ring_is_dropped(self):
        """Three copies of one point is not a triangle.

        Test scenario:
            `Polygon([(0, 0)] * 3)` has four coordinates once shapely closes it, so the raw
            length check saw three after the closing duplicate came off and kept it. The ring
            then deduplicated to a single node and the face's three slots all pointed at it.
        """
        degenerate = Polygon([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])

        _, _, faces, _ = from_geodataframe(_frame([_UNIT, degenerate]))

        assert faces.shape[0] == 1, (
            f"only the unit square is a face, got {faces.shape[0]} faces: {faces.tolist()}"
        )

    def test_a_two_point_ring_is_dropped(self):
        """Two distinct points repeated is not a triangle either.

        Test scenario:
            `[(0,0), (1,1), (0,0)]` is three raw coordinates spanning two distinct vertices --
            a degenerate sliver with no area. It survived the raw-length check for the same
            reason.
        """
        degenerate = Polygon([(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)])

        _, _, faces, _ = from_geodataframe(_frame([_UNIT, degenerate]))

        assert faces.shape[0] == 1, (
            f"only the unit square is a face, got {faces.shape[0]} faces: {faces.tolist()}"
        )

    def test_no_face_repeats_a_node(self):
        """The surviving faces never name the same node twice.

        Test scenario:
            The consequence the count above stands in for: a face with a duplicated node index
            has zero area and breaks the edge derivation, which assumes consecutive nodes
            differ.
        """
        degenerate = Polygon([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])

        _, _, faces, _ = from_geodataframe(_frame([_UNIT, degenerate]))

        for row in np.asarray(faces.data):
            nodes = [int(node) for node in row if int(node) != -1]
            assert len(nodes) == len(set(nodes)), f"face {row.tolist()} repeats a node"

    def test_the_attributes_stay_aligned_to_the_kept_faces(self):
        """Dropping a ring drops its attribute row with it.

        Test scenario:
            The alignment is the part a silent drop can corrupt: if the degenerate row were
            kept in `data` while its face was dropped, every value after it would shift onto
            the wrong face.
        """
        degenerate = Polygon([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])
        second = Polygon([(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)])

        _, _, faces, data = from_geodataframe(
            _frame([_UNIT, degenerate, second], values=[10.0, 20.0, 30.0])
        )

        assert faces.shape[0] == 2, f"two faces should survive, got {faces.shape[0]}"
        assert data["v"].tolist() == [10.0, 30.0], (
            f"the dropped ring's value must go with it, got {data['v'].tolist()}"
        )


class TestMultiPolygon:
    """A MultiPolygon row."""

    def test_it_is_dropped(self):
        """A MultiPolygon has no single exterior ring, so it is skipped.

        Test scenario:
            `_polygon_ring` reads `geom.exterior`, which a MultiPolygon does not have, so the
            row is dropped along with its attributes. Defensible for a single-ring face model,
            but it was silent and undocumented -- a caller handed a dissolved layer would get
            a mesh quietly missing faces.
        """
        multi = MultiPolygon(
            [
                Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
                Polygon([(4.0, 0.0), (5.0, 0.0), (5.0, 1.0), (4.0, 1.0)]),
            ]
        )

        _, _, faces, data = from_geodataframe(
            _frame([_UNIT, multi], values=[10.0, 20.0])
        )

        assert faces.shape[0] == 1, (
            f"the MultiPolygon contributes no face, got {faces.shape[0]}"
        )
        assert data["v"].tolist() == [10.0], (
            f"its attribute row goes with it, got {data['v'].tolist()}"
        )

    def test_exploding_it_first_keeps_both_parts(self):
        """The documented way to keep a MultiPolygon's parts.

        Test scenario:
            `GeoDataFrame.explode` turns each part into its own row, which is what the
            single-ring face model needs -- so the drop is a conversion step the caller owns,
            not data loss they cannot avoid.
        """
        multi = MultiPolygon(
            [
                Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
                Polygon([(4.0, 0.0), (5.0, 0.0), (5.0, 1.0), (4.0, 1.0)]),
            ]
        )
        frame = _frame([_UNIT, multi], values=[10.0, 20.0])

        _, _, faces, data = from_geodataframe(frame.explode(ignore_index=True))

        assert faces.shape[0] == 3, (
            f"the square plus the two exploded parts, got {faces.shape[0]}"
        )
        assert data["v"].tolist() == [10.0, 20.0, 20.0], (
            f"explode copies the attribute onto each part, got {data['v'].tolist()}"
        )
