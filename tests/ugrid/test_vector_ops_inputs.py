"""The vector input shapes and value shapes `vector_ops` documents, exercised.

Three of the module's claims were about inputs no test covered: that a pyramids
`FeatureCollection` is accepted, that `polygonize` drops the faces with no value, and what
`burn` does with a geometry mixing a polygon and a line. The first worked for a different
reason than the docstring gave, and the other two were narrower than they read.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid import vector_ops as vo
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.core

_SQUARE = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _connectivity(data, cf_role):
    """A connectivity over `data` tagged with `cf_role`, zero-based and `-1`-padded."""
    return Connectivity(
        data=np.asarray(data, dtype="int64"),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


@pytest.fixture
def strip():
    """A 1 x 3 strip of unit quads along x, with centroids at 0.5, 1.5 and 2.5."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
        node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity(
            [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]], "face_node_connectivity"
        ),
    )


class TestFeatureCollectionInput:
    """A pyramids `FeatureCollection` as the geometry argument."""

    def test_it_is_accepted_as_a_geodataframe(self, strip):
        """`burn` takes a `FeatureCollection` directly.

        Test scenario:
            The docstring said this worked "via its `_gdf`". There is no such attribute -- it
            works because `FeatureCollection` subclasses `GeoDataFrame`, so the
            `isinstance(geometry, gpd.GeoDataFrame)` branch claims it. Naming a private
            attribute that does not exist sends a reader looking for a wrapper that is not
            there, and hides the one property the support actually rests on.
        """
        assert issubclass(FeatureCollection, gpd.GeoDataFrame), (
            "the GeoDataFrame branch is what accepts a FeatureCollection"
        )
        assert not hasattr(FeatureCollection, "_gdf"), (
            "there is no `_gdf` wrapper attribute to go through"
        )

        collection = FeatureCollection(geometry=[_SQUARE], crs=4326)
        out = vo.burn(strip, collection, 1.0, fill=0.0, all_touched=False)

        assert out.tolist() == [1.0, 0.0, 0.0], (
            f"the unit square covers only face 0's centroid, got {out.tolist()}"
        )

    def test_the_reduction_unions_every_row(self, strip):
        """A multi-row collection is unioned, not just read for its first geometry."""
        collection = FeatureCollection(
            geometry=[
                _SQUARE,
                Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
            ],
            crs=4326,
        )
        out = vo.burn(strip, collection, 1.0, fill=0.0, all_touched=False)

        assert out.tolist() == [1.0, 0.0, 1.0], (
            f"both end faces' centroids are covered, got {out.tolist()}"
        )


class TestPolygonizeDropsValuelessFaces:
    """Which faces `polygonize` leaves out."""

    def test_infinities_are_dropped_as_well_as_nan(self, strip):
        """`+inf` and `-inf` are dropped too, which "NaN are dropped" did not say.

        Test scenario:
            The filter is `np.isfinite`, so an infinite value is dropped exactly as NaN is. A
            reader taking the docstring literally would expect an `inf` row in the output.
        """
        frame = vo.polygonize(strip, np.array([1.0, np.inf, -np.inf]))

        assert frame["value"].tolist() == [1.0], (
            f"only the finite face should survive, got {frame['value'].tolist()}"
        )

    def test_decimals_is_ignored_for_a_label_field(self, strip):
        """`decimals` cannot round a label, so it is ignored rather than raising.

        Test scenario:
            `polygonize` recommends itself for a classified / label field, and `decimals` is
            meaningless there -- `np.round` on a string array raises. Passing both is accepted
            and the rounding simply does not apply, which was not documented.
        """
        labels = np.array(["a", "b", "b"], dtype=object)

        frame = vo.polygonize(strip, labels, decimals=2)

        assert sorted(frame["value"].tolist()) == ["a", "b"], (
            f"the labels should group unchanged, got {frame['value'].tolist()}"
        )


class TestBurnMixedGeometry:
    """A geometry mixing an area part with a zero-area part."""

    def test_a_line_alone_takes_the_intersection_rule(self, strip):
        """On its own, a line burns every face it crosses whatever `all_touched` says.

        Test scenario:
            The baseline the next test contrasts with. `geom.area == 0.0`, so the zero-area
            branch selects every intersecting candidate and `all_touched` is ignored -- as the
            docstring says.
        """
        line = LineString([(2.1, 0.7), (2.9, 0.7)])

        out = vo.burn(strip, line, 1.0, fill=0.0, all_touched=False)

        assert out.tolist() == [0.0, 0.0, 1.0], (
            f"the line crosses face 2 only, got {out.tolist()}"
        )

    def test_the_same_line_in_a_mixed_geometry_selects_nothing(self, strip):
        """Put that line beside a polygon and `all_touched=False` drops it.

        Test scenario:
            A `GeometryCollection` holding the square and the line has `area == 1.0`, so the
            whole geometry takes the polygon rule, and centroid-containment finds nothing on a
            line 0.2 above face 2's centroid. The identical line burned face 2 when it
            travelled alone. Which rule applies is decided by the *collection's* total area,
            not per part.
        """
        mixed = GeometryCollection([_SQUARE, LineString([(2.1, 0.7), (2.9, 0.7)])])

        out = vo.burn(strip, mixed, 1.0, fill=0.0, all_touched=False)

        assert out.tolist() == [1.0, 0.0, 0.0], (
            f"only the polygon's centroid hit survives, got {out.tolist()}"
        )

    def test_a_line_through_a_centroid_is_the_knife_edge(self, strip):
        """It is not that the line is ignored -- it is that only an exact hit counts.

        Test scenario:
            Move the same line onto face 2's centroid (y = 0.5) and `shapely.contains` finds
            the point on it, so face 2 burns after all. The zero-area part is not discarded; it
            just selects only a centroid lying exactly on it, which is a measure-zero
            coincidence rather than a usable rule.
        """
        mixed = GeometryCollection([_SQUARE, LineString([(2.1, 0.5), (2.9, 0.5)])])

        out = vo.burn(strip, mixed, 1.0, fill=0.0, all_touched=False)

        assert out.tolist() == [1.0, 0.0, 1.0], (
            f"the centroid lies on this line, so face 2 burns, got {out.tolist()}"
        )

    def test_all_touched_true_is_the_way_to_catch_both_parts(self, strip):
        """The documented remedy: intersection selection for the whole geometry.

        Test scenario:
            Face 1 burns too, because `intersects` includes the square's boundary touch at
            x = 1 -- which is what `all_touched=True` means.
        """
        mixed = GeometryCollection([_SQUARE, LineString([(2.1, 0.7), (2.9, 0.7)])])

        out = vo.burn(strip, mixed, 1.0, fill=0.0, all_touched=True)

        assert out.tolist() == [1.0, 1.0, 1.0], (
            f"every intersected face should burn, got {out.tolist()}"
        )

    def test_a_point_ignores_all_touched_either_way(self, strip):
        """A zero-area geometry takes the intersection rule under both settings."""
        point = Point(1.5, 0.5)

        strict = vo.burn(strip, point, 1.0, fill=0.0, all_touched=False)
        loose = vo.burn(strip, point, 1.0, fill=0.0, all_touched=True)

        assert strict.tolist() == [0.0, 1.0, 0.0], (
            f"a point burns its own face, got {strict.tolist()}"
        )
        assert loose.tolist() == strict.tolist(), (
            f"all_touched should not change a point's answer, got {loose.tolist()}"
        )
