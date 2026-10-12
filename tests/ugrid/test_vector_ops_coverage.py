"""Input normalisation, guards and degenerate faces in ``pyramids.netcdf.ugrid.vector_ops``.

``burn`` accepts any of four vector-input shapes and funnels them through ``_as_single_geometry``;
the mainline tests only ever hand it a shapely geometry or a ``GeoDataFrame``, leaving the
``_gdf``-wrapper, ``GeoSeries`` and bare-iterable shapes — and the empty-geometry short circuit —
unexercised. These tests pin all four shapes, the ``sample_line`` line-coercion and ``polygonize``
length guards, and the one degenerate-mesh case where a face is offered by the STRtree but has no
real intersection with the line.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, Point

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.vector_ops import burn, polygonize, sample_line

pytestmark = pytest.mark.core


@pytest.fixture
def grid_2x2() -> Mesh2d:
    """2x2 quad mesh over ``[0, 2] x [0, 2]``, faces in row-major order."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    return UgridDataset.from_arrays(node_x, node_y, faces).mesh


@pytest.fixture
def collapsed_face_mesh() -> Mesh2d:
    """A unit quad plus a second face whose four nodes all sit on the point ``(2, 0.5)``.

    The collapsed face has zero extent, so its polygon is degenerate: the STRtree still offers it
    as an ``intersects`` candidate for a line through ``(2, 0.5)``, yet the actual intersection is
    empty.
    """
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [4, 4, 4, 4]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


class TestGeometryInputShapes:
    """The vector-input shapes ``burn`` normalises before selecting faces."""

    def test_a_geoseries_is_unioned(self, grid_2x2):
        """A ``GeoSeries`` of points burns every face containing one of them."""
        series = gpd.GeoSeries([Point(0.5, 0.5), Point(1.5, 1.5)])
        burned = burn(grid_2x2, series, 2.0, fill=0.0)
        assert burned.tolist() == [2.0, 0.0, 0.0, 2.0]

    def test_a_plain_iterable_of_geometries_is_unioned(self, grid_2x2):
        """A bare list of shapely geometries is unioned and burned."""
        burned = burn(grid_2x2, [Point(0.5, 0.5), Point(1.5, 0.5)], 3.0, fill=0.0)
        assert burned.tolist() == [3.0, 3.0, 0.0, 0.0]

    def test_an_empty_geometry_burns_nothing(self, grid_2x2):
        """An empty geometry short-circuits: every face keeps the fill value."""
        burned = burn(grid_2x2, GeometryCollection(), 9.0, fill=0.0)
        assert burned.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_an_empty_geoseries_burns_nothing(self, grid_2x2):
        """An empty ``GeoSeries`` unions to an empty geometry and burns nothing."""
        burned = burn(grid_2x2, gpd.GeoSeries([], dtype=object), 9.0, fill=0.0)
        assert burned.tolist() == [0.0, 0.0, 0.0, 0.0]


class TestLineCoercion:
    """``sample_line`` coercing its ``line`` argument."""

    def test_a_non_line_geometry_is_rejected(self, grid_2x2):
        """A shapely geometry that is not a ``LineString`` raises, naming its geometry type."""
        point = Point(0.5, 0.5)
        with pytest.raises(TypeError, match="line must be a LineString, got Point"):
            sample_line(grid_2x2, np.zeros(4), point)

    def test_a_polygon_is_rejected_too(self, grid_2x2):
        """The same guard rejects a polygon rather than silently sampling its boundary."""
        polygon = shapely.box(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(TypeError, match="got Polygon"):
            sample_line(grid_2x2, np.zeros(4), polygon)


class TestPolygonizeLengthGuard:
    """``polygonize`` checking the value array against the face count."""

    def test_a_value_array_shorter_than_n_face_is_rejected(self, grid_2x2):
        """The error names both the given length and the mesh's face count."""
        with pytest.raises(ValueError, match="values length 3 does not match n_face 4"):
            polygonize(grid_2x2, np.array([1.0, 2.0, 3.0]))


class TestDegenerateFaceSampling:
    """``sample_line`` over a mesh holding a zero-extent face."""

    def test_a_collapsed_face_on_the_line_contributes_no_sample(
        self, collapsed_face_mesh
    ):
        """The degenerate face is skipped; only the real quad the line crosses is sampled."""
        line = LineString([(0.5, 0.5), (3.0, 0.5)])
        distances, sampled, faces = sample_line(
            collapsed_face_mesh, np.array([10.0, 20.0]), line
        )

        np.testing.assert_array_equal(faces, [0])
        np.testing.assert_array_equal(sampled, [10.0])
        assert distances.shape == (1,)
        assert distances[0] == pytest.approx(0.25)
