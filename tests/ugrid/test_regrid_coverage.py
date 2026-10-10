"""Error paths, degenerate inputs and weight metadata of ``ugrid.regrid``.

``test_regrid.py`` pins the happy paths of the four B-epic capabilities, and
``test_regrid_conservation.py`` / ``test_regrid_nodata.py`` / ``test_rasterize_rotated.py`` /
``test_sample_boundary.py`` pin their numerical contracts. What none of them reach are the guard
clauses and the one-element corners, which is what this file pins:

- every ``ValueError`` :func:`sample`, :func:`rasterize_like`, :func:`interp_time` and
  :class:`Regridder` raise, asserted on the message so a reworded guard cannot silently become a
  different error;
- the all-points-outside short circuit of the barycentric sampler, which must hand back the
  caller's ``nodata`` fill rather than an empty array;
- the cross-CRS branch of :func:`rasterize_like`, where the mesh source points are reprojected
  into the reference grid's CRS before sampling;
- the single-source-point IDW path, where ``cKDTree.query(k=1)`` returns 1-D arrays that have to
  be re-expanded;
- the ``method`` / ``n_source`` / ``n_target`` accessors a stored weight matrix is reused through.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.regrid import Regridder, interp_time, rasterize_like, sample

pytestmark = pytest.mark.core

#: Unit square split by the diagonal from (0, 0) to (1, 1).
NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)

FACE_VALUES = np.array([10.0, 20.0])
NODE_VALUES = np.array([1.0, 3.0, 6.0, 4.0])


@pytest.fixture
def triangles() -> UgridDataset:
    """The two-triangle unit square, carrying one face variable and one node variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"f": FACE_VALUES, "z": NODE_VALUES},
        data_locations={"f": "face", "z": "node"},
        epsg=4326,
    )


@pytest.fixture
def grid_2x2() -> UgridDataset:
    """A 2x2 grid of unit quads over ``[0, 2] x [0, 2]``, one value per face."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.intp
        ),
        data={"v": np.array([1.0, 2.0, 3.0, 4.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )


@pytest.fixture
def one_quad() -> UgridDataset:
    """A single-face mesh covering ``[0, 2] x [0, 2]`` -- exactly one source point per location."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 2.0, 2.0, 0.0]),
        node_y=np.array([0.0, 0.0, 2.0, 2.0]),
        face_node_connectivity=np.array([[0, 1, 2, 3]], dtype=np.intp),
        data={"v": np.array([7.0])},
        data_locations={"v": "face"},
        epsg=4326,
    )


def _reference(geo: tuple[float, ...], rows: int, cols: int, epsg: int) -> Dataset:
    """A zero-filled reference raster carrying ``geo`` / ``epsg``, for ``rasterize_like``."""
    return Dataset.from_array(
        np.zeros((rows, cols), dtype="float64"),
        no_data_value=np.nan,
        geo_ref=GeoReference(geo=geo, epsg=epsg),
    )


class TestSampleGuards:
    """The location / method guards of :func:`sample`."""

    def test_an_unknown_location_is_refused(self, triangles):
        """Only 'face' and 'node' are samplable; 'edge' is refused by name."""
        with pytest.raises(ValueError, match="'face' and 'node' locations"):
            sample(triangles.mesh, FACE_VALUES, 0.5, 0.25, location="edge")

    def test_nearest_on_face_data_takes_the_closest_centroid(self, triangles):
        """`nearest` on face data resolves an outside point to the nearest face centroid."""
        got = sample(
            triangles.mesh,
            FACE_VALUES,
            np.array([5.0]),
            np.array([0.0]),
            method="nearest",
            location="face",
        )
        assert got[0] == 10.0

    def test_an_unknown_method_on_face_data_is_refused(self, triangles):
        """An unrecognised method on face data names the three it accepts."""
        with pytest.raises(ValueError, match="'contains', 'nearest' or 'linear'"):
            sample(
                triangles.mesh, FACE_VALUES, 0.5, 0.25, method="bogus", location="face"
            )

    def test_contains_on_node_data_is_refused(self, triangles):
        """Point-in-face has no meaning for node data, so `contains` is refused there."""
        with pytest.raises(ValueError, match="only valid for face data"):
            sample(
                triangles.mesh,
                NODE_VALUES,
                0.5,
                0.25,
                method="contains",
                location="node",
            )

    def test_an_unknown_method_on_node_data_is_refused(self, triangles):
        """An unrecognised method on node data names the two it accepts."""
        with pytest.raises(ValueError, match="'nearest' or 'linear' for node data"):
            sample(
                triangles.mesh, NODE_VALUES, 0.5, 0.25, method="bogus", location="node"
            )

    def test_barycentric_with_every_point_outside_returns_the_nodata_fill(
        self, triangles
    ):
        """When no query point lands in any face the barycentric sampler returns the fill."""
        got = sample(
            triangles.mesh,
            NODE_VALUES,
            np.array([50.0, 60.0]),
            np.array([50.0, 60.0]),
            method="linear",
            location="node",
            nodata=-1.0,
        )
        np.testing.assert_array_equal(got, [-1.0, -1.0])

    def test_barycentric_keeps_the_time_axis_when_all_points_are_outside(
        self, triangles
    ):
        """The all-outside short circuit preserves the leading time axis of the fill array."""
        values = np.tile(NODE_VALUES, (3, 1))
        got = sample(
            triangles.mesh,
            values,
            np.array([50.0, 60.0]),
            np.array([50.0, 60.0]),
            method="linear",
            location="node",
            nodata=-1.0,
        )
        assert got.shape == (3, 2)
        np.testing.assert_array_equal(got[0], [-1.0, -1.0])


class TestRasterizeLikeGuards:
    """The method guard and the two less-travelled interpolators of :func:`rasterize_like`."""

    def test_an_unknown_method_is_refused(self, grid_2x2):
        """An unrecognised method names the three it accepts."""
        reference = _reference((0.0, 0.5, 0.0, 2.0, 0.0, -0.5), 4, 4, 4326)
        with pytest.raises(ValueError, match="'nearest', 'linear' or 'idw'"):
            rasterize_like(grid_2x2.mesh, grid_2x2["v"].data, reference, method="bogus")

    def test_the_linear_method_stays_within_the_source_range(self, grid_2x2):
        """Linear interpolation of face values produces interior values bounded by the source."""
        reference = _reference((0.0, 0.5, 0.0, 2.0, 0.0, -0.5), 4, 4, 4326)
        out = rasterize_like(
            grid_2x2.mesh,
            grid_2x2["v"].data,
            reference,
            method="linear",
            nodata=-9999.0,
        )
        arr = out.read_array()
        assert arr.shape == (4, 4)
        interior = arr[arr != -9999.0]
        assert interior.size > 0
        assert interior.min() >= 1.0
        assert interior.max() <= 4.0

    def test_a_differing_source_epsg_reprojects_the_mesh_points(self, triangles):
        """With `source_epsg` set the mesh points are reprojected into the reference CRS."""
        # The mesh spans [0, 1] degrees, which is ~[0, 111_320] m in EPSG:3857. The reference
        # grid lives in those metres, so the far corner of the grid is only reachable once the
        # source points have been pushed out to their projected position.
        reference = _reference((0.0, 40000.0, 0.0, 120000.0, 0.0, -40000.0), 3, 3, 3857)
        reprojected = rasterize_like(
            triangles.mesh,
            FACE_VALUES,
            reference,
            method="nearest",
            nodata=-9999.0,
            source_epsg=4326,
        )
        untouched = rasterize_like(
            triangles.mesh, FACE_VALUES, reference, method="nearest", nodata=-9999.0
        )

        values = reprojected.read_array()
        assert set(np.unique(values)) <= {-9999.0, 10.0, 20.0}
        # Top-right cell centre is (100_000, 100_000) m: in range of a projected source point,
        # out of range of an unprojected one.
        assert values[0, 2] in (10.0, 20.0)
        assert untouched.read_array()[0, 2] == -9999.0

    def test_a_single_source_point_idw_fills_every_cell_in_range(self, one_quad):
        """One source point makes `cKDTree.query(k=1)` return 1-D arrays; IDW still fills."""
        reference = _reference((0.0, 1.0, 0.0, 2.0, 0.0, -1.0), 2, 2, 4326)
        out = rasterize_like(
            one_quad.mesh,
            one_quad["v"].data,
            reference,
            method="idw",
            nodata=-9999.0,
        )
        np.testing.assert_allclose(out.read_array(), np.full((2, 2), 7.0))

    def test_a_single_source_point_idw_honours_max_distance(self, one_quad):
        """The k=1 path still refuses cells whose only source lies beyond `max_distance`."""
        reference = _reference((0.0, 1.0, 0.0, 2.0, 0.0, -1.0), 2, 2, 4326)
        out = rasterize_like(
            one_quad.mesh,
            one_quad["v"].data,
            reference,
            method="idw",
            nodata=-9999.0,
            max_distance=0.1,
        )
        np.testing.assert_array_equal(out.read_array(), np.full((2, 2), -9999.0))


class TestInterpTimeGuards:
    """The four input guards of :func:`interp_time`."""

    def test_a_two_dimensional_time_coordinate_is_refused(self):
        """`times` must be 1-D; a 2-D coordinate is refused before anything else is checked."""
        with pytest.raises(ValueError, match="`times` must be 1-D"):
            interp_time(np.zeros((2, 2)), np.zeros((2, 3)), np.array([0.5]))

    def test_a_length_mismatch_on_the_time_axis_is_refused(self):
        """A `values` time axis shorter than `times` is reported with both lengths."""
        with pytest.raises(ValueError, match="steps on axis 0"):
            interp_time(np.array([0.0, 1.0, 2.0]), np.zeros((2, 4)), np.array([0.5]))

    def test_an_unknown_method_is_refused(self):
        """An unrecognised method names the two it accepts."""
        with pytest.raises(ValueError, match="'linear' or 'nearest'"):
            interp_time(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.5]),
                method="cubic",
            )

    def test_an_unknown_bounds_policy_is_refused(self):
        """An unrecognised out-of-range policy names the two it accepts."""
        with pytest.raises(ValueError, match="bounds must be 'nan' or 'clamp'"):
            interp_time(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.5]),
                bounds="extrapolate",
            )


class TestRegridderMetadata:
    """The accessors and guards around the reusable weight matrix."""

    @pytest.fixture
    def target(self) -> UgridDataset:
        """A single-face target covering the whole 2x2 source extent."""
        return UgridDataset.from_arrays(
            node_x=np.array([0.0, 2.0, 2.0, 0.0]),
            node_y=np.array([0.0, 0.0, 2.0, 2.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]], dtype=np.intp),
        )

    def test_the_method_is_reported(self, grid_2x2, target):
        """A regridder remembers the method it was built with."""
        assert (
            Regridder(grid_2x2.mesh, target.mesh, method="conservative").method
            == "conservative"
        )

    def test_the_element_counts_are_reported(self, grid_2x2, target):
        """`n_source` / `n_target` report the face counts of the two meshes."""
        regridder = Regridder(grid_2x2.mesh, target.mesh, method="overlap")
        assert regridder.n_source == 4
        assert regridder.n_target == 1

    def test_a_reused_matrix_reports_custom_and_its_own_shape(self, grid_2x2, target):
        """`from_weights` reports method 'custom' and the element counts of the matrix."""
        stored = Regridder(grid_2x2.mesh, target.mesh, method="overlap").weights
        reused = Regridder.from_weights(stored, target.mesh)
        assert reused.method == "custom"
        assert reused.n_source == 4
        assert reused.n_target == 1

    def test_a_non_face_source_location_is_refused(self, grid_2x2, target):
        """Only face-located source data can be regridded."""
        with pytest.raises(ValueError, match="face-located source data"):
            Regridder(grid_2x2.mesh, target.mesh, source_location="node")

    def test_a_wrong_source_length_is_refused(self, grid_2x2, target):
        """Applying the weights to the wrong number of source elements is refused."""
        regridder = Regridder(grid_2x2.mesh, target.mesh, method="overlap")
        with pytest.raises(ValueError, match="expected n_source=4"):
            regridder.regrid(np.array([1.0, 2.0, 3.0]))

    def test_centroid_locator_is_an_alias_of_centroid(self, grid_2x2, target):
        """The 'centroid_locator' spelling builds the same weights as 'centroid'."""
        alias = Regridder(grid_2x2.mesh, target.mesh, method="centroid_locator")
        plain = Regridder(grid_2x2.mesh, target.mesh, method="centroid")
        np.testing.assert_allclose(
            alias.weights.toarray(), plain.weights.toarray(), atol=1e-12
        )
