"""`rasterize_like` against a rotated reference grid.

The sample points came from `GeoTransform.x_axis` / `y_axis`, which document that they ignore the
rotation terms — "on a rotated grid there is no single x per column". The full geotransform,
rotation included, was then stamped onto the output, so the values were sampled on an
axis-aligned lattice but labelled as a rotated one. Cell centres now come from the full affine,
which is identical on an axis-aligned grid and correct on a rotated one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.regrid import rasterize_like

AXIS_ALIGNED = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
ROTATED = (0.0, 1.0, 0.5, 4.0, 0.25, -1.0)


@pytest.fixture
def mesh() -> UgridDataset:
    """A 4x4 extent meshed as two triangles, carrying a field linear in x."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 4.0, 4.0, 0.0]),
        node_y=np.array([0.0, 0.0, 4.0, 4.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"v": np.array([1.0, 2.0])},
    )


def _reference(geo: tuple[float, ...]) -> Dataset:
    """A 4x4 raster carrying `geo` as its geotransform."""
    return Dataset.from_array(
        np.zeros((4, 4), dtype="float64"),
        no_data_value=np.nan,
        geo_ref=GeoReference(geo=geo, epsg=4326),
    )


class TestRotatedReferenceGrid:
    def test_the_output_keeps_the_reference_geotransform(self, mesh):
        out = rasterize_like(mesh.mesh, np.array([1.0, 2.0]), _reference(ROTATED))
        assert tuple(out.geotransform) == pytest.approx(ROTATED)

    def test_the_output_has_the_reference_shape(self, mesh):
        out = rasterize_like(mesh.mesh, np.array([1.0, 2.0]), _reference(ROTATED))
        assert (out.rows, out.columns) == (4, 4)

    def test_a_rotated_grid_differs_from_the_axis_aligned_one(self, mesh):
        # The two grids cover different ground, so sampling the same mesh must not give the
        # identical array -- which is exactly what happened when rotation was ignored.
        rotated = rasterize_like(
            mesh.mesh, np.array([1.0, 2.0]), _reference(ROTATED), method="idw"
        )
        aligned = rasterize_like(
            mesh.mesh, np.array([1.0, 2.0]), _reference(AXIS_ALIGNED), method="idw"
        )
        assert not np.allclose(
            np.nan_to_num(rotated.read_array()), np.nan_to_num(aligned.read_array())
        )


class TestAxisAlignedIsUnchanged:
    def test_values_are_produced(self, mesh):
        out = rasterize_like(
            mesh.mesh, np.array([1.0, 2.0]), _reference(AXIS_ALIGNED), method="idw"
        )
        assert np.isfinite(out.read_array()).any()

    def test_the_geotransform_round_trips(self, mesh):
        out = rasterize_like(mesh.mesh, np.array([1.0, 2.0]), _reference(AXIS_ALIGNED))
        assert tuple(out.geotransform) == pytest.approx(AXIS_ALIGNED)

    def test_nearest_still_works(self, mesh):
        out = rasterize_like(
            mesh.mesh, np.array([1.0, 2.0]), _reference(AXIS_ALIGNED), method="nearest"
        )
        assert out.read_array().shape == (4, 4)


class TestRotatedSamplePositions:
    """Where the samples are actually taken, not merely that they differ."""

    @staticmethod
    def _plane(node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
        """The field `2x + 3y`, which a barycentric interpolation reproduces exactly."""
        return 2.0 * node_x + 3.0 * node_y

    @staticmethod
    def _centres(geo: tuple[float, ...], rows: int, cols: int):
        """Cell centre coordinates from the full affine, rotation terms included."""
        column_positions = np.arange(cols, dtype="float64") + 0.5
        row_positions = np.arange(rows, dtype="float64") + 0.5
        cc, rr = np.meshgrid(column_positions, row_positions)
        x = geo[0] + cc * geo[1] + rr * geo[2]
        y = geo[3] + cc * geo[4] + rr * geo[5]
        return x, y

    @pytest.fixture
    def plane_mesh(self) -> UgridDataset:
        """A square large enough to contain every rotated cell centre, carrying a plane."""
        return UgridDataset.from_arrays(
            node_x=np.array([-10.0, 20.0, 20.0, -10.0]),
            node_y=np.array([-10.0, -10.0, 20.0, 20.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"v": np.array([1.0, 2.0])},
        )

    def test_the_values_are_the_plane_at_the_true_rotated_centres(self, plane_mesh):
        """Every cell holds `2x + 3y` evaluated where that cell really is.

        Test scenario:
            `not np.allclose(rotated, aligned)` only says the two grids disagree, which a
            half-applied rotation also satisfies. A field linear in x and y is reproduced
            exactly by the barycentric interpolation, so each cell's value *is* the sample
            position -- and comparing against the plane at the centres derived from the full
            affine pins the position itself.
        """
        mesh = plane_mesh.mesh
        node_values = self._plane(np.asarray(mesh.node_x), np.asarray(mesh.node_y))

        out = rasterize_like(
            mesh,
            node_values,
            _reference(ROTATED),
            location="node",
            method="linear",
            nodata=np.nan,
        )
        true_x, true_y = self._centres(ROTATED, 4, 4)

        np.testing.assert_allclose(out.read_array(), self._plane(true_x, true_y))

    def test_it_would_fail_against_centres_that_ignore_the_rotation(self, plane_mesh):
        """The same comparison rejects the rotation-ignoring positions.

        Test scenario:
            This is what makes the test above load-bearing: dropping the `geo[2]` / `geo[4]`
            cross terms moves the centres far enough that the plane differs by up to 6.125 --
            a difference the previous `not allclose` check could not distinguish from the
            correct answer.
        """
        mesh = plane_mesh.mesh
        node_values = self._plane(np.asarray(mesh.node_x), np.asarray(mesh.node_y))

        out = rasterize_like(
            mesh,
            node_values,
            _reference(ROTATED),
            location="node",
            method="linear",
            nodata=np.nan,
        )
        flat_x, _ = self._centres((ROTATED[0], ROTATED[1], 0.0, 0.0, 0.0, 0.0), 4, 4)
        _, flat_y = self._centres((0.0, 0.0, 0.0, ROTATED[3], 0.0, ROTATED[5]), 4, 4)
        unrotated = self._plane(flat_x, flat_y)

        assert not np.allclose(out.read_array(), unrotated), (
            "the sampled values must not match the rotation-ignoring centres"
        )
        assert np.abs(out.read_array() - unrotated).max() == pytest.approx(6.125), (
            "the two sets of centres differ by 6.125 on this plane, which is the margin the "
            "assertion above relies on"
        )
