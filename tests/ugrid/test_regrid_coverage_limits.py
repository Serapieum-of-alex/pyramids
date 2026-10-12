"""Which regrid methods can answer NaN for an uncovered target, and which cannot.

`regrid`'s own docstring says a target with no source contribution is NaN. That is a
property of the weight matrix, and two of the four methods do not produce an empty row for
an uncovered target: `centroid` always finds a nearest source face, however far away, and
`barycentric` triangulates the source *centroids*, whose hull is strictly smaller than the
source footprint. Both are defensible, and neither was stated.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.regrid import Regridder

pytestmark = pytest.mark.core


def _connectivity(data, cf_role):
    """A connectivity over `data` tagged with `cf_role`, zero-based and `-1`-padded."""
    return Connectivity(
        data=np.asarray(data, dtype="int64"),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


def _grid_mesh(x0, y0, nx, ny, step=1.0):
    """An `nx` x `ny` quad mesh whose lower-left corner is at `(x0, y0)`."""
    xs = x0 + step * np.arange(nx + 1, dtype="float64")
    ys = y0 + step * np.arange(ny + 1, dtype="float64")
    grid_x, grid_y = np.meshgrid(xs, ys)
    node_x = grid_x.ravel()
    node_y = grid_y.ravel()
    faces = []
    for row in range(ny):
        for col in range(nx):
            lower_left = row * (nx + 1) + col
            faces.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + nx + 2,
                    lower_left + nx + 1,
                ]
            )
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=_connectivity(faces, "face_node_connectivity"),
    )


class TestCentroidReach:
    """`centroid` copies from the nearest source face at any distance."""

    def test_a_distant_target_still_gets_a_value(self):
        """A target nowhere near the source is filled, not NaN.

        Test scenario:
            The source is a single unit face at the origin and the target a single unit face a
            thousand units away. `locate_nearest_face` has no cutoff, so the target's row holds
            weight 1.0 on that one source face -- the nearest-source copy is unbounded, and no
            row is ever empty.
        """
        source = _grid_mesh(0.0, 0.0, 1, 1)
        target = _grid_mesh(1000.0, 1000.0, 1, 1)

        out = Regridder(source, target, method="centroid").regrid(np.array([7.0]))

        assert out.tolist() == [7.0], (
            f"the nearest-source copy reaches any distance, so expected [7.0], got "
            f"{out.tolist()}"
        )

    def test_no_weight_row_is_empty(self):
        """Every target row carries exactly one unit weight.

        Test scenario:
            This is why `centroid` can never answer NaN: `regrid` NaNs only the rows with no
            weight, and this method gives every row one.
        """
        source = _grid_mesh(0.0, 0.0, 2, 2)
        target = _grid_mesh(50.0, 50.0, 2, 2)

        weights = Regridder(source, target, method="centroid").weights
        row_sums = np.asarray(weights.sum(axis=1)).ravel()

        assert np.allclose(row_sums, 1.0), (
            f"every target row should sum to 1.0, got {row_sums.tolist()}"
        )


class TestBarycentricHull:
    """`barycentric`'s hull is the hull of the source centroids, not of the source mesh."""

    def test_a_target_inside_the_source_can_still_be_nan(self):
        """A target face within the source footprint, but outside the centroid hull, is NaN.

        Test scenario:
            A 3x3 unit source over [0, 3]^2 has centroids at 0.5/1.5/2.5, so their convex hull
            is only [0.5, 2.5]^2. A target face occupying [0, 1] x [0, 1] has its centroid at
            (0.5, 0.5) -- a hull *corner*, inside -- while one at [0, 0.4]^2 has its centroid
            at (0.2, 0.2), inside the source mesh yet outside that hull. The docstring's
            "targets outside the hull are NaN" reads as the source footprint and is not.
        """
        source = _grid_mesh(0.0, 0.0, 3, 3)
        corner_target = _grid_mesh(0.0, 0.0, 1, 1, step=0.4)

        out = Regridder(source, corner_target, method="barycentric").regrid(
            np.arange(9.0)
        )

        assert np.isnan(out).all(), (
            f"a target centroid at (0.2, 0.2) sits inside the source mesh but outside the "
            f"centroid hull [0.5, 2.5]^2, so it is NaN; got {out.tolist()}"
        )

    def test_a_target_inside_the_centroid_hull_is_interpolated(self):
        """Within the centroid hull the interpolation is the ordinary one.

        Test scenario:
            The contrast case: a target face centred at (1.5, 1.5) is well inside
            [0.5, 2.5]^2, so it interpolates rather than returning NaN.
        """
        source = _grid_mesh(0.0, 0.0, 3, 3)
        inner_target = _grid_mesh(1.0, 1.0, 1, 1)

        out = Regridder(source, inner_target, method="barycentric").regrid(
            np.full(9, 5.0)
        )

        assert not np.isnan(out).any(), (
            f"expected an interpolated value, got {out.tolist()}"
        )
        assert out == pytest.approx([5.0]), (
            f"barycentric weights sum to 1, so a constant field stays constant; got "
            f"{out.tolist()}"
        )
