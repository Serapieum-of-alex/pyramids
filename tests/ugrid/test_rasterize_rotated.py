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
