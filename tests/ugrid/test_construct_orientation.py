"""Face winding out of `from_structured`, across every geotransform orientation.

UGRID wants counter-clockwise faces, and `Mesh2d.face_areas`, `graph._face_polygons` and
`bounding_polygon` all rely on it. Orientation was keyed on the signs of the two pixel sizes,
which says nothing when both are 0.0 — an exact 90-degree rotation then read as north-up and every
face came out clockwise. The sign of the Jacobian determinant is what actually decides it.

The earlier suite only exercised the north-up geotransform, so none of this was visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid.construct import from_structured

ORIENTATIONS = {
    "north_up": (0.0, 1.0, 0.0, 2.0, 0.0, -1.0),
    "both_positive": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    "both_negative": (0.0, -1.0, 0.0, 2.0, 0.0, -1.0),
    "mirrored": (0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    "rotated_exactly_90": (0.0, 0.0, 3.0, 0.0, 2.0, 0.0),
    "rotated_exactly_minus_90": (0.0, 0.0, -3.0, 0.0, -2.0, 0.0),
    "rotated_270": (0.0, 0.0, -3.0, 0.0, 2.0, 0.0),
    "rotated_45": (0.0, 0.7, 0.7, 0.0, 0.7, -0.7),
}


def _reference(geo: tuple[float, ...]) -> Dataset:
    """A 2x3 raster carrying `geo`."""
    return Dataset.from_array(
        np.zeros((2, 3), dtype="float64"),
        no_data_value=np.nan,
        geo_ref=GeoReference(geo=geo, epsg=4326),
    )


def _signed_areas(geo: tuple[float, ...]) -> list[float]:
    """The shoelace signed area of every face built from `geo`; positive is counter-clockwise."""
    node_x, node_y, faces, _ = from_structured(_reference(geo))
    areas = []
    for face in faces:
        xs, ys = node_x[face], node_y[face]
        areas.append(float(0.5 * np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)))
    return areas


class TestFacesAreCounterClockwise:
    @pytest.mark.parametrize("name", sorted(ORIENTATIONS))
    def test_every_face_is_counter_clockwise(self, name):
        areas = _signed_areas(ORIENTATIONS[name])
        assert len(areas) == 6
        assert all(area > 0.0 for area in areas), areas

    @pytest.mark.parametrize("name", sorted(ORIENTATIONS))
    def test_every_face_has_the_same_area(self, name):
        areas = _signed_areas(ORIENTATIONS[name])
        assert areas == pytest.approx([areas[0]] * 6)

    def test_an_exact_90_degree_rotation_has_the_pixel_area(self):
        # 3 by 2 pixels rotated onto the axes: each face covers 6 square units.
        assert _signed_areas(ORIENTATIONS["rotated_exactly_90"])[0] == pytest.approx(
            6.0
        )

    def test_the_north_up_case_is_unchanged(self):
        assert _signed_areas(ORIENTATIONS["north_up"])[0] == pytest.approx(1.0)
