"""The `fixed=` mask on the morphology facades.

The kernel has always accepted a `fixed` mask -- faces held at their input value through every
iteration -- but the dataset facades did not forward it, so it was unreachable from the public API
and untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def grid() -> UgridDataset:
    """A 3x3 quad grid with only face 0 set."""
    coords = np.arange(4, dtype=float)
    grid_x, grid_y = np.meshgrid(coords, coords)
    faces = [
        [row * 4 + col, row * 4 + col + 1, row * 4 + col + 5, row * 4 + col + 4]
        for row in range(3)
        for col in range(3)
    ]
    seed = np.zeros(9)
    seed[0] = 1.0
    return UgridDataset.from_arrays(
        node_x=grid_x.ravel(),
        node_y=grid_y.ravel(),
        face_node_connectivity=np.array(faces),
        data={"m": seed},
        data_locations={"m": "face"},
    )


def _mask(dataset: UgridDataset) -> list[int]:
    """The boolean face mask as 0/1."""
    return np.asarray(dataset["m"].data).astype(int).tolist()


class TestFixedMask:
    def test_dilation_grows_without_a_fixed_mask(self, grid):
        assert _mask(grid.binary_dilation("m", iterations=1)) == [
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ]

    def test_a_fixed_face_is_held_at_its_input(self, grid):
        held = np.zeros(9, dtype=bool)
        held[1] = True
        assert _mask(grid.binary_dilation("m", iterations=1, fixed=held)) == [
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ]

    def test_erosion_accepts_a_fixed_mask(self, grid):
        held = np.zeros(9, dtype=bool)
        held[0] = True
        assert _mask(grid.binary_erosion("m", iterations=1, fixed=held))[0] == 1

    def test_fixing_every_face_is_a_no_op(self, grid):
        held = np.ones(9, dtype=bool)
        assert _mask(grid.binary_dilation("m", iterations=2, fixed=held)) == _mask(grid)
