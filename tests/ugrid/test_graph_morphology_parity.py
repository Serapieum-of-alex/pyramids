"""Mesh morphology against `scipy.ndimage` on a mesh that *is* a regular grid.

A quad mesh laid out as an NxN grid has exactly the edge-neighbour structure of
`scipy.ndimage`'s default cross structuring element, so the two must agree cell for cell. That
makes `ndimage` usable as ground truth — in particular for `border_value`, where imputing the
exterior and propagating across neighbours have to happen in the *same* pass. Applying the
exterior first and then propagating moves two rings per iteration instead of one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from pyramids.netcdf.ugrid import graph
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d


def _grid_mesh(size: int) -> Mesh2d:
    """A `size` x `size` grid of quads, face `r * size + c` at row r, column c."""
    coords = np.arange(size + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(coords, coords)
    faces = []
    stride = size + 1
    for row in range(size):
        for col in range(size):
            base = row * stride + col
            faces.append([base, base + 1, base + stride + 1, base + stride])
    return Mesh2d(
        node_x=grid_x.ravel(),
        node_y=grid_y.ravel(),
        face_node_connectivity=Connectivity(
            data=np.array(faces),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


def _as_grid(flat: np.ndarray, size: int) -> np.ndarray:
    """Reshape a face-ordered mask back to its grid layout."""
    return np.asarray(flat, dtype=bool).reshape(size, size)


@pytest.fixture
def size() -> int:
    """Grid edge length used by the parity tests."""
    return 3


class TestDilationMatchesNdimage:
    def test_border_value_true_preserves_the_centre(self, size):
        mesh = _grid_mesh(size)
        seed = np.zeros(size * size, dtype=bool)
        mine = _as_grid(
            graph.binary_dilation(mesh, seed, iterations=1, border_value=True), size
        )
        theirs = ndimage.binary_dilation(
            _as_grid(seed, size), iterations=1, border_value=True
        )
        np.testing.assert_array_equal(mine, theirs)

    def test_a_single_seed_grows_one_ring(self, size):
        mesh = _grid_mesh(size)
        seed = np.zeros(size * size, dtype=bool)
        seed[4] = True
        mine = _as_grid(graph.binary_dilation(mesh, seed, iterations=1), size)
        theirs = ndimage.binary_dilation(_as_grid(seed, size), iterations=1)
        np.testing.assert_array_equal(mine, theirs)

    def test_two_iterations_match(self):
        mesh = _grid_mesh(5)
        seed = np.zeros(25, dtype=bool)
        seed[12] = True
        mine = _as_grid(graph.binary_dilation(mesh, seed, iterations=2), 5)
        theirs = ndimage.binary_dilation(_as_grid(seed, 5), iterations=2)
        np.testing.assert_array_equal(mine, theirs)

    def test_border_value_true_on_a_larger_grid(self):
        mesh = _grid_mesh(5)
        seed = np.zeros(25, dtype=bool)
        mine = _as_grid(
            graph.binary_dilation(mesh, seed, iterations=1, border_value=True), 5
        )
        theirs = ndimage.binary_dilation(
            _as_grid(seed, 5), iterations=1, border_value=True
        )
        np.testing.assert_array_equal(mine, theirs)


class TestErosionMatchesNdimage:
    def test_the_default_border_preserves_the_centre(self, size):
        # border_value defaults to False, which equals `value` for erosion, so this is the
        # default-path case: the outside erodes the boundary ring and nothing further.
        mesh = _grid_mesh(size)
        full = np.ones(size * size, dtype=bool)
        mine = _as_grid(graph.binary_erosion(mesh, full, iterations=1), size)
        theirs = ndimage.binary_erosion(
            _as_grid(full, size), iterations=1, border_value=False
        )
        np.testing.assert_array_equal(mine, theirs)

    def test_erosion_of_a_patch_matches(self):
        mesh = _grid_mesh(5)
        patch = np.zeros((5, 5), dtype=bool)
        patch[1:4, 1:4] = True
        mine = _as_grid(graph.binary_erosion(mesh, patch.ravel(), iterations=1), 5)
        theirs = ndimage.binary_erosion(patch, iterations=1, border_value=False)
        np.testing.assert_array_equal(mine, theirs)

    def test_border_value_true_matches(self):
        mesh = _grid_mesh(5)
        patch = np.zeros((5, 5), dtype=bool)
        patch[1:4, 1:4] = True
        mine = _as_grid(
            graph.binary_erosion(mesh, patch.ravel(), iterations=1, border_value=True),
            5,
        )
        theirs = ndimage.binary_erosion(patch, iterations=1, border_value=True)
        np.testing.assert_array_equal(mine, theirs)


class TestInputIsNotMutated:
    def test_dilation_leaves_the_mask_alone(self, size):
        mesh = _grid_mesh(size)
        seed = np.zeros(size * size, dtype=bool)
        seed[4] = True
        before = seed.copy()
        graph.binary_dilation(mesh, seed, iterations=2, border_value=True)
        np.testing.assert_array_equal(seed, before)
