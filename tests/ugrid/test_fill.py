"""Tests for mesh-space gap filling (``pyramids.netcdf.ugrid.fill``).

Covers nearest-valid spatial fill (:func:`fill_na_spatial`) and Laplace
interpolation (:func:`laplace_interpolate`) over small meshes built with
:meth:`UgridDataset.from_arrays` and hand-built :class:`Mesh2d` objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Connectivity, Mesh2d, UgridDataset
from pyramids.netcdf.ugrid.fill import fill_na_spatial, laplace_interpolate


def _quad_grid(
    nx: int, ny: int, x0: float = 0.0, node_offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular ``nx`` by ``ny`` grid of unit quad faces.

    Args:
        nx: Number of faces along x.
        ny: Number of faces along y.
        x0: X-origin of the lower-left node (used to offset a second grid).
        node_offset: Added to every node index in the face table (used when the
            grid's nodes are concatenated after another grid's nodes).

    Returns:
        Tuple ``(node_x, node_y, faces)`` where ``faces`` is an
        ``(nx * ny, 4)`` connectivity table.
    """
    xs = np.arange(nx + 1, dtype=float) + x0
    ys = np.arange(ny + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)
    node_x = grid_x.ravel()
    node_y = grid_y.ravel()
    stride = nx + 1
    faces = []
    for r in range(ny):
        for c in range(nx):
            n0 = r * stride + c + node_offset
            faces.append([n0, n0 + 1, n0 + 1 + stride, n0 + stride])
    return node_x, node_y, np.array(faces, dtype=np.intp)


def _grid_mesh(nx: int, ny: int) -> Mesh2d:
    """Return the :class:`Mesh2d` of a regular ``nx`` by ``ny`` quad grid."""
    node_x, node_y, faces = _quad_grid(nx, ny)
    return UgridDataset.from_arrays(node_x, node_y, faces).mesh


class TestFillNaSpatial:
    """Nearest-valid spatial fill."""

    def test_single_hole_fills_to_constant(self):
        """A single gap in a constant face field fills to that constant."""
        mesh = _grid_mesh(3, 3)
        values = np.full(mesh.n_face, 5.0)
        values[4] = np.nan
        filled = fill_na_spatial(mesh, values, "face")
        np.testing.assert_array_almost_equal(filled, np.full(mesh.n_face, 5.0))

    def test_picks_nearest_valid_value(self, triangle_mesh):
        """The gap takes the value of the nearest valid element."""
        filled = fill_na_spatial(triangle_mesh, np.array([5.0, np.nan]), "face")
        assert filled[1] == 5.0
        filled = fill_na_spatial(triangle_mesh, np.array([np.nan, 7.0]), "face")
        assert filled[0] == 7.0

    def test_max_distance_leaves_far_gap_nan(self, triangle_mesh):
        """A gap farther than ``max_distance`` from any valid element stays NaN."""
        cx, _ = triangle_mesh.face_centroids
        gap = abs(cx[1] - cx[0])
        filled = fill_na_spatial(
            triangle_mesh, np.array([5.0, np.nan]), "face", max_distance=0.5 * gap
        )
        assert np.isnan(filled[1])
        filled = fill_na_spatial(
            triangle_mesh, np.array([5.0, np.nan]), "face", max_distance=2.0 * gap
        )
        assert filled[1] == 5.0

    def test_node_location(self):
        """Nearest fill works for node-located data."""
        mesh = _grid_mesh(3, 3)
        values = np.full(mesh.n_node, 2.0)
        values[5] = np.nan
        filled = fill_na_spatial(mesh, values, "node")
        np.testing.assert_array_almost_equal(filled, np.full(mesh.n_node, 2.0))

    def test_edge_location(self):
        """Nearest fill works for edge-located data using edge centers."""
        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.0, 1.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 3, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            edge_x=np.array([0.0, 1.0, 5.0]),
            edge_y=np.array([0.0, 0.0, 0.0]),
        )
        filled = fill_na_spatial(mesh, np.array([10.0, np.nan, 30.0]), "edge")
        assert filled[1] == 10.0

    def test_missing_edge_coords_raises(self, triangle_mesh):
        """Edge fill without edge coordinates raises ValueError."""
        with pytest.raises(ValueError, match="Edge coordinates"):
            fill_na_spatial(triangle_mesh, np.array([1.0, np.nan]), "edge")

    def test_temporal_filled_per_slab(self):
        """Each time slab is filled independently with its own null mask."""
        mesh = _grid_mesh(3, 3)
        values = np.full((2, mesh.n_face), 5.0)
        values[0, 1] = np.nan
        values[1, 7] = np.nan
        filled = fill_na_spatial(mesh, values, "face")
        np.testing.assert_array_almost_equal(filled, np.full((2, mesh.n_face), 5.0))

    def test_input_not_mutated(self):
        """The input array is not modified in place."""
        mesh = _grid_mesh(3, 3)
        values = np.full(mesh.n_face, 5.0)
        values[4] = np.nan
        before = values.copy()
        fill_na_spatial(mesh, values, "face")
        np.testing.assert_array_equal(
            values[~np.isnan(values)], before[~np.isnan(before)]
        )
        assert np.isnan(values[4])


class TestLaplaceInterpolate:
    """Laplace interpolation over the mesh graph."""

    def test_linear_field_recovered_face(self):
        """A linear face field with an interior block of gaps is recovered to 1e-6."""
        mesh = _grid_mesh(5, 5)
        cx, cy = mesh.face_centroids
        field = 2.0 * cx + 3.0 * cy + 1.0
        holed = field.copy()
        rows = np.arange(5)[:, None]
        cols = np.arange(5)[None, :]
        interior = ((rows >= 1) & (rows <= 3) & (cols >= 1) & (cols <= 3)).ravel()
        holed[interior] = np.nan
        filled = laplace_interpolate(mesh, holed, "face")
        np.testing.assert_allclose(filled, field, atol=1e-6)

    def test_linear_field_recovered_node(self):
        """A linear node field with interior gaps is recovered to 1e-6."""
        mesh = _grid_mesh(5, 5)
        field = 2.0 * mesh.node_x + 3.0 * mesh.node_y
        holed = field.copy()
        stride = 6
        interior = np.zeros(mesh.n_node, dtype=bool)
        for r in range(1, 5):
            for c in range(1, 5):
                interior[r * stride + c] = True
        holed[interior] = np.nan
        filled = laplace_interpolate(mesh, holed, "node")
        np.testing.assert_allclose(filled, field, atol=1e-6)

    def test_fully_null_component_stays_nan(self):
        """A connected component that is entirely NaN is left untouched."""
        nx0, nodes0, faces0 = _quad_grid(3, 3)
        n_nodes0 = len(nx0)
        nx1, nodes1, faces1 = _quad_grid(3, 3, x0=20.0, node_offset=n_nodes0)
        node_x = np.concatenate([nx0, nx1])
        node_y = np.concatenate([nodes0, nodes1])
        faces = np.concatenate([faces0, faces1], axis=0)
        mesh = UgridDataset.from_arrays(node_x, node_y, faces).mesh
        cx, cy = mesh.face_centroids
        values = 2.0 * cx + 3.0 * cy + 1.0
        values[4] = np.nan  # interior gap in the first component
        values[9:] = np.nan  # second component entirely NaN
        filled = laplace_interpolate(mesh, values, "face")
        assert np.isfinite(filled[4])
        np.testing.assert_allclose(
            filled[4], 2.0 * cx[4] + 3.0 * cy[4] + 1.0, atol=1e-6
        )
        assert np.all(np.isnan(filled[9:]))

    def test_edge_location_raises(self):
        """Edge-located Laplace is not supported and raises ValueError."""
        mesh = _grid_mesh(2, 2)
        with pytest.raises(ValueError, match="edge"):
            laplace_interpolate(mesh, np.full(mesh.n_face, np.nan), "edge")

    def test_all_nan_raises(self):
        """An all-NaN input raises ValueError."""
        mesh = _grid_mesh(3, 3)
        with pytest.raises(ValueError, match="every value is NaN"):
            laplace_interpolate(mesh, np.full(mesh.n_face, np.nan), "face")

    def test_direct_and_cg_agree(self):
        """The direct (spsolve) and CG paths produce the same solution."""
        mesh = _grid_mesh(5, 5)
        cx, cy = mesh.face_centroids
        field = 2.0 * cx - 1.5 * cy + 0.5
        holed = field.copy()
        rows = np.arange(5)[:, None]
        cols = np.arange(5)[None, :]
        interior = ((rows >= 1) & (rows <= 3) & (cols >= 1) & (cols <= 3)).ravel()
        holed[interior] = np.nan
        direct = laplace_interpolate(mesh, holed.copy(), "face", direct_solve=True)
        iterative = laplace_interpolate(mesh, holed.copy(), "face", direct_solve=False)
        np.testing.assert_allclose(direct, iterative, atol=1e-6)

    def test_uniform_weights_path(self):
        """The ``use_weights=False`` (uniform) path also recovers a linear field."""
        mesh = _grid_mesh(5, 5)
        cx, cy = mesh.face_centroids
        field = cx + cy
        holed = field.copy()
        holed[12] = np.nan
        filled = laplace_interpolate(mesh, holed, "face", use_weights=False)
        np.testing.assert_allclose(filled, field, atol=1e-6)

    def test_temporal_broadcast(self):
        """Each time slab is solved independently."""
        mesh = _grid_mesh(5, 5)
        cx, cy = mesh.face_centroids
        field = 2.0 * cx + 3.0 * cy
        values = np.stack([field, field + 10.0])
        values[0, 12] = np.nan
        values[1, 7] = np.nan
        filled = laplace_interpolate(mesh, values, "face")
        expected = np.stack([field, field + 10.0])
        np.testing.assert_allclose(filled, expected, atol=1e-6)

    def test_input_not_mutated(self):
        """The input array is not modified in place."""
        mesh = _grid_mesh(5, 5)
        cx, cy = mesh.face_centroids
        field = cx + cy
        values = field.copy()
        values[12] = np.nan
        laplace_interpolate(mesh, values, "face")
        assert np.isnan(values[12])
        np.testing.assert_array_equal(
            values[~np.isnan(values)], field[~np.isnan(values)]
        )
