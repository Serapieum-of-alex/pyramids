"""Error paths, pre-built-connectivity paths and solver corners of ``pyramids.netcdf.ugrid.fill``.

The two fill entry points normally *derive* their adjacency from ``face_node_connectivity`` via
``graph.sparse_adjacency``. A mesh that already carries ``face_face_connectivity`` or
``edge_node_connectivity`` takes a second code path that reads those arrays directly, drops their
fill values and symmetrises the result — a path the derived-adjacency tests never touch. These
tests pin it against the derived path, and pin the remaining corners: the argument-validation
errors, the short circuit that leaves an entirely-null connected component alone, and the warning
raised when the CG solve is cut off before it converges.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.fill import fill_na_spatial, laplace_interpolate
from pyramids.netcdf.ugrid.mesh import Mesh2d

GRID_2X2_FACES = np.array(
    [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.intp
)
"""Row-major face table of the 2x2 quad grid: f0 lower-left, f1 lower-right, f2/f3 the top row."""

PADDED_FACE_FACE = np.array(
    [[1, 2, -1], [0, 3, -1], [0, 3, -1], [1, 2, -1]], dtype=np.intp
)
"""The 2x2 grid's edge-sharing neighbours, padded with one fill column per row."""

TRIANGLE_NODE_X = np.array([0.0, 1.0, 0.0])
TRIANGLE_NODE_Y = np.array([0.0, 0.0, 1.0])


def _quad_grid(
    nx: int, ny: int, x0: float = 0.0, node_offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular ``nx`` by ``ny`` grid of unit quad faces.

    Args:
        nx: Number of faces along x.
        ny: Number of faces along y.
        x0: X-origin of the lower-left node (used to offset a second, disjoint grid).
        node_offset: Added to every node index in the face table (used when this grid's nodes are
            concatenated after another grid's nodes).

    Returns:
        Tuple ``(node_x, node_y, faces)`` where ``faces`` is an ``(nx * ny, 4)`` table.
    """
    xs = np.arange(nx + 1, dtype=float) + x0
    ys = np.arange(ny + 1, dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)
    stride = nx + 1
    faces = []
    for row in range(ny):
        for col in range(nx):
            first = row * stride + col + node_offset
            faces.append([first, first + 1, first + 1 + stride, first + stride])
    return grid_x.ravel(), grid_y.ravel(), np.array(faces, dtype=np.intp)


def _grid_mesh(nx: int, ny: int) -> Mesh2d:
    """Return the :class:`Mesh2d` of a regular ``nx`` by ``ny`` quad grid, with no derived tables."""
    node_x, node_y, faces = _quad_grid(nx, ny)
    return UgridDataset.from_arrays(node_x, node_y, faces).mesh


def _connectivity(data: np.ndarray, cf_role: str) -> Connectivity:
    """Wrap a 0-indexed index array as a ``-1``-filled :class:`Connectivity`.

    Args:
        data: The index array.
        cf_role: The UGRID ``cf_role`` the array plays.

    Returns:
        Connectivity: The wrapped array.
    """
    return Connectivity(
        data=np.asarray(data, dtype=np.intp),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


def _triangle_mesh(
    face_face: np.ndarray | None = None, edge_node: np.ndarray | None = None
) -> Mesh2d:
    """A single right triangle, optionally carrying hand-written neighbour/edge tables.

    Args:
        face_face: Optional ``face_face_connectivity`` index array.
        edge_node: Optional ``edge_node_connectivity`` index array.

    Returns:
        Mesh2d: The one-face mesh.
    """
    return Mesh2d(
        node_x=TRIANGLE_NODE_X,
        node_y=TRIANGLE_NODE_Y,
        face_node_connectivity=_connectivity(
            np.array([[0, 1, 2]]), "face_node_connectivity"
        ),
        edge_node_connectivity=(
            None
            if edge_node is None
            else _connectivity(edge_node, "edge_node_connectivity")
        ),
        face_face_connectivity=(
            None
            if face_face is None
            else _connectivity(face_face, "face_face_connectivity")
        ),
    )


@pytest.fixture
def two_triangles() -> Mesh2d:
    """A unit square split into two triangles sharing the ``(0, 0)-(1, 1)`` diagonal."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity(
            np.array([[0, 1, 2], [0, 2, 3]]), "face_node_connectivity"
        ),
    )


@pytest.fixture
def disjoint_grids() -> Mesh2d:
    """Two 2x2 quad grids 20 units apart: one mesh, two connected components of 4 faces each."""
    first_x, first_y, first_faces = _quad_grid(2, 2)
    second_x, second_y, second_faces = _quad_grid(
        2, 2, x0=20.0, node_offset=len(first_x)
    )
    return UgridDataset.from_arrays(
        np.concatenate([first_x, second_x]),
        np.concatenate([first_y, second_y]),
        np.concatenate([first_faces, second_faces], axis=0),
    ).mesh


class TestArgumentValidation:
    """The location and element-axis guards of both entry points."""

    def test_nearest_fill_rejects_an_unknown_location(self, two_triangles):
        """``fill_na_spatial`` names the offending location in the error."""
        with pytest.raises(ValueError, match="Unknown location: 'vertex'"):
            fill_na_spatial(two_triangles, np.array([1.0, np.nan]), "vertex")

    def test_nearest_fill_rejects_a_mismatched_element_axis(self, two_triangles):
        """A last axis longer than the face count is rejected, reporting both counts."""
        with pytest.raises(
            ValueError, match=r"values last axis \(3\).*face elements \(2\)"
        ):
            fill_na_spatial(two_triangles, np.array([1.0, 2.0, np.nan]), "face")

    def test_laplace_rejects_an_unknown_location(self, two_triangles):
        """``laplace_interpolate`` rejects a location that is neither face, node nor edge."""
        with pytest.raises(ValueError, match="Unknown location: 'vertex'"):
            laplace_interpolate(two_triangles, np.array([1.0, np.nan]), "vertex")

    def test_laplace_rejects_a_mismatched_element_axis(self, two_triangles):
        """A node-located array handed to a face solve is rejected before any solve runs."""
        with pytest.raises(
            ValueError, match=r"values last axis \(4\).*face elements \(2\)"
        ):
            laplace_interpolate(
                two_triangles, np.array([1.0, 2.0, 3.0, np.nan]), "face"
            )


class TestPrebuiltFaceAdjacency:
    """``laplace_interpolate`` reading an existing ``face_face_connectivity``."""

    def test_matches_the_derived_adjacency(self):
        """Pre-building the neighbour table changes nothing about the filled values."""
        derived_mesh = _grid_mesh(4, 4)
        prebuilt_mesh = _grid_mesh(4, 4)
        prebuilt_mesh.build_face_face_connectivity()
        assert prebuilt_mesh.face_face_connectivity is not None

        centroid_x, centroid_y = derived_mesh.face_centroids
        field = 2.0 * centroid_x + 3.0 * centroid_y + 1.0
        holed = field.copy()
        holed[[5, 6]] = np.nan

        derived = laplace_interpolate(derived_mesh, holed, "face")
        prebuilt = laplace_interpolate(prebuilt_mesh, holed, "face")

        np.testing.assert_allclose(prebuilt, derived, atol=1e-10)
        np.testing.assert_allclose(prebuilt, field, atol=1e-6)

    def test_fill_values_in_the_neighbour_table_are_ignored(self):
        """A padded neighbour row fills exactly like the same table without the padding."""
        node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        padded = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=_connectivity(
                GRID_2X2_FACES, "face_node_connectivity"
            ),
            face_face_connectivity=_connectivity(
                PADDED_FACE_FACE, "face_face_connectivity"
            ),
        )
        unpadded = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=_connectivity(
                GRID_2X2_FACES, "face_node_connectivity"
            ),
            face_face_connectivity=_connectivity(
                PADDED_FACE_FACE[:, :2], "face_face_connectivity"
            ),
        )
        values = np.array([np.nan, 4.5, 5.5, 7.0])

        with_padding = laplace_interpolate(padded, values, "face")
        without_padding = laplace_interpolate(unpadded, values, "face")

        np.testing.assert_allclose(with_padding, without_padding, atol=1e-10)
        assert np.isfinite(with_padding).all()

    def test_a_one_sided_neighbour_table_is_symmetrised(self, two_triangles):
        """A neighbour listed in one direction only still couples both faces."""
        one_sided = Mesh2d(
            node_x=two_triangles.node_x,
            node_y=two_triangles.node_y,
            face_node_connectivity=two_triangles.face_node_connectivity,
            face_face_connectivity=_connectivity(
                np.array([[1], [-1]]), "face_face_connectivity"
            ),
        )
        filled = laplace_interpolate(one_sided, np.array([5.0, np.nan]), "face")
        assert filled[1] == pytest.approx(5.0)


class TestPrebuiltNodeAdjacency:
    """``laplace_interpolate`` reading an existing ``edge_node_connectivity``."""

    def test_matches_the_derived_adjacency(self):
        """Pre-building the edge table changes nothing about the filled node values."""
        derived_mesh = _grid_mesh(4, 4)
        prebuilt_mesh = _grid_mesh(4, 4)
        prebuilt_mesh.build_edge_connectivity()
        assert prebuilt_mesh.edge_node_connectivity is not None

        field = 2.0 * derived_mesh.node_x + 3.0 * derived_mesh.node_y
        holed = field.copy()
        holed[12] = np.nan

        derived = laplace_interpolate(derived_mesh, holed, "node")
        prebuilt = laplace_interpolate(prebuilt_mesh, holed, "node")

        np.testing.assert_allclose(prebuilt, derived, atol=1e-10)
        np.testing.assert_allclose(prebuilt, field, atol=1e-6)

    def test_fill_values_in_the_edge_table_are_ignored(self):
        """A half-written edge row (one endpoint is the fill value) is dropped, not treated as node -1."""
        clean = _triangle_mesh(edge_node=np.array([[0, 1], [1, 2], [0, 2]]))
        with_fill = _triangle_mesh(
            edge_node=np.array([[0, 1], [1, 2], [0, 2], [2, -1]])
        )
        values = np.array([1.0, 2.0, np.nan])

        expected = laplace_interpolate(clean, values, "node", use_weights=False)
        actual = laplace_interpolate(with_fill, values, "node", use_weights=False)

        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert actual[2] == pytest.approx(1.5)

    def test_a_one_directional_edge_table_is_symmetrised(self):
        """A node that only ever appears as an edge's second endpoint still gets a degree."""
        mesh = _triangle_mesh(edge_node=np.array([[0, 1], [1, 2], [0, 2]]))
        filled = laplace_interpolate(
            mesh, np.array([1.0, 2.0, np.nan]), "node", use_weights=False
        )
        assert np.isfinite(filled).all()
        assert filled[2] == pytest.approx(1.5)


class TestEntirelyNullComponent:
    """A connected component with no valid element anywhere in the slab."""

    def test_an_all_null_component_is_skipped_without_solving(self, disjoint_grids):
        """The valid component is returned untouched and the null one stays NaN, with no warning."""
        values = np.arange(disjoint_grids.n_face, dtype=np.float64)
        values[4:] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            filled = laplace_interpolate(disjoint_grids, values, "face")

        np.testing.assert_array_equal(filled[:4], np.arange(4, dtype=np.float64))
        assert np.all(np.isnan(filled[4:]))


class TestCgConvergence:
    """The conjugate-gradient path's non-convergence warning."""

    def test_a_truncated_cg_solve_warns(self):
        """Cutting CG off after one iteration warns and reports the solver's info code."""
        mesh = _grid_mesh(12, 12)
        centroid_x, centroid_y = mesh.face_centroids
        field = np.sin(centroid_x) * np.cos(centroid_y)
        holed = field.copy()
        holed[np.arange(mesh.n_face) % 3 == 1] = np.nan

        with pytest.warns(UserWarning, match=r"CG solver did not converge \(info=1\)"):
            filled = laplace_interpolate(mesh, holed, "face", rtol=1e-14, maxiter=1)

        assert np.isfinite(filled).all()
