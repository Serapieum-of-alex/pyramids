"""Unit tests for pyramids.netcdf.ugrid.plot.

Covers plot_mesh_data (tripcolor/tricontourf) and plot_mesh_outline
(wireframe). Tests marked with 'plot' marker for optional skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.plot import (
    _map_face_to_triangle_values,
    plot_mesh_data,
    plot_mesh_outline,
)


@pytest.mark.plot
class TestPlotMeshData:
    """Tests for plot_mesh_data() function (UGRID-13)."""

    def test_face_data_plot(self, triangle_mesh):
        """Test plotting face-centered data.

        Test scenario:
            Should create a tripcolor plot without errors.
        """
        import matplotlib
        matplotlib.use("Agg")

        data = np.array([1.0, 2.0])
        ax = plot_mesh_data(triangle_mesh, data, location="face")
        assert ax is not None, "Should return an Axes object"

    def test_node_data_plot(self, triangle_mesh):
        """Test plotting node-centered data.

        Test scenario:
            Should create a tricontourf plot without errors.
        """
        import matplotlib
        matplotlib.use("Agg")

        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ax = plot_mesh_data(triangle_mesh, data, location="node")
        assert ax is not None, "Should return an Axes object"

    def test_invalid_location_raises(self, triangle_mesh):
        """Test that invalid location raises ValueError.

        Test scenario:
            location='edge' should raise ValueError.
        """
        import matplotlib
        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="not supported"):
            plot_mesh_data(triangle_mesh, np.array([1.0]), location="edge")

    def test_mixed_mesh_face_plot(self, mixed_mesh):
        """Test plotting face data on mixed mesh.

        Test scenario:
            Mixed mesh with 3 faces (1 quad + 2 tri) should work.
        """
        import matplotlib
        matplotlib.use("Agg")

        data = np.array([10.0, 20.0, 30.0])
        ax = plot_mesh_data(mixed_mesh, data, location="face")
        assert ax is not None, "Should handle mixed mesh plotting"

    def test_with_title_and_colorbar(self, triangle_mesh):
        """Test plot with title and colorbar options.

        Test scenario:
            Should create plot with title and colorbar.
        """
        import matplotlib
        matplotlib.use("Agg")

        data = np.array([1.0, 2.0])
        ax = plot_mesh_data(
            triangle_mesh, data, location="face",
            title="Test", colorbar=True, cmap="coolwarm",
        )
        assert ax.get_title() == "Test", f"Expected title 'Test', got '{ax.get_title()}'"


@pytest.mark.plot
class TestPlotMeshOutline:
    """Tests for plot_mesh_outline() function (UGRID-14)."""

    def test_wireframe_plot(self, triangle_mesh):
        """Test plotting mesh wireframe.

        Test scenario:
            Should create a line collection plot without errors.
        """
        import matplotlib
        matplotlib.use("Agg")

        ax = plot_mesh_outline(triangle_mesh)
        assert ax is not None, "Should return an Axes object"

    def test_wireframe_mixed_mesh(self, mixed_mesh):
        """Test wireframe plot on mixed mesh.

        Test scenario:
            Mixed mesh with quads and triangles should render all edges.
        """
        import matplotlib
        matplotlib.use("Agg")

        ax = plot_mesh_outline(mixed_mesh, color="blue", linewidth=1.0)
        assert ax is not None, "Should return an Axes object"


class TestMapFaceToTriangleValues:
    """Tests for _map_face_to_triangle_values() helper."""

    def test_pure_triangles(self, triangle_mesh):
        """Test mapping for pure triangular mesh.

        Test scenario:
            2 triangles -> 2 triangles, each gets its face value.
        """
        face_values = np.array([10.0, 20.0])
        result = _map_face_to_triangle_values(triangle_mesh, face_values)
        assert len(result) == 2, f"Expected 2 triangle values, got {len(result)}"
        assert result[0] == 10.0, f"Expected 10.0, got {result[0]}"
        assert result[1] == 20.0, f"Expected 20.0, got {result[1]}"

    def test_mixed_mesh(self, mixed_mesh):
        """Test mapping for mixed mesh (quad + triangles).

        Test scenario:
            1 quad (2 tris) + 2 triangles (1 tri each) = 4 total.
            Quad value should appear twice.
        """
        face_values = np.array([10.0, 20.0, 30.0])
        result = _map_face_to_triangle_values(mixed_mesh, face_values)
        assert len(result) == 4, f"Expected 4 triangle values, got {len(result)}"
        assert result[0] == 10.0, "First triangle from quad should be 10"
        assert result[1] == 10.0, "Second triangle from quad should be 10"
