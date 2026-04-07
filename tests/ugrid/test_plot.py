"""Unit tests for pyramids.netcdf.ugrid.plot (cleopatra wrapper).

Tests that the thin wrapper correctly delegates to cleopatra's
MeshGlyph and that the UgridDataset.plot()/plot_outline() methods work.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.plot import plot_mesh_data, plot_mesh_outline


@pytest.mark.plot
class TestPlotMeshData:
    """Tests for plot_mesh_data() wrapper."""

    def test_face_data_plot(self, triangle_mesh):
        """Test plotting face-centered data returns Axes."""
        data = np.array([1.0, 2.0])
        ax = plot_mesh_data(triangle_mesh, data, location="face")
        assert ax is not None, "Should return an Axes object"

    def test_node_data_plot(self, triangle_mesh):
        """Test plotting node-centered data returns Axes."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ax = plot_mesh_data(triangle_mesh, data, location="node")
        assert ax is not None, "Should return an Axes object"

    def test_invalid_location_raises(self, triangle_mesh):
        """Test that invalid location raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            plot_mesh_data(triangle_mesh, np.array([1.0]), location="edge")

    def test_mixed_mesh_face_plot(self, mixed_mesh):
        """Test plotting face data on mixed mesh."""
        data = np.array([10.0, 20.0, 30.0])
        ax = plot_mesh_data(mixed_mesh, data, location="face")
        assert ax is not None, "Should handle mixed mesh plotting"

    def test_with_title(self, triangle_mesh):
        """Test plot with title."""
        data = np.array([1.0, 2.0])
        ax = plot_mesh_data(
            triangle_mesh, data, location="face", title="Test",
        )
        assert ax.get_title() == "Test", (
            f"Expected title 'Test', got '{ax.get_title()}'"
        )


@pytest.mark.plot
class TestPlotMeshOutline:
    """Tests for plot_mesh_outline() wrapper."""

    def test_wireframe_plot(self, triangle_mesh):
        """Test plotting mesh wireframe returns Axes."""
        ax = plot_mesh_outline(triangle_mesh)
        assert ax is not None, "Should return an Axes object"

    def test_wireframe_mixed_mesh(self, mixed_mesh):
        """Test wireframe on mixed mesh."""
        ax = plot_mesh_outline(mixed_mesh, color="blue", linewidth=1.0)
        assert ax is not None, "Should return an Axes object"


@pytest.mark.plot
class TestUgridDatasetPlotMethods:
    """Tests for UgridDataset.plot() and plot_outline()."""

    def test_dataset_plot(self):
        """Test UgridDataset.plot() method."""
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.0])},
            data_locations={"depth": "face"},
        )
        ax = ds.plot("depth")
        assert ax is not None, "plot() should return Axes"
        assert ax.get_title() == "depth", (
            f"Default title should be variable name, got '{ax.get_title()}'"
        )

    def test_dataset_plot_outline(self):
        """Test UgridDataset.plot_outline() method."""
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        ax = ds.plot_outline()
        assert ax is not None, "plot_outline() should return Axes"
