"""UGRID mesh visualization (delegates to cleopatra.mesh_glyph).

Thin wrapper around ``cleopatra.mesh_glyph.MeshGlyph`` that accepts
pyramids ``Mesh2d`` objects and delegates all rendering to cleopatra.
Requires the ``viz`` optional extra (``pip install pyramids-gis[viz]``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.base._utils import import_cleopatra
from pyramids.netcdf.ugrid.mesh import Mesh2d

_CLEOPATRA_MSG = (
    "Mesh plotting requires the cleopatra package. "
    "Install it with: pip install pyramids-gis[viz] "
    "or see https://github.com/serapeum-org/cleopatra"
)


def _mesh_to_glyph(mesh: Mesh2d) -> Any:
    """Convert a Mesh2d to a cleopatra MeshGlyph.

    Args:
        mesh: pyramids Mesh2d topology object.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph instance.

    Raises:
        OptionalPackageDoesNotExist: If cleopatra is not installed.
    """
    import_cleopatra(_CLEOPATRA_MSG)
    from cleopatra.mesh_glyph import MeshGlyph

    edge_nodes = None
    if mesh.edge_node_connectivity is not None:
        edge_nodes = mesh.edge_node_connectivity.data

    result = MeshGlyph(
        node_x=mesh.node_x,
        node_y=mesh.node_y,
        face_node_connectivity=mesh.face_node_connectivity.data,
        fill_value=mesh.face_node_connectivity.fill_value,
        edge_node_connectivity=edge_nodes,
    )
    return result


def plot_mesh_data(
    mesh: Mesh2d,
    data: np.ndarray,
    location: str = "face",
    ax: Any = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolor: str = "none",
    colorbar: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot mesh data using cleopatra MeshGlyph.

    For face-centered data: uses tripcolor. For node-centered data:
    uses tricontourf. Delegates to ``cleopatra.mesh_glyph.MeshGlyph.plot``.

    Args:
        mesh: Mesh2d topology.
        data: 1D data array matching the mesh location count.
        location: "face" or "node".
        ax: matplotlib Axes. Created if None.
        cmap: Colormap name.
        vmin: Minimum color scale value.
        vmax: Maximum color scale value.
        edgecolor: Edge color for face rendering.
        colorbar: Whether to add a colorbar.
        title: Plot title.
        **kwargs: Additional keyword arguments passed to the underlying
            matplotlib tripcolor/tricontourf call.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph instance with the plot rendered.
            Use the returned object to access the underlying Figure/Axes
            or to call additional MeshGlyph methods.

    Raises:
        ValueError: If location is not "face" or "node".
    """
    glyph = _mesh_to_glyph(mesh)
    glyph.plot(
        data, location=location, ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolor=edgecolor,
        colorbar=colorbar, title=title, **kwargs,
    )
    return glyph


def plot_mesh_outline(
    mesh: Mesh2d,
    ax: Any = None,
    color: str = "black",
    linewidth: float = 0.3,
    **kwargs: Any,
) -> Any:
    """Plot mesh edges as a wireframe using cleopatra MeshGlyph.

    Args:
        mesh: Mesh2d topology.
        ax: matplotlib Axes. Created if None.
        color: Edge color.
        linewidth: Edge line width.
        **kwargs: Additional keyword arguments passed to LineCollection.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph instance with the wireframe rendered.
    """
    glyph = _mesh_to_glyph(mesh)
    glyph.plot_outline(
        ax=ax, color=color, linewidth=linewidth, **kwargs,
    )
    return glyph
