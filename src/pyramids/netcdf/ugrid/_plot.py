"""UGRID mesh visualization.

Provides functions for plotting mesh data using matplotlib
triangulation (tripcolor/tricontourf) and mesh wireframe
rendering using LineCollection.

Depends on:
    - _mesh.py: Mesh2d (uses mesh.triangulation property)
    - matplotlib (optional dependency via cleopatra/viz extra)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.netcdf.ugrid._mesh import Mesh2d


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
    **kwargs,
) -> Any:
    """Plot mesh data using matplotlib triangulation.

    For face-centered data: uses tripcolor (each triangle colored
    by value). For node-centered data: uses tricontourf (interpolated
    contours). The mesh.triangulation property handles mixed meshes
    by decomposing each polygon into triangles.

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
        **kwargs: Additional keyword arguments passed to tripcolor/tricontourf.

    Returns:
        matplotlib Axes with the plot.

    Raises:
        ValueError: If location is not "face" or "node".
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    tri = mesh.triangulation

    if location == "face":
        tri_values = _map_face_to_triangle_values(mesh, data)
        tpc = ax.tripcolor(
            tri, facecolors=tri_values, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolors=edgecolor, **kwargs
        )
    elif location == "node":
        contour_kw = {"cmap": cmap, "levels": 20}
        if vmin is not None:
            contour_kw["vmin"] = vmin
        if vmax is not None:
            contour_kw["vmax"] = vmax
        tpc = ax.tricontourf(tri, data, **contour_kw, **kwargs)
    else:
        raise ValueError(
            f"Plotting not supported for location='{location}'. "
            f"Use 'face' or 'node'."
        )

    if colorbar:
        plt.colorbar(tpc, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")

    result = ax
    return result


def _map_face_to_triangle_values(
    mesh: Mesh2d,
    face_values: np.ndarray,
) -> np.ndarray:
    """Map per-face values to per-triangle values after fan decomposition.

    The fan decomposition creates (nodes_per_face - 2) triangles per
    face. All triangles from the same face get the same data value.

    Args:
        mesh: Mesh2d topology.
        face_values: 1D array of values, one per face.

    Returns:
        1D array of values, one per triangle in the triangulation.
    """
    fnc = mesh.face_node_connectivity
    counts = fnc.nodes_per_element()
    n_triangles = int(np.sum(counts - 2))
    tri_values = np.empty(n_triangles)

    tri_idx = 0
    for face_idx in range(mesh.n_face):
        n_tris = counts[face_idx] - 2
        tri_values[tri_idx:tri_idx + n_tris] = face_values[face_idx]
        tri_idx += n_tris

    return tri_values


def plot_mesh_outline(
    mesh: Mesh2d,
    ax: Any = None,
    color: str = "black",
    linewidth: float = 0.3,
    **kwargs,
) -> Any:
    """Plot mesh edges as a wireframe.

    Uses matplotlib LineCollection for efficient rendering of
    thousands of edges.

    Args:
        mesh: Mesh2d topology.
        ax: matplotlib Axes. Created if None.
        color: Edge color.
        linewidth: Edge line width.
        **kwargs: Additional keyword arguments passed to LineCollection.

    Returns:
        matplotlib Axes with the wireframe plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    segments: list[list[tuple[float, float]]] = []

    if mesh.edge_node_connectivity is not None:
        enc = mesh.edge_node_connectivity
        for i in range(enc.n_elements):
            nodes = enc.get_element(i)
            n1, n2 = int(nodes[0]), int(nodes[1])
            segments.append([
                (mesh.node_x[n1], mesh.node_y[n1]),
                (mesh.node_x[n2], mesh.node_y[n2]),
            ])
    else:
        seen_edges: set[tuple[int, int]] = set()
        for i in range(mesh.n_face):
            nodes = mesh.face_node_connectivity.get_element(i)
            n = len(nodes)
            for j in range(n):
                n1, n2 = int(nodes[j]), int(nodes[(j + 1) % n])
                edge_key = (min(n1, n2), max(n1, n2))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    segments.append([
                        (mesh.node_x[n1], mesh.node_y[n1]),
                        (mesh.node_x[n2], mesh.node_y[n2]),
                    ])

    lc = LineCollection(segments, colors=color, linewidths=linewidth, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")

    result = ax
    return result
