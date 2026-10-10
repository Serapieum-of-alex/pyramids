# Network1d

UGRID `topology_dimension == 1` data: river networks and 1-D hydraulic models.

Nodes plus an edge-node connectivity table, with **no faces** — the core element is the edge, so
there are no centroids or areas. `Network1d` is a standalone sibling of `Mesh2d`, holding plain
numpy arrays with no GDAL objects and no dependency on `Mesh2d`.

It offers edge lengths and midpoints, bounds, nearest-neighbour KD-trees on both nodes and edge
midpoints, and a topological sort for upstream/downstream ordering of a directed network.

**In-memory only.** There is no 1-D reader or writer yet: the UGRID topology parser recognises
`topology_dimension == 2` meshes and `UgridDataset` carries a `Mesh2d`, so a network is built from
arrays rather than read from a file, and does not travel through `read_file` / `to_file`.

::: pyramids.netcdf.ugrid.network.Network1d
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
