# Graph & Connectivity

Derived connectivity tables, graph traversal, and binary morphology on a mesh.

Everything here is **pure**: each function reads a `Mesh2d` and returns new arrays or a new
`Connectivity`, never writing a derived table back onto the mesh it was handed. The edge tables are
derived from the faces; when a file declares its own `edge_node_connectivity` those edges keep their
positions (edge-located variables index into them) and any face-implied edge the file omits is
appended after them, so a partial edge list still yields a complete face topology.

`sparse_adjacency` is the shared primitive: `connected_components`, `reverse_cuthill_mckee` and the
binary morphology all read it.

One morphology pass is simultaneous — the neighbour test is taken from the mask as it stands at the
start of the pass, and the `border_value` imputation joins that same update — so one iteration moves
exactly one ring, matching `scipy.ndimage`.

::: pyramids.netcdf.ugrid.graph
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
