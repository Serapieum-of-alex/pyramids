# Gap Filling

Filling missing values across the mesh, as opposed to along time.

`fill_na_spatial` takes each gap from the nearest valid element; `laplace_interpolate` solves
Laplace's equation on the mesh adjacency graph, which reproduces a field linear in `x`/`y` exactly.
Where `UgridDataset.interpolate_na` fills along the time axis, these fill across space.

Both are pure with respect to the mesh they are given: the adjacency is derived through
`graph.sparse_adjacency` rather than built onto the mesh, which matters because `UgridDataset.copy()`
shares its `Mesh2d` — a mutating fill on a copy would reach back into the original.

::: pyramids.netcdf.ugrid.fill
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
