# Mesh Construction

Building a mesh from something that is not already a mesh.

A structured grid becomes a quad mesh, and a `GeoDataFrame` of polygons becomes a mesh of those
polygons with shared vertices merged. These back `UgridDataset.from_structured` and
`UgridDataset.from_geodataframe`.

::: pyramids.netcdf.ugrid.construct
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
