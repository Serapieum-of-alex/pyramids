# Vector Operations

Converting between mesh fields and vector geometry.

`polygonize` dissolves faces sharing a value into one (possibly multi-part) geometry — the mesh
analogue of the raster `to_feature_collection` — and is meant for fields with few distinct values,
including string label fields. `burn` is the inverse direction: it stamps a value onto every face a
geometry touches, sizing its output from both the value and the fill so a fractional value is not
truncated by an integer fill. `sample_line` reads a field along a line.

::: pyramids.netcdf.ugrid.vector_ops
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
