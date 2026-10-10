# Regridding & Sampling

Moving a field from one mesh to another, to points, or onto a raster.

`Regridder` builds a sparse weight matrix once and applies it to any number of fields. The two
area-based methods differ only in their normalisation, and they agree whenever the target is fully
covered by the source — so the distinction only shows on a partially covered target:

| method | weight | answers |
|---|---|---|
| `overlap` | `area(t, s) / sum_s area(t, s)` | the mean of the source it covers; rows sum to 1 |
| `conservative` | `area(t, s) / target_area[t]` | preserves `sum(value * area)` when the target
covers the whole source |

`centroid` copies from the nearest source centroid and `barycentric` interpolates in the Delaunay
triangulation, reproducing a plane exactly.


`conservative` conserves only when the target's footprint contains the source's. Source area the
target does not reach has its mass dropped — a 3x3 unit source over `[0, 3]²` onto a 2x2 unit target
over `[0, 2]²` goes from an integral of 45.0 to 12.0 — and because an uncovered target element is
NaN, the target integral is `nan` whenever the target reaches past the source.

`sample` reads a field at points; a point lying exactly on a face boundary — an interior shared edge,
or one of the mesh's own nodes — belongs to a face that meets there, not to nothing.
`rasterize_like` interpolates onto another raster's grid, taking the cell centres from that grid's
full affine so a rotated reference is sampled where it actually lies. An uncovered target element
comes back as the declared no-data value.

::: pyramids.netcdf.ugrid.regrid
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
