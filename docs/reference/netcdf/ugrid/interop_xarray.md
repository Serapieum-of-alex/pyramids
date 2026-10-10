# xarray & Zarr Interop

Handing a mesh to xarray, and reading one back, in the UGRID-CF encoding.

`xarray`, `zarr` and `dask` are **peer dependencies**: they are imported inside these functions
through the optional-extras helpers, never at module import, so a bare install never needs them.
There is no `xugrid` dependency — the UGRID-CF encoding is written and read directly.

The encoding trap this module exists to handle is the connectivity round trip: a file's
`start_index` and `_FillValue` are restored on the way out and normalised back to the internal
0-indexed / `-1`-fill form on the way in. The dummy topology and grid-mapping variables carry a
length-1 dimension rather than being true scalars, because GDAL's multidimensional reader does not
expose a dimensionless variable — without it the exported dataset could be written to NetCDF but
not read back.

::: pyramids.netcdf.ugrid.interop_xarray
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["!^_"]
