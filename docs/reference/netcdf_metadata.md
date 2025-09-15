# NetCDF MDArray metadata

This page describes how to enumerate and normalize all metadata from NetCDF files using GDAL's Multidimensional (MDim) API in Pyramids.

Overview:
- Open NetCDF files as MDArray-backed datasets
- Traverse groups, arrays, dimensions, and attributes
- Produce a JSON-serializable metadata object
- Keep compatibility with existing dimension parser exposed via `NetCDF.meta_data`

## Usage

Basic usage for reading all metadata:

```python
from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.metadata import to_json

# Open the file in MDIM mode
nc = NetCDF.read_file("tests/data/netcdf/pyramids-netcdf-3d.nc", open_as_multi_dimensional=True)

# Read everything (groups, arrays, dimensions, attributes)
md = nc.get_all_metadata()

# Convert to JSON
s = to_json(md)
print(s)
```

You can also pass open options (persisted into the result for provenance):

```python
md = nc.get_all_metadata(open_options={"OPEN_SHARED": "YES"})
```

## Dimension overview

For convenience and backward compatibility, the returned metadata includes a `dimension_overview` section summarizing parsed dimensions using the existing `dimensions.MetaData` logic.

Shape:
- names: list[str]
- sizes: dict[str, int]
- attrs: dict[str, dict[str, str]]
- values: dict[str, list[int|float|string]] | None

This mirrors `nc.meta_data` and provides a compact CF-friendly view.

## Notes

- The feature uses GDAL's MDim API starting at `dataset.GetRootGroup()`.
- Attributes are normalized to JSON-friendly scalars or vectors; bytes are decoded as UTF-8.
- Convenience fields on arrays include: unit, nodata (`_FillValue`/`missing_value` precedence), scale/offset, CRS (WKT/PROJJSON), structural info, block size, and coordinate variables.
- No array data values are read; only metadata.
- The module provides helpers to serialize to/from JSON and to a plain dict.

## References

- GDAL Multidimensional API: https://gdal.org/api/python/osgeo.gdal_array.MDArray-class.html
- netCDF driver: https://gdal.org/drivers/raster/netcdf.html
- gdal_mdim_info utility: https://gdal.org/programs/gdalmdiminfo.html
