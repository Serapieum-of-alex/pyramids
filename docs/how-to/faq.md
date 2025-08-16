# FAQ & Troubleshooting

## Installation issues (GDAL)
- Symptom: Errors importing GDAL or installing wheels.
- Tips:
  - Prefer conda-forge: `conda install -c conda-forge gdal pyramids`
  - Ensure Python version matches available GDAL builds.

## File not found or unsupported format
- Check the path. Windows users: escape backslashes in Python strings or use raw strings, e.g., `r"C:\\path\\file.tif"`.
- Verify the file extension is supported (GeoTIFF/ASC/NetCDF for rasters; GeoJSON/Shapefile/GPKG for vectors).

## Different raster sizes in a datacube
- Ensure all rasters share the same extent, resolution, and CRS before stacking.
- Align inputs using the reference raster and resample if needed.

## Writing outputs fails
- Confirm write permissions in the destination folder.
- For ASCII export, ensure `no_data_value` and `cell_size` are valid (see API docs for `_io.to_ascii`).

## CRS or transform confusion
- Use the `meta` and `transform` attributes on `Dataset` to inspect georeferencing.
- Reproject or align datasets prior to arithmetic operations.

## Performance tips
- Work with windows/tiles for large rasters.
- Use compressed GeoTIFFs when appropriate and keep I/O on local SSD.
