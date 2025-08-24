# ADR-0003: I/O format choices

- Status: Accepted
- Date: 2025-08-16

## Context
`pyramids` processes raster and vector geospatial data and must interoperate with common GIS tools and libraries.

## Decision
- Raster: prioritize GeoTIFF for read/write; support ASCII Grid for simple interchange; support NetCDF for multi-dimensional datasets when appropriate.
- Vector: prioritize GeoJSON for web interoperability; support Shapefile and GeoPackage for desktop GIS.

## Consequences
- Users can rely on standard formats across workflows.
- Additional drivers may be enabled via GDAL installation; docs should clarify environment requirements.
