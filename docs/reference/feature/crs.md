# CRS & Reprojection Helpers

CRS-handling helpers on `pyramids.feature.crs`. Re-exposed as
`FeatureCollection` static methods for convenience
(`FeatureCollection.reproject_coordinates` delegates to
`pyramids.feature.crs.reproject_coordinates`).

## Design notes

- **ARC-2** — `pyproj.Proj(init="epsg:N")` is gone. All reprojection
  goes through `pyproj.Transformer.from_crs(..., always_xy=True)`.
- **ARC-14** — the two historical helpers (`reproject_points`,
  `reproject_points2`) had inconsistent axis order and were deleted
  outright. `reproject_coordinates` is the single entry point and
  uses `(x, y)` throughout.
- **ARC-7** — `get_epsg_from_prj("")` raises
  `pyramids.base._errors.CRSError` instead of silently defaulting to
  EPSG:4326. An empty projection string signals a configuration
  error, not "I'd like WGS84, please".
- **M1** — `reproject_coordinates` catches only
  `(pyproj.exceptions.CRSError, TypeError, ValueError)` and wraps
  them into `pyramids.base._errors.CRSError`. Other exceptions
  (`AttributeError`, `ImportError`, …) propagate unchanged — they
  indicate a bug, not a bad user input.

## Functions

::: pyramids.feature.crs.create_sr_from_proj
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.crs.get_epsg_from_prj
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids.feature.crs.reproject_coordinates
    options:
        show_root_heading: true
        heading_level: 3
