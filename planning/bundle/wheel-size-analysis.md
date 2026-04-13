# Wheel Size Analysis (Phase 6 / Task 6.1)

## Measured — 2026-04-13

Wheel: `pyramids_gis-0.13.0-cp312-cp312-manylinux_2_39_x86_64.whl`

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Compressed size | **63.6 MB** | ≤ 80 MB | ✅ Under budget |
| Uncompressed size | 188.4 MB | — | OK |
| File count | 469 | — | OK |

## Breakdown by category

| Category | Files | Uncompressed (MB) | % of total | Notes |
|----------|------:|------------------:|-----------:|-------|
| Bundled `.so` (auditwheel) | 77 | 145.0 | **77%** | libgdal + proj + geos + tiff + hdf5 + hdf4 + netcdf + curl + openssl + ... |
| `osgeo` Python bindings (SWIG) | 26 | 25.5 | 13.5% | `_gdal.so` (~11 MB), `_ogr.so` (6.5 MB), `_osr.so` (3 MB), etc. |
| PROJ_DATA | 17 | 10.4 | 5.5% | Dominated by `proj.db` (10 MB) — EPSG/WKT catalog |
| GDAL_DATA | 94 | 2.3 | 1.2% | Driver metadata, XSD schemas, small CSVs |
| `osgeo_utils` | 182 | 2.1 | 1.1% | Pure Python CLI helpers (gdal_polygonize.py, etc.) |
| GDAL plugins | 5 | 2.1 | 1.1% | `gdal_netCDF.so`, `gdal_HDF4.so`, `gdal_HDF5.so`, `drivers.ini` |
| pyramids Python source | 61 | 0.9 | 0.5% | The actual package code |
| Wheel metadata | 7 | 0.1 | < 0.1% | METADATA, RECORD, WHEEL |

## Comparison with peers

| Package | Wheel size | GDAL included | HDF4 / NetCDF |
|---------|-----------:|:-------------:|:-------------:|
| **pyramids-gis** | **63.6 MB** | ✅ | ✅ / ✅ |
| rasterio | ~26 MB | ✅ | ❌ / ❌ |
| pyogrio | ~18 MB | ✅ (vector only) | ❌ / ❌ |
| pyproj | ~7 MB | ❌ | ❌ / ❌ |
| shapely | ~3 MB | ❌ | ❌ / ❌ |

Our wheel is ~2.5× larger than rasterio because we include HDF4, HDF5, and
NetCDF drivers + their dependency chains (libhdf4, libmfhdf, libhdf5,
libnetcdf, libaec, libsz). rasterio explicitly skips these.

## Optimization opportunities

Sorted by savings potential.

### Large wins (10+ MB each)

**1. Trim `proj.db` (10 MB → 2–3 MB possible).** The full PROJ database
includes EPSG, IGNF, ESRI, IAU, NKG, and OGC authorities plus grid
metadata. Most users need only EPSG + maybe ESRI. PROJ has no official
slimmer distribution, but we could:
- Ship `proj.db` as-is for now (10 MB is acceptable)
- Revisit if total wheel size becomes a problem

**2. Strip more aggressively from bundled `.so` files.** We run
`strip --strip-unneeded` on Linux. We could additionally:
- Drop SIMD kernel duplicates in libgdal (marginal)
- Use `strip --strip-all` instead — but may break exception handling
- Net expected savings: 2–5 MB

**3. Disable unused GDAL drivers** via conda-forge feature selectors.
This is the biggest potential win but requires a custom conda-forge
build of `libgdal-core` with fewer drivers. Not practical without
fork-and-maintain overhead.

### Medium wins (1–3 MB each)

**4. Skip `osgeo_utils` (2.1 MB).** Contains CLI entry points
(`gdal_polygonize.py`, `samples/`). Not used by pyramids code. Could
be dropped with minimal impact — remove the vendor step in
`ci/install-and-vendor-osgeo.py`.
- **Caution:** Third-party code importing `osgeo_utils` via pyramids
  would break. Low risk since the module is separate from `osgeo`.

**5. Prune GDAL_DATA to essentials (2.3 → 0.5 MB possible).** Keep only:
- `stateplane.csv`, `gt_datum.csv`, `gcs.csv`, `pcs.csv` (core tables)
- `gdalvrt.xsd`, `gml_registry.xml` (driver metadata)
Drop:
- `default.rsc`, `cubewerx_extra.wkt`, `*.schema.json` (rarely used)
- Logo SVGs, misc XSDs
- **Net savings:** ~1.5 MB, ~60 file count reduction

### Small wins (< 1 MB each)

**6. Compress `proj.db` at rest.** SQLite supports `VACUUM`; could
shrink by ~5% (0.5 MB).

**7. Deduplicate `libcurl` / `libssl`.** If two deps ship their own
copies with different sonames, auditwheel vendors both. Check via
`auditwheel show` output.

### Not worth pursuing

- **UPX compression** — breaks `dlopen` symbol resolution on Linux.
- **Remove libstdcxx-ng** — breaks GCC 13 ABI required by libgdal.
- **Smaller libgdal** — would need custom conda-forge build.

## Recommendation

**Ship 63.6 MB as-is.** We're well under the 80 MB target, well under
PyPI's practical limit (200+ MB wheels are tolerated). Revisit only if:

1. Wheel exceeds 100 MB after a dep bump
2. User complaints about install time
3. PyPI adds a hard per-wheel ceiling

Track the size in CI — if a future GDAL version bumps us past 100 MB,
trigger the optimization playbook above.

## CI size monitoring

Add to `build-wheels.yml` after the repair step:

```yaml
- name: Report wheel size
  shell: bash
  run: |
    for whl in wheelhouse/*.whl; do
      size=$(stat -c%s "$whl" 2>/dev/null || stat -f%z "$whl")
      size_mb=$(echo "scale=1; $size / 1048576" | bc)
      echo "::notice::$(basename $whl): ${size_mb} MB"
      if [ "$size" -gt $((120 * 1024 * 1024)) ]; then
        echo "::error::Wheel exceeds 120 MB threshold"
        exit 1
      fi
    done
```

This adds a hard 120 MB gate and a notice annotation per wheel.
