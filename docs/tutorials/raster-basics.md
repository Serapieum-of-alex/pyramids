# Tutorial: Raster basics

This tutorial demonstrates opening a raster, reading values, and saving to a new file.

Prerequisites: install pyramids and have a sample raster (we use a test file path from this repo).

```python
from pyramids.dataset import Dataset

src = "tests\\data\\geotiff\\dem.tif"  # adjust as needed

# Open
ds = Dataset.read_file(src)
print("Size:", ds.width, ds.height)

# Read full array
arr = ds.read()
print(arr.shape, arr.dtype)

# Save copy
out = "tutorial_dem_copy.tif"
ds.to_file(out)
print("Saved:", out)
```

Expected outcome: an output GeoTIFF is written and can be opened by common GIS tools.
