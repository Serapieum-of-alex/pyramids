# Tutorial: Datacube basics

This tutorial shows how to construct a simple datacube from a folder of rasters and compute a basic aggregation.

```python
from pyramids.datacube import MultiDataset


# Prepare a folder with a few demo rasters (use repo tests or your own)
folder = "tests\\data\\geotiff\\rhine"  # adjust as needed

# Parse files and build cube (order: numeric part in filenames)
dc = MultiDataset.read_multiple_files(folder, with_order=True, regex_string=r"\d+", date=False)
print(dc)

# Open bands into the cube's memory structure (if required by API)
# dc.open_datacube()

# Example: compute per-pixel mean over time (pseudo-code; see your API methods)
# mean_arr = dc.mean(axis=0)
# print(mean_arr.shape)
```

Notes:
- Ensure the rasters share the same shape and georeferencing.
- Use regex and format options to parse timestamps when needed.
