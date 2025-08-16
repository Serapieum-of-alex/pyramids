# How-to Recipes

Practical snippets for common tasks with pyramids.

## Read a GeoTIFF and print stats

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("tests\\data\\geotiff\\dem.tif")
arr = ds.read()
print(arr.min(), arr.max(), arr.mean())
```

## Convert a raster to ASCII

```python
from pyramids.dataset import Dataset

ds = Dataset.read_file("tests\\data\\geotiff\\dem.tif")
ds.to_file("out.asc")
```

## Crop a raster to bounding box

```python
from pyramids.dataset import Dataset

xmin, ymin, xmax, ymax = 6.8, 50.3, 7.2, 50.6  # example bbox
bbox = (xmin, ymin, xmax, ymax)
src = "tests\\data\\geotiff\\dem.tif"

ds = Dataset.read_file(src)
# ds.crop_bbox is illustrative; see actual API for cropping/windowing
# cropped = ds.crop_bbox(bbox)
# cropped.to_file("dem_cropped.tif")
```

## Read multiple rasters into a datacube

```python
from pyramids.datacube import Datacube

dc = Datacube.read_multiple_files("tests\\data\\geotiff\\rhine", with_order=True, regex_string=r"\\d+", date=False)
print(dc)
```

## Zonal statistics with a polygon layer

```python
from pyramids.dataset import Dataset
from pyramids.featurecollection import FeatureCollection

raster = Dataset.read_file("tests\\data\\geotiff\\dem.tif")
polys = FeatureCollection.read_file("tests\\data\\geometries\\polygons.geojson")
# table = raster.zonal_stats(polys)  # replace with the actual method name in your API
# print(table.head())
```
