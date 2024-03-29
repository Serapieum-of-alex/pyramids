import numpy as np
from pyramids.dataset import Dataset
from digitalearth.static import StaticGlyph
from osgeo import gdal

datapath = "examples/data"
# path = rf"{datapath}/asci_example.asc"
path = rf"{datapath}/zip/asci_example.asc"
gdal.UseExceptions()
# %% read ascii
# arr, geotransform = Dataset.readASCII(path, dtype=1)
src_obj = Dataset.read_file(path)
assert src_obj.band_count == 1
assert src_obj.epsg == 6326
assert isinstance(src_obj.raster, gdal.Dataset)
assert src_obj.geotransform == (
    432968.1206170588,
    4000.0,
    0.0,
    520007.787999178,
    0.0,
    -4000.0,
)
fig, ax = StaticGlyph.plot(
    src_obj,
    title="Read ASCII file",
    color_scale=2,
    ticks_spacing=0.01,
    nodataval=None,
)
#%% write ascii
# arr = src_obj.raster.ReadAsArray()
# arr[~np.isclose(arr, geotransform[-1], rtol=0.001)] = 0.03
path2 = rf"{datapath}/roughness0.asc"
src_obj._to_ascii(path2)
