"""Created on Fri Oct 11 15:43:22 2019.

@author: mofarrag

match two rasters
"""
#%links
import os

rpath = r"C:\MyComputer\01Algorithms\gis\pyramids"
os.chdir(f"{rpath}/examples")

from osgeo import gdal

# import datetime as dt
# import pandas as pd
from pyramids.raster import Raster

# import matplotlib.pyplot as plt
#%% inputs
RasterAPath = f"{rpath}/examples/data/DEM5km_Rhine_burned_acc.tif"
RasterBPath = f"{rpath}/examples/data/MSWEP_1979010100_reprojected.tif"
SaveTo = f"{rpath}/examples/data/MSWEP_1979010100_Matched.tif"
#%% Read the Input rasters
# the source raster is of the ASCII format
src = gdal.Open(RasterAPath)
src_Array = src.ReadAsArray()
print("Shape of source raster = " + str(src_Array.shape))

# read destination array
dst = gdal.Open(RasterBPath)
Dst_Array = dst.ReadAsArray()
print("Shape of distnation raster Before matching = " + str(Dst_Array.shape))

### Match the alignment of both rasters
NewRasterB = Raster.matchRasterAlignment(src, dst)
NewRasterB_array = NewRasterB.ReadAsArray()
print("Shape of distnation  raster after matching = " + str(NewRasterB_array.shape))

message = "Error the shape of the result raster does not match the source raster"
assert (
    NewRasterB_array.shape[0] == src_Array.shape[0]
    and NewRasterB_array.shape[1] == src_Array.shape[1]
), message

# %% Match the NODataValue
# TODO : fix bug in nearestneighbor
NewRasterB_ND = Raster.cropAlligned(src, NewRasterB)

NoDataValue = NewRasterB_ND.GetRasterBand(1).GetNoDataValue()

assert (
    src.GetRasterBand(1).GetNoDataValue() == NoDataValue
), "NoData Value does not match"

# NewRasterB_ND_array =NewRasterB_ND.ReadAsArray()

# f = NewRasterB_ND_array[NewRasterB_ND_array == NoDataValue]
# g = src_Array[src_Array == NoDataValue]

# %%
Raster.saveRaster(NewRasterB_ND, SaveTo)
