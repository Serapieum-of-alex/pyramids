"""
Created on Fri Feb 19 17:04:28 2021

@author: mofarrag
"""
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from osgeo import gdal, osr

from pyramids.raster import Raster

#%% directory of the examples folder
"chenge the current path to the examples directory inside the repo"
rpath = r"C:\MyComputer\01Algorithms\gis\pyramids\examples"
os.chdir(rpath)
#%%
RasterApath = "data/DEM5km_Rhine_burned_acc.tif"
RasterBpath = "data/MSWEP_1979010100.tif"
# RasterBpath = "F:/01Algorithms/Hydrology/HAPI/examples/data/GIS/MSWEP_4746epsg.tif"
SaveTo = "data/MSWEP_1979010100_reprojected.tif"

RasterA = gdal.Open(RasterApath)
RasterB = gdal.Open(RasterBpath)
#%%
# get the array and the nodatavalue in the raster
RasterA_arr, nodataval = Raster.getRasterData(RasterA, band=1)

plt.imshow(RasterA_arr, cmap="CMRmap", vmax=RasterA_arr.max(), vmin=RasterA_arr.min())
plt.colorbar()
#%%
# we need number of rows and cols from src A and data from src B to store both in dst
RasterA_proj = RasterA.GetProjection()
RasterA_epsg = osr.SpatialReference(wkt=RasterA_proj)

to_epsg = int(RasterA_epsg.GetAttrValue("AUTHORITY", 1))
RasterB_reprojected = Raster.projectRaster(
    RasterB, to_epsg, resample_technique="cubic", option=1
)
# GET THE GEOTRANSFORM
RasterB_gt = RasterB.GetGeoTransform()
# GET NUMBER OF columns
RasterB_x = RasterB.RasterXSize
# get number of rows
RasterB_y = RasterB.RasterYSize
# we need number of rows and cols from src A and data from src B to store both in dst
RasterB_proj = RasterB.GetProjection()
RasterB_epsg = osr.SpatialReference(wkt=RasterB_proj)
# GET THE GEOTRANSFORM
RasterB_reprojected_gt = RasterB_reprojected.GetGeoTransform()
# GET NUMBER OF columns
RasterB_reprojected_x = RasterB_reprojected.RasterXSize
# get number of rows
RasterB_reprojected_y = RasterB_reprojected.RasterYSize
# we need number of rows and cols from src A and data from src B to store both in dst
RasterB_reprojected_proj = RasterB_reprojected.GetProjection()
RasterB_reprojected_epsg = osr.SpatialReference(wkt=RasterB_reprojected_proj)

RasterB_reprojected_array = RasterB_reprojected.ReadAsArray()
#%% save the raster
Raster.saveRaster(RasterB_reprojected, SaveTo)
#%%
RasterB_reprojected = Raster.projectRaster(
    RasterB,
    int(RasterA_epsg.GetAttrValue("AUTHORITY", 1)),
    resample_technique="cubic",
    option=2,
)
SaveTo = "data/GIS/MSWEP_1979010100_reprojected2.tif"
Raster.saveRaster(RasterB_reprojected, SaveTo)
