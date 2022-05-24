"""
Created on Fri Mar 26 21:44:12 2021

@author: mofarrag
"""
import os

Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "/Coello/HAPI/Data")
# from gdalconst import GA_ReadOnly
# import osr
# from osgeo import gdal, gdalconst

from Hapi.gis.raster import Raster

SourceRasterPath = "00inputs/GIS/4000/acc4000.tif"
RasterTobeClippedPath = (
        Comp + "/Documents/01Algorithms/HAPI/Hapi/Parameters/01/Par_BETA.tif"
)
output = "F:/Users/mofarrag/distributed/Hapi/Data/output.tif"
# %%

Raster.ClipRasterWithRaster(RasterTobeClippedPath, SourceRasterPath, output, Save=True)
