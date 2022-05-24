"""
Created on Sun Aug  2 22:30:48 2020

@author: mofarrag

make sure to change the directory to the examples folder in the repo
"""
import os

import matplotlib.pyplot as plt
import rasterio
from Hapi.gis.raster import Raster
from mpl_toolkits.axes_grid1 import make_axes_locatable

ParentPath = "F:/Users/mofarrag/Documents/01Algorithms/HAPI/examples/"
# %% Netcdf file that contains only one layer
FileName = ParentPath + "/data/GIS/MSWEP_1979010100.nc"
SaveTo = ParentPath + "/data/GIS/"
VarName = None

Raster.NCtoTiff(FileName, SaveTo, Separator="_")

# %plot

src = rasterio.open(SaveTo + "MSWEP_1979010100.nc")
fig = plt.figure(figsize=(12, 8))
im = plt.imshow(src.read(1) / 100.0, cmap="gist_rainbow")
plt.title("Monthly mean sea level pressure")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)
plt.tight_layout()
plt.show()
# %% Netcdf file that contains multiple layer
FileName = ParentPath + "/data/GIS/precip.1979.nc"
SaveTo = ParentPath + "/data/GIS/Save_prec_netcdf_multiple/"

Raster.NCtoTiff(FileName, SaveTo, Separator=".")
# %% list of files
Path = ParentPath + "/data/GIS/netcdf files/"
SaveTo = ParentPath + "/data/GIS/Save_prec_netcdf_multiple/"

files = os.listdir(Path)
Paths = [Path + i for i in files]
for i in range(len(files)):
    FileName = Path + "/" + files[i]
    Raster.NCtoTiff(FileName, SaveTo, Separator=".")
