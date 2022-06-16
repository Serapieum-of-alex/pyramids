"""
Created on Sun Aug  2 22:30:48 2020

@author: mofarrag

make sure to change the directory to the examples folder in the repo
"""
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyramids.convert import Convert

rpath = r"C:\MyComputer\01Algorithms\gis\pyramids"
ParentPath = f"{rpath}/examples/"
# %% Netcdf file that contains only one layer
FileName = ParentPath + "/data/MSWEP_1979010100.nc"
SaveTo = ParentPath + "/data/"
# VarName = None
Convert.nctoTiff(FileName, SaveTo, separator="_")
#%% plot
src = rasterio.open(SaveTo + "MSWEP_1979010100.nc")
# fig = plt.figure(figsize=(12, 8))
# im = plt.imshow(src.read(1) / 100.0, cmap="gist_rainbow")
# plt.title("Monthly mean sea level pressure")
# divider = make_axes_locatable(plt.gca())
# cax = divider.append_axes("right", "5%", pad="3%")
# plt.colorbar(im, cax=cax)
# plt.tight_layout()
# plt.show()
# %% Netcdf file that contains multiple layer
FileName = ParentPath + "/data/precip.1979.nc"
SaveTo = ParentPath + "/data/Save_prec_netcdf_multiple/"

Convert.nctoTiff(FileName, SaveTo, separator=".")
# %% list of files
Path = ParentPath + "/data/GIS/netcdf files/"
SaveTo = ParentPath + "/data/GIS/Save_prec_netcdf_multiple/"

files = os.listdir(Path)
Paths = [Path + i for i in files]
for i in range(len(files)):
    FileName = Path + "/" + files[i]
    Convert.nctoTiff(FileName, SaveTo, separator=".")
