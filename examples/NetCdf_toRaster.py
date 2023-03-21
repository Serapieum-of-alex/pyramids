"""Created on Sun Aug  2 22:30:48 2020.

@author: mofarrag

make sure to change the directory to the examples folder in the repo
"""
import os
import matplotlib

matplotlib.use("TkAgg")
from pyramids.convert import Convert

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
rdir = f"examples/"
# %% Netcdf file that contains only one layer
FileName = f"{rdir}/data/MSWEP_1979010100.nc"
SaveTo = f"{rdir}/data/"
# VarName = None
Convert.nctoTiff(FileName, SaveTo, separator="_")
# %% Netcdf file that contains multiple layer
FileName = f"{rdir}/data/precip.1979.nc"
SaveTo = f"{rdir}/data/Save_prec_netcdf_multiple/"

Convert.nctoTiff(FileName, SaveTo, separator=".")
# %% list of files
Path = rdir + "/data/GIS/netcdf files/"
SaveTo = rdir + "/data/GIS/Save_prec_netcdf_multiple/"

files = os.listdir(Path)
Paths = [Path + i for i in files]
for i in range(len(files)):
    FileName = Path + "/" + files[i]
    Convert.nctoTiff(FileName, SaveTo, separator=".")
