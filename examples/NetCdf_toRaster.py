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
file_name = f"{rdir}/data/nc/MSWEP_1979010100.nc"
save_to = f"{rdir}/data/"
# VarName = None
# Convert.nctoTiff(file_name, save_to, separator="_")
# %% Netcdf file that contains multiple layer
# file_name = f"{rdir}/data/precip.1979.nc"
file_name = f"{rdir}/data/nc/202205_monthly_precipitation_amount_1hour_Accumulation.nc"
save_to = f"{rdir}/data/nc/Save_prec_netcdf_multiple/"

Convert.nctoTiff(file_name, save_to, separator=".")
# %% list of files
Path = rdir + "/data/GIS/netcdf files/"
save_to = rdir + "/data/GIS/Save_prec_netcdf_multiple/"

files = os.listdir(Path)
Paths = [Path + i for i in files]
for i in range(len(files)):
    file_name = Path + "/" + files[i]
    Convert.nctoTiff(file_name, save_to, separator=".")
