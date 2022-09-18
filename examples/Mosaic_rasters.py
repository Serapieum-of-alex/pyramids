"""Created on Fri May 15 19:30:03 2020.

@author: mofarrag
"""
import glob
import os

from rasterio.plot import show

from pyramids.raster import Raster

os.chdir("F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/srtms")

# File and folder paths
dirpath = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/srtms"
out_fp = os.path.join(dirpath, "DEM_Germany.tif")
# %% Make a search criteria to select the DEM files
search_criteria = "*.tif"
filelist = os.path.join(dirpath, search_criteria)
print(filelist)

# glob function can be used to list files from a directory with specific criteria
dem_fps = glob.glob(filelist)

dst, dst_meta = Raster.mosaic(dem_fps, save=True, path=out_fp)

# %% Plot the result
show(dst, cmap="terrain")
