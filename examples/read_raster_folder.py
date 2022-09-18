"""Created on Sun May  9 03:35:25 2021.

@author: mofarrag
"""
import os

import numpy as np

rpath = r"C:\MyComputer\01Algorithms\gis\pyramids"
os.chdir(rpath)


from pyramids.raster import Raster

Path = "examples/data/raster-folder/"
F = Raster.readRastersFolder(Path)
assert np.shape(F) == (125, 93, 6)

start = "1979-01-02"
end = "1979-01-05"
fmt = "%Y-%m-%d"

F = Raster.readRastersFolder(Path, start=start, end=end, fmt=fmt)
assert np.shape(F) == (125, 93, 4)

# "5_MSWEP_1979.01.06.tif".
