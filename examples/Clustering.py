# import numpy as np
import sys

from osgeo import gdal

from pyramids.dem import DEM as GC

# Path = "F:/04Private/990110182315.csv"  # Book1.csv
rpath = r"C:\MyComputer\01Algorithms\gis\pyramids"
Path = f"{rpath}/examples/data/DEM5km_Rhine_burned_acc.tif"
# data = np.loadtxt(Path, delimiter=',')

Data = gdal.Open(Path)
DataArr = Data.ReadAsArray()
NoDataValue = Data.GetRasterBand(1).GetNoDataValue()

print(sys.getrecursionlimit())
# sys.setrecursionlimit(6000)
# print(sys.getrecursionlimit())
# %%
lowervalue = 0.1  # DataArr[DataArr != NoDataValue].min()
uppervalue = 20  # DataArr[DataArr != NoDataValue].max()

ClusterArray, count, Position, Values = GC.cluster(DataArr, lowervalue, uppervalue)
