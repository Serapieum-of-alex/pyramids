# from IPython import get_ipython
from osgeo import gdal, gdalconst, ogr

# get_ipython().magic("reset -f")
# import os


# os.chdir("")


rasterF = "/home/zeito/pyqgis_data/utah_demUTM2.tif"
shpF = "/home/zeito/pyqgis_data/polygon8.shp"
output = "/home/zeito/pyqgis_data/my.tif"
NoData_value = -999999

raster = gdal.Open(rasterF, gdalconst.GA_ReadOnly)
geo_transform = raster.GetGeoTransform()
# source_layer = data.GetLayer()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * raster.RasterXSize
y_min = y_max + geo_transform[5] * raster.RasterYSize

x_res = raster.RasterXSize
y_res = raster.RasterYSize

mb_v = ogr.Open(shpF)
mb_l = mb_v.GetLayer()
pixel_width = geo_transform[1]


target_ds = gdal.GetDriverByName("GTiff").Create(output, x_res, y_res, 1, gdal.GDT_Byte)

target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
band = target_ds.GetRasterBand(1)

band.SetNoDataValue(NoData_value)
band.FlushCache()
gdal.RasterizeLayer(target_ds, [1], mb_l)  # , options=["ATTRIBUTE=hedgerow"]

target_ds = None
