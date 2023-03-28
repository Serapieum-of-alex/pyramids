import geopandas as gpd
from pyramids.dataset import Dataset
from osgeo import gdal

datapath = "examples/data"
Basinshp = f"{datapath}/basin.geojson"
aligned_raster = f"{datapath}/Evaporation_ECMWF_ERA-Interim_mm_daily_2009.01.01.tif"
soilmappath = f"{datapath}/soil_raster.tif"
#%%
gdal.UseExceptions()
gdal.ErrorReset()
#%%
poly = gpd.read_file(Basinshp)

epsg = poly.crs.to_crs()
src = Dataset.read_file(soilmappath)
src_reprojected = src.to_crs(epsg)
cropped_obj = src_reprojected.crop(poly)
cropped_obj.to_file(f"{datapath}/out_cropped_raster1333.tif")
#%%
poly_path = "tests/data/polygon_germany.geojson"
poly = gpd.read_file("tests/data/polygon_germany.geojson")
xmin, ymin, xmax, ymax = poly.bounds.values.tolist()[0]
window = (xmin, ymax, xmax, ymin)

path = "tests/data/raster-folder/1_MSWEP_1979.01.02.tif"
src = gdal.Open(path)
# dst = gdal.GetDriverByName("MEM").CreateCopy("", src, 0)
# gdal.Translate(dst, src, projWin=window)
dst = gdal.Warp("", src, cutline=poly_path, format="VRT")
dst_obj = Dataset(dst)
dst_obj.to_file(f"{datapath}/out_cropped_raster1333.tif")
