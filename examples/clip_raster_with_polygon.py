import geopandas as gpd
from pyramids.raster import Raster
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

epsg = poly.crs.to_epsg()
src = Raster.open(soilmappath)
src_reprojected = src.reproject(epsg)
cropped_obj = src_reprojected.crop(poly)
cropped_obj.ToGeotiff(f"{datapath}/out_cropped_raster1333.tif")