"""
Created on Sat May 16 00:02:08 2020

@author: mofarrag
"""


import geopandas as gpd
import Hapi.gis.raster as Raster

Raster_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/DEM_Germany.tif"
shapefile_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/cropDEM.shp"

shpfile = gpd.read_file(shapefile_path)


output_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/DEM_GermanyC.tif"

Raster.Clip(Raster_path, shapefile_path, save=True, output_path=output_path)
