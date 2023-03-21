from pyramids.raster import Raster
from pyramids.convert import Convert
import geopandas as gpd

# rpath = "C:/gdrive/01Algorithms/gis/rastertodataframe/tests/data/"
raster_path = f"tests/mo/raster_to_df/raster.tif"
vector_path = f"tests/mo/raster_to_df/vector.geojson"
rasterized_vector_path = f"tests/mo/raster_to_df/resulted_rasterized_vector.tif"
raster_wgs84_path = f"tests/mo/raster_to_df/raster_epsg4326.tif"
single_band_raster = f"tests/mo/raster_to_df/oneband.tif"
#%%

vector_mask = Convert.polygonToRaster(
    raster_wgs84_path, vector_path, rasterized_vector_path, vector_field="value"
)
src = Raster.read(raster_wgs84_path)
arr = src.ReadAsArray()
tile_arr = src.ReadAsArray(xoff=0, yoff=0, xsize=58, ysize=39)
# out_df = Convert.rasterToDataframe(raster_wgs84_path, vector_path=vector_path)
out_df = Convert.rasterToGeoDataFrame(raster_wgs84_path)
