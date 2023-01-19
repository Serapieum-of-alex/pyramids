import os
import geopandas as gpd
from osgeo.gdal import Dataset
from pyramids.convert import Convert

def test_polygonize(
        test_image: Dataset,
        polygonized_raster_path: str
):
    Convert.polygonize(test_image, polygonized_raster_path)
    assert os.path.exists(polygonized_raster_path)
    gdf = gpd.read_file(polygonized_raster_path)
    assert len(gdf) == 4
    assert all(gdf.geometry.geom_type == "Polygon")
    os.remove(polygonized_raster_path)