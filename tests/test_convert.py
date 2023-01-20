import os
import numpy as np
import geopandas as gpd
from osgeo.gdal import Dataset

from pyramids.raster import Raster
from pyramids.convert import Convert


def test_polygonize(test_image: Dataset, polygonized_raster_path: str):
    Convert.polygonize(test_image, polygonized_raster_path)
    assert os.path.exists(polygonized_raster_path)
    gdf = gpd.read_file(polygonized_raster_path)
    assert len(gdf) == 4
    assert all(gdf.geometry.geom_type == "Polygon")
    os.remove(polygonized_raster_path)

def test_rasterize_vector(
        vector_mask_path,
        raster_to_df: str,
        raster_to_df_dataset: Dataset,
        rasterized_mask_path: str,
        rasterized_mask_array: np.ndarray
):
    src = Convert.rasterize(vector_mask_path, raster_to_df, rasterized_mask_path)
    assert Raster.getEPSG(src) == 32618
    geo = src.GetGeoTransform()
    geo_source = raster_to_df_dataset.GetGeoTransform()
    assert geo == geo_source
    arr, no_vata_val = Raster.getRasterData(src)
    assert no_vata_val == 0.0
    values = arr[arr[:,:] == 1.0]
    assert values.shape[0] == 16

# class TestRasterToDataFrame:
#     def test_raster_to_dataframe(
#             self,
#             raster_to_df: str,
#             vector_mask_path
#     ):
#         df = Convert.rasterToDataframe(raster_to_df, vector_mask_path)
#         print(df)
