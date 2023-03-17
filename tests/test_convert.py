import os

import geopandas as gpd
import numpy as np
from geopandas.geodataframe import DataFrame, GeoDataFrame
from osgeo import gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource

from pyramids.convert import Convert
from pyramids.raster import Raster




class TestConvertDataSourceAndGDF:
    def test_ds_to_gdf(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        gdf = Convert._ogrDataSourceToGeoDF(data_source)
        assert all(gdf == ds_geodataframe)

    def test_gdf_to_ds(self, data_source: DataSource, ds_geodataframe: GeoDataFrame):
        ds = Convert._gdfToOgrDataSource(ds_geodataframe)
        assert isinstance(ds, DataSource)
        assert ds.name == "memory"
