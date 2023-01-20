import os
from osgeo.ogr import DataSource
from geopandas.geodataframe import GeoDataFrame
from pyramids.vector import Vector

class TestOpenVector:
    def test_open_vector(
            self,
            test_vector_path: str
    ):
        ds = Vector.openVector(test_vector_path)
        assert isinstance(ds, DataSource)
        assert ds.name == test_vector_path

    def test_open_geodataframe(
            self,
            test_vector_path: str
    ):
        gdf = Vector.openVector(test_vector_path, geodataframe=True)
        assert isinstance(gdf, GeoDataFrame)

class TestCreateDataSource:
    def test_create_geojson_data_source(
            self,
            create_vector_path: str
    ):
        Vector.createDataSource(driver="GeoJSON", path=create_vector_path)
        assert os.path.exists(create_vector_path), "the geojson vector driver was not created in the given path"
        # clean created files
        os.remove(create_vector_path)

    def test_create_memory_data_source(
            self,
    ):
        ds = Vector.createDataSource(driver="MEMORY")
        assert isinstance(ds, DataSource), "the in memory ogr data source object was not created correctly"
        assert ds.name == "memData"

