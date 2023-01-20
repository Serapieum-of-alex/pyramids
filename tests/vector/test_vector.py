import os
from osgeo.ogr import DataSource
from pyramids.vector import Vector
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

