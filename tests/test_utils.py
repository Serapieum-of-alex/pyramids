import numpy as np
from osgeo import gdal

from pyramids._errors import DriverNotExistError
from pyramids._utils import numpy_to_gdal_dtype, Catalog


def test_numpy_to_gdal_dtype(arr: np.ndarray):
    gdal_type = numpy_to_gdal_dtype(arr)
    assert gdal_type is gdal.GDT_Float32


class TestCatalog:
    def test_create_instance(self):
        catalog = Catalog()
        assert hasattr(catalog, "catalog")

    def test_get_driver(self):
        catalog = Catalog()
        driver = catalog.get_driver("memory")
        assert isinstance(driver, dict)

    def test_get_driver_by_ext(self):
        catalog = Catalog()
        driver = catalog.get_driver_by_extension("nc")
        assert driver.get("GDAL Name") == "netCDF"
        try:
            catalog.get_driver_by_extension("mm")
        except DriverNotExistError:
            pass

    def test_get_gdal_name(self):
        catalog = Catalog()
        name = catalog.get_gdal_name("memory")
        assert name == "MEM"

    def test_exists(self):
        catalog = Catalog()
        assert catalog.exists("memory")
        assert not catalog.exists("MEM")

    def test_get_extension(self):
        catalog = Catalog()
        ext = catalog.get_extension("geotiff")
        assert ext == "tif"

    def test_get_driver_name(self):
        catalog = Catalog()
        name = catalog.get_driver_name("AAIGrid")
        assert name == "ascii"
