import numpy as np
from osgeo import gdal, ogr

from pyramids._errors import DriverNotExistError
from pyramids._utils import (
    numpy_to_gdal_dtype,
    gdal_to_numpy_dtype,
    gdal_to_ogr_dtype,
    Catalog,
    ogr_ds_togdal_dataset,
    ogr_to_numpy_dtype,
)


def test_numpy_to_gdal_dtype(arr: np.ndarray):
    # test with array input
    gdal_type = numpy_to_gdal_dtype(arr)
    assert gdal_type is gdal.GDT_Float32
    # test with  a dtye input
    gdal_type = numpy_to_gdal_dtype(arr.dtype)
    assert gdal_type is gdal.GDT_Float32


def test_gdal_to_numpy_dtype():
    assert gdal_to_numpy_dtype(6) == "float32"
    assert gdal_to_numpy_dtype(7) == "float64"
    assert gdal_to_numpy_dtype(2) == "uint16"
    try:
        gdal_to_numpy_dtype(20)
    except ValueError:
        pass


def test_gdal_to_ogr_dtype(test_image: gdal.Dataset, src: gdal.Dataset):
    assert gdal_to_ogr_dtype(test_image) == 0
    assert gdal_to_ogr_dtype(src) == 2


def test_ogr_to_numpy_dtype():
    assert ogr_to_numpy_dtype(0) == np.int32
    try:
        ogr_to_numpy_dtype(1)
    except ValueError:
        pass


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


def test_ogr_ds_togdal_dataset(data_source: ogr.DataSource):
    gdal_ds = ogr_ds_togdal_dataset(data_source)
    assert isinstance(gdal_ds, gdal.Dataset)
