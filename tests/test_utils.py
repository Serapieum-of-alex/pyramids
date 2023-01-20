import numpy as np
from osgeo import gdal
from pyramids.utils import numpy_to_gdal_dtype

def test_numpy_to_gdal_dtype(arr:np.ndarray):
    gdal_type = numpy_to_gdal_dtype(arr)
    assert gdal_type is gdal.GDT_Float32