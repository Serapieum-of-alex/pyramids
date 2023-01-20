from pathlib import Path

import pytest
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset

from tests.catchment.conftest import *
from tests.raster.conftest import *
from tests.utils.conftest import *
from tests.vector.conftest import *


@pytest.fixture(scope="session")
def test_image_path() -> Path:
    return Path("tests/data/test_image.tif").absolute()


@pytest.fixture(scope="session")
def test_image(test_image_path: Path) -> Dataset:
    return gdal.Open(str(test_image_path))


@pytest.fixture(scope="session")
def polygonized_raster_path() -> str:
    return "tests/data/polygonized.geojson"


@pytest.fixture(scope="session")
def arr() -> np.ndarray:
    return np.array([[1,2], [3,4]], dtype=np.float32)
