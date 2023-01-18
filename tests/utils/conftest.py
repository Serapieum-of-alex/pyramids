import pytest


@pytest.fixture(scope="module")
def compressed_raster() -> str:
    return "tests/data/chirps-v2.0.2009.01.01.tif.gz"


@pytest.fixture(scope="module")
def uncompressed_output() -> str:
    return "tests/data/chirps-v2.0.2009.01.01.tif"
