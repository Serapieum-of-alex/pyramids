import pytest


@pytest.fixture(scope="module")
def noah_nc_path() -> str:
    return "tests/data/netcdf/noah-precipitation-1979.nc"


@pytest.fixture(scope="module")
def pyramids_created_nc_3d() -> str:
    return "tests/data/netcdf/pyramids-netcdf-3d.nc"
