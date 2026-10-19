"""Longitude conversion: wrap_longitude.

(``convert_units`` physical value-unit conversion is covered in
``tests/dataset/unit/test_unit_properties.py``, not here.)
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_wrap_longitude_returns_container(sample):
    """``wrap_longitude`` shifts 0–360 longitudes into the -180/180 frame."""
    nc = NetCDF.read_file(sample("coards__5v__1d4-4d1__y-desc.nc"))
    try:
        result = nc.wrap_longitude()
        assert isinstance(result, NetCDF)
        assert (result.lon < 0).any(), (
            "expected negative longitudes after 0-360 → -180/180 wrap"
        )
    finally:
        nc.close()
