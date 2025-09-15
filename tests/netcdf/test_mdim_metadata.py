import pytest

from pyramids.netcdf.netcdf import NetCDF
from pyramids.netcdf.metadata import (
    get_mdim_metadata,
    to_json,
    from_json,
    to_dict,
    flatten_for_index,
)


@pytest.mark.parametrize("fixture_name", [
    "pyramids_created_nc_3d",
])
def test_mdim_metadata_basic(request, fixture_name: str):
    """Open a small NetCDF in MDIM mode and ensure traversal returns content.

    Also verifies that NetCDF.get_all_metadata delegates to the module API and that
    JSON round-trip preserves the structure.
    """
    path = request.getfixturevalue(fixture_name)

    # Open via library entrypoint in MDIM mode
    nc = NetCDF.read_file(path, open_as_multi_dimensional=True)

    # Use instance method
    md1 = nc.get_all_metadata(open_options={"OPEN_SHARED": "YES"})
    assert md1.driver and isinstance(md1.driver, str)
    assert md1.structural is not None
    assert isinstance(md1.groups, dict) and len(md1.groups) >= 1
    assert isinstance(md1.arrays, dict) and len(md1.arrays) >= 1
    assert isinstance(md1.dimensions, dict) and len(md1.dimensions) >= 1

    # dimension_overview should mirror nc.meta_data
    dov = md1.dimension_overview
    assert isinstance(dov, dict)
    assert set(["names", "sizes", "attrs", "values"]).issuperset(dov.keys()) or set(["names", "sizes", "attrs"]).issuperset(dov.keys())
    names = dov.get("names", [])
    sizes = dov.get("sizes", {})
    assert isinstance(names, list)
    assert isinstance(sizes, dict)
    # names from overview should be a subset of meta_data.names
    assert set(names).issubset(set(list(nc.meta_data.names)))

    # JSON round-trip
    s = to_json(md1)
    md2 = from_json(s)
    assert to_dict(md1) == to_dict(md2)

    # flatten_for_index returns basic keys
    flat = flatten_for_index(md1)
    assert flat["driver"] == md1.driver
    assert flat["array_count"] >= 1
    assert isinstance(flat.get("arrays"), list)


def test_get_mdim_metadata_from_path_and_instance(pyramids_created_nc_3d: str):
    """Ensure both path and instance inputs are supported and equivalent."""
    nc = NetCDF.read_file(pyramids_created_nc_3d, open_as_multi_dimensional=True)
    md_from_instance = get_mdim_metadata(nc)
    md_from_path = get_mdim_metadata(pyramids_created_nc_3d)

    # They should describe the same file structure (allowing for open_options differences)
    # Compare flattened summaries which should be equivalent regardless of ephemeral differences
    f1 = flatten_for_index(md_from_instance)
    f2 = flatten_for_index(md_from_path)
    assert f1 == f2
