import geopandas as gpd


def basin_polygon():
    assert gpd.read_file("tests/basin.geojson")
