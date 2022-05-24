import os

import geopandas as gpd


def basin_polygon():
    print(os.getcwd())
    print(os.listdir())
    print(os.listdir("tests/"))
    assert gpd.read_file("tests/basin.geojson")
