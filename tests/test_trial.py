import os

import geopandas as gpd


def test_basin_polygon():
    print(os.getcwd())
    print(os.listdir())
    print(os.listdir("tests/"))
    print(os.listdir("examples/data"))
    gpd.read_file("tests/basin.geojson")
