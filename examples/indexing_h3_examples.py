from geopandas.geodataframe import GeoDataFrame
from shapely import wkt

from pyramids.indexing import H3

#%% geometryToIndex
"""
get the index of the hexagon that the coordinates lie inside at a certain resolution level.

Parameters
----------
lat: float
    Latitude or y coordinate
lon: float
    longitude or x coordinate
resolution: [int]
    resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

Returns
-------
str:
    hexadecimal number
"""
lat_lon = (89.83, -157.30)
h3_resolution = 5
hex = H3.geometryToIndex(lat_lon[0], lat_lon[1], h3_resolution)
print(hex)

#%% getIndex
"""
Get Hexagon index.

    get the index of the hexagon that the coordinates lie inside at a certain resolution level.

Parameters
----------
gdf: [GeoDataFrame]
    GeoDataFrame
resolution: [int]
    resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

Returns
-------
Pandas Series:
    pandas series with a column with the h3 hex index.
"""
geom_list = [
    "POINT (-75.13333 21.93333)",
    "POINT (108.00000 3.00000)",
    "POINT (90.00000 -1.00000)",
    "POINT (42.00000 14.00000)",
]
geoms = list(map(wkt.loads, geom_list))
gdf = GeoDataFrame(geometry=geoms)
gdf.set_crs(epsg=4326, inplace=True)
# gdf.to_file("examples/data/indexing_data/indexing_h3_points.geojson")
print(gdf)
"""
                     geometry
0  POINT (-75.13333 21.93333)
1   POINT (108.00000 3.00000)
2   POINT (90.00000 -1.00000)
3   POINT (42.00000 14.00000)
"""
h3_resolution = 1
gdf["h3"] = H3.getIndex(gdf, h3_resolution)
print(gdf["h3"])
"""
                     geometry               h3
0  POINT (-75.13333 21.93333)  814cbffffffffff
1   POINT (108.00000 3.00000)  8169bffffffffff
2   POINT (90.00000 -1.00000)  8186bffffffffff
3   POINT (42.00000 14.00000)  8152bffffffffff
"""
h3_resolution = 0
gdf["h3"] = H3.getIndex(gdf, h3_resolution)
print(gdf["h3"])
"""
                     geometry               h3
0  POINT (-75.13333 21.93333)  804dfffffffffff
1   POINT (108.00000 3.00000)  8069fffffffffff
2   POINT (90.00000 -1.00000)  8087fffffffffff
3   POINT (42.00000 14.00000)  8053fffffffffff
"""
#%% indexToPolygon
"""indexToPolygon.

    return the polygon corresponding to the given hexagon index

Parameters
----------
hex_index: [str]
    hexagon index (hexadecimal format)

Returns
-------
Shapely Polygon
"""
hex_index = "854c91cffffffff"
geom = H3.indexToPolygon(hex_index)
#%% getGeometry
"""Get the Hexagon polygon geometry form a hexagon index.

Parameters
----------
gdf: [GeoDataFrame]
    geodataframe with a column filled with hexagon index

index_column: [str]
    column where the hexagon index is stored

Returns
-------
Pandas Series
    polygon geometries corespondint to the hexagon index.
"""
geom = H3.getGeometry(gdf, index_column="h3")
# geom.to_file("examples/data/indexing_data/indexing_h3_resolution1_polys.geojson")
