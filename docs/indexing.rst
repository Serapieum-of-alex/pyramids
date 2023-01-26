########
Indexing
########


**
H3
**

.. code-block:: py
    :linenos:

    from pyramids.indexing import H3

geometryToIndex
===============
get the index of the hexagon that the coordinates lie inside at a certain resolution level.

Parameters:
    lat: float
        Latitude or y coordinate
    lon: float
        longitude or x coordinate
    resolution: [int]
        resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

Returns:
    str:
        hexadecimal number

.. code-block:: py
    :linenos:

    lat_lon = (89.83, -157.30)
    h3_resolution = 5
    hex = H3.geometryToIndex(lat_lon[0], lat_lon[1], h3_resolution)
    print(hex)
    >>> 8503262bfffffff

getIndex
========

- get the index of the hexagon that the coordinates lie inside at a certain resolution level.

Parameters:
    gdf: [GeoDataFrame]
        GeoDataFrame
    resolution: [int]
        resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

Returns:
    Pandas Series:
        pandas series with a column with the h3 hex index.

.. code-block:: py
    :linenos:

    from geopandas.geodataframe import GeoDataFrame
    from pyramids.indexing import H3
    from shapely import wkt

    geom_list = [
        "POINT (-75.13333 21.93333)",
        "POINT (108.00000 3.00000)",
        "POINT (90.00000 -1.00000)",
        "POINT (42.00000 14.00000)",
    ]
    geoms = list(map(wkt.loads, geom_list))
    gdf = GeoDataFrame(geometry=geoms)
    gdf.set_crs(epsg=4326, inplace=True)

    print(gdf)

    >>>                      geometry
    >>> 0  POINT (-75.13333 21.93333)
    >>> 1   POINT (108.00000 3.00000)
    >>> 2   POINT (90.00000 -1.00000)
    >>> 3   POINT (42.00000 14.00000)

.. image:: images/indexing/points.png
    :width: 40pt

.. code-block:: py
    :linenos:

    h3_resolution = 1
    gdf["h3"] = H3.getIndex(gdf, h3_resolution)
    print(gdf["h3"])

    >>> 0    814cbffffffffff
    >>> 1    8169bffffffffff
    >>> 2    8186bffffffffff
    >>> 3    8152bffffffffff
    >>> Name: h3, dtype: object

.. image:: images/indexing/hexagons_level1.png
    :width: 40pt

.. code-block:: py
    :linenos:

    h3_resolution = 0
    gdf["h3"] = H3.getIndex(gdf, h3_resolution)
    print(gdf["h3"])

    >>> 0    804dfffffffffff
    >>> 1    8069fffffffffff
    >>> 2    8087fffffffffff
    >>> 3    8053fffffffffff
    >>>  Name: h3, dtype: object

.. image:: images/indexing/hexagons_level0.png
    :width: 40pt


indexToPolygon
==============

- Return the polygon corresponding to the given hexagon index

Parameters:
    hex_index: [str]
        hexagon index (hexadecimal format)

Returns:
    Shapely Polygon

.. code-block:: py
    :linenos:

    hex_index = "854c91cffffffff"
    geom = H3.indexToPolygon(hex_index)
    print(geom)

    >>> <shapely.geometry.polygon.Polygon object at 0x000002102C981330>

getGeometry
===========

- Get the Hexagon polygon geometry form a hexagon index.

Parameters:
    gdf: [GeoDataFrame]
        geodataframe with a column filled with hexagon index

    index_column: [str]
        column where the hexagon index is stored

Returns:
    Pandas Series
        polygon geometries corespondint to the hexagon index.

.. code-block:: py
    :linenos:

    geom = H3.getGeometry(gdf, index_column = "h3")

    print(geom)
    >>> 0    POLYGON ((-62.02253 31.87789, -74.64047 30.219...
    >>> 1    POLYGON ((107.74262 15.06330, 106.41168 2.7703...
    >>> 2    POLYGON ((94.36893 -13.83685, 95.22669 -2.5046...
    >>> 3    POLYGON ((37.16428 29.33908, 31.19369 18.24201...
    >>> dtype: geometry