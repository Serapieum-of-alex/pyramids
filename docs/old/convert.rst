#######
Convert
#######
The convert module contains all the conversion function between different data types (vector and raster data types)

***************
rasterToPolygon
***************
`RasterToPolygon` takes a gdal Dataset object and group neighboring cells with the same value into one
polygon, the resulted vector will be saved to disk as a geojson file

===============================
Case 1: Save the result to disk
===============================
- If you are working with big raster data with small spatial resolution, you should consider saving the output
    polygon directly to disk.
- If you give the second parameter a path, the function will save the resulted polygon to disk.

.. code-block:: py

    src_path = gdal.Open("examples/data/convert_data/test_image.tif")
    polygonized_raster_path = "examples/data/convert_data/polygonized.geojson"
    Convert.rasterToPolygon(src_path, polygonized_raster_path, driver="GeoJSON")

=================================
Case 2: Return the output Polygon
=================================

- If you want the output polygon to be returned as a `GeoDataFrame`

.. code-block:: py

    gdf = Convert.rasterToPolygon(src_path)
    print(gdf)
    >>> id                                           geometry
    >>> 0  310  POLYGON ((-92.03658 41.27464, -92.03658 41.272...
    >>> 1  376  POLYGON ((-92.03450 41.27464, -92.03450 41.272...
    >>> 2  410  POLYGON ((-92.03658 41.27256, -92.03658 41.270...
    >>> 3  475  POLYGON ((-92.03450 41.27256, -92.03450 41.270...


.. image:: _images/convert/raster_to_polygon.png
    :width: 500pt

***************
polygonToRaster
***************
Covert a vector into raster.

- The raster cell values will be taken from the column name given in the `vector_filed` in the vector file.
- All the new raster geo-transform data will be copied from the given raster.
- Both the raster and the vector should have the same projection.

==============================================================
Case 1: First two input parameters are paths for files on disk
==============================================================

- Path for output raster. if given the resulted raster will be saved to disk.

.. code-block:: py

    input_vector_path = "examples/data/convert_data/mask.geojson"
    src_raster_path = "examples/data/convert_data/raster_to_df.tif"
    output_raster = "examples/data/convert_data/rasterized_mask.tif"

    Convert.polygonToRaster(
        input_vector_path, src_raster_path, output_raster
    )


.. image:: _images/convert/raster_to_polygon.png
    :width: 500pt

.. note::
    Currently the code does not reproject any of the raster or the vector if they have different projections


=================================================
Case 2: The input vector is a GeoDataFrame object
=================================================

.. code-block:: py

    gdf = gpd.read_file(input_vector_path)
    print(gdf)

    >>>    fid                                           geometry
    >>>     0    1  POLYGON ((432933.947 520034.455, 448910.957 51...

    Convert.polygonToRaster(
        gdf, src_raster_path, output_raster
    )

================================
Case 3: Return the output raster
================================

There is no given path to save the output raster to disk to it will be returned as an output.

.. code-block:: py

    src = Convert.polygonToRaster(gdf, src_raster_path)
    type(src)
    >>> <class 'osgeo.gdal.Datacube'>


********************
rasterToGeoDataFrame
********************

The function do the following
- Flatten the array in each band in the raster then mask the values if a vector
file is given otherwise it will flatten all values.

- Put the values for each band in a column in a dataframe under the name of the raster band, but if no meta
    data in the raster band exists, an index number will be used [1, 2, 3, ...]
- The values in the dataframe will be ordered row by row from top to bottom
- The function has a add_geometry parameter with two possible values ["point", "polygon"], which you can
    specify the type of shapely geometry you want to create from each cell,
        - If point is chosen, the created point will be at the center of each cell
        - If a polygon is chosen, a square polygon will be created that covers the entire cell.

==========
Parameters
==========
    src : [str/gdal Dataset]
        Path to raster file.
    vector : Optional[GeoDataFrame/str]
        GeoDataFrame for the vector file path to vector file. If given, it will be used to clip the raster
    add_geometry: [str]
        "Polygon", or "Point" if you want to add a polygon geometry of the cells as  column in dataframe.
        Default is None.
    tile: [bool]
        True to use tiles in extracting the values from the raster. Default is False.
    tile_size: [int]
        tile size. Default is 1500.


.. code-block:: py

    src_raster_path = "examples/data/convert_data/raster_to_df.tif"
    gdf = Convert.rasterToGeoDataFrame(src_raster_path, add_geometry="Point")

- The resulted geodataframe will have the band value under the name of the band (if the raster file has a metadata,
    if not, the bands will be indexed from 1 to the number of bands)

.. code-block:: py

    print(gdf)
    >>> Band_1                       geometry
    >>> 0         1  POINT (434968.121 518007.788)
    >>> 1         2  POINT (438968.121 518007.788)
    >>> 2         3  POINT (442968.121 518007.788)
    >>> 3         4  POINT (446968.121 518007.788)
    >>> 4         5  POINT (450968.121 518007.788)
    >>> ..      ...                            ...
    >>> 177     178  POINT (470968.121 470007.788)
    >>> 178     179  POINT (474968.121 470007.788)
    >>> 179     180  POINT (478968.121 470007.788)
    >>> 180     181  POINT (482968.121 470007.788)
    >>> 181     182  POINT (486968.121 470007.788)
    >>> [182 rows x 2 columns]

.. image:: _images/convert/raster_to_geodataframe.png
    :width: 500pt

***********************
Case 2: Mask the raster
***********************

.. code-block:: py

    gdf = gpd.read_file(input_vector_path)
    df = Convert.rasterToGeoDataFrame(src_raster_path, gdf)

    print(df)

    >>>     Band_1  fid
    >>> 0        1    1
    >>> 1        2    1
    >>> 2        3    1
    >>> 3        4    1
    >>> 4       15    1
    >>> 5       16    1
    >>> 6       17    1
    >>> 7       18    1
    >>> 8       29    1
    >>> 9       30    1
    >>> 10      31    1
    >>> 11      32    1
    >>> 12      43    1
    >>> 13      44    1
    >>> 14      45    1
    >>> 15      46    1

.. image:: _images/convert/raster_to_df_with_mask.png
    :width: 500pt
