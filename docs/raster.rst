######
raster
######

raster module contains one class `Raster` which have all the functions that deals with gdal
and rasterio objest. The module contains function that falls in one of the following categories.

- to import the raster module

.. code:: py

    from pyramids.raster import Raster


- The main parameter for most of the functions in the `raster` module is a `gdal.Dataset`

.. code:: py

    raster_path = "examples/data/acc4000.tif"
    src = gdal.Open(RasterAPath)
    fig, ax = Map.plot(src, title="Flow Accumulation")

.. image:: /images/flow_accumulation.png
   :width: 500pt

.. note::

    * change the directory of your code to point at the repository root directory to be able to read the raster files
    * the visualization in this documentatin uses digitalearth package to install it `Digital-Earth`_

- The module contains function that falls in one of the following categories.

#.`Raster Data`_
#.`Raster Operations`_
#.`Raster Dataset`_

***********
Raster Data
***********
function that are related to spatial resolution, projection and coordinates of a raster.


getRasterData
-------------
- the definition of the function is as follow.

.. code:: py

    get the basic data inside a raster (the array and the nodatavalue)

    Parameters
    ----------
    src: [gdal.Dataset]
        a gdal.Dataset is a raster already been read using gdal
    band : [integer]
        the band you want to get its data. Default is 1

    Returns
    -------
    array : [array]
        array with all the values in the flow path length raster
    nodataval : [numeric]
        value stored in novalue cells

- To use the function use the `gda.Dataset` you read using `gdal.Open` method

.. code:: py

    arr, nodataval = Raster.getRasterData(src)
    print(arr)
    array([[-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38,  0.000000e+00,  0.000000e+00,  0.000000e+00,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38,  1.000000e+00,  0.000000e+00,  2.000000e+00,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38, -3.402823e+38,  0.000000e+00,
         0.000000e+00,  2.000000e+00,  0.000000e+00,  4.000000e+00,
         0.000000e+00,  0.000000e+00, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38, -3.402823e+38,  0.000000e+00,
         4.000000e+00,  4.000000e+00,  0.000000e+00,  5.000000e+00,
         2.000000e+00,  0.000000e+00, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38, -3.402823e+38,  0.000000e+00,
         0.000000e+00,  1.100000e+01,  0.000000e+00,  0.000000e+00,
         1.000000e+01,  1.000000e+00, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38, -3.402823e+38,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  1.500000e+01,  0.000000e+00,  0.000000e+00,
         0.000000e+00,  1.300000e+01, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38,  0.000000e+00,  1.000000e+00,  1.000000e+00,
         1.500000e+01,  2.300000e+01,  4.500000e+01,  1.000000e+00,
         0.000000e+00,  1.500000e+01, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38,  0.000000e+00,  1.000000e+00,  1.100000e+01,
         6.000000e+00,  0.000000e+00,  2.000000e+00,  4.900000e+01,
         5.400000e+01,  0.000000e+00,  1.600000e+01,  1.700000e+01,
         0.000000e+00, -3.402823e+38],
       [-3.402823e+38,  0.000000e+00,  6.000000e+00,  4.000000e+00,
         0.000000e+00,  1.000000e+00,  1.000000e+00,  0.000000e+00,
         0.000000e+00,  5.500000e+01,  1.000000e+00,  2.000000e+00,
         8.600000e+01, -3.402823e+38],
       [ 0.000000e+00,  4.000000e+00,  2.000000e+00,  0.000000e+00,
         0.000000e+00,  0.000000e+00, -3.402823e+38,  0.000000e+00,
         1.000000e+00,  2.000000e+00,  5.900000e+01,  6.300000e+01,
         0.000000e+00,  8.800000e+01],
       [ 0.000000e+00,  1.000000e+00,  1.000000e+00, -3.402823e+38,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38,  0.000000e+00,  1.000000e+00,  0.000000e+00,
        -3.402823e+38, -3.402823e+38],
       [-3.402823e+38,  0.000000e+00,  0.000000e+00, -3.402823e+38,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38,
        -3.402823e+38, -3.402823e+38]], dtype=float32)

    print(nodataval)
    -3.4028230607370965e+38


getProjectionData
-----------------

- GetProjectionData returns the projection details of a given gdal.Dataset

Parameters
==========
src: [gdal.Dataset]
    raster read by gdal

Returns
=======
epsg: [integer]
     integer reference number that defines the projection (https://epsg.io/)
geo: [tuple]
    geotransform data of the upper left corner of the raster
    (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation, pixelsize).


.. code:: py

    epsg, geo = Raster.getProjectionData(src)
    print("EPSG = " + str(epsg))
    EPSG = 32618
    print(geo)
    (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)


getEPSG
-------


getCellCoords
-------------


openArrayInfo
-------------



*****************
Raster Operations
*****************
- saveRaster
- createRaster
- rasterLike
- rasterFill
- mapAlgebra
- resampleRaster
- projectRaster
- cropAlligned
- crop
- clipRasterWithPolygon
- clip2
- changeNoDataValue
- matchRasterAlignment
- nearestNeighbour
- readASCII
- writeASCII
- mosaic
- extractValues
- overlayMap
- normalize

**************
Raster Dataset
**************
- reprojectDataset
- cropAlignedFolder
- readASCIIsFolder
- rastersLike
- matchDataAlignment
- folderCalculator
- readRastersFolder
- overlayMaps

*****************
helping functions
*****************
- stringSpace


Zonal Statistics
****************

one of the most frequent used function in geospatial analysis is zonal
statistics, where you overlay a shapefile contains some polygons with
some maps and you want each polygon to extract the values that locates
inside it from the map, `raster` module in `Hapi` contains a similar
function `OverlayMap` where you can convert the polygon shapefile into
a raster first and use it as a base map to overlay with other maps

You don't need to copy and paste the code in this page you can find it
in the examples `Zonal Statistics <https://github.com/MAfarrag/Hapi/blob/master/Examples/GIS/ZonalStatistics.py/>`_.


OverlayMap one map
------------------

The `OverlayMap` function takes two ascii files the `BaseMap` which is the
raster/asc file of the polygons and the secon is the asc file you want to
extract its values.


.. code:: py

    def overlayMap(path: str,
            classes_map: Union[str, np.ndarray],
            exclude_value: Union[float, int],
            compressed: bool=False,
            occupied_Cells_only: bool=True) -> Tuple[Dict[List[float], List[float]], int]:
    """
    """OverlayMap.

            OverlayMap extracts and return a list of all the values in an ASCII file,
            if you have two maps one with classes, and the other map contains any type of values,
            and you want to know the values in each class

    Parameters
    ----------
    path: [str]
        a path to ascii file.
    classes_map: [str/array]
        a path includng the name of the ASCII and extention, or an array
        >>> path = "classes.asc"
    exclude_value: [Numeric]
        values you want to exclude from extracted values.
    compressed: [Bool]
        if the map you provided is compressed.
    occupied_Cells_only: [Bool]
        if you want to count only cells that is not zero.

    Returns
    -------
    ExtractedValues: [Dict]
        dictonary with a list of values in the basemap as keys
            and for each key a list of all the intersected values in the
            maps from the path.
    NonZeroCells: [dataframe]
        the number of cells in the map.
    """

To extract the

.. code:: py

    import Hapi.raster as R

    Path = "F:/02Case studies/Hapi Examples/"
    SavePath  = Path + "results/ZonalStatistics"
    BaseMapF = Path + "data/Polygons.tif"
    ExcludedValue = 0
    Compressed = True
    OccupiedCellsOnly = False

    ExtractedValues, Cells = R.OverlayMap(Path+"DepthMax22489.zip", BaseMapF,ExcludedValue, Compressed,OccupiedCellsOnly)


OverlayMap Several maps
===================
The `OverlayMaps` function takes path to the folder where more than one map exist instead of a path to one file, it also takes an extra parameter `FilePrefix`, this prefix is used to name the files in the given path and all the file has to start with the prefix

.. code:: py

    FilePrefix = "Map"
    # several maps
    ExtractedValues, Cells = R.OverlayMaps(Path+"data", BaseMapF, FilePrefix,ExcludedValue, Compressed,OccupiedCellsOnly)

both methods `OverlayMap` and `OverlayMaps` returns the values as a `dict`, the difference is in the number of cells `OverlayMaps` returns a single integer number while `OverlayMap` returns a `dataframe` with two columns the first in the map name and the second is the number of occupied cell in each map.

Save extracted values
===================

.. code:: py
    # save extracted values in different files
    Polygons = list(ExtractedValues.keys())
    for i in range(len(Polygons)):
        np.savetxt(SavePath +"/" + str(Polygons[i]) + ".txt",
                   ExtractedValues[Polygons[i]],fmt="%4.2f")


**********
References
**********

.. target-notes::
.. _`Digital-Earth`:
   https://github.com/MAfarrag/Digital-Earth
