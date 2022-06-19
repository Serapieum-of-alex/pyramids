######
raster
######

raster module contains all the functions that deals with gdal and rasterio
objest. The module contains function that falls in one of the following
categories.

- The main parameter for most of the functions in the `raster` module
is the g


***********
Raster Data
***********
function that are related to spatial resolution, projection and
coordinates of a raster.


getRasterData
-------------


getProjectionData
-----------------


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
