*************
GIS Functions
*************

Zonal Statistics
################

one of the most frequent used function in geospatial analysis is zonal statistics, where you overlay a shapefile contains some polygons with some maps and you want each polygon to extract the values that locates inside it from the map, `raster` module in `Hapi` contains a similar function `OverlayMap` where you can convert the polygon shapefile into a raster first and use it as a base map to overlay with other maps

You don't need to copy and paste the code in this page you can find it in the examples `Zonal Statistics <https://github.com/MAfarrag/Hapi/blob/master/Examples/GIS/ZonalStatistics.py/>`_.

OverlayMap one map
-----------------------------------

The `OverlayMap` function takes two ascii files the `BaseMap` which is the raster/asc file of the polygons and the secon is the asc file you want to extract its values.


.. code:: py

    def OverlayMap(Path, BaseMap, ExcludeValue, Compressed = False, OccupiedCellsOnly=True):
    """
    =================================================================
        (Path, BaseMap, ExcludeValue, Compressed = False, OccupiedCellsOnly=True)
    =================================================================
    this function is written to extract and return a list of all the values
    in an ASCII file

    Inputs:
        1-Path:
            [String] a path to ascii file (inclusing the extension).
        2-BaseMap:
            [String/array] a path includng the name of the ASCII/raster and extention like BaseMap="data/cropped.asc".
        3-ExcludedValue:
            [Numeric] values you want to exclude from extracted values.
        4-Compressed:
            [Bool] if the map you provided is compressed.
        5-OccupiedCellsOnly:
            [Bool] if you want to count only cells that is not zero.
    Outputs:
        1- ExtractedValues:
            [Dict] dictonary with a list of values in the basemap as keys
                and for each key a list of all the intersected values in the
                maps from the path.
        2- NonZeroCells:
            [dataframe] dataframe with the first column as the "file" name
            and the second column is the number of cells in each map.
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
