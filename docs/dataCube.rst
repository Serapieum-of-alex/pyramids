#######
dataset
#######

- dataset module contains Two classes `Dataset` and `DataCube`.

.. digraph:: Linking

    dataset -> Dataset;
    dataset -> DataCube;
    dpi=200;

- Dataset represent a raster object which could be created from reading a geotiff, netcdf, ascii or any file
    format/driver supported by gdal.
- The raster could have single or multi bands.
- The raster could have different variables (like netcdf file) and these variable can have similar or different
    dimensions.

- DataCube represent a stack of rasters which have the same dimensions, contains data that have same dimensions (rows
    & columns).


*******
Dataset
*******

.. image:: /images/dataset.svg
   :width: 200pt

The dataset object has some attributes and methods to help

- To import the raster module

.. code:: py

    from pyramids.dataset import Dataset


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

#. `Raster Data`_
#. `Raster Operations`_
#. `Raster Dataset`_


********
DataCube
********


crop
====

- `cropAlignedFolder`_ matches the location of nodata value from src raster to dst raster, Mask is where the
    nodatavalue will be taken and the location of this value src_dir is path to the folder where rasters exist where we
    need to put the NoDataValue of the mask in RasterB at the same locations.

Parameters
----------
    src_dir : [String]
        path of the folder of the rasters you want to set Nodata Value on the same location of NodataValue of Raster A,
        the folder should not have any other files except the rasters
    mask : [String/gdal.Dataset]
        path/gdal.Dataset of the mask raster to crop the rasters (to get the NoData value and it location in the array)
        Mask should include the name of the raster and the extension like "data/dem.tif", or you can read the mask raster
        using gdal and use is the first parameter to the function.
    saveto : [String]
        path where new rasters are going to be saved with exact same old names

Returns
-------
    new rasters have the values from rasters in B_input_path with the NoDataValue in the same
    locations like raster A

.. code:: py

    # The folder should contain tif files only (check example here `cropAlignedFolder`_)
    saveto = "examples/data/crop_aligned_folder/"
    Raster.cropAlignedFolder(aligned_raster_folder, src, saveto)

****************
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


OverlayMap Several maps
=======================
The `overlayMaps` function takes path to the folder where more than one map exist instead of a path to one file, it also takes an extra parameter `FilePrefix`, this prefix is used to name the files in the given path and all the file has to start with the prefix

.. code:: py

    FilePrefix = "Map"
    # several maps
    ExtractedValues, Cells = R.overlayMaps(Path+"data", BaseMapF, FilePrefix,ExcludedValue, Compressed,OccupiedCellsOnly)

both methods `OverlayMap` and `overlayMaps` returns the values as a `dict`, the difference is in the number of cells `overlayMaps` returns a single integer number while `OverlayMap` returns a `dataframe` with two columns the first in the map name and the second is the number of occupied cell in each map.

Save extracted values
=====================

.. code:: py

    # save extracted values in different files
    Polygons = list(ExtractedValues.keys())
    for i in range(len(Polygons)):
        np.savetxt(SavePath +"/" + str(Polygons[i]) + ".txt",
                   ExtractedValues[Polygons[i]],fmt="%4.2f")
