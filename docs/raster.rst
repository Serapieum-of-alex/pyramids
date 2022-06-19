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

#. `Raster Data`_
#. `Raster Operations`_
#. `Raster Dataset`_


***********
Raster Data
***********
function that are related to spatial resolution, projection and coordinates of a raster.


getRasterData
-------------
- `getRasterData` get the basic data inside a raster (the array and the nodatavalue)

Parameters
==========
    src: [gdal.Dataset]
        a gdal.Dataset is a raster already been read using gdal
    band : [integer]
        the band you want to get its data. Default is 1

Returns
=======
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
- `getProjectionData` returns the projection details of a given gdal.Dataset

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
# TODO

getCellCoords
-------------

- `getCellCoords` returns the coordinates of all cell centres inside the domain (only the cells that
        does not have nodatavalue)

Parameters
==========
    src : [gdal_Dataset]
        Get the data from the gdal datasetof the DEM

Returns
=======
    coords : array
        Array with a list of the coordinates to be interpolated, without the Nan
    mat_range : array
        Array with all the centres of cells in the domain of the DEM


.. code:: py

    coords, centers_coords = Raster.getCellCoords(src)
    print(coords)
    array([[434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],
       [434968.12061706, 520007.78799918],

    print(centers_coords)
    array([[[434968.12061706, 520007.78799918],
        [438968.12061706, 520007.78799918],
        [442968.12061706, 520007.78799918],
        [446968.12061706, 520007.78799918],
        [450968.12061706, 520007.78799918],
        [454968.12061706, 520007.78799918],
        [458968.12061706, 520007.78799918],


openArrayInfo
-------------
# TODO


*****************
Raster Operations
*****************

- clipRasterWithPolygon
- clip2
- changeNoDataValue

- nearestNeighbour
- readASCII
- writeASCII
- mosaic
- extractValues
- overlayMap
- normalize

saveRaster
-------------
- `saveRaster` saves a raster to a path

Parameters
==========
    raster: [gdal object]
        gdal dataset opbject
    path: [string]
        a path includng the name of the raster and extention like
        path="data/cropped.tif"

Returns
=======
    the function does not return and data but only save the raster to the hard drive

.. code:: py

    path = "examples/data/save_raster_test.tif"
    Raster.saveRaster(src, path)


createRaster
-------------
- `createRaster` method creates a raster from a given array and geotransform data
and save the tif file if a Path is given or it will return the gdal.Dataset

Parameters
==========
    path : [str], optional
        Path to save the Raster, if '' is given a memory raster will be returned. The default is ''.
    arr : [array], optional
        numpy array. The default is ''.
    geo : [list], optional
        geotransform list [minimum lon, pixelsize, rotation, maximum lat, rotation,
            pixelsize]. The default is ''.
    nodatavalue : TYPE, optional
        DESCRIPTION. The default is -9999.
    epsg: [integer]
        integer reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )

Returns
=======
    dst : [gdal.Dataset/save raster to drive].
        if a path is given the created raster will be saved to drive, if not
        a gdal.Dataset will be returned.

- If we take the array we obtained from the `getRasterData`, do some arithmetic operation in it, then we created a
`gdal.DataSet` out of it

.. code:: py

    src = Raster.createRaster(arr=arr, geo=geo, epsg=str(epsg), nodatavalue=nodataval)
    Map.plot(src, title="Flow Accumulation")


.. image:: /images/flow_accumulation.png
   :width: 500pt

rasterLike
----------
- `rasterLike` method creates a Geotiff raster like another input raster, new raster will have the same projection,
coordinates or the top left corner of the original raster, cell size, nodata velue, and number of rows and columns
the raster and the dem should have the same number of columns and rows

Parameters
==========
    src : [gdal.dataset]
        source raster to get the spatial information
    array : [numpy array]
        to store in the new raster
    path : [String]
        path to save the new raster including new raster name and extension (.tif)
    pixel_type : [integer]
        type of the data to be stored in the pixels,default is 1 (float32)
        for example pixel type of flow direction raster is unsigned integer
        1 for float32
        2 for float64
        3 for Unsigned integer 16
        4 for Unsigned integer 32
        5 for integer 16
        6 for integer 32

Returns
=======
    save the new raster to the given path

- If we have made some calculation on raster array and we want to save the array back in the raster

.. code:: py

    arr2 = np.ones(shape=arr.shape, dtype=np.float64) * nodataval
    arr2[~np.isclose(arr, nodataval, rtol=0.001)] = 5

    path = "examples/data/rasterlike.tif"
    Raster.rasterLike(src, arr2, path)

- Now to check the raster that has been saved we can read it again with `gda.Open`

.. code:: py

    dst = gdal.Open(path)
    Map.plot(dst, title="Flow Accumulation", color_scale=1)


.. image:: /images/raster_like.png
   :width: 500pt

mapAlgebra
-------------

- `mapAlgebra` executes a mathematical operation on raster array and returns the result

Parameters
==========
    src : [gdal.dataset]
        source raster to that you want to make some calculation on its values
    fun: [function]
        defined function that takes one input which is the cell value

Returns
=======
    Dataset
        gdal dataset object

.. code:: py

    def classify(val):
        if val < 20:
            val = 1
        elif val < 40:
            val = 2
        elif val < 60:
            val = 3
        elif val < 80:
            val = 4
        elif val < 100:
            val = 5
        else:
            val = 0
        return val


    dst = Raster.mapAlgebra(src, classify)
    Map.plot(dst, title="Classes", color_scale=4, ticks_spacing=1)

.. image:: /images/map_algebra.png
   :width: 500pt



rasterFill
----------

- `rasterFill` takes a raster and fill it with one value.

Parameters
==========
    src : [gdal.dataset]
        source raster
    val: [numeric]
        numeric value
    save_to : [str]
        path including the extension (.tif)

Returns
=======
    raster : [saved on disk]
        the raster will be saved directly to the path you provided.

.. code:: py

    path = "examples/data/fillrasterexample.tif"
    value = 20
    Raster.rasterFill(src, value, save_to=path)

    "now the resulted raster is saved to disk"
    dst = gdal.Open(path)
    Map.plot(dst, title="Flow Accumulation")

.. image:: /images/raster_fill.png
   :width: 500pt

resampleRaster
-------------

- `resampleRaster` reproject a raster to any projection (default the WGS84 web mercator projection, without
resampling) The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

Parameters
==========
    src : [gdal.Dataset]
         gdal raster (src=gdal.Open("dem.tif"))
    cell_size : [integer]
         new cell size to resample the raster.
        (default empty so raster will not be resampled)
    resample_technique : [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear

Returns
=======
    raster : [gdal.Dataset]
         gdal object (you can read it by ReadAsArray)


.. code:: py

    print("Original Cell Size =" + str(geo[1]))
    cell_size = 100
    dst = Raster.resampleRaster(src, cell_size, resample_technique="bilinear")

    dst_arr, _ = Raster.getRasterData(dst)
    _, newgeo = Raster.getProjectionData(dst)
    print("New cell size is " + str(newgeo[1]))
    Map.plot(dst, title="Flow Accumulation")

    Original Cell Size =4000.0
    New cell size is 100.0


.. image:: /images/resample.png
   :width: 500pt

projectRaster
-------------

- `projectRaster` reprojects a raster to any projection (default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

Parameters
==========
    src: [gdal object]
        gdal dataset (src=gdal.Open("dem.tif"))
    to_epsg: [integer]
        reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    resample_technique: [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear
    option : [1 or 2]
        option 2 uses the gda.wrap function, option 1 uses the gda.ReprojectImage function

Returns
=======
    raster:
        gdal dataset (you can read it by ReadAsArray)

.. code:: py

    print("current EPSG - " + str(epsg))
    to_epsg = 4326
    dst = Raster.projectRaster(src, to_epsg=to_epsg, option=1)
    newepsg, newgeo = Raster.getProjectionData(dst)
    print("New EPSG - " + str(newepsg))
    print("New Geotransform - " + str(newgeo))

    current EPSG - 32618
    New EPSG - 4326
    New Geotransform - (-75.60441, 0.03606600000000526, 0.0, 4.704305, 0.0, -0.03606600000000526)


- Option 2

.. code:: py

    dst = Raster.projectRaster(src, to_epsg=to_epsg, option=2)
    newepsg, newgeo = Raster.getProjectionData(dst)
    print("New EPSG - " + str(newepsg))
    print("New Geotransform - " + str(newgeo))

    New EPSG - 4326
    New Geotransform - (-75.60441003848668, 0.03611587177268461, 0.0, 4.704560448076901, 0.0, -0.03611587177268461)

cropAlligned
-------------
- If you have an array and you want to clip/crop it using another raster/array.

Crop array using a raster
=========================
- `cropAlligned` clip/crop (matches the location of nodata value from src raster to dst raster), Both rasters have to
have the same dimensions (no of rows & columns) so MatchRasterAlignment should be used prior to this function to
align both rasters.

Parameters
^^^^^^^^^^
    src: [gdal.dataset/np.ndarray]
        raster you want to clip/store NoDataValue in its cells
        exactly the same like mask raster
    mask: [gdal.dataset/np.ndarray]
        mask raster to get the location of the NoDataValue and
        where it is in the array
    mask_noval: [numeric]
        in case the mask is np.ndarray, the mask_noval have to be given.

Returns
^^^^^^^
    dst: [gdal.dataset]
        the second raster with NoDataValue stored in its cells
        exactly the same like src raster


.. code:: py

    aligned_raster = "examples/data/Evaporation_ECMWF_ERA-Interim_mm_daily_2009.01.01.tif"
    dst = gdal.Open(aligned_raster)
    dst_arr, dst_nodataval = Raster.getRasterData(dst)

    Map.plot(
        dst_arr,
        nodataval=dst_nodataval,
        title="Before Cropping-Evapotranspiration",
        color_scale=1,
        ticks_spacing=0.01,
    )

.. image:: /images/before_cropping.png
   :width: 500pt


.. code:: py

    dst_arr_cropped = Raster.cropAlligned(dst_arr, src)
    Map.plot(
        dst_arr_cropped,
        nodataval=nodataval,
        title="Cropped array",
        color_scale=1,
        ticks_spacing=0.01,
    )

.. image:: /images/cropped_array.png
   :width: 500pt

Crop raster using another raster while preserving the alignment
===============================================================
- cropping rasters may  change the alignment of the cells and to keep the alignment during cropping a raster we will
crop the same previous raster but will give the input to the function as a gdal.dataset object.


.. code:: py

    dst_cropped = Raster.cropAlligned(dst, src)
    Map.plot(dst_cropped, title="Cropped raster", color_scale=1, ticks_spacing=0.01)


.. image:: /images/cropped_aligned_raster.png
   :width: 500pt


Crop raster using array
=======================

.. code:: py

    dst_cropped = Raster.cropAlligned(dst, arr, mask_noval=nodataval)
    Map.plot(dst_cropped, title="Cropped array", color_scale=1, ticks_spacing=0.01)

.. image:: /images/crop_raster_using_array.png
   :width: 500pt

crop
----
- `crop` method crops a raster using another raster (both rasters does not have to be aligned).

Parameters
==========
    src: [string/gdal.Dataset]
        the raster you want to crop as a path or a gdal object
    mask : [string/gdal.Dataset]
        the raster you want to use as a mask to crop other raster,
        the mask can be also a path or a gdal object.
    output_path : [string]
        if you want to save the cropped raster directly to disk
        enter the value of the OutputPath as the path.
    save : [boolen]
        True if you want to save the cropped raster directly to disk.

Returns
=======
    dst : [gdal.Dataset]
        the cropped raster will be returned, if the save parameter was True,
        the cropped raster will also be saved to disk in the OutputPath
        directory.


.. code:: py

    RasterA = gdal.Open(aligned_raster)
    epsg, geotransform = Raster.getProjectionData(RasterA)
    print("Raster EPSG = " + str(epsg))
    print("Raster Geotransform = " + str(geotransform))
    Map.plot(RasterA, title="Raster to be cropped", color_scale=1, ticks_spacing=1)

    Raster EPSG = 32618
    Raster Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)

.. image:: /images/raster_tobe_cropped.png
   :width: 500pt


- We will use the soil raster from the previous example as a mask so the projection is different between the raster
and the mask and the cell size is also different

.. code:: py

    dst = Raster.crop(RasterA, soil_raster)
    dst_epsg, dst_geotransform = Raster.getProjectionData(dst)
    print("resulted EPSG = " + str(dst_epsg))
    print("resulted Geotransform = " + str(dst_geotransform))
    Map.plot(dst, title="Cropped Raster", color_scale=1, ticks_spacing=1)

    resulted EPSG = 32618
    resulted Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)

.. image:: /images/cropped_raster.png
   :width: 500pt


matchRasterAlignment
--------------------
- `matchRasterAlignment` method matches the coordinate system and the number of of rows & columns between two rasters
alignment_src is the source of the coordinate system, number of rows, number of columns & cell size data_src is the
source of data values in cells the result will be a raster with the same structure like alignment_src but with values
from data_src using Nearest Neighbour interpolation algorithm

Parameters
==========
    alignment_src : [gdal.dataset/string]
        spatial information source raster to get the spatial information
        (coordinate system, no of rows & columns)
    data_src : [gdal.dataset/string]
        data values source raster to get the data (values of each cell)

Returns
=======
    dst : [gdal.dataset]
        result raster in memory

.. code:: py

    soil_raster = gdal.Open(soilmappath)
    epsg, geotransform = Raster.getProjectionData(soil_raster)
    print("Before alignment EPSG = " + str(epsg))
    print("Before alignment Geotransform = " + str(geotransform))
    # cell_size = geotransform[1]
    Map.plot(soil_raster, title="To be aligned", color_scale=1, ticks_spacing=1)

    Before alignment EPSG = 3116
    Before alignment Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)

.. image:: /images/soil_map.png
   :width: 500pt


.. code:: py

    soil_aligned = Raster.matchRasterAlignment(src, soil_raster)
    New_epsg, New_geotransform = Raster.getProjectionData(soil_aligned)
    print("After alignment EPSG = " + str(New_epsg))
    print("After alignment Geotransform = " + str(New_geotransform))
    Map.plot(soil_aligned, title="After alignment", color_scale=1, ticks_spacing=1)

    After alignment EPSG = 32618
    After alignment Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)

.. image:: /images/soil_map_aligned.png
   :width: 500pt


**************
Raster Dataset
**************

- reprojectDataset
- readASCIIsFolder
- rastersLike
- matchDataAlignment
- folderCalculator
- readRastersFolder
- overlayMaps

cropAlignedFolder
-----------------

- `cropAlignedFolder` matches the location of nodata value from src raster to dst raster, Mask is where the
nodatavalue will be taken and the location of this value src_dir is path to the folder where rasters exist where we
need to put the NoDataValue of the mask in RasterB at the same locations.

Parameters
==========
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
=======
    new rasters have the values from rasters in B_input_path with the NoDataValue in the same
    locations like raster A

.. code:: py

    # The folder should contain tif files only (check example here `cropAlignedFolder`_)
    saveto = "examples/data/crop_aligned_folder/"
    Raster.cropAlignedFolder(aligned_raster_folder, src, saveto)


*****************
helping functions
*****************
- stringSpace


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

.. _`cropAlignedFolder`:
   https://github.com/MAfarrag/pyramids/tree/main/examples/data/crop_aligned_folder
