import os
import sys

# os.chdir("examples/")
# debug
# rootpath = os.path.abspath(os.getcwd())
# datapath = os.path.join(rootpath, "examples/GIS/data")
# sys.path.append(rootpath + "/src")
# datapath = os.path.join(rootpath, "examples/data/GIS/Hapi_GIS_Data")
# run
rootpath = r"C:\MyComputer\01Algorithms\Hydrology\Hapi"
datapath = os.path.join(rootpath, "examples/GIS/data")
os.chdir(rootpath)

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from osgeo import gdal

matplotlib.use("TkAgg")
from pyramids.catchment import Catchment as GC
from pyramids.raster import Raster

# from Hapi.plot.map import Map
# from Hapi.plot.visualizer import Visualize as vis

#%% Paths
RasterAPath = datapath + "/acc4000.tif"
RasterBPath = datapath + "/dem_100_f.tif"
pointsPath = datapath + "/points.csv"
aligned_raster_folder = datapath + "/aligned_rasters/"
aligned_raster = datapath + "/Evaporation_ECMWF_ERA-Interim_mm_daily_2009.01.01.tif"
soilmappath = datapath + "/soil_raster.tif"
Basinshp = datapath + "/basin.geojson"
#%%
"""
you need to define the TEMP path in your environment variable as some of the metods in the raster
module do some preprocessing in the TEMP path

also if you have installed qgis define the directory to the bin folder inside the installation directory
of qgis in the environment variable with a name "qgis"
"""
# %% read the raster
src = gdal.Open(RasterAPath)
# fig, ax = Map.PlotArray(src, Title="Flow Accumulation")
# %% GetRasterData
"""
get the basic data inside a raster (the array and the nodatavalue)

Inputs:
----------
    Input: [gdal.Dataset]
        a gdal.Dataset is a raster already been read using gdal
    band : [integer]
        the band you want to get its data. Default is 1
Outputs:
----------
    1- mask:[array]
        array with all the values in the flow path length raster
    2- no_val: [numeric]
        value stored in novalue cells
"""
arr, nodataval = Raster.getRasterData(src)
# %%
"""GetProjectionData.

GetProjectionData returns the projection details of a given gdal.Dataset

Inputs:
-------
    1- src : [gdal.Dataset]
        raster read by gdal

Returns:
    1- epsg : [integer]
         integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    2- geo : [tuple]
        geotransform data (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation,
                            pixelsize). The default is ''.
"""
epsg, geo = Raster.getProjectionData(src)
print("EPSG = " + str(epsg))
print(geo)
# %% GetCoords
"""GetCoords.

Returns the coordinates of the cell centres (only the cells that
does not have nodata value)

Parameters
----------

dem : [gdal_Dataset]
    Get the data from the gdal datasetof the DEM

Returns
-------
coords : array
    Array with a list of the coordinates to be interpolated, without the Nan
mat_range : array
    Array with all the centres of cells in the domain of the DEM

"""
coords, centerscoords = Raster.getCellCoords(src)
# %% SaveRaster
"""SaveRaster.

SaveRaster saves a raster to a path

inputs:
----------
    1- raster:
        [gdal object]
    2- path:
        [string] a path includng the name of the raster and extention like
        path="data/cropped.tif"

Outputs:
----------
    the function does not return and data but only save the raster to the hard drive

EX:
----------
    SaveRaster(raster,output_path)
"""
path = datapath + "/save_raster_test.tif"
Raster.saveRaster(src, path)
# %%` CreateRaster
"""
We will recreate the raster that we have already read using the 'GetRasterData' method at the
top from the array and the projection data we obtained using the 'GetProjectionData' method
"""

"""CreateRaster.

CreateRaster method creates a raster from a given array and geotransform data
and save the tif file if a Path is given or it will return the gdal.Dataset

Parameters
----------
Path : [str], optional
    Path to save the Raster, if '' is given a memory raster will be returned. The default is ''.
arr : [array], optional
    numpy array. The default is ''.
geo : [list], optional
    geotransform list [minimum lon, pixelsize, rotation, maximum lat, rotation,
        pixelsize]. The default is ''.
NoDataValue : TYPE, optional
    DESCRIPTION. The default is -9999.
EPSG: [integer]
    integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )

Returns
-------
1- dst : [gdal.Dataset/save raster to drive].
            if a path is given the created raster will be saved to drive, if not
            a gdal.Dataset will be returned.
"""

src = Raster.createRaster(arr=arr, geo=geo, EPSG=str(epsg), NoDataValue=nodataval)
# Map.PlotArray(src, Title="Flow Accumulation")
# %%` RasterLike
"""RasterLike.

RasterLike method creates a Geotiff raster like another input raster, new raster
will have the same projection, coordinates or the top left corner of the original
raster, cell size, nodata velue, and number of rows and columns
the raster and the dem should have the same number of columns and rows

inputs:
-------
    1- src : [gdal.dataset]
        source raster to get the spatial information
    2- array:
        [numpy array]to store in the new raster
    3- path : [String]
        path to save the new raster including new raster name and extension (.tif)
    4- pixel_type : [integer]
        type of the data to be stored in the pixels,default is 1 (float32)
        for example pixel type of flow direction raster is unsigned integer
        1 for float32
        2 for float64
        3 for Unsigned integer 16
        4 for Unsigned integer 32
        5 for integer 16
        6 for integer 32

outputs:
--------
    1- save the new raster to the given path

Ex:
----------
    data=np.load("RAIN_5k.npy")
    src=gdal.Open("DEM.tif")
    name="rain.tif"
    RasterLike(src,data,name)
"""

"""
If we have made some calculation on raster array and we want to save the array back in the raster
"""
arr2 = np.ones(shape=arr.shape, dtype=np.float64) * nodataval
arr2[~np.isclose(arr, nodataval, rtol=0.001)] = 5

path = datapath + "/rasterlike.tif"
Raster.rasterLike(src, arr2, path)
dst = gdal.Open(path)
# Map.PlotArray(dst, Title="Flow Accumulation", ColorScale=1)
# %%
"""MapAlgebra.

MapAlgebra executes a mathematical operation on raster array and returns
the result

inputs:
----------
    1-src : [gdal.dataset]
        source raster to that you want to make some calculation on its values
    3-function:
        defined function that takes one input which is the cell value

Example :
----------
    A=gdal.Open(evap.tif)
    func=np.abs
    new_raster=MapAlgebra(A,func)
"""


def func1(val):
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


dst = Raster.mapAlgebra(src, func1)
# Map.PlotArray(dst, Title="Classes", ColorScale=4, TicksSpacing=1)
# %%
"""RasterFill.

RasterFill takes a raster and fill it with one value


inputs:
----------
    1- src : [gdal.dataset]
        source raster
    2- Val: [numeric]
        numeric value
    3- SaveTo : [str]
        path including the extension (.tif)

Returns:
--------
    1- raster : [saved on disk]
        the raster will be saved directly to the path you provided.
"""
path = datapath + "/fillrasterexample.tif"
value = 20
Raster.rasterFill(src, value, SaveTo=path)

"now the resulted raster is saved to disk"
dst = gdal.Open(path)
# Map.PlotArray(dst, Title="Flow Accumulation")
# %%
"""ResampleRaster.

this function reproject a raster to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster : [gdal.Dataset]
         gdal raster (src=gdal.Open("dem.tif"))
    3-cell_size : [integer]
         new cell size to resample the raster.
        (default empty so raster will not be resampled)
    4- resample_technique : [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear

Outputs:
----------
    1-raster : [gdal.Dataset]
         gdal object (you can read it by ReadAsArray)
"""
print("Original Cell Size =" + str(geo[1]))
cell_size = 100
dst = Raster.resampleRaster(src, cell_size, resample_technique="bilinear")

dst_arr, _ = Raster.getRasterData(dst)
_, newgeo = Raster.getProjectionData(dst)
print("New cell size is " + str(newgeo[1]))
# Map.PlotArray(dst, Title="Flow Accumulation")
# %%
"""ProjectRaster.

ProjectRaster reprojects a raster to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster: [gdal object]
        gdal dataset (src=gdal.Open("dem.tif"))
    2-to_epsg: [integer]
        reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    3- resample_technique: [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear
    4- Option : [1 or 2]


Outputs:
----------
    1-raster:
        gdal dataset (you can read it by ReadAsArray)

Example :
----------
    projected_raster=project_dataset(src, to_epsg=3857)
"""
print("current EPSG - " + str(epsg))
to_epsg = 4326
dst = Raster.projectRaster(src, to_epsg=to_epsg, Option=1)
newepsg, newgeo = Raster.getProjectionData(dst)
print("New EPSG - " + str(newepsg))
print("New Geotransform - " + str(newgeo))
"""Option 2"""
print("Option 2")
dst = Raster.projectRaster(src, to_epsg=to_epsg, Option=2)
newepsg, newgeo = Raster.getProjectionData(dst)
print("New EPSG - " + str(newepsg))
print("New Geotransform - " + str(newgeo))
# %%
"""ReprojectDataset.

ReprojectDataset reprojects and resamples a folder of rasters to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster:
        gdal dataset (src=gdal.Open("dem.tif"))
    2-to_epsg:
        integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    3-cell_size:
        integer number to resample the raster cell size to a new cell size
        (default empty so raster will not be resampled)
    4- resample_technique:
        [String] resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear

Outputs:
----------
    1-raster:
        gdal dataset (you can read it by ReadAsArray)
"""
# to_epsg = 4326
# cell_size = 0.05
# dst = Raster.ReprojectDataset(src, to_epsg=to_epsg, cell_size=cell_size, resample_technique="Nearest")
# arr , noval = Raster.GetRasterData(dst)
# newepsg, newgeo = Raster.GetProjectionData(dst)
# print("New EPSG - " + str(newepsg))
# print("New Geotransform - " + str(newgeo))
# Map.PlotArray(dst, Title="Flow Accumulation")
# %% CropAlligned
"""if you have an array and you want clip/crop it using another raster/array"""

"""CropAlligned.

CropAlligned clip/crop (matches the location of nodata value from src raster to dst
raster), Both rasters have to have the same dimensions (no of rows & columns)
so MatchRasterAlignment should be used prior to this function to align both
rasters

inputs:
-------
    1-src : [gdal.dataset/np.ndarray]
        raster you want to clip/store NoDataValue in its cells
        exactly the same like mask raster
    2-mask : [gdal.dataset/np.ndarray]
        mask raster to get the location of the NoDataValue and
        where it is in the array
    3-mask_noval : [numeric]
        in case the mask is np.ndarray, the mask_noval have to be given.
Outputs:
--------
    1- dst:
        [gdal.dataset] the second raster with NoDataValue stored in its cells
        exactly the same like src raster
"""
# crop array using a raster
dst = gdal.Open(aligned_raster)
dst_arr, dst_nodataval = Raster.getRasterData(dst)

# Map.PlotArray(
#     dst_arr,
#     nodataval=dst_nodataval,
#     Title="Before Cropping-Evapotranspiration",
#     ColorScale=1,
#     TicksSpacing=0.01,
# )
dst_arr_cropped = Raster.cropAlligned(dst_arr, src)
# Map.PlotArray(
#     dst_arr_cropped,
#     nodataval=nodataval,
#     Title="Cropped array",
#     ColorScale=1,
#     TicksSpacing=0.01,
# )
# %% clip raster using another raster while preserving the alignment
"""
cropping rasters may  change the alignment of the cells and to keep the alignment during cropping a raster
we will crop the same previous raster but will give the input to the function as a gdal.dataset object
"""
dst_cropped = Raster.cropAlligned(dst, src)
# Map.PlotArray(dst_cropped, Title="Cropped raster", ColorScale=1, TicksSpacing=0.01)
# %% crop raster using array
"""
we can also crop a raster using an array in condition that we enter the value of the nodata stored in the
array
we can repeat the previous example but
"""
dst_cropped = Raster.cropAlligned(dst, arr, mask_noval=nodataval)
# Map.PlotArray(dst_cropped, Title="Cropped array", ColorScale=1, TicksSpacing=0.01)
# %% clip a folder of rasters using another raster while preserving the alignment
"""
you can perform the previous step on multiple rasters using the CropAlignedFolder
"""
"""CropAlignedFolder.

CropAlignedFolder matches the location of nodata value from src raster to dst
raster
Raster A is where the NoDatavalue will be taken and the location of this value
B_input_path is path to the folder where Raster B exist where  we need to put
the NoDataValue of RasterA in RasterB at the same locations

Inputs:
----------
    1- Mask_path:
        [String] path to the source raster/mask to get the NoData value and it location in the array
        A_path should include the name of the raster and the extension like "data/dem.tif"
    2- src_dir:
        [String] path of the folder of the rasters you want to set Nodata Value
        on the same location of NodataValue of Raster A, the folder should
        not have any other files except the rasters
    3- new_B_path:
        [String] [String] path where new rasters are going to be saved with exact
        same old names

Outputs:
----------
    1- new rasters have the values from rasters in B_input_path with the NoDataValue in the same
    locations like raster A

Example:
----------
    dem_path="01GIS/inputs/4000/acc4000.tif"
    temp_in_path="03Weather_Data/new/4km/temp/"
    temp_out_path="03Weather_Data/new/4km_f/temp/"
    MatchDataNoValuecells(dem_path,temp_in_path,temp_out_path)

"""
saveto = datapath + "/crop_aligned_folder/"
Raster.cropAlignedFolder(aligned_raster_folder, src, saveto)
# %%
"""MatchRasterAlignment.

MatchRasterAlignment method matches the coordinate system and the number of of rows & columns
between two rasters
alignment_src is the source of the coordinate system, number of rows, number of columns & cell size
RasterB is the source of data values in cells
the result will be a raster with the same structure like alignment_src but with
values from RasterB using Nearest Neighbour interpolation algorithm

Inputs:
----------
    1- RasterA : [gdal.dataset/string]
        spatial information source raster to get the spatial information
        (coordinate system, no of rows & columns)
    2- RasterB : [gdal.dataset/string]
        data values source raster to get the data (values of each cell)

Outputs:
----------
    1- dst : [gdal.dataset]
        result raster in memory

Example:
----------
    A=gdal.Open("dem4km.tif")
    B=gdal.Open("P_CHIRPS.v2.0_mm-day-1_daily_2009.01.06.tif")
    matched_raster = MatchRasterAlignment(A,B)
"""
# we want to align the soil raster similar to the alignment in the src raster
soil_raster = gdal.Open(soilmappath)
epsg, geotransform = Raster.getProjectionData(soil_raster)
print("Before alignment EPSG = " + str(epsg))
print("Before alignment Geotransform = " + str(geotransform))
# cell_size = geotransform[1]
# Map.PlotArray(soil_raster, Title="To be aligned", ColorScale=1, TicksSpacing=1)

soil_aligned = Raster.matchRasterAlignment(src, soil_raster)
New_epsg, New_geotransform = Raster.getProjectionData(soil_aligned)
print("After alignment EPSG = " + str(New_epsg))
print("After alignment Geotransform = " + str(New_geotransform))
# Map.PlotArray(soil_aligned, Title="After alignment", ColorScale=1, TicksSpacing=1)
# %%
"""Crop.

crop method crops a raster sing another raster.

Parameters:
-----------
    1-src: [string/gdal.Dataset]
        the raster you want to crop as a path or a gdal object
    2- Mask : [string/gdal.Dataset]
        the raster you want to use as a mask to crop other raster,
        the mask can be also a path or a gdal object.
    3- OutputPath : [string]
        if you want to save the cropped raster directly to disk
        enter the value of the OutputPath as the path.
    3- Save : [boolen]
        True if you want to save the cropped raster directly to disk.
Output:
-------
    1- dst : [gdal.Dataset]
        the cropped raster will be returned, if the Save parameter was True,
        the cropped raster will also be saved to disk in the OutputPath
        directory.
"""
RasterA = gdal.Open(aligned_raster)
epsg, geotransform = Raster.getProjectionData(RasterA)
print("Raster EPSG = " + str(epsg))
print("Raster Geotransform = " + str(geotransform))
# Map.PlotArray(RasterA, Title="Raster to be cropped", ColorScale=1, TicksSpacing=1)
"""
We will use the soil raster from the previous example as a mask
so the projection is different between the raster and the mask and the cell size is also different
"""

dst = Raster.crop(RasterA, soil_raster)
dst_epsg, dst_geotransform = Raster.getProjectionData(dst)
print("resulted EPSG = " + str(dst_epsg))
print("resulted Geotransform = " + str(dst_geotransform))
# Map.PlotArray(dst, Title="Cropped Raster", ColorScale=1, TicksSpacing=1)
# %%
src_aligned = gdal.Open(aligned_raster)
# # arr, nodataval = Raster.GetRasterData(src_aligned)
# Map.PlotArray(src_aligned, Title="Before Cropping-Evapotranspiration", ColorScale=1,
#               TicksSpacing=0.01)
# %%
"""ClipRasterWithPolygon.

ClipRasterWithPolygon method clip a raster using polygon shapefile

inputs:
----------
    1- Raster_path : [String]
        path to the input raster including the raster extension (.tif)
    2- shapefile_path : [String]
        path to the input shapefile including the shapefile extension (.shp)
    3-save : [Boolen]
        True or False to decide whether to save the clipped raster or not
        default is False
    3- output_path : [String]
        path to the place in your drive you want to save the clipped raster
        including the raster name & extension (.tif), default is None

Outputs:
----------
    1- projected_raster:
        [gdal object] clipped raster
    2- if save is True function is going to save the clipped raster to the output_path

EX:
----------
    Raster_path = r"data/Evaporation_ERA-Interim_2009.01.01.tif"
    shapefile_path ="data/"+"Outline.shp"
    clipped_raster = plf.ClipRasterWithPolygon(Raster_path,shapefile_path)
    or
    output_path = r"data/cropped.tif"
    clipped_raster=ClipRasterWithPolygon(Raster_path,shapefile_path,True,output_path)
"""
shp = gpd.read_file(Basinshp)
src = gdal.Open(aligned_raster)

dst = Raster.clipRasterWithPolygon(aligned_raster, Basinshp, save=False, output_path=None)
dst = Raster.clip2(aligned_raster, Basinshp, save=False, output_path=None)
# Map.PlotArray(dst, Title="After Cropping-Evapotranspiration by a shapefile", ColorScale=1,
#               TicksSpacing=0.01)
# %% ReadASCII.
"""ReadASCII.

ReadASCII reads an ASCII file

Inputs:
    1-ASCIIFileName:
        [String] name of the ASCII file you want to convert and the name
        should include the extension ".asc"

    2-pixel_type:
        [Integer] type of the data to be stored in the pixels,default is 1 (float32)
        for example pixel type of flow direction raster is unsigned integer
        1 for float32
        2 for float64
        3 for Unsigned integer 16
        4 for Unsigned integer 32
        5 for integer 16
        6 for integer 32
Outputs:
    1-ASCIIValues:
        [numpy array] 2D arrays containing the values stored in the ASCII
        file

    2-ASCIIDetails:
        [List] list of the six spatial information of the ASCII file
        [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner,
        CellSize, NoValue]
Example:
    Elevation_values,DEMSpatialDetails = ReadASCII("dem.asc",1)
"""
path = datapath + r"/asci_example.asc"
arr, geotransform = Raster.readASCII(path, pixel_type=1)
# Map.PlotArray(arr, geotransform[-1], Title="Cropped Raster", ColorScale=2, TicksSpacing=200)
arr[~np.isclose(arr, geotransform[-1], rtol=0.001)] = 0.03
path2 = datapath + r"/roughness.asc"
Raster.writeASCII(path2, geotransform, arr)
# %% read the points
points = pd.read_csv(pointsPath)
points["row"] = np.nan
points["col"] = np.nan
points.loc[:, ["row", "col"]] = GC.nearestCell(src, points[["x", "y"]][:]).values
# %%

from osgeo import ogr, osr

band = dst.GetRasterBand(1)
src_proj = dst.GetProjection()
src_epsg = osr.SpatialReference(wkt=src_proj)

# create a new vector dataset and layer in it
drv = ogr.GetDriverByName("ESRI Shapefile")
outfile = drv.CreateDataSource(datapath + r"\polygonizedRaster.shp")
outlayer = outfile.CreateLayer(
    "polygonized raster", srs=src_epsg, geom_type=ogr.wkbMultiPolygon
)
# add a new field ‘DN’ to the layer for storing the raster values for each of the polygons
newField = ogr.FieldDefn("DN", ogr.OFTReal)
outlayer.CreateField(newField)
# call Polygonize(...) and provide the band and the output layer as parameters plus a few additional parameters needed
gdal.Polygonize(band, None, outlayer, 0, [])
# 2nd parameter With the None for the second parameter we say that we don’t want to provide a mask for the operation.
# 4th parameter The 0 for the fourth parameter is the index of the field to which the raster values shall be written,
# so the index of the newly added ‘DN’ field in this case.
# The last parameter allows for passing additional options to the function but we do not make use of this,
# so we provide an empty list.
outfile = None
import os

# The second line "outfile = None" is for closing the new shapefile and making sure that all
# data has been written to it
# %%
from osgeo import gdal, gdalnumeric, ogr
# import Image, ImageDraw
# from PIL.Image import core as Image
from PIL import Image, ImageDraw

# Exceptions will get raised on anything >= gdal.CE_Failure
gdal.UseExceptions()


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    # rtnX = geoMatrix[2]
    # rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)


# This function will convert the rasterized clipper shapefile
# to a mask for use within GDAL.
def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tostring(), "b")
    a.shape = i.im.size[1], i.im.size[0]
    return a


def arrayToImage(a):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i = Image.fromstring("L", (a.shape[1], a.shape[0]), (a.astype("b")).tostring())
    return i


#
#  EDIT: this is basically an overloaded
#  version of the gdal_array.OpenArray passing in xoff, yoff explicitly
#  so we can pass these params off to CopyDatasetInfo
#
def OpenArray(array, prototype_ds=None, xoff=0, yoff=0):
    ds = gdal.Open(gdalnumeric.GetArrayFilename(array))

    if ds is not None and prototype_ds is not None:
        if type(prototype_ds).__name__ == "str":
            prototype_ds = gdal.Open(prototype_ds)
        if prototype_ds is not None:
            gdalnumeric.CopyDatasetInfo(prototype_ds, ds, xoff=xoff, yoff=yoff)
    return ds


def histogram(a, bins=range(0, 256)):
    """
    Histogram function for multi-dimensional array.
    a = array
    bins = range of numbers to match
    """
    fa = a.flat
    n = gdalnumeric.searchsorted(gdalnumeric.sort(fa), bins)
    n = gdalnumeric.concatenate([n, [len(fa)]])
    hist = n[1:] - n[:-1]
    return hist


def stretch(a):
    """
    Performs a histogram stretch on a gdalnumeric array image.
    """
    hist = histogram(a)
    im = arrayToImage(a)
    lut = []
    for b in range(0, len(hist), 256):
        # step size
        step = reduce(operator.add, hist[b: b + 256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + hist[i + b]
    im = im.point(lut)
    return imageToArray(im)


shapefile_path = Basinshp
raster_path = aligned_raster

# def main( shapefile_path, raster_path ):
# Load the source data as a gdalnumeric array
srcArray = gdalnumeric.LoadFile(raster_path)
# Also load as a gdal image to get geotransform
# (world file) info
srcImage = gdal.Open(raster_path)
geoTrans = srcImage.GetGeoTransform()

# Create an OGR layer from a boundary shapefile
shapef = ogr.Open(shapefile_path)
lyr = shapef.GetLayer(os.path.split(os.path.splitext(shapefile_path)[0])[1])
poly = lyr.GetNextFeature()

# Convert the layer extent to image pixel coordinates
minX, maxX, minY, maxY = lyr.GetExtent()

ulX, ulY = world2Pixel(geoTrans, minX, maxY)
lrX, lrY = world2Pixel(geoTrans, maxX, minY)

# Calculate the pixel size of the new image
pxWidth = int(lrX - ulX)
pxHeight = int(lrY - ulY)

clip = srcArray[ulY:lrY, ulX:lrX]

#
# EDIT: create pixel offset to pass to new image Projection info
#
xoffset = ulX
yoffset = ulY
print("Xoffset, Yoffset = ( %f, %f )" % (xoffset, yoffset))

# Create a new geomatrix for the image
geoTrans = list(geoTrans)
geoTrans[0] = minX
geoTrans[3] = maxY

# Map points to pixels for drawing the
# boundary on a blank 8-bit,
# black and white, mask image.
points = []
pixels = []
geom = poly.GetGeometryRef()
pts = geom.GetGeometryRef(0)
for p in range(pts.GetPointCount()):
    points.append((pts.GetX(p), pts.GetY(p)))

for p in points:
    pixels.append(world2Pixel(geoTrans, p[0], p[1]))

rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
rasterize = ImageDraw.Draw(rasterPoly)
rasterize.polygon(pixels, 0)
mask = imageToArray(rasterPoly)

# Clip the image using the mask
clip = gdalnumeric.choose(mask, (clip, 0)).astype(gdalnumeric.uint8)

# This image has 3 bands so we stretch each one to make them
# visually brighter
for i in range(3):
    clip[i, :, :] = stretch(clip[i, :, :])

# Save new tiff
#
#  EDIT: instead of SaveArray, let's break all the
#  SaveArray steps out more explicity so
#  we can overwrite the offset of the destination
#  raster
#
### the old way using SaveArray
#
# gdalnumeric.SaveArray(clip, "OUTPUT.tif", format="GTiff", prototype=raster_path)
#
###
#
gtiffDriver = gdal.GetDriverByName("GTiff")
if gtiffDriver is None:
    raise ValueError("Can't find GeoTiff Driver")
gtiffDriver.CreateCopy(
    "OUTPUT.tif", OpenArray(clip, prototype_ds=raster_path, xoff=xoffset, yoff=yoffset)
)

# Save as an 8-bit jpeg for an easy, quick preview
clip = clip.astype(gdalnumeric.uint8)
gdalnumeric.SaveArray(clip, "OUTPUT.jpg", format="JPEG")

gdal.ErrorReset()
# %%
