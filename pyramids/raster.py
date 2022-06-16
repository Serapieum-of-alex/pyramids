"""
GISpy contains python functions to handle raster data align them together
based on a source raster, perform any algebric operation on cell's values

@author: Mostafa
"""
import datetime as dt
import json
import os
import zipfile
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, gdalconst, osr
from osgeo.gdal import Dataset
from rasterio.mask import mask as rio_mask

from pyramids.vector import Vector


class Raster:
    """
    Raster class contains methods to deal with rasters and netcdf files,
    change projection and coordinate systems.

    Methods:
        1-GetMask
        2-AddMask
        3-GetTargets
        4-SaveRaster
        5-GetRasterData
        6-MapAlgebra
        7-RasterFill
        8-ResampleRaster
        9-ProjectRaster
        10-ReprojectDataset
        11-RasterLike
        12-CropAlligned
        13-ChangeNoDataValue
        14-MatchRasterAlignment
        15-NearestNeighbour
        16-ReadASCII
        17-StringSpace
        18-WriteASCII
        19-ASCIItoRaster
        20-ClipRasterWithPolygon
        21-Clip2
        22-ClipRasterWithRaster
        23-Mosaic
        24-ReadASCIIsFolder
        25-ASCIIFoldertoRaster
        26-RastersLike
        27-MatchDataAlignment
        28-MatchDataNoValuecells
        29-FolderCalculator
        30-ReadRastersFolder
        31-ExtractValues
        32-OverlayMap
        33-OverlayMaps
        34-Normalize
        35-GetEpsg
        36-NCdetails
        37-NCtoTiff
        38-Convert_nc_to_tiff
        39-Convert_grb2_to_nc
        40-Convert_adf_to_tiff
        41-Convert_bil_to_tiff
        42-Convert_hdf5_to_tiff
        45-SaveNC
        46-Create_NC_name
        47-Create_new_NC_file
        48-Add_NC_Array_Variable
        49-Add_NC_Array_Static
        50-Convert_dict_to_array
        51-Open_array_info
        53-Open_nc_info
        54-Open_nc_array
        55-Open_bil_array
        56-Open_ncs_array
        57-Open_nc_dict
        58-Clip_Dataset_GDAL
        59-clip_data
        60-reproject_dataset_epsg
        61-reproject_MODIS
        62-reproject_dataset_example
        63-resize_array_example
        64-Get_epsg
        65-gap_filling
        66-Vector_to_Raster
        67-Moving_average
        68-Get_ordinal
        69-ListAttributes
    """
    def __init__(self):
        pass


    @staticmethod
    def getRasterData(
            src: Dataset,
            band: int = 1
    ) -> Tuple[np.ndarray, Union[int, float]]:
        """
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
        no_val : [numeric]
            value stored in novalue cells
        """
        if not isinstance(src, Dataset):
            raise TypeError("please enter a valib gdal object (raster has been read using gdal.Open)")

        # get the value stores in novalue cells
        nodatavalue = np.float32(src.GetRasterBand(band).GetNoDataValue())
        arr = src.GetRasterBand(band).ReadAsArray()

        return arr, nodatavalue


    @staticmethod
    def getProjectionData(src: Dataset) -> Tuple[int, tuple]:
        """GetProjectionData.

        GetProjectionData returns the projection details of a given gdal.Dataset

        Parameters
        ----------
        src: [gdal.Dataset]
            raster read by gdal

        Returns
        -------
        epsg: [integer]
             integer reference number that defines the projection (https://epsg.io/)
        geo: [tuple]
            geotransform data of the upper left corner of the raster
            (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation, pixelsize).
        """
        geo = src.GetGeoTransform()
        # GET PROJECTION
        src_proj = src.GetProjection()
        # spatial ref
        sr_src = osr.SpatialReference(wkt=src_proj)
        epsg = int(sr_src.GetAttrValue("AUTHORITY", 1))

        return epsg, geo


    @staticmethod
    def getEPSG(proj, extension: str="tiff"):
        """GetEPSG.

            This function reads the projection of a GEOGCS file or tiff file

        Parameters
        ----------
        proj : TYPE
            projection read from the netcdf file.
        extension : [string], optional
            tiff or GEOGCS . The default is 'tiff'.

        Returns
        -------
        epsg : [integer]
            epsg number
        """
        try:
            if extension == "tiff":
                # Get info of the dataset that is used for transforming
                g_proj = proj.GetProjection()
                Projection = g_proj.split('EPSG","')
            if extension == "GEOGCS":
                Projection = proj
            epsg = int((str(Projection[-1]).split("]")[0])[0:-1])
        except:
            epsg = 4326

        return epsg


    @staticmethod
    def getCellCoords(src: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """GetCoords.

        Returns the coordinates of the cell centres inside the domain (only the cells that
        does not have nodata value)

        Parameters
        ----------
        src : [gdal_Dataset]
            Get the data from the gdal datasetof the DEM

        Returns
        -------
        coords : array
            Array with a list of the coordinates to be interpolated, without the Nan
        mat_range : array
            Array with all the centres of cells in the domain of the DEM

        """
        # Getting data for the whole grid
        x_init, xx_span, xy_span, y_init, yy_span, yx_span = src.GetGeoTransform()
        shape_dem = src.ReadAsArray().shape

        # Getting data of the mask
        no_val = src.GetRasterBand(1).GetNoDataValue()
        mask = src.ReadAsArray()

        # Adding 0.5*cell size to get the centre
        x = np.array([x_init + xx_span * (i + 0.5) for i in range(shape_dem[0])])
        y = np.array([y_init + yy_span * (i + 0.5) for i in range(shape_dem[1])])
        mat_range = [[(xi, yi) for xi in x] for yi in y]

        # applying the mask
        coords = []

        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                if np.isclose(mask[i, j], no_val, rtol=0.001):
                    coords.append(mat_range[j][i])

        return np.array(coords), np.array(mat_range)


    @staticmethod
    def saveRaster(raster: Dataset, path: str) -> None:
        """SaveRaster.

            SaveRaster saves a raster to a path

        Parameters
        ----------
        raster: [gdal object]
            gdal dataset opbject
        path: [string]
            a path includng the name of the raster and extention like
            path="data/cropped.tif"

        Returns
        -------
        the function does not return and data but only save the raster to the hard drive

        Examples
        --------
        >>> gdal_raster_obj = gdal.Open("path/to/file/***.tif")
        >>> output_path = "examples/GIS/data/save_raster_test.tif"
        >>> Raster.saveRaster(gdal_raster_obj, output_path)
        """
        if not isinstance(raster, gdal.Dataset):
            raise TypeError("raster parameter should be read using gdal dataset please read it using gdal")

        if not isinstance(path, str):
            raise TypeError("Raster_path input should be string type")
        # input values
        ext = path[-4:]
        if not ext == ".tif":
            raise ValueError("please add the extension at the end of the path input")

        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.CreateCopy(path, raster, 0)
        dst_ds = None  # Flush the dataset to disk


    @staticmethod
    def createRaster(
            path: str = "",
            arr: Union[str, Dataset, np.ndarray] = "",
            geo: Union[str, tuple] = "",
            epsg: Union[str, int] = "",
            nodatavalue: Any = -9999
    ) -> Union[Dataset, None]:
        """CreateRaster.

        CreateRaster method creates a raster from a given array and geotransform data
        and save the tif file if a Path is given or it will return the gdal.Dataset

        Parameters
        ----------
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
        -------
        dst : [gdal.Dataset/save raster to drive].
            if a path is given the created raster will be saved to drive, if not
            a gdal.Dataset will be returned.
        """
        if np.isnan(nodatavalue):
            nodatavalue = -9999

        if path == "":
            driver_type = "MEM"
            compress = []
        else:
            if not isinstance(path, str):
                raise TypeError("first parameter Path should be string")

            driver_type = "GTiff"
            compress = ["COMPRESS=LZW"]

        driver = gdal.GetDriverByName(driver_type)
        dst_ds = driver.Create(
            path, int(arr.shape[1]), int(arr.shape[0]), 1, gdal.GDT_Float32, compress
        )
        srse = osr.SpatialReference()

        if epsg == "":
            # if no epsg assume WGS 94 (4326)
            srse.SetWellKnownGeogCS("WGS84")
        else:
            try:
                if not srse.SetWellKnownGeogCS(epsg) == 6:
                    srse.SetWellKnownGeogCS(epsg)
                else:
                    try:
                        srse.ImportFromEPSG(int(epsg))
                    except:
                        srse.ImportFromWkt(epsg)
            except:
                try:
                    srse.ImportFromEPSG(int(epsg))
                except:
                    srse.ImportFromWkt(epsg)

        dst_ds.SetProjection(srse.ExportToWkt())
        try:
            # if the nodata value gives error because of the type.
            dst_ds.GetRasterBand(1).SetNoDataValue(nodatavalue)
        except TypeError:
            # TypeError: in method 'Band_SetNoDataValue', argument 2 of type 'double'
            dst_ds.GetRasterBand(1).SetNoDataValue(np.float64(nodatavalue))

        dst_ds.SetGeoTransform(geo)
        dst_ds.GetRasterBand(1).WriteArray(arr)

        if path == "":
            return dst_ds
        else:
            dst_ds = None
            return


    @staticmethod
    def rasterLike(
            src: Dataset,
            array: np.ndarray,
            path: str,
            pixel_type: int = 1
    ) -> None:
        """RasterLike.

        RasterLike method creates a Geotiff raster like another input raster, new raster
        will have the same projection, coordinates or the top left corner of the original
        raster, cell size, nodata velue, and number of rows and columns
        the raster and the dem should have the same number of columns and rows

        Parameters
        ----------
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
        -------
            save the new raster to the given path

        Example
        -------
        >>> data = np.load("RAIN_5k.npy")
        >>> src_raster = gdal.Open("DEM.tif")
        >>> name = "rain.tif"
        >>> Raster.rasterLike(src_raster, data, name)
        """
        if not isinstance(src, gdal.Dataset):
            raise TypeError("src should be read using gdal (gdal dataset please read it using gdal library) ")

        if not isinstance(array, np.ndarray):
            raise TypeError("array should be of type numpy array")

        if not isinstance(path, str):
            raise TypeError("Raster_path input should be string type")

        if not isinstance(pixel_type, int):
            raise TypeError("pixel type input should be integer type please check documentations")

        # input values
        #    assert os.path.exists(path), path+ " you have provided does not exist"
        ext = path[-4:]
        assert ext == ".tif", "please add the extension at the end of the path input"
        #    assert os.path.exists(path), "source raster you have provided does not exist"

        prj = src.GetProjection()
        cols = src.RasterXSize
        rows = src.RasterYSize
        gt = src.GetGeoTransform()
        noval = src.GetRasterBand(1).GetNoDataValue()
        if pixel_type == 1:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_Float32
            )
        elif pixel_type == 2:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_Float64
            )
        elif pixel_type == 3:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_UInt16
            )
        elif pixel_type == 4:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_UInt32
            )
        elif pixel_type == 5:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_Int16
            )
        elif pixel_type == 6:
            dst = gdal.GetDriverByName("GTiff").Create(
                path, cols, rows, 1, gdal.GDT_Int32
            )

        dst.SetGeoTransform(gt)
        dst.SetProjection(prj)
        # setting the NoDataValue does not accept double precision numbers
        try:
            dst.GetRasterBand(1).SetNoDataValue(noval)
            dst.GetRasterBand(1).Fill(noval)
        except:
            noval = -999999
            dst.GetRasterBand(1).SetNoDataValue(noval)
            dst.GetRasterBand(1).Fill(noval)
            # assert False, "please change the NoDataValue in the source raster as it is not accepted by Gdal"
            print(
                "please change the NoDataValue in the source raster as it is not accepted by Gdal"
            )

        dst.GetRasterBand(1).WriteArray(array)
        dst.FlushCache()
        dst = None


    @staticmethod
    def mapAlgebra(src: Dataset, fun) -> Dataset:
        """MapAlgebra.

        MapAlgebra executes a mathematical operation on raster array and returns
        the result

        Parameters
        ----------
        src : [gdal.dataset]
            source raster to that you want to make some calculation on its values
        fun: [function]
            defined function that takes one input which is the cell value

        Returns
        -------
        Dataset

        Examples
        --------
        >>> src_raster = gdal.Open("evap.tif")
        >>> func = np.abs
        >>> new_raster = Raster.mapAlgebra(src_raster, func)
        """
        # input data validation
        # data type
        assert isinstance(
            src, gdal.Dataset
        ), "src should be read using gdal (gdal dataset please read it using gdal library) "
        assert callable(fun), "second argument should be a function"

        src_gt = src.GetGeoTransform()
        src_proj = src.GetProjection()
        src_row = src.RasterYSize
        src_col = src.RasterXSize
        noval = np.float32(src.GetRasterBand(1).GetNoDataValue())
        src_sref = osr.SpatialReference(wkt=src_proj)
        src_array = src.ReadAsArray()

        # fill the new array with the nodata value
        new_array = np.ones((src_row, src_col)) * noval
        # execute the function on each cell
        for i in range(src_row):
            for j in range(src_col):
                if not np.isclose(src_array[i, j], noval, rtol=0.001):
                    new_array[i, j] = fun(src_array[i, j])

        # create the output raster
        mem_drv = gdal.GetDriverByName("MEM")
        dst = mem_drv.Create(
            "", src_col, src_row, 1, gdalconst.GDT_Float32
        )  # ,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation

        # set the geotransform
        dst.SetGeoTransform(src_gt)
        # set the projection
        dst.SetProjection(src_sref.ExportToWkt())
        # set the no data value
        dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
        # initialize the band with the nodata value instead of 0
        dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
        dst.GetRasterBand(1).WriteArray(new_array)

        return dst


    @staticmethod
    def rasterFill(
            src: Dataset,
            val: Union[float, int],
            save_to: str
    ) -> None:
        """RasterFill.

            RasterFill takes a raster and fill it with one value

        Parameters
        ----------
        src : [gdal.dataset]
            source raster
        val: [numeric]
            numeric value
        save_to : [str]
            path including the extension (.tif)

        Returns
        -------
        raster : [saved on disk]
            the raster will be saved directly to the path you provided.
        """
        assert isinstance(
            src, gdal.Dataset
        ), "src should be read using gdal (gdal dataset please read it using gdal library) "

        NoDataVal = src.GetRasterBand(1).GetNoDataValue()
        src_array = src.ReadAsArray()

        if NoDataVal is None:
            NoDataVal = np.nan

        if not np.isnan(NoDataVal):
            src_array[~np.isclose(src_array, NoDataVal, rtol=0.001)] = val
        else:
            src_array[~np.isnan(src_array)] = val
        # TODO : make this function returns the resulted raster
        #  if the save_to parameter is empty
        Raster.rasterLike(src, src_array, save_to, pixel_type=1)


    @staticmethod
    def resampleRaster(
            src: Dataset,
            cell_size: Union[int, float],
            resample_technique: str="Nearest"
    ) -> Dataset:
        """ResampleRaster.

        this function reproject a raster to any projection
        (default the WGS84 web mercator projection, without resampling)
        The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

        Parameters
        ----------
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
        -------
        raster : [gdal.Dataset]
             gdal object (you can read it by ReadAsArray)
        """
        if not isinstance(src, gdal.Dataset):
            raise TypeError("src should be read using gdal (gdal dataset please read it using gdal library) ")

        if not isinstance(resample_technique, str):
            raise TypeError(" please enter correct resample_technique more information see docmentation ")

        if resample_technique == "Nearest":
            resample_technique = gdal.GRA_NearestNeighbour
        elif resample_technique == "cubic":
            resample_technique = gdal.GRA_Cubic
        elif resample_technique == "bilinear":
            resample_technique = gdal.GRA_Bilinear

        #    # READ THE RASTER
        # GET PROJECTION
        src_proj = src.GetProjection()
        # GET THE GEOTRANSFORM
        src_gt = src.GetGeoTransform()
        # GET NUMBER OF columns
        src_x = src.RasterXSize
        # get number of rows
        src_y = src.RasterYSize
        # spatial ref
        sr_src = osr.SpatialReference(wkt=src_proj)

        ulx = src_gt[0]
        uly = src_gt[3]
        # transform the right lower corner point
        lrx = src_gt[0] + src_gt[1] * src_x
        lry = src_gt[3] + src_gt[5] * src_y

        pixel_spacing = cell_size
        # new geotransform
        new_geo = (
            src_gt[0],
            pixel_spacing,
            src_gt[2],
            src_gt[3],
            src_gt[4],
            -1 * pixel_spacing,
        )
        # create a new raster
        mem_drv = gdal.GetDriverByName("MEM")
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))
        dst = mem_drv.Create("", cols, rows, 1, gdalconst.GDT_Float32)
        # ['COMPRESS=LZW']
        # LZW is a lossless compression method achieve the highst compression but with lot of computation

        # set the geotransform
        dst.SetGeoTransform(new_geo)
        # set the projection
        dst.SetProjection(sr_src.ExportToWkt())
        # set the no data value
        dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
        # initialize the band with the nodata value instead of 0
        dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
        # perform the projection & resampling
        gdal.ReprojectImage(
            src, dst, sr_src.ExportToWkt(), sr_src.ExportToWkt(), resample_technique
        )

        return dst


    @staticmethod
    def projectRaster(
            src: Dataset,
            to_epsg: int,
            resample_technique = "Nearest",
            option: int = 2
    ) -> Dataset:
        """ProjectRaster.

        ProjectRaster reprojects a raster to any projection
        (default the WGS84 web mercator projection, without resampling)
        The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

        Parameters
        ----------
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
        -------
        raster:
            gdal dataset (you can read it by ReadAsArray)

        Examples
        --------
        >>> projected_raster = Raster.projectRaster(src, to_epsg=3857)
        """
        # input data validation
        # data type
        if not isinstance(src, gdal.Dataset):
            raise TypeError("src should be read using gdal (gdal dataset please read it using gdal"
                            f" library) given {type(src)}")
        if not isinstance(to_epsg, int):
            raise TypeError("please enter correct integer number for to_epsg more information "
                            f"https://epsg.io/, given {type(to_epsg)}")
        if not isinstance(resample_technique, str):
            raise TypeError("please enter correct resample_technique more information see "
                            "docmentation ")

        if resample_technique == "Nearest":
            resample_technique = gdal.GRA_NearestNeighbour
        elif resample_technique == "cubic":
            resample_technique = gdal.GRA_Cubic
        elif resample_technique == "bilinear":
            resample_technique = gdal.GRA_Bilinear

        if option == 1:
            # GET PROJECTION
            src_proj = src.GetProjection()
            # GET THE GEOTRANSFORM
            src_gt = src.GetGeoTransform()
            # GET NUMBER OF columns
            src_x = src.RasterXSize
            # get number of rows
            src_y = src.RasterYSize
            # spatial ref
            src_sr = osr.SpatialReference(wkt=src_proj)
            src_epsg = src_sr.GetAttrValue("AUTHORITY", 1)

            ### distination raster
            # spatial ref
            dst_sr = osr.SpatialReference()
            dst_sr.ImportFromEPSG(to_epsg)

            # in case the source crs is GCS and longitude is in the west hemisphere gdal
            # reads longitude fron 0 to 360 and transformation factor wont work with values
            # greater than 180
            if src_epsg != str(to_epsg):
                if src_epsg == "4326" and src_gt[0] > 180:
                    lng_new = src_gt[0] - 360
                    # transformation factors
                    tx = osr.CoordinateTransformation(src_sr, dst_sr)
                    # transform the right upper corner point
                    (ulx, uly, ulz) = tx.TransformPoint(lng_new, src_gt[3])
                    # transform the right lower corner point
                    (lrx, lry, lrz) = tx.TransformPoint(
                        lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                    )
                else:
                    xs = [src_gt[0], src_gt[0] + src_gt[1] * src_x]
                    ys = [src_gt[3], src_gt[3] + src_gt[5] * src_y]

                    from_epsg = int(src_epsg)
                    [uly, lry], [ulx, lrx] = Vector.reprojectPoints(
                        ys, xs, from_epsg=from_epsg, to_epsg=to_epsg
                    )
            else:
                ulx = src_gt[0]
                uly = src_gt[3]
                lrx = src_gt[0] + src_gt[1] * src_x
                lry = src_gt[3] + src_gt[5] * src_y

            # get the cell size in the source raster and convert it to the new crs
            # x coordinates or longitudes
            xs = [src_gt[0], src_gt[0] + src_gt[1]]
            # y coordinates or latitudes
            ys = [src_gt[3], src_gt[3]]

            if src_epsg != str(to_epsg):
                # transform the two points coordinates to the new crs to calculate the new cell size
                from_epsg = int(src_epsg)
                new_ys, new_xs = Vector.reprojectPoints(
                    ys, xs, from_epsg=from_epsg, to_epsg=to_epsg, precision=6
                )  # int(dst_epsg.GetAttrValue('AUTHORITY',1))
                # new_xs, new_ys= Vector.ReprojectPoints_2(ys,xs,from_epsg=int(src_epsg.GetAttrValue('AUTHORITY',1)),
                #                                  to_epsg=int(dst_epsg.GetAttrValue('AUTHORITY',1)))
            else:
                new_xs = xs
                new_ys = ys

            pixel_spacing = np.abs(new_xs[0] - new_xs[1])

            # create a new raster
            mem_drv = gdal.GetDriverByName("MEM")
            cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
            rows = int(np.round(abs(uly - lry) / pixel_spacing))
            dst = mem_drv.Create("", cols, rows, 1, gdalconst.GDT_Float32)
            # ['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation

            # new geotransform
            new_geo = (
                ulx,
                pixel_spacing,
                src_gt[2],
                uly,
                src_gt[4],
                np.sign(src_gt[-1]) * pixel_spacing,
            )
            # set the geotransform
            dst.SetGeoTransform(new_geo)
            # set the projection
            dst.SetProjection(dst_sr.ExportToWkt())
            # set the no data value
            dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
            # initialize the band with the nodata value instead of 0
            dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
            # perform the projection & resampling
            gdal.ReprojectImage(
                src,
                dst,
                src_sr.ExportToWkt(),
                dst_sr.ExportToWkt(),
                resample_technique,
            )

        else:
            dst = gdal.Warp("", src, dstSRS="EPSG:" + str(to_epsg), format="VRT")

        return dst


    # TODO: merge ReprojectDataset and ProjectRaster they are almost the same
    # TODO: still needs to be tested
    @staticmethod
    def reprojectDataset(
            src: Dataset,
            to_epsg: int = 3857,
            cell_size = [],
            resample_technique: str = "Nearest"
    ):
        """ReprojectDataset.

        ReprojectDataset reprojects and resamples a folder of rasters to any projection
        (default the WGS84 web mercator projection, without resampling)

        Parameters
        ----------
        src: [gdal dataset]
            gdal dataset object (src=gdal.Open("dem.tif"))
        to_epsg: [integer]
             reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        cell_size: [integer]
             number to resample the raster cell size to a new cell size
            (default empty so raster will not be resampled)
        resample_technique: [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "Nearest" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear

        Returns
        -------
        raster: [gdal Dataset]
             a GDAL in-memory file object, where you can ReadAsArray etc.
        """
        if not isinstance(src, gdal.Dataset):
            raise TypeError("src should be read using gdal (gdal dataset please read it using gdal"
                            f" library) given {type(src)}")
        if not isinstance(to_epsg, int):
            raise TypeError("please enter correct integer number for to_epsg more information "
                            f"https://epsg.io/, given {type(to_epsg)}")
        if not isinstance(resample_technique, str):
            raise TypeError("please enter correct resample_technique more information see "
                            "docmentation ")

        if cell_size:
            assert isinstance(cell_size, int) or isinstance(
                cell_size, float
            ), "please enter an integer or float cell size"

        if resample_technique == "Nearest":
            resample_technique = gdal.GRA_NearestNeighbour
        elif resample_technique == "cubic":
            resample_technique = gdal.GRA_Cubic
        elif resample_technique == "bilinear":
            resample_technique = gdal.GRA_Bilinear

        #    # READ THE RASTER
        # GET PROJECTION
        src_proj = src.GetProjection()
        # GET THE GEOTRANSFORM
        src_gt = src.GetGeoTransform()
        # GET NUMBER OF columns
        src_x = src.RasterXSize
        # get number of rows
        src_y = src.RasterYSize
        # spatial ref
        src_sr = osr.SpatialReference(wkt=src_proj)
        src_epsg = src_sr.GetAttrValue("AUTHORITY", 1)

        # distination
        # spatial ref
        dst_epsg = osr.SpatialReference()
        dst_epsg.ImportFromEPSG(to_epsg)
        # transformation factors
        tx = osr.CoordinateTransformation(src_sr, dst_epsg)

        # incase the source crs is GCS and longitude is in the west hemisphere gdal
        # reads longitude fron 0 to 360 and transformation factor wont work with valeus
        # greater than 180
        if src_epsg == "4326" and src_gt[0] > 180:
            lng_new = src_gt[0] - 360
            # transform the right upper corner point
            (ulx, uly, ulz) = tx.TransformPoint(lng_new, src_gt[3])
            # transform the right lower corner point
            (lrx, lry, lrz) = tx.TransformPoint(
                lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
            )
        else:
            # transform the right upper corner point
            (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
            # transform the right lower corner point
            (lrx, lry, lrz) = tx.TransformPoint(
                src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
            )

        if not cell_size:
            # the result raster has the same pixcel size as the source
            # check if the coordinate system is GCS convert the distance from angular to metric
            if src_epsg == "4326":
                coords_1 = (src_gt[3], src_gt[0])
                coords_2 = (src_gt[3], src_gt[0] + src_gt[1])
                #            pixel_spacing=geopy.distance.vincenty(coords_1, coords_2).m
                pixel_spacing = Vector.GCSDistance(coords_1, coords_2)
            else:
                pixel_spacing = src_gt[1]
        else:
            # if src_epsg.GetAttrValue('AUTHORITY', 1) != "4326":
            #     assert (cell_size > 1), "please enter cell size greater than 1"
            # if the user input a cell size resample the raster
            pixel_spacing = cell_size

        # create a new raster
        mem_drv = gdal.GetDriverByName("MEM")
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))
        dst = mem_drv.Create("", cols, rows, 1, gdalconst.GDT_Float32)
        # ['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation

        # new geotransform
        new_geo = (ulx, pixel_spacing, src_gt[2], uly, src_gt[4], -pixel_spacing)
        # set the geotransform
        dst.SetGeoTransform(new_geo)
        # set the projection
        dst.SetProjection(dst_epsg.ExportToWkt())
        # set the no data value
        dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
        # initialize the band with the nodata value instead of 0
        dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
        # perform the projection & resampling
        gdal.ReprojectImage(
            src, dst, src_sr.ExportToWkt(), dst_epsg.ExportToWkt(), resample_technique
        )

        return dst



    @staticmethod
    def cropAlligned(
            src: Union[Dataset, np.ndarray],
            mask: Union[Dataset, np.ndarray],
            mask_noval: Union[int, float] = None
    ) -> Union[np.ndarray, Dataset]:
        """CropAlligned.

        CropAlligned clip/crop (matches the location of nodata value from src raster to dst
        raster), Both rasters have to have the same dimensions (no of rows & columns)
        so MatchRasterAlignment should be used prior to this function to align both
        rasters

        Parameters
        ----------
        src: [gdal.dataset/np.ndarray]
            raster you want to clip/store NoDataValue in its cells
            exactly the same like mask raster
        mask: [gdal.dataset/np.ndarray]
            mask raster to get the location of the NoDataValue and
            where it is in the array
        mask_noval: [numeric]
            in case the mask is np.ndarray, the mask_noval have to be given.

        Returns
        -------
        dst: [gdal.dataset]
            the second raster with NoDataValue stored in its cells
            exactly the same like src raster
        """
        # if the mask object is raster
        if isinstance(mask, gdal.Dataset):
            mask_gt = mask.GetGeoTransform()
            mask_proj = mask.GetProjection()
            mask_sref = osr.SpatialReference(wkt=mask_proj)
            mask_epsg = int(mask_sref.GetAttrValue("AUTHORITY", 1))

            row = mask.RasterYSize
            col = mask.RasterXSize
            mask_noval = mask.GetRasterBand(1).GetNoDataValue()
            mask_array = mask.ReadAsArray()
        elif isinstance(mask, np.ndarray):
            msg = " You have to enter the value of the no_val parameter when the mask is a numpy array"
            assert mask_noval is not None, msg
            mask_array = mask.copy()
        else:
            raise TypeError("The second parameter 'mask' has to be either gdal.Dataset or numpy array"
                            f"given - {type(mask)}")

        # if the to be clipped object is raster
        if isinstance(src, gdal.Dataset):
            src_gt = src.GetGeoTransform()
            src_proj = src.GetProjection()
            row = src.RasterYSize
            col = src.RasterXSize
            src_noval = src.GetRasterBand(1).GetNoDataValue()
            src_sref = osr.SpatialReference(wkt=src_proj)
            src_epsg = int(src_sref.GetAttrValue("AUTHORITY", 1))
            src_array = src.ReadAsArray()
        elif isinstance(src, np.ndarray):
            # if the object to be cropped is array
            src_array = src.copy()

        # check proj
        if not mask_array.shape == src_array.shape:
            raise ValueError("two rasters has different no of columns or rows please resample or match both rasters")

        # if both inputs are rasters
        if isinstance(mask, gdal.Dataset) and isinstance(src, gdal.Dataset):
            if not src_gt == mask_gt:
                raise ValueError("location of upper left corner of both rasters are not the same or cell size is "
                                 "different please match both rasters first ")

            if not mask_epsg == src_epsg:
                raise ValueError("Raster A & B are using different coordinate system please reproject one of them to "
                                 "the other raster coordinate system")

        src_array[np.isclose(mask_array, mask_noval, rtol=0.001)] = mask_noval

        # align function only equate the no of rows and columns only
        # match nodatavalue inserts nodatavalue in src raster to all places like mask
        # still places that has nodatavalue in the src raster but it is not nodatavalue in the mask
        # and now has to be filled with values
        # compare no of element that is not nodatavalue in both rasters to make sure they are matched
        # if both inputs are rasters
        if isinstance(mask, gdal.Dataset) and isinstance(src, gdal.Dataset):
            # there might be cells that are out of domain in the src but not out of domain in the mask
            # so change all the src_noval to mask_noval in the src_array
            src_array[np.isclose(src_array, src_noval, rtol=0.001)] = mask_noval
            # then count them (out of domain cells) in the src_array
            elem_src = np.size(src_array[:, :]) - np.count_nonzero(
                (src_array[np.isclose(src_array, mask_noval, rtol=0.001)])
            )
            # count the out of domian cells in the mask
            elem_mask = np.size(mask_array[:, :]) - np.count_nonzero(
                (mask_array[np.isclose(mask_array, mask_noval, rtol=0.001)])
            )

            # if not equal then store indices of those cells that doesn't matchs
            if elem_mask > elem_src:
                rows = [
                    i
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], mask_noval, rtol=0.001)
                       and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
                cols = [
                    j
                    for i in range(row)
                    for j in range(col)
                    if np.isclose(src_array[i, j], mask_noval, rtol=0.001)
                       and not np.isclose(mask_array[i, j], mask_noval, rtol=0.001)
                ]
            # interpolate those missing cells by nearest neighbour
            if elem_mask > elem_src:
                src_array = Raster.nearestNeighbour(src_array, mask_noval, rows, cols)

        # if the dst is a raster
        if isinstance(src, gdal.Dataset):
            mem_drv = gdal.GetDriverByName("MEM")
            dst = mem_drv.Create("", col, row, 1, gdalconst.GDT_Float32)
            # ,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression
            # but with lot of computation
            # if the mask is an array and the mask_gt is not defined use the src_gt as both the mask and the src
            # are aligned so they have the sam gt
            try:
                # set the geotransform
                dst.SetGeoTransform(mask_gt)
                # set the projection
                dst.SetProjection(mask_sref.ExportToWkt())
            except UnboundLocalError:
                dst.SetGeoTransform(src_gt)
                dst.SetProjection(src_sref.ExportToWkt())

            # set the no data value
            try:
                # if the nodata value gives error because of the type.
                dst.GetRasterBand(1).SetNoDataValue(mask_noval)
                # initialize the band with the nodata value instead of 0
                dst.GetRasterBand(1).Fill(mask_noval)
            except TypeError:
                # TypeError: in method 'Band_SetNoDataValue', argument 2 of type 'double'
                dst.GetRasterBand(1).SetNoDataValue(np.float64(mask_noval))
                dst.GetRasterBand(1).Fill(np.float64(mask_noval))

            dst.GetRasterBand(1).WriteArray(src_array)

            return dst
        else:
            return src_array


    @staticmethod
    def cropAlignedFolder(
            src_dir: str,
            mask: Union[Dataset, str],
            saveto: str,
    ) -> None:
        """CropAlignedFolder.

            CropAlignedFolder matches the location of nodata value from src raster to dst
            raster, Mask is where the NoDatavalue will be taken and the location of
            this value src_dir is path to the folder where rasters exist where we
            need to put the NoDataValue of the mask in RasterB at the same locations

        Parameters
        ----------
        src_dir : [String]
            path of the folder of the rasters you want to set Nodata Value
            on the same location of NodataValue of Raster A, the folder should
            not have any other files except the rasters
        mask : [String/gdal.Dataset]
            path/gdal.Dataset of the mask raster to crop the rasters (to get the NoData value
            and it location in the array) Mask should include the name of the raster and the
            extension like "data/dem.tif", or you can read the mask raster using gdal and use
            is the first parameter to the function.
        saveto : [String]
            path where new rasters are going to be saved with exact
            same old names

        Returns
        -------
        new rasters have the values from rasters in B_input_path with the NoDataValue in the same
        locations like raster A

        Examples
        --------
        >>> dem_path = "examples/GIS/data/acc4000.tif"
        >>> src_path = "examples/GIS/data/aligned_rasters/"
        >>> out_path = "examples/GIS/data/crop_aligned_folder/"
        >>> Raster.cropAlignedFolder(dem_path, src_path, out_path)
        """
        # if the mask is a string
        if isinstance(mask, str):
            # check wether the path exists or not
            if not os.path.exists(mask):
                raise FileNotFoundError("source raster you have provided does not exist")

            ext = mask[-4:]
            if not ext == ".tif":
                raise TypeError("Please add the extension '.tif' at the end of the mask input")

            mask = gdal.Open(mask)
        else:
            mask = mask

        # assert isinstance(Mask_path, str), "Mask_path input should be string type"
        if not isinstance(src_dir, str):
            raise TypeError("src_dir input should be string type")

        if not isinstance(saveto, str):
            raise TypeError("saveto input should be string type")

        # check wether the path exists or not
        if not os.path.exists(src_dir):
                raise FileNotFoundError(f"the {src_dir} path you have provided does not exist")

        if not os.path.exists(saveto):
            raise FileNotFoundError(f"the {saveto} path you have provided does not exist")
        # check wether the folder has the rasters or not
        if not len(os.listdir(src_dir)) > 0:
            raise FileNotFoundError(f"{src_dir} folder you have provided is empty")

        files_list = os.listdir(src_dir)
        if "desktop.ini" in files_list:
            files_list.remove("desktop.ini")

        print("New Path- " + saveto)
        for i in range(len(files_list)):
            if files_list[i][-4:] == ".tif":
                print(f"{i + 1}/{len(files_list)} - {saveto}{files_list[i]}")
                B = gdal.Open(src_dir + files_list[i])
                new_B = Raster.cropAlligned(B, mask)
                Raster.saveRaster(new_B, saveto + files_list[i])


    @staticmethod
    def crop(
            src: Union[Dataset, str],
            mask: Union[Dataset, str],
            output_path: str="",
            save: bool=False,
            # Resample: bool=True
    ):
        """Crop.

            crop method crops a raster using another raster (both rasters does not have to be aligned).

        Parameters
        -----------
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
        -------
        dst : [gdal.Dataset]
            the cropped raster will be returned, if the save parameter was True,
            the cropped raster will also be saved to disk in the OutputPath
            directory.
        """
        # get information from the mask raster
        if isinstance(mask, str):
            mask = gdal.Open(mask)
        elif isinstance(mask, gdal.Dataset):
            mask = mask
        else:
            print("Second parameter has to be either path to the mask raster"
                "or a gdal.Dataset object")
            return

        if isinstance(src, str):
            src = gdal.Open(src)
        elif isinstance(src, gdal.Dataset):
            src = src
        else:
            print("first parameter has to be either path to the raster to be cropped "
                "or a gdal.Dataset object")
            return

        # first align the mask with the src raster
        mask_aligned = Raster.matchRasterAlignment(src, mask)
        # crop the src raster with the aligned mask
        dst = Raster.cropAlligned(src, mask_aligned)

        # mask_proj = mask.GetProjection()
        # # GET THE GEOTRANSFORM
        # mask_gt = mask.GetGeoTransform()
        # # GET NUMBER OF columns
        # mask_x = mask.RasterXSize
        # # get number of rows
        # mask_y = mask.RasterYSize
        #
        # mask_epsg = osr.SpatialReference(wkt=mask_proj)
        #
        # mem_drv = gdal.GetDriverByName("MEM")
        # dst = mem_drv.Create("", mask_x, mask_y, 1,
        #                      gdalconst.GDT_Float32)
        # # ,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
        # # set the geotransform
        # dst.SetGeoTransform(mask_gt)
        # # set the projection
        # dst.SetProjection(mask_epsg.ExportToWkt())
        # # set the no data value
        # dst.GetRasterBand(1).SetNoDataValue(mask.GetRasterBand(1).GetNoDataValue())
        # # initialize the band with the nodata value instead of 0
        # dst.GetRasterBand(1).Fill(mask.GetRasterBand(1).GetNoDataValue())
        # # perform the projection & resampling
        # resample_technique = gdal.GRA_NearestNeighbour  # gdal.GRA_NearestNeighbour
        #
        # # reproject the src raster to the mask projection
        # Reprojected_src = gdal.Warp('', src,
        #                               dstSRS='EPSG:' + mask_epsg.GetAttrValue('AUTHORITY', 1), format='VRT')
        # if Resample:
        #     # resample
        #     gdal.ReprojectImage(Reprojected_src, dst, mask_epsg.ExportToWkt(), mask_epsg.ExportToWkt(),
        #                         resample_technique)

        if save:
            Raster.saveRaster(dst, output_path)

        return dst


    @staticmethod
    def clipRasterWithPolygon(
            raster_path: str,
            shapefile_path: str,
            save: bool = False,
            output_path: str = None
    ):
        """ClipRasterWithPolygon.

            ClipRasterWithPolygon method clip a raster using polygon shapefile

        Parameters
        ----------
        raster_path : [String]
            path to the input raster including the raster extension (.tif)
        shapefile_path : [String]
            path to the input shapefile including the shapefile extension (.shp)
        save : [Boolen]
            True or False to decide whether to save the clipped raster or not
            default is False
        output_path : [String]
            path to the place in your drive you want to save the clipped raster
            including the raster name & extension (.tif), default is None

        Returns
        -------
        projected_raster:
            [gdal object] clipped raster
        if save is True function is going to save the clipped raster to the output_path

        Examples
        --------
        >>> src_path = r"data/Evaporation_ERA-Interim_2009.01.01.tif"
        >>> shp_path = "data/"+"Outline.shp"
        >>> clipped_raster = Raster.clipRasterWithPolygon(raster_path,shapefile_path)
        or
        >>> dst_path = r"data/cropped.tif"
        >>> clipped_raster = Raster.clipRasterWithPolygon(src_path, shp_path, True, dst_path)
        """
        if isinstance(raster_path, str):
            src = gdal.Open(raster_path)
        elif isinstance(raster_path, gdal.Dataset):
            src = raster_path
        else:
            raise TypeError("Raster_path input should be string type")

        if isinstance(shapefile_path, str):
            poly = gpd.read_file(shapefile_path)
        elif isinstance(shapefile_path, gpd.geodataframe.GeoDataFrame):
            poly = shapefile_path
        else:
            raise TypeError("shapefile_path input should be string type")

        if not isinstance(save, bool):
            raise TypeError("save input should be bool type (True or False)")

        if save:
            if not isinstance(output_path, str):
                raise ValueError("Pleaase enter a path to save the clipped raster")
        # inputs value
        if save:
            ext = output_path[-4:]
            if not ext == ".tif":
                raise TypeError("please add the extention at the end of the output_path input")

        proj = src.GetProjection()
        src_epsg = osr.SpatialReference(wkt=proj)
        gt = src.GetGeoTransform()

        # first check if the crs is GCS if yes check whether the long is greater than 180
        # geopandas read -ve longitude values if location is west of the prime meridian
        # while rasterio and gdal not
        if src_epsg.GetAttrValue("AUTHORITY", 1) == "4326" and gt[0] > 180:
            # reproject the raster to web mercator crs
            raster = Raster.reprojectDataset(src)
            out_transformed = os.environ["Temp"] + "/transformed.tif"
            # save the raster with the new crs
            Raster.saveRaster(raster, out_transformed)
            raster = rasterio.open(out_transformed)
            # delete the transformed raster
            os.remove(out_transformed)
        else:
            # crs of the raster was not GCS or longitudes less than 180
            if isinstance(raster_path, str):
                raster = rasterio.open(raster_path)
            else:
                raster = rasterio.open(raster_path.GetDescription())

        ### Cropping the raster with the shapefile
        # Re-project into the same coordinate system as the raster data
        shpfile = poly.to_crs(crs=raster.crs.data)

        # Get the geometry coordinates by using the function.
        coords = Vector.getFeatures(shpfile)

        out_img, out_transform = rio_mask(
            dataset=raster, shapes=coords, crop=True
        )

        # copy the metadata from the original data file.
        out_meta = raster.meta.copy()

        # Next we need to parse the EPSG value from the CRS so that we can create
        # a Proj4 string using PyCRS library (to ensure that the projection information is saved correctly).
        epsg_code = int(raster.crs.data["init"][5:])

        # close the transformed raster
        raster.close()

        # Now we need to update the metadata with new dimensions, transform (affine) and CRS (as Proj4 text)
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
            }
        )

        # save the clipped raster.
        temp_path = os.environ["Temp"] + "/cropped.tif"
        with rasterio.open(temp_path, "w", **out_meta) as dest:
            dest.write(out_img)
            dest.close()
            dest = None

        # read the clipped raster
        raster = gdal.Open(temp_path, gdal.GA_ReadOnly)
        # reproject the clipped raster back to its original crs
        projected_raster = Raster.projectRaster(
            raster, int(src_epsg.GetAttrValue("AUTHORITY", 1))
        )
        raster = None
        # delete the clipped raster
        # try:
        # TODO: fix ClipRasterWithPolygon as it does not delete the the cropped.tif raster from the temp_path
        # the following line through an error
        os.remove(temp_path)
        # except:
        #     print(temp_path + " - could not be deleted")

        # write the raster to the file
        if save:
            Raster.saveRaster(projected_raster, output_path)

        return projected_raster


    @staticmethod
    def clip2(
            src: Union[rasterio.io.DatasetReader, str],
            poly: Union[GeoDataFrame, str],
            save: bool = False,
            output_path: str = "masked.tif"
    ):
        """Clip2.

            Clip function takes a rasterio object and clip it with a given geodataframe
            containing a polygon shapely object

        Parameters
        ----------
        src : [rasterio.io.DatasetReader]
            the raster read by rasterio .
        poly : [geodataframe]
            geodataframe containing the polygon you want clip the raster based on.
        save : [Bool], optional
            to save the clipped raster to your drive. The default is False.
        output_path : [String], optional
            path iincluding the extention (.tif). The default is 'masked.tif'.

        Returns
        -------
        out_img : [rasterio object]
            the clipped raster.
        metadata : [dictionay]
                dictionary containing number of bands, coordinate reference system crs
                dtype, geotransform, height and width of the raster
        """
        ### 1- Re-project the polygon into the same coordinate system as the raster data.
        # We can access the crs of the raster using attribute .crs.data:
        if isinstance(poly, str):
            # read the shapefile
            poly = gpd.read_file(poly)
        elif isinstance(poly, gpd.geodataframe.GeoDataFrame):
            poly = poly
        else:
            raise TypeError("Polygongdf input should be string type")

        if isinstance(src, str):
            src = rasterio.open(src)
        elif isinstance(src, rasterio.io.DatasetReader):
            src = src
        else:
            raise TypeError("Rasterobj input should be string type")

        # Project the Polygon into same CRS as the grid
        poly = poly.to_crs(crs=src.crs.data)

        # Print crs
        # geo.crs
        ### 2- Convert the polygon into GeoJSON format for rasterio.

        # Get the geometry coordinates by using the function.
        coords = [json.loads(poly.to_json())["features"][0]["geometry"]]

        # print(coords)

        ### 3-Clip the raster with Polygon
        out_img, out_transform = rasterio.mask.mask(
            dataset=src, shapes=coords, crop=True
        )

        ### 4- update the metadata
        # Copy the old metadata
        out_meta = src.meta.copy()
        # print(out_meta)

        # Next we need to parse the EPSG value from the CRS so that we can create
        # a Proj4 -string using PyCRS library (to ensure that the projection
        # information is saved correctly).

        # Parse EPSG code
        epsg_code = int(src.crs.data["init"][5:])
        # print(epsg_code)

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
            }
        )
        if save:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_img)

        return out_img, out_meta


    @staticmethod
    def changeNoDataValue(
            src: Dataset,
            dst: Dataset
    ):
        """ChangeNoDataValue.

        ChangeNoDataValue changes the cells of nodata value in a dst raster to match
        a src raster.

        Parameters
        ----------
        src: [gdal.dataset]
            source raster to get the location of the NoDataValue and
            where it is in the array
        dst: [gdal.dataset]
            raster you want to store NoDataValue in its cells
            exactly the same like src raster

        Returns
        -------
        dst: [gdal.dataset]
            the second raster with NoDataValue stored in its cells
            exactly the same like src raster
        """
        # input data validation
        # data type
        assert (
                type(src) == gdal.Dataset
        ), "src should be read using gdal (gdal dataset please read it using gdal library) "
        assert (
                type(dst) == gdal.Dataset
        ), "dst should be read using gdal (gdal dataset please read it using gdal library) "

        src_noval = np.float32(src.GetRasterBand(1).GetNoDataValue())

        dst_gt = dst.GetGeoTransform()
        dst_proj = dst.GetProjection()
        dst_row = dst.RasterYSize
        dst_col = dst.RasterXSize
        dst_noval = np.float32(dst.GetRasterBand(1).GetNoDataValue())
        dst_sref = osr.SpatialReference(wkt=dst_proj)
        #    dst_epsg = int(dst_sref.GetAttrValue('AUTHORITY',1))

        dst_array = dst.ReadAsArray()

        for i in range(dst_array.shape[0]):
            for j in range(dst_array.shape[1]):
                if np.isclose(dst_array[i, j], dst_noval, rtol=0.001):
                    dst_array[i, j] = src_noval

        # dst_array[dst_array==dst_noval]=src_noval

        mem_drv = gdal.GetDriverByName("MEM")
        dst = mem_drv.Create(
            "", dst_col, dst_row, 1, gdalconst.GDT_Float32
        )  # ,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation

        # set the geotransform
        dst.SetGeoTransform(dst_gt)
        # set the projection
        dst.SetProjection(dst_sref.ExportToWkt())
        # set the no data value
        dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
        # initialize the band with the nodata value instead of 0
        dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
        dst.GetRasterBand(1).WriteArray(dst_array)

        return dst


    @staticmethod
    def matchRasterAlignment(
            alignment_src: Union[Dataset, str],
            data_src: Union[Dataset, str]
    ) -> Dataset:
        """MatchRasterAlignment.

        MatchRasterAlignment method matches the coordinate system and the number of of rows & columns
        between two rasters
        alignment_src is the source of the coordinate system, number of rows, number of columns & cell size
        data_src is the source of data values in cells
        the result will be a raster with the same structure like alignment_src but with
        values from data_src using Nearest Neighbour interpolation algorithm

        Parameters
        ----------
        alignment_src : [gdal.dataset/string]
            spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns)
        data_src : [gdal.dataset/string]
            data values source raster to get the data (values of each cell)

        Returns
        -------
        dst : [gdal.dataset]
            result raster in memory

        Examples
        --------
        >>> A = gdal.Open("examples/GIS/data/acc4000.tif")
        >>> B = gdal.Open("examples/GIS/data/soil_raster.tif")
        >>> RasterBMatched = Raster.matchRasterAlignment(A,B)
        """
        if isinstance(alignment_src, gdal.Dataset):
            src = alignment_src
        elif isinstance(alignment_src, str):
            src = gdal.Open(alignment_src)
        else:
            raise TypeError("First parameter should be a raster read using gdal (gdal dataset please read it "
                            f"using gdal library) or a path to the raster, given {type(alignment_src)}")

        if isinstance(data_src, gdal.Dataset):
            RasterB = data_src
        elif isinstance(data_src, str):
            RasterB = gdal.Open(data_src)
        else:
            raise TypeError("Second parameter should be a raster read using gdal (gdal dataset please read it "
                            f"using gdal library) or a path to the raster, given {type(data_src)}")

        # we need number of rows and cols from src A and data from src B to store both in dst
        src_proj = src.GetProjection()
        # GET THE GEOTRANSFORM
        src_gt = src.GetGeoTransform()
        # GET NUMBER OF columns
        src_x = src.RasterXSize
        # get number of rows
        src_y = src.RasterYSize

        src_sr = osr.SpatialReference(wkt=src_proj)
        src_epsg = int(src_sr.GetAttrValue("AUTHORITY", 1))

        # reproject the RasterB to match the projection of alignment_src
        reprojected_RasterB = Raster.projectRaster(RasterB, src_epsg)

        # create a new raster
        mem_drv = gdal.GetDriverByName("MEM")
        dst = mem_drv.Create("", src_x, src_y, 1, gdalconst.GDT_Float32)
        # ,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
        # set the geotransform
        dst.SetGeoTransform(src_gt)
        # set the projection
        dst.SetProjection(src_sr.ExportToWkt())
        # set the no data value
        dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
        # initialize the band with the nodata value instead of 0
        dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
        # perform the projection & resampling
        resample_technique = gdal.GRA_NearestNeighbour  # gdal.GRA_NearestNeighbour
        # resample the reprojected_RasterB
        gdal.ReprojectImage(
            reprojected_RasterB,
            dst,
            src_sr.ExportToWkt(),
            src_sr.ExportToWkt(),
            resample_technique,
        )

        return dst


    @staticmethod
    def nearestNeighbour(
            array: np.ndarray,
            nodatavalue: Union[float, int],
            rows: list,
            cols: list
    ):
        """
        nearestNeighbour filles cells of a given indices in rows and cols with
        the value of the nearest neighbour.
        as the raster grid is square so the 4 perpendicular direction are of the same
        close so the function give priority to the right then left then bottom then top
        and the same for 45 degree inclined direction right bottom then left bottom
        then left Top then right Top

        Parameters
        ----------
        array: [numpy.array]
            Array to fill some of its cells with Nearest value.
        nodatavalue: [float32]
            value stored in cells that is out of the domain
        rows: [List]
            list of the row index of the cells you want to fill it with
            nearest neighbour.
        cols: [List]
            list of the column index of the cells you want to fill it with
            nearest neighbour.

        Returns
        -------
        array: [numpy array]
            Cells of given indices will be filled with value of the Nearest neighbour

        Examples
        --------
        >>> raster = gdal.Open("dem.tif")
        >>> req_rows = [3,12]
        >>> req_cols = [9,2]
        >>> new_array = Raster.nearestNeighbour(raster, req_rows, req_cols)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("src should be read using gdal (gdal dataset please read it using gdal library) ")
        assert type(rows) == list, "rows input has to be of type list"
        assert type(cols) == list, "cols input has to be of type list"

        #    array=raster.ReadAsArray()
        #    nodatavalue=np.float32(raster.GetRasterBand(1).GetNoDataValue())

        no_rows = np.shape(array)[0]

        no_cols = np.shape(array)[1]

        for i in range(len(rows)):
            # give the cell the value of the cell that is at the right
            if array[rows[i], cols[i] + 1] != nodatavalue and cols[i] + 1 <= no_cols:
                array[rows[i], cols[i]] = array[rows[i], cols[i] + 1]

            elif array[rows[i], cols[i] - 1] != nodatavalue and cols[i] - 1 > 0:
                # give the cell the value of the cell that is at the left
                array[rows[i], cols[i]] = array[rows[i], cols[i] - 1]

            elif array[rows[i] - 1, cols[i]] != nodatavalue and rows[i] - 1 > 0:
                # give the cell the value of the cell that is at the bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i]]

            elif array[rows[i] + 1, cols[i]] != nodatavalue and rows[i] + 1 <= no_rows:
                # give the cell the value of the cell that is at the Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i]]

            elif (
                    array[rows[i] - 1, cols[i] + 1] != nodatavalue
                    and rows[i] - 1 > 0
                    and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] + 1]

            elif (
                    array[rows[i] - 1, cols[i] - 1] != nodatavalue
                    and rows[i] - 1 > 0
                    and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] - 1]

            elif (
                    array[rows[i] + 1, cols[i] - 1] != nodatavalue
                    and rows[i] + 1 <= no_rows
                    and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] - 1]

            elif (
                    array[rows[i] + 1, cols[i] + 1] != nodatavalue
                    and rows[i] + 1 <= no_rows
                    and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] + 1]
            else:
                print("the cell is isolated (No surrounding cells exist)")
        return array


    @staticmethod
    def readASCII(
            ascii_file: str,
            pixel_type: int = 1
    ) -> Tuple[np.ndarray, tuple]:
        """ReadASCII.

            ReadASCII reads an ASCII file

        Parameters
        ----------
        ascii_file: [str]
            name of the ASCII file you want to convert and the name
            should include the extension ".asc"
        pixel_type: [Integer]
            type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32

        Returns
        -------
        ascii_values: [numpy array]
            2D arrays containing the values stored in the ASCII file
        ascii_details: [List]
            list of the six spatial information of the ASCII file
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner,
            CellSize, NoValue]

        Examples
        --------
        >>> Elevation_values, DEMSpatialDetails = Raster.readASCII("dem.asc",1)
        """
        if not isinstance(ascii_file, str):
            raise TypeError("ascii_file input should be string type")

        if not isinstance(pixel_type, int):
            raise TypeError("pixel type input should be integer type please check documentations")

        # input values
        ASCIIExt = ascii_file[-4:]
        if not ASCIIExt == ".asc":
            raise ValueError("please add the extension at the end of the path input")

        if not os.path.exists(ascii_file):
            raise FileNotFoundError("ASCII file path you have provided does not exist")

        ### read the ASCII file
        File = open(ascii_file)
        Wholefile = File.readlines()
        File.close()

        cols = int(Wholefile[0].split()[1])
        rows = int(Wholefile[1].split()[1])

        XLeftSide = float(Wholefile[2].split()[1])
        YLowerSide = float(Wholefile[3].split()[1])
        CellSize = float(Wholefile[4].split()[1])
        NoValue = float(Wholefile[5].split()[1])

        arr = np.ones((rows, cols), dtype=np.float32)
        try:
            for i in range(rows):
                x = Wholefile[6 + i].split()
                arr[i, :] = list(map(float, x))
        except:
            try:
                for j in range(len(x)):
                    float(x[j])
            except:
                print(f"Error reading the ARCII file please check row {i + 6 + 1}, column {j}")
                print(f"A value of {x[j]} , is stored in the ASCII file ")

        geotransform = (rows, cols, XLeftSide, YLowerSide, CellSize, NoValue)

        return arr, geotransform


    @staticmethod
    def stringSpace(inp):
        return str(inp) + "  "


    @staticmethod
    def writeASCII(
            ascii_file: str,
            geotransform: tuple,
            arr: np.ndarray
    ):
        """WriteASCII.

            WriteASCII reads an ASCII file the spatial information

        Parameters
        ----------
        ascii_file: [str]
            name of the ASCII file you want to convert and the name
            should include the extension ".asc"
        geotransform: [tuple]
            list of the six spatial information of the ASCII file
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner,
            CellSize, NoValue]
        arr: [np.ndarray]
            [numpy array] 2D arrays containing the values stored in the ASCII
            file

        Returns
        -------
        None

        Examples
        --------
        >>> Elevation_values, DEMSpatialDetails = Raster.readASCII("dem.asc",1)
        """
        if not isinstance(ascii_file, str):
            raise TypeError("ascii_file input should be string type")

        # input values
        ASCIIExt = ascii_file[-4:]
        if not ASCIIExt == ".asc":
            raise ValueError("please add the extension at the end of the path input")

        try:
            File = open(ascii_file, "w")
        except FileExistsError:
            raise FileExistsError(f"path you have provided does not exist please check {ascii_file}")

        # write the the ASCII file details
        File.write("ncols         " + str(geotransform[1]) + "\n")
        File.write("nrows         " + str(geotransform[0]) + "\n")
        File.write("xllcorner     " + str(geotransform[2]) + "\n")
        File.write("yllcorner     " + str(geotransform[3]) + "\n")
        File.write("cellsize      " + str(geotransform[4]) + "\n")
        File.write("NODATA_value  " + str(geotransform[5]) + "\n")

        # write the array
        for i in range(np.shape(arr)[0]):
            File.writelines(list(map(Raster.stringSpace, arr[i, :])))
            File.write("\n")

        File.close()


    @staticmethod
    def mosaic(
            raster_list: list,
            save: bool = False,
            path: str = "MosaicedRaster.tif"):
        """mosaic.

        Parameters
        ----------
        raster_list : [list]
            list of the raster files to mosaic.
        save : [Bool], optional
            to save the clipped raster to your drive. The default is False.
        path : [String], optional
            Path iincluding the extention (.tif). The default is 'MosaicedRaster.tif'.

        Returns
        -------
        Mosaiced raster: [Rasterio object]
            the whole mosaiced raster
        metadata : [dictionay]
            dictionary containing number of bands, coordinate reference system crs
            dtype, geotransform, height and width of the raster
        """
        # List for the source files
        RasterioObjects = []

        # Iterate over raster files and add them to source -list in 'read mode'
        for file in raster_list:
            src = rasterio.open(file)
            RasterioObjects.append(src)

        # Merge function returns a single mosaic array and the transformation info
        dst, dst_trans = rasterio.merge.merge(RasterioObjects)

        # Copy the metadata
        dst_meta = src.meta.copy()
        epsg_code = int(src.crs.data["init"][5:])
        # Update the metadata
        dst_meta.update(
            {
                "driver": "GTiff",
                "height": dst.shape[1],
                "width": dst.shape[2],
                "transform": dst_trans,
                "crs": pyproj.CRS.from_epsg(epsg_code).to_wkt(),
            }
        )

        if save:
            # Write the mosaic raster to disk
            with rasterio.open(path, "w", **dst_meta) as dest:
                dest.write(dst)

        return dst, dst_meta


    @staticmethod
    def readASCIIsFolder(path: str, pixel_type: int):
        """readASCIIsFolder.

        this function reads rasters from a folder and creates a 3d arraywith the same
        2d dimensions of the first raster in the folder and len as the number of files
        inside the folder.
        - all rasters should have the same dimensions
        - folder should only contain raster files

        Parameters
        ----------
        path: [String]
            path of the folder that contains all the rasters.
        pixel_type: [int]

        Returns
        -------
        arr_3d: [numpy.ndarray]
            3d array contains arrays read from all rasters in the folder.

        ASCIIDetails: [List]
            list of the six spatial information of the ASCII file
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner,
            CellSize, NoValue]
        files: [list]
            list of names of all files inside the folder

        Examples
        --------
        >>> raster_dir = "ASCII folder/"
        >>> pixel_type = 1
        >>> ASCIIArray, ASCIIDetails, NameList = Raster.readASCIIsFolder(raster_dir, pixel_type)
        """
        # input data validation
        # data type
        assert type(path) == str, "A_path input should be string type"
        # input values
        # check wether the path exist or not
        assert os.path.exists(path), "the path you have provided does not exist"
        # check whether there are files or not inside the folder
        assert os.listdir(path) != "", "the path you have provided is empty"
        # get list of all files
        files = os.listdir(path)
        if "desktop.ini" in files:
            files.remove("desktop.ini")
        # check that folder only contains rasters
        assert all(
            f.endswith(".asc") for f in files
        ), "all files in the given folder should have .tif extension"
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        ASCIIValues, ASCIIDetails = Raster.readASCII(path + "/" + files[0], pixel_type)
        noval = ASCIIDetails[5]
        # fill the array with noval data
        arr_3d = np.ones((ASCIIDetails[0], ASCIIDetails[1], len(files))) * noval

        for i in range(len(files)):
            # read the tif file
            f, _ = Raster.readASCII(path + "/" + files[0], pixel_type)
            arr_3d[:, :, i] = f

        return arr_3d, ASCIIDetails, files



    @staticmethod
    def rastersLike(
            src: Dataset,
            array: np.ndarray,
            path: List[str]=None
    ):
        """
        this function creates a Geotiff raster like another input raster, new raster
        will have the same projection, coordinates or the top left corner of the original
        raster, cell size, nodata velue, and number of rows and columns
        the raster and the dem should have the same number of columns and rows

        Parameters
        ----------
        src: [gdal.dataset]
            source raster to get the spatial information
        array: [numpy array]
            3D array to be stores as a rasters, the dimensions should be
            [rows, columns, timeseries length]
        path: [String]
            list of names to save the new rasters
            like ["results/surfaceDischarge_2012_08_13_23.tif","results/surfaceDischarge_2012_08_14_00.tif"]
            Default value is None

        Returns
        -------
        save the new raster to the given path

        Examples
        --------
        >>> src_raster = gdal.Open("DEM.tif")
        >>> name = ["Q_2012_01_01_01.tif","Q_2012_01_01_02.tif","Q_2012_01_01_03.tif","Q_2012_01_01_04.tif"]
        >>> Raster.rastersLike(src_raster, data, name)
        """
        # input data validation
        # length of the 3rd dimension of the array
        try:
            l = np.shape(array)[2]
        except IndexError:
            assert (
                False
            ), "the array you have entered is 2D you have to use RasterLike function not RastersLike"

        # check length of the list of names to be equal to 3rd dimension of the array
        if path is not None:  # paths are given
            assert len(path) == np.shape(array)[2], (
                    "length of list of names "
                    + str(len(path))
                    + "should equal the 3d dimension of the array-"
                    + str(np.shape(array)[2])
            )
        else:  # paths are not given
            # try to create a folder called results at the current working directory to store resulted rasters
            try:
                os.makedirs(os.path.join(os.getcwd(), "result_rasters"))
            except WindowsError:
                assert (
                    False
                ), "please either to provide your own paths including folder name and rasternames.tif in a list or rename the folder called result_rasters"
            # careate list of names
            path = ["result_rasters/" + str(i) + ".tif" for i in range(l)]

        for i in range(l):
            Raster.rasterLike(src, array[:, :, i], path[i])


    @staticmethod
    def matchDataAlignment(
            src_alignment: str,
            rasters_dir: str,
            save_to: str
    ):
        """matchDataAlignment.

        this function matches the coordinate system and the number of of rows & columns
        between two rasters
        Raster A is the source of the coordinate system, no of rows and no of columns & cell size
        rasters_dir is path to the folder where Raster B exist where  Raster B is
        the source of data values in cells
        the result will be a raster with the same structure like RasterA but with
        values from RasterB using Nearest Neighbour interpolation algorithm

        Parameters
        ----------
        src_alignment: [String]
            path to the spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns) src_alignment should include the name of the raster
            and the extension like "data/dem.tif"
        rasters_dir: [String]
            path of the folder of the rasters (Raster B) you want to adjust their
            no of rows, columns and resolution (alignment) like raster A
            the folder should not have any other files except the rasters
        save_to: [String]
            path where new rasters are going to be saved with exact
            same old names

        Returns
        -------
        new rasters:
            ٌRasters have the values from rasters in rasters_dir with the same
            cell size, no of rows & columns, coordinate system and alignment like raster A

        Examples
        --------
        >>> dem_path = "01GIS/inputs/4000/acc4000.tif"
        >>> prec_in_path = "02Precipitation/CHIRPS/Daily/"
        >>> prec_out_path = "02Precipitation/4km/"
        >>> Raster.matchDataAlignment(dem_path,prec_in_path,prec_out_path)
        """
        # input data validation
        # data type
        assert type(src_alignment) == str, "src_alignment input should be string type"
        assert type(rasters_dir) == str, "rasters_dir input should be string type"
        assert type(save_to) == str, "save_to input should be string type"
        # input values
        ext = src_alignment[-4:]
        assert ext == ".tif", "please add the extension at the end of the path input"

        A = gdal.Open(src_alignment)
        files_list = os.listdir(rasters_dir)
        if "desktop.ini" in files_list:
            files_list.remove("desktop.ini")

        print("New Path- " + save_to)
        for i in range(len(files_list)):
            if files_list[i][-4:] == ".tif":
                print(
                    str(i + 1)
                    + "/"
                    + str(len(files_list))
                    + " - "
                    + save_to
                    + files_list[i]
                )
                B = gdal.Open(rasters_dir + files_list[i])
                new_B = Raster.matchRasterAlignment(A, B)
                Raster.saveRaster(new_B, save_to + files_list[i])


    @staticmethod
    def folderCalculator(
            rasters_dir: str,
            save_to: str,
            function
    ):
        """folderCalculator.

        this function matches the location of nodata value from src raster to dst
        raster
        Raster A is where the NoDatavalue will be taken and the location of this value
        B_input_path is path to the folder where Raster B exist where  we need to put
        the NoDataValue of RasterA in RasterB at the same locations

        Parameters
        ----------
        rasters_dir: [String]
            path of the folder of rasters you want to execute a certain function on all
            of them
        save_to: [String]
            path of the folder where resulted raster will be saved
        function: [function]
            callable function (builtin or user defined)

        Returns
        -------
        new rasters will be saved to the save_to

        Examples
        --------
        >>> def func(args):
        ...    A = args[0]
        ...    funcion = np.abs
        ...    path = args[1]
        ...    B = Raster.mapAlgebra(A, funcion)
        ...    Raster.saveRaster(B, path)

        >>> rasters_dir = "03Weather_Data/new/4km_f/evap/"
        >>> save_to = "03Weather_Data/new/4km_f/new_evap/"
        >>> Raster.folderCalculator(rasters_dir, save_to, func)
        """
        # input data validation
        # data type
        assert type(rasters_dir) == str, "A_path input should be string type"
        assert type(save_to) == str, "B_input_path input should be string type"
        assert callable(function), "second argument should be a function"

        assert os.path.exists(rasters_dir), (
                rasters_dir + "the path you have provided does not exist"
        )
        assert os.path.exists(save_to), (
                save_to + "the path you have provided does not exist"
        )
        # check whether there are files or not inside the folder
        assert os.listdir(rasters_dir) != "", (
                rasters_dir + "the path you have provided is empty"
        )

        # check if you can create the folder
        # try:
        #     os.makedirs(os.path.join(os.environ['TEMP'],"AllignedRasters"))
        # except WindowsError :
        #     # if not able to create the folder delete the folder with the same name and create one empty
        #     shutil.rmtree(os.path.join(os.environ['TEMP']+"/AllignedRasters"))
        #     os.makedirs(os.path.join(os.environ['TEMP'],"AllignedRasters"))

        # get names of rasters
        files_list = os.listdir(rasters_dir)
        if "desktop.ini" in files_list:
            files_list.remove("desktop.ini")

        # execute the function on each raster
        for i in range(len(files_list)):
            print(str(i + 1) + "/" + str(len(files_list)) + " - " + files_list[i])
            B = gdal.Open(rasters_dir + files_list[i])
            args = [B, save_to + files_list[i]]
            function(args)


    @staticmethod
    def readRastersFolder(
            path: str,
            band: int=1,
            with_order: bool=True,
            start: str="",
            end: str="",
            fmt: str="",
            freq: str="daily",
            # separator: str = "."
    ):
        """ReadRastersFolder.

        this function reads rasters from a folder and creates a 3d array with the same
        2d dimensions of the first raster in the folder and len as the number of files

        inside the folder.
        - all rasters should have the same dimensions
        - folder should only contain raster files
        - raster file name should have the date at the end of the file name before the extension directly
          with the YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD
          >>> "50_MSWEP_1979.01.01.tif"

        Parameters
        ----------
        path:[String/list]
            path of the folder that contains all the rasters or
            a list contains the paths of the rasters to read.
        band: [int]
            number of the band you want to read default is 1.
        with_order: [bool]
            True if the rasters follows a certain order, then the rasters names should have a
            number at the beginning indicating the order.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        freq: [str]
            frequency of the rasters "daily", Hourly, monthly

        Returns
        -------
        arr_3d: [numpy.ndarray]
            3d array contains arrays read from all rasters in the folder.

        Example
        -------
        >>> raster_folder = "examples/GIS/data/raster-folder"
        >>> prec = Raster.readRastersFolder(raster_folder)

        >>> import glob
        >>> search_criteria = "*.tif"
        >>> file_list = glob.glob(os.path.join(raster_folder, search_criteria))
        >>> prec = Raster.readRastersFolder(file_list, with_order=False)
        """
        # input data validation
        # data type
        if not isinstance(path, str)  and not isinstance(path, list):
            raise TypeError(f"path input should be string/list type, given{type(path)}")

        # input values
        if isinstance(path, str):
            # check wether the path exist or not
            if not os.path.exists(path):
                raise FileNotFoundError("The path you have provided does not exist")
            # get list of all files
            files = os.listdir(path)
            files = [i for i in files if i.endswith(".tif")]
            # files = glob.glob(os.path.join(path, "*.tif"))
            # check whether there are files or not inside the folder
            if len(files) < 1:
                raise FileNotFoundError("The path you have provided is empty")

            if "desktop.ini" in files:
                files.remove("desktop.ini")
        else:
            files = path[:]

        # to sort the files in the same order as the first number in the name
        if with_order:
            try:
                filesNo = [int(i.split("_")[0]) for i in files]
            except ValueError:
                ErrorMsg = (
                    "please include a number at the beginning of the"
                    "rasters name to indicate the order of the rasters. to do so please"
                    "use the Inputs.RenameFiles method to solve this issue and don't "
                    "include any other files in the folder with the rasters"
                )
                assert False, ErrorMsg

            filetuple = sorted(zip(filesNo, files))
            files = [x for _, x in filetuple]

        if start != "" or end != "":
            start = dt.datetime.strptime(start, fmt)
            end = dt.datetime.strptime(end, fmt)

            # get the dates for each file
            dates = list()
            for i in range(len(files)):
                if freq == "daily":
                    l = len(files[i]) - 4
                    day = int(files[i][l - 2: l])
                    month = int(files[i][l - 5:l - 3])
                    year = int(files[i][l - 10: l - 6])
                    dates.append(dt.datetime(year, month, day))
                elif freq == "hourly":
                    year = int(files[i].split("_")[-4])
                    month = int(files[i].split("_")[-3])
                    day = int(files[i].split("_")[-2])
                    hour = int(files[i].split("_")[-1].split(".")[0])
                    dates.append(dt.datetime(year, month, day, hour))

            starti = dates.index(start)
            endi = dates.index(end) + 1
            assert all(
                f.endswith(".tif") for f in files[starti:endi]
            ), "all files in the given folder should have .tif extension"
        else:
            starti = 0
            endi = len(files)
            # check that folder only contains rasters
            assert all(
                f.endswith(".tif") for f in files
            ), "all files in the given folder should have .tif extension"
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        if type(path) == list:
            sample = gdal.Open(files[starti])
        else:
            sample = gdal.Open(path + "/" + files[starti])
        # check the given band number
        if band > sample.RasterCount:
            raise ValueError(f"the raster has only {sample.RasterCount} check the given band number")

        dim = sample.GetRasterBand(band).ReadAsArray().shape
        naval = sample.GetRasterBand(band).GetNoDataValue()
        # fill the array with noval data
        arr_3d = np.ones((dim[0], dim[1], len(range(starti, endi))))
        arr_3d[:, :, :] = naval

        if type(path) == list:
            for i in range(starti, endi):
                # read the tif file
                f = gdal.Open(files[i])
                arr_3d[:, :, i] = f.GetRasterBand(band).ReadAsArray()
        else:
            for i in enumerate(range(starti, endi)):
                # read the tif file
                f = gdal.Open(path + "/" + files[i[1]])
                arr_3d[:, :, i[0]] = f.GetRasterBand(band).ReadAsArray()

        return arr_3d

    @staticmethod
    def extractValues(
            path: str,
            exclude_value,
            compressed: bool=True,
            occupied_Cells_only: bool=True
    ):
        """extractValues.

        this function is written to extract and return a list of all the values
        in a map
        #TODO (an ASCII for now to be extended later to read also raster)

        Parameters
        ----------
        path: [String]
            a path includng the name of the ASCII and extention like
            path="data/cropped.asc"
        exclude_value: [Numeric]
            values you want to exclude from exteacted values
        compressed: [Bool]
             if the map you provided is compressed
        occupied_Cells_only:

        """
        # input data validation
        # data type
        assert type(path) == str, "path input should be string type" + str(path)
        assert type(compressed) == bool, "compressed input should be Boolen type"
        # input values
        # check wether the path exist or not
        assert os.path.exists(path), "the path you have provided does not exist" + str(
            path
        )
        # check wether the path has the extention or not
        if compressed:
            assert path.endswith(".zip"), "file" + path + " should have .asc extension"
        else:
            assert path.endswith(".asc"), "file" + path + " should have .asc extension"

        ExtractedValues = list()

        try:
            # open the zip file
            if compressed:
                Compressedfile = zipfile.ZipFile(path)
                # get the file name
                fname = Compressedfile.infolist()[0]
                # ASCIIF = Compressedfile.open(fname)
                # SpatialRef = ASCIIF.readlines()[:6]
                ASCIIF = Compressedfile.open(fname)
                ASCIIRaw = ASCIIF.readlines()[6:]
                rows = len(ASCIIRaw)
                cols = len(ASCIIRaw[0].split())
                MapValues = np.ones((rows, cols), dtype=np.float32) * 0
                # read the ascii file
                for i in range(rows):
                    x = ASCIIRaw[i].split()
                    MapValues[i, :] = list(map(float, x))

            else:
                MapValues, SpatialRef = Raster.readASCII(path)

            # count nonzero cells
            NonZeroCells = np.count_nonzero(MapValues)

            if occupied_Cells_only:
                ExtractedValues = 0
                return ExtractedValues, NonZeroCells

            # get the position of cells that is not zeros
            rows = np.where(MapValues[:, :] != exclude_value)[0]
            cols = np.where(MapValues[:, :] != exclude_value)[1]

        except:
            print("Error Opening the compressed file")
            NonZeroCells = -1
            ExtractedValues = -1
            return ExtractedValues, NonZeroCells

        # get the values of the filtered cells
        for i in range(len(rows)):
            ExtractedValues.append(MapValues[rows[i], cols[i]])

        return ExtractedValues, NonZeroCells


    @staticmethod
    def overlayMap(
            path: str,
            classes_map: Union[str, np.ndarray],
            exclude_value: Union[float, int],
            compressed: bool=False,
            occupied_Cells_only: bool=True
    ) -> Tuple[Dict[List[float], List[float]], int]:
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
        if not isinstance(path, str):
            raise TypeError(f"Path input should be string type - given{type(path)}")

        if not isinstance(compressed, bool):
            raise TypeError(f"Compressed input should be Boolen type given {type(compressed)}")

        # check wether the path exist or not
        if not os.path.exists(path):
            raise FileNotFoundError(f"the path {path} you have provided does not exist")

        # read the base map
        if isinstance(classes_map, str):
            if classes_map.endswith(".asc"):
                BaseMapV, _ = Raster.readASCII(classes_map)
            else:
                BaseMap = gdal.Open(classes_map)
                BaseMapV = BaseMap.ReadAsArray()
        else:
            BaseMapV = classes_map

        ExtractedValues = dict()

        try:
            # open the zip file
            if compressed:
                Compressedfile = zipfile.ZipFile(path)
                # get the file name
                fname = Compressedfile.infolist()[0]
                ASCIIF = Compressedfile.open(fname)
                #                SpatialRef = ASCIIF.readlines()[:6]
                ASCIIF = Compressedfile.open(fname)
                ASCIIRaw = ASCIIF.readlines()[6:]
                rows = len(ASCIIRaw)
                cols = len(ASCIIRaw[0].split())
                MapValues = np.ones((rows, cols), dtype=np.float32) * 0
                # read the ascii file
                for row in range(rows):
                    x = ASCIIRaw[row].split()
                    MapValues[row, :] = list(map(float, x))

            else:
                MapValues, SpatialRef = Raster.readASCII(path)
            # count number of nonzero cells
            NonZeroCells = np.count_nonzero(MapValues)

            if occupied_Cells_only:
                ExtractedValues = 0
                return ExtractedValues, NonZeroCells

            # get the position of cells that is not zeros
            rows = np.where(MapValues[:, :] != exclude_value)[0]
            cols = np.where(MapValues[:, :] != exclude_value)[1]

        except:
            print("Error Opening the compressed file")
            NonZeroCells = -1
            ExtractedValues = -1
            return ExtractedValues, NonZeroCells

        # extract values
        for i in range(len(rows)):
            # first check if the sub-basin has a list in the dict if not create a list
            if BaseMapV[rows[i], cols[i]] not in list(ExtractedValues.keys()):
                ExtractedValues[BaseMapV[rows[i], cols[i]]] = list()

            #            if not np.isnan(MapValues[rows[i],cols[i]]):
            ExtractedValues[BaseMapV[rows[i], cols[i]]].append(
                MapValues[rows[i], cols[i]]
            )
        #            else:
        # if the value is nan
        #                NanList.append(FilteredList[i])

        return ExtractedValues, NonZeroCells


    @staticmethod
    def overlayMaps(
            path: str,
            basemap_file: str,
            file_prefix: str,
            exclude_value: Union[float, int],
            compressed: bool=False,
            occupied_cells_only: bool=True,
    ):
        """
        this function is written to extract and return a list of all the values
        in an ASCII file

        Parameters
        ----------
        path: [String]
            a path to the folder includng the maps.
        basemap_file: [String]
            a path includng the name of the ASCII and extention like
            path="data/cropped.asc"
        file_prefix: [String]
            a string that make the files you want to filter in the folder
            uniq
        exclude_value: [Numeric]
            values you want to exclude from exteacted values
        compressed:
            [Bool] if the map you provided is compressed
        occupied_cells_only:
            [Bool] if you want to count only cells that is not zero

        Returns
        -------
        ExtractedValues:
            [Dict] dictonary with a list of values in the basemap as keys
                and for each key a list of all the intersected values in the
                maps from the path
        NonZeroCells:
            [dataframe] dataframe with the first column as the "file" name
            and the second column is the number of cells in each map
        """
        assert type(path) == str, "Path input should be string type"
        assert type(file_prefix) == str, "Path input should be string type"
        assert type(compressed) == bool, "Compressed input should be Boolen type"
        assert type(basemap_file) == str, "basemap_file input should be string type"
        # input values
        # check wether the path exist or not
        assert os.path.exists(path), "the path you have provided does not exist"
        # check whether there are files or not inside the folder
        assert os.listdir(path) != "", "the path you have provided is empty"
        # get list of all files
        Files = os.listdir(path)

        FilteredList = list()

        # filter file list with the File prefix input
        for i in range(len(Files)):
            if Files[i].startswith(file_prefix):
                FilteredList.append(Files[i])

        NonZeroCells = pd.DataFrame()
        NonZeroCells["files"] = FilteredList
        NonZeroCells["cells"] = 0
        # read the base map
        if basemap_file.endswith(".asc"):
            BaseMapV, _ = Raster.readASCII(basemap_file)
        else:
            BaseMap = gdal.Open(basemap_file)
            BaseMapV = BaseMap.ReadAsArray()

        ExtractedValues = dict()
        FilesNotOpened = list()

        for i in range(len(FilteredList)):
            print("File " + FilteredList[i])
            if occupied_cells_only:
                ExtractedValuesi, NonZeroCells.loc[i, "cells"] = Raster.overlayMap(
                    path + "/" + FilteredList[i],
                    BaseMapV,
                    exclude_value,
                    compressed,
                    occupied_cells_only,
                )
            else:
                ExtractedValuesi, NonZeroCells.loc[i, "cells"] = Raster.overlayMap(
                    path + "/" + FilteredList[i],
                    BaseMapV,
                    exclude_value,
                    compressed,
                    occupied_cells_only,
                )

                # these are the destinct values from the BaseMap which are keys in the
                # ExtractedValuesi dict with each one having a list of values
                BaseMapValues = list(ExtractedValuesi.keys())

                for j in range(len(BaseMapValues)):
                    if BaseMapValues[j] not in list(ExtractedValues.keys()):
                        ExtractedValues[BaseMapValues[j]] = list()

                    ExtractedValues[BaseMapValues[j]] = (
                            ExtractedValues[BaseMapValues[j]]
                            + ExtractedValuesi[BaseMapValues[j]]
                    )

            if ExtractedValuesi == -1 or NonZeroCells.loc[i, "cells"] == -1:
                FilesNotOpened.append(FilteredList[i])
                continue

        return ExtractedValues, NonZeroCells


    @staticmethod
    def normalize(array: np.ndarray):
        """
        Normalizes numpy arrays into scale 0.0 - 1.0

        Parameters
        ----------
        array : [array]
            numpy array

        Returns
        -------
        array
            DESCRIPTION.
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return val



    @staticmethod
    def openArrayInfo(fname: str = ""):
        """openArrayInfo.

        Opening a tiff info, for example size of array, projection and transform matrix.

        Parameters
        ----------
        fname: [str]
            path to the tiff file

        Returns
        -------
        geotransform: [Tuple]
            geotransform data of the upper left corner of the raster
            (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation, pixelsize).
        proj: [str]
            projection as a well known text.
        size_x: [float]

        size_y: [float]
        """
        src = gdal.Open(fname)
        if src is None:
            print(f"{fname} does not exists")
        else:
            geo_out = src.GetGeoTransform()
            proj = src.GetProjection()
            size_X = src.RasterXSize
            size_Y = src.RasterYSize
            src = None
        return geo_out, proj, size_X, size_Y


    def listAttributes(self):
        """
        Print Attributes List
        """

        print("\n")
        print(
            f"Attributes List of: { repr(self.__dict__['name'])} - {self.__class__.__name__}  Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")
