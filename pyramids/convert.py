"""Convert data from one form to another"""
import os

import netCDF4
import numpy as np
from osgeo import gdal, osr

from pyramids.netcdf import NC
from pyramids.raster import Raster


class Convert:
    """
    Convert data from one form to another
    """
    def __init__(self):
        pass


    @staticmethod
    def asciiToRaster(
            ascii_file: str,
            save_path: str,
            pixel_type: int = 1,
            raster_file=None,
            epsg=None
    ):
        """ASCIItoRaster.

            ASCIItoRaster convert an ASCII file into a raster format and in takes  all
            the spatial reference information (projection, coordinates of the corner point), and
            number of rows and columns from raster file or you have to define the epsg corresponding
            to the you coordinate system and projection

        Parameters
        ----------
        ascii_file: [str]
            name of the ASCII file you want to convert and the name
            should include the extension ".asc"
        save_path: [str]
            path to save the new raster including new raster name and extension (.tif)
        pixel_type: [int]
            type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32

        raster_file: [str]
            source raster to get the spatial information, both ASCII
            file and source raster should have the same number of rows, and
            same number of columns default value is [None].

        epsg:
            EPSG stands for European Petroleum Survey Group and is an organization
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units
            and such alike (https://epsg.io/) default value is [None].

        Returns
        -------
        a New Raster will be saved in the savePath containing the values
        of the ASCII file

        Example
        -------
        1- ASCII to raster given a raster file:
        >>> asc_file = "soiltype.asc"
        >>> raster_file = "DEM.tif"
        >>> save_to = "Soil_raster.tif"
        >>> pixeltype = 1
        >>> Convert.asciiToRaster(asc_file,  save_to, pixeltype, raster_file)
        2- ASCII to Raster given an EPSG number
        >>> asc_file = "soiltype.asc"
        >>> save_to = "Soil_raster.tif"
        >>> pixeltype = 1
        >>> epsg_number = 4647
        >>> Convert.asciiToRaster(asc_file, save_to, pixeltype, epsg = epsg_number)
        """
        if not isinstance(ascii_file, str):
            raise TypeError(f"ascii_file input should be string type - given{type(ascii_file)}")

        if not isinstance(save_path, str):
            raise TypeError(f"save_path input should be string type - given {type(save_path)}")

        if not isinstance(pixel_type, int):
            raise TypeError(f"pixel type input should be integer type please check documentations "
                            f"- given {pixel_type}")

        # input values
        ASCIIExt = ascii_file[-4:]
        if not ASCIIExt == ".asc":
            raise ValueError("please add the extension at the end of the path input")

        message = """ you have to enter one of the following inputs
            - raster_file : if you have a raster with the same spatial information
                (projection, coordinate system), and have the same number of rows,
                and columns
            - epsg : if you have the EPSG number (https://epsg.io/) refering to
                the spatial information of the ASCII file
            """
        assert raster_file is not None or epsg is not None, message

        ### read the ASCII file
        ASCIIValues, ASCIIDetails = Raster.readASCII(ascii_file, pixel_type)
        ASCIIRows = ASCIIDetails[0]
        ASCIIColumns = ASCIIDetails[1]

        # check the optional inputs
        if raster_file is not None:
            assert type(raster_file) == str, "raster_file input should be string type"

            RasterExt = raster_file[-4:]
            assert (
                    RasterExt == ".tif"
            ), "please add the extension at the end of the path input"
            # read the raster file
            src = gdal.Open(raster_file)
            RasterColumns = src.RasterXSize
            RasterRows = src.RasterYSize

            assert (
                    ASCIIRows == RasterRows and ASCIIColumns == RasterColumns
            ), " Data in both ASCII file and Raster file should have the same number of row and columns"

            Raster.rasterLike(src, ASCIIValues, save_path, pixel_type)
        elif epsg is not None:
            assert (
                    type(epsg) == int
            ), "epsg input should be integer type please check documentations"
            # coordinates of the lower left corner
            XLeftSide = ASCIIDetails[2]
            #        YLowSide = ASCIIDetails[3]

            CellSize = ASCIIDetails[4]
            NoValue = ASCIIDetails[5]
            # calculate Geotransform coordinates for the raster
            YUpperSide = ASCIIDetails[3] + ASCIIRows * CellSize

            dst_gt = (XLeftSide, CellSize, 0.0, YUpperSide, 0.0, -1 * CellSize)
            dst_epsg = osr.SpatialReference()
            dst_epsg.ImportFromEPSG(epsg)

            if pixel_type == 1:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_Float32
                )
            elif pixel_type == 2:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_Float64
                )
            elif pixel_type == 3:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_UInt16
                )
            elif pixel_type == 4:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_UInt32
                )
            elif pixel_type == 5:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_Int16
                )
            elif pixel_type == 6:
                dst = gdal.GetDriverByName("GTiff").Create(
                    save_path, ASCIIColumns, ASCIIRows, 1, gdal.GDT_Int32
                )

            dst.SetGeoTransform(dst_gt)
            dst.SetProjection(dst_epsg.ExportToWkt())
            dst.GetRasterBand(1).SetNoDataValue(NoValue)
            dst.GetRasterBand(1).Fill(NoValue)
            dst.GetRasterBand(1).WriteArray(ASCIIValues)
            dst.FlushCache()
            dst = None


    @staticmethod
    def asciiFoldertoRaster(
            path: str,
            save_path: str,
            pixel_type: int = 1,
            Rraster_file=None,
            epsg=None
    ):
        """
        This function takes the path of a folder contains ASCII files and convert
        them into a raster format and in takes  all the spatial information
        (projection, coordinates of the corner point), and number of rows
        and columns from raster file or you have to define the epsg corresponding
        to the you coordinate system and projection

        Parameters
        ----------
        path: [str]
            [String] path to the folder containing the ASCII files
        save_path:
            [String] path to save the new raster including new raster name and extension (.tif)
        pixel_type:
            [Integer] type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32

        Rraster_file:
            [String] source raster to get the spatial information, both ASCII
            file and source raster should have the same number of rows, and
            same number of columns default value is [None].

        epsg:
            EPSG stands for European Petroleum Survey Group and is an organization
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units
            and such alike (https://epsg.io/) default value is [None].

        Returns
        -------
        a New Raster will be saved in the savePath containing the values
        of the ASCII file

        Examples
        --------
        ASCII to raster given a raster file:
        >>> ascii_file = "soiltype.asc"
        >>> RasterFile = "DEM.tif"
        >>> savePath = "Soil_raster.tif"
        >>> pixel_type = 1
        >>> Convert.asciiFoldertoRaster(ascii_file,  savePath, pixel_type, RasterFile)
        ASCII to Raster given an EPSG number
        >>> ascii_file = "soiltype.asc"
        >>> savePath = "Soil_raster.tif"
        >>> pixel_type = 1
        >>> Convert.asciiFoldertoRaster(path, savePath, pixel_type=5, epsg=4647)
        """
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

        for i in range(len(files)):
            ASCIIFile = path + "/" + files[i]
            name = save_path + "/" + files[i].split(".")[0] + ".tif"
            Convert.asciiToRaster(
                ASCIIFile, name, pixel_type, raster_file=None, epsg=epsg
            )


    @staticmethod
    def nctoTiff(input_nc, save_to: str, separator: str = "_"):
        """nctoTiff.

        Parameters
        ----------
        input_nc : [string/list]
            a path of the netcdf file of a list of the netcdf files' names.
        save_to : [str]
            Path to where you want to save the files.
        separator : [string]
            separator in the file name that separate the name from the date.
            Default is "_"

        Returns
        -------
        None.
        """
        if type(input_nc) == str:
            nc = netCDF4.Dataset(input_nc)
        elif type(input_nc) == list:
            nc = netCDF4.MFDataset(input_nc)
        else:
            raise TypeError("first parameter to the nctoTiff function should be either str or list")

        # get the variable
        Var = list(nc.variables.keys())[-1]
        # extract the data
        All_Data = nc[Var]
        # get the details of the file
        (
            geo,
            epsg,
            size_X,
            size_Y,
            size_Z,
            Time,
            NoDataValue,
            datatype,
        ) = NC.ncDetails(nc)

        # Create output folder if needed
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        for i in range(0, size_Z):
            if (
                    All_Data.shape[0] and All_Data.shape[0] > 1
            ):  # type(time) == np.ndarray: #not time == -9999
                time_one = Time[i]
                # d = dt.date.fromordinal(int(time_one))
                name = os.path.splitext(os.path.basename(input_nc))[0]
                nameparts = name.split(separator)[0]  # [0:-2]
                name_out = os.path.join(
                    save_to
                    + "/"
                    + nameparts
                    + "_%d.%02d.%02d.tif"
                    % (time_one.year, time_one.month, time_one.day)
                )
                data = All_Data[i, :, :]
            else:
                name = os.path.splitext(os.path.basename(input_nc))[0]
                name_out = os.path.join(save_to, name + ".tif")
                data = All_Data[0, :, :]

            driver = gdal.GetDriverByName("GTiff")
            # driver = gdal.GetDriverByName("MEM")

            if datatype == np.float32:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_Float32,
                    ["COMPRESS=LZW"],
                )
            elif datatype == np.float64:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_Float64,
                )
            elif datatype == np.uint16:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_UInt16,
                    ["COMPRESS=LZW"],
                )
            elif datatype == np.uint32:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_UInt32,
                    ["COMPRESS=LZW"],
                )
            elif datatype == np.int16:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_Int16,
                    ["COMPRESS=LZW"],
                )
            elif datatype == np.int32:
                dst = driver.Create(
                    name_out,
                    int(data.shape[1]),
                    int(data.shape[0]),
                    1,
                    gdal.GDT_Int32,
                    ["COMPRESS=LZW"],
                )

            srse = osr.SpatialReference()
            if epsg == "":
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

            # set the geotransform
            dst.SetGeoTransform(geo)
            # set the projection
            dst.SetProjection(srse.ExportToWkt())
            # setting the NoDataValue does not accept double precision numbers
            try:
                dst.GetRasterBand(1).SetNoDataValue(NoDataValue)
                # initialize the band with the nodata value instead of 0
                dst.GetRasterBand(1).Fill(NoDataValue)
            except:
                NoDataValue = -9999
                dst.GetRasterBand(1).SetNoDataValue(NoDataValue)
                dst.GetRasterBand(1).Fill(NoDataValue)
                # assert False, "please change the NoDataValue in the source raster as it is not accepted by Gdal"
                print(
                    "the NoDataValue in the source Netcdf is double precission and as it is not accepted by Gdal"
                )
                print("the NoDataValue now is et to -9999 in the raster")

            dst.GetRasterBand(1).WriteArray(data)
            dst.FlushCache()
            dst = None
