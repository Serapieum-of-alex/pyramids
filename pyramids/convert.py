"""Convert data from one form to another"""
import os
import numpy as np
from osgeo import gdal, osr
import netCDF4
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
    def nctoTiff(input_nc, SaveTo, Separator="_"):
        """
        Parameters
        ----------
        input_nc : [string/list]
            a path of the netcdf file of a list of the netcdf files' names.
        SaveTo : TYPE
            Path to where you want to save the files.
        Separator : [string]
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
        ) = Raster.ncDetails(nc)

        # Create output folder if needed
        if not os.path.exists(SaveTo):
            os.mkdir(SaveTo)

        for i in range(0, size_Z):
            if (
                    All_Data.shape[0] and All_Data.shape[0] > 1
            ):  # type(Time) == np.ndarray: #not Time == -9999
                time_one = Time[i]
                # d = dt.date.fromordinal(int(time_one))
                name = os.path.splitext(os.path.basename(input_nc))[0]
                nameparts = name.split(Separator)[0]  # [0:-2]
                name_out = os.path.join(
                    SaveTo
                    + "/"
                    + nameparts
                    + "_%d.%02d.%02d.tif"
                    % (time_one.year, time_one.month, time_one.day)
                )
                data = All_Data[i, :, :]
            else:
                name = os.path.splitext(os.path.basename(input_nc))[0]
                name_out = os.path.join(SaveTo, name + ".tif")
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

    # def Convert_nc_to_tiff(input_nc, output_folder):
    #     """
    #     This function converts the nc file into tiff files
    #
    #     Keyword Arguments:
    #     input_nc -- name, name of the adf file
    #     output_folder -- Name of the output tiff file
    #     """
    #
    #     # All_Data = Raster.Open_nc_array(input_nc)
    #
    #     if type(input_nc) == str:
    #         nc = netCDF4.Dataset(input_nc)
    #     elif type(input_nc) == list:
    #         nc = netCDF4.MFDataset(input_nc)
    #
    #     Var = nc.variables.keys()[-1]
    #     All_Data = nc[Var]
    #
    #     geo_out, epsg, size_X, size_Y, size_Z, Time = Raster.Open_nc_info(input_nc)
    #
    #     if epsg == 4326:
    #         epsg = "WGS84"
    #
    #     # Create output folder if needed
    #     if not os.path.exists(output_folder):
    #         os.mkdir(output_folder)
    #
    #     for i in range(0, size_Z):
    #         if not Time == -9999:
    #             time_one = Time[i]
    #             d = dt.fromordinal(time_one)
    #             name = os.path.splitext(os.path.basename(input_nc))[0]
    #             nameparts = name.split("_")[0:-2]
    #             name_out = os.path.join(
    #                 output_folder,
    #                 "_".join(nameparts)
    #                 + "_%d.%02d.%02d.tif" % (d.year, d.month, d.day),
    #             )
    #             Data_one = All_Data[i, :, :]
    #         else:
    #             name = os.path.splitext(os.path.basename(input_nc))[0]
    #             name_out = os.path.join(output_folder, name + ".tif")
    #             Data_one = All_Data[:, :]
    #
    #         Raster.createRaster(name_out, Data_one, geo_out, epsg)
    #
    #     return ()
    #
    #
    # def Convert_grb2_to_nc(input_wgrib, output_nc, band):
    #
    #     # Get environmental variable
    #     qgis_path = os.environ["qgis"].split(";")
    #     GDAL_env_path = qgis_path[0]
    #     GDAL_TRANSLATE_PATH = os.path.join(GDAL_env_path, "gdal_translate.exe")
    #
    #     # Create command
    #     fullCmd = " ".join(
    #         [
    #             '"%s" -of netcdf -b %d' % (GDAL_TRANSLATE_PATH, band),
    #             input_wgrib,
    #             output_nc,
    #         ]
    #     )  # -r {nearest}
    #
    #     Raster.Run_command_window(fullCmd)
    #
    #     return ()
    #
    #
    # def Convert_adf_to_tiff(input_adf, output_tiff):
    #     """
    #     This function converts the adf files into tiff files
    #
    #     Keyword Arguments:
    #     input_adf -- name, name of the adf file
    #     output_tiff -- Name of the output tiff file
    #     """
    #
    #     # Get environmental variable
    #     qgis_path = os.environ["qgis"].split(";")
    #     GDAL_env_path = qgis_path[0]
    #     GDAL_TRANSLATE_PATH = os.path.join(GDAL_env_path, "gdal_translate.exe")
    #
    #     # convert data from ESRI GRID to GeoTIFF
    #     fullCmd = (
    #                   '"%s" -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ' "ZLEVEL=1 -of GTiff %s %s"
    #               ) % (GDAL_TRANSLATE_PATH, input_adf, output_tiff)
    #
    #     Raster.Run_command_window(fullCmd)
    #
    #     return output_tiff
    #
    #
    # def Convert_bil_to_tiff(input_bil, output_tiff):
    #     """
    #     This function converts the bil files into tiff files
    #
    #     Keyword Arguments:
    #     input_bil -- name, name of the bil file
    #     output_tiff -- Name of the output tiff file
    #     """
    #
    #     gdal.GetDriverByName("EHdr").Register()
    #     dest = gdal.Open(input_bil, gdalconst.GA_ReadOnly)
    #     Array = dest.GetRasterBand(1).ReadAsArray()
    #     geo_out = dest.GetGeoTransform()
    #     Raster.createRaster(output_tiff, Array, geo_out, "WGS84")
    #
    #     return output_tiff
    #
    #
    # def Convert_hdf5_to_tiff(
    #         inputname_hdf, Filename_tiff_end, Band_number, scaling_factor, geo_out
    # ):
    #     """
    #     This function converts the hdf5 files into tiff files
    #
    #     Keyword Arguments:
    #     input_adf -- name, name of the adf file
    #     output_tiff -- Name of the output tiff file
    #     Band_number -- bandnumber of the hdf5 that needs to be converted
    #     scaling_factor -- factor multipied by data is the output array
    #     geo -- [minimum lon, pixelsize, rotation, maximum lat, rotation,
    #             pixelsize], (geospatial dataset)
    #     """
    #
    #     # Open the hdf file
    #     g = gdal.Open(inputname_hdf, gdal.GA_ReadOnly)
    #
    #     #  Define temporary file out and band name in
    #     name_in = g.GetSubDatasets()[Band_number][0]
    #
    #     # Get environmental variable
    #     qgis_path = os.environ["qgis"].split(";")
    #     GDAL_env_path = qgis_path[0]
    #     GDAL_TRANSLATE = os.path.join(GDAL_env_path, "gdal_translate.exe")
    #
    #     # run gdal translate command
    #     FullCmd = "%s -of GTiff %s %s" % (GDAL_TRANSLATE, name_in, Filename_tiff_end)
    #     Raster.Run_command_window(FullCmd)
    #
    #     # Get the data array
    #     dest = gdal.Open(Filename_tiff_end)
    #     Data = dest.GetRasterBand(1).ReadAsArray()
    #     dest = None
    #
    #     # If the band data is not SM change the DN values into PROBA-V values and write into the spectral_reflectance_PROBAV
    #     Data_scaled = Data * scaling_factor
    #
    #     # Save the PROBA-V as a tif file
    #     Raster.createRaster(Filename_tiff_end, Data_scaled, geo_out, "WGS84")
    #
    #     return ()


    # def Vector_to_Raster(Dir, shapefile_name, reference_raster_data_name):
    #     """
    #     This function creates a raster of a shp file
    #
    #     Keyword arguments:
    #     Dir --
    #         str: path to the basin folder
    #     shapefile_name -- 'C:/....../.shp'
    #         str: Path from the shape file
    #     reference_raster_data_name -- 'C:/....../.tif'
    #         str: Path to an example tiff file (all arrays will be reprojected to this example)
    #     """
    #     geo, proj, size_X, size_Y = Raster.Open_array_info(reference_raster_data_name)
    #
    #     x_min = geo[0]
    #     x_max = geo[0] + size_X * geo[1]
    #     y_min = geo[3] + size_Y * geo[5]
    #     y_max = geo[3]
    #     pixel_size = geo[1]
    #
    #     # Filename of the raster Tiff that will be created
    #     Dir_Basin_Shape = os.path.join(Dir, "Basin")
    #     if not os.path.exists(Dir_Basin_Shape):
    #         os.mkdir(Dir_Basin_Shape)
    #
    #     Basename = os.path.basename(shapefile_name)
    #     Dir_Raster_end = os.path.join(
    #         Dir_Basin_Shape, os.path.splitext(Basename)[0] + ".tif"
    #     )
    #
    #     # Open the data source and read in the extent
    #     source_ds = ogr.Open(shapefile_name)
    #     source_layer = source_ds.GetLayer()
    #
    #     # Create the destination data source
    #     x_res = int(round((x_max - x_min) / pixel_size))
    #     y_res = int(round((y_max - y_min) / pixel_size))
    #
    #     # Create tiff file
    #     target_ds = gdal.GetDriverByName("GTiff").Create(
    #         Dir_Raster_end, x_res, y_res, 1, gdal.GDT_Float32, ["COMPRESS=LZW"]
    #     )
    #     target_ds.SetGeoTransform(geo)
    #     srse = osr.SpatialReference()
    #     srse.SetWellKnownGeogCS(proj)
    #     target_ds.SetProjection(srse.ExportToWkt())
    #     band = target_ds.GetRasterBand(1)
    #     target_ds.GetRasterBand(1).SetNoDataValue(-9999)
    #     band.Fill(-9999)
    #
    #     # Rasterize the shape and save it as band in tiff file
    #     gdal.RasterizeLayer(
    #         target_ds, [1], source_layer, None, None, [1], ["ALL_TOUCHED=TRUE"]
    #     )
    #     target_ds = None
    #
    #     # Open array
    #     Raster_Basin = Raster.getRasterData(Dir_Raster_end)
    #
    #     return Raster_Basin