"""Convert data from one form to another."""
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, Union

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, ogr, osr
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource
from pandas import DataFrame

# import fiona
from pyramids.array import getPixels
from pyramids.netcdf import NC
from pyramids.raster import Raster
from pyramids.utils import gdal_to_ogr_dtype
from pyramids.vector import Vector


class Convert:
    """Convert data from one form to another."""

    def __init__(self):
        self.vector_catalog: Dict[str, str] = Vector.getCatalog()
        pass

    @staticmethod
    def asciiToRaster(
        ascii_file: str,
        save_path: str,
        pixel_type: int = 1,
        raster_file=None,
        epsg=None,
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
            raise TypeError(
                f"ascii_file input should be string type - given{type(ascii_file)}"
            )

        if not isinstance(save_path, str):
            raise TypeError(
                f"save_path input should be string type - given {type(save_path)}"
            )

        if not isinstance(pixel_type, int):
            raise TypeError(
                f"pixel type input should be integer type please check documentations "
                f"- given {pixel_type}"
            )

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
        path: str, save_path: str, pixel_type: int = 1, Rraster_file=None, epsg=None
    ):
        """This function takes the path of a folder contains ASCII files and convert them into a raster format and in takes  all the spatial information (projection, coordinates of the corner point), and number of rows and columns from raster file or you have to define the epsg corresponding to the you coordinate system and projection.

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
    def nctoTiff(
        input_nc,
        save_to: str,
        separator: str = "_",
        time_var_name: str = None,
        prefix: str = None,
    ):
        """nctoTiff.

        Parameters
        ----------
        input_nc : [string/list]
            a path of the netcdf file or a list of the netcdf files' names.
        save_to : [str]
            Path to where you want to save the files.
        separator : [string]
            separator in the file name that separate the name from the date.
            Default is "_".
        time_var_name: [str]
            name of the time variable in the dataset, as it does not have a unified name. the function will check
            the name time and temporal_resolution if the time_var parameter is not given
        prefix: [str]
            file name prefix. Default is None.

        Returns
        -------
        None.
        """
        if isinstance(input_nc, str):
            nc = netCDF4.Dataset(input_nc)
        elif isinstance(input_nc, list):
            nc = netCDF4.MFDataset(input_nc)
        else:
            raise TypeError(
                "First parameter to the nctoTiff function should be either str or list"
            )

        # get the variable
        Var = list(nc.variables.keys())[-1]
        # extract the data
        dataset = nc[Var]
        # get the details of the file
        geo, epsg, _, _, time_len, time_var, no_data_value, datatype = NC.ncDetails(
            nc, time_var_name=time_var_name
        )

        # Create output folder if needed
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        if prefix is None and time_len > 1:
            # if there is no prefix take the first par of the fname
            fname_prefix = os.path.splitext(os.path.basename(input_nc))[0]
            nameparts = fname_prefix.split(separator)[0]
        else:
            fname_prefix = prefix
            nameparts = fname_prefix

        for i in range(time_len):
            if (
                time_len > 1
            ):  # dataset.shape[0] and # type(temporal_resolution) == np.ndarray: #not temporal_resolution == -9999
                time_one = time_var[i]
                name_out = os.path.join(
                    save_to,
                    f"{nameparts}_{time_one.strftime('%Y')}.{time_one.strftime('%m')}."
                    f"{time_one.strftime('%d')}.{time_one.strftime('%H')}."
                    f"{time_one.strftime('%M')}.{time_one.strftime('%S')}.tif",
                )
            else:
                name_out = os.path.join(save_to, f"{fname_prefix}.tif")

            data = dataset[i, :, :]
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

            sr = Raster._createSRfromEPSG(epsg=epsg)
            # set the geotransform
            dst.SetGeoTransform(geo)
            # set the projection
            dst.SetProjection(sr.ExportToWkt())
            dst = Raster.setNoDataValue(dst)
            dst.GetRasterBand(1).WriteArray(data)
            dst.FlushCache()
            dst = None

    @staticmethod
    def rasterize(vector: str, raster: str, out_path: str, vector_field=None):
        """Covert a vector into raster.

            - The raster cell values will be taken from the column name given in the vector_filed in the vector file.
            - all the new raster geotransform data will be copied from the given raster.
            - raster and vector should have the same projection

        Parameters
        ----------
        raster : [str/gdal Dataset]
            raster path, or gdal Dataset, the raster will only be used as a source for the geotransform (
            projection, rows, columns, location) data to be copied to the rasterized vector.
        vector : [str/ogr DataSource]
            vector path
        out_path : [str]
            Path for output raster. Format and Datatype are the same as ``ras``.
        vector_field : str or None
            Name of a field in the vector to burn values from. If None, all vector
            features are burned with a constant value of 1.

        Returns
        -------
        gdal.Dataset
            Single band raster with vector geometries burned.
        """
        if isinstance(raster, str):
            src = Raster.openDataset(raster)
        else:
            src = raster

        if isinstance(vector, str):
            ds = Vector.openVector(vector)
        else:
            ds = vector

        # Check EPSG are same, if not reproject vector.
        src_epsg = Raster.getEPSG(src)
        ds_epsg = Vector.getEPSG(ds)
        if src_epsg != ds_epsg:
            # TODO: reproject the vector to the raster projection instead of raising an error.
            raise ValueError(
                f"Raster and vector are not the same EPSG. {src_epsg} != {ds_epsg}"
            )

        dr = Raster.createEmptyDriver(src, out_path, bands=1, no_data_value=0)

        if vector_field is None:
            # Use a constant value for all features.
            burn_values = [1]
            attribute = None
        else:
            # Use the values given in the vector field.
            burn_values = None
            attribute = vector_field

        rasterize_opts = gdal.RasterizeOptions(
            bands=[1], burnValues=burn_values, attribute=attribute, allTouched=True
        )
        _ = gdal.Rasterize(dr, vector, options=rasterize_opts)

        dr.FlushCache()
        dr = None
        # read the rasterized vector
        src = Raster.openDataset(out_path)
        return src

    @staticmethod
    def rasterToPolygon(
        src: Dataset,
        path: str = None,
        band: int = 1,
        col_name: Any = "id",
        driver: str = "MEMORY",
    ) -> Union[GeoDataFrame, None]:
        """polygonize.

            polygonize takes a gdal band object and group neighboring cells with the same value into one polygon,
            the resulted vector will be saved to disk as a geojson file

        Parameters
        ----------
        src:
            gdal Dataset
        band: [int]
            raster band index [1,2,3,..]
        path:[str]
            path where you want to save the polygon, the path should include the extension at the end
            (i.e. path/vector_name.geojson)
        col_name:
            name of the column where the raster data will be stored.
        driver: [str]
            vector driver, for all possible drivers check https://gdal.org/drivers/vector/index.html .
            Default is "GeoJSON".

        Returns
        -------
        None
        """
        band = src.GetRasterBand(band)
        prj = src.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        if path is None:
            dst_layername = "id"
        else:
            dst_layername = path.split(".")[0].split("/")[-1]

        dst_ds = Vector.createDataSource(driver, path)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
        dtype = gdal_to_ogr_dtype(src)
        newField = ogr.FieldDefn(col_name, dtype)
        dst_layer.CreateField(newField)
        gdal.Polygonize(band, band, dst_layer, 0, [], callback=None)
        if path:
            dst_layer = None
            dst_ds = None
        else:
            gdf = Convert.ogrDataSourceToGeoDF(dst_ds)
            return gdf


    @staticmethod
    def rasterToDataframe(src: str, vector=None) -> DataFrame:
        """Convert a raster to a DataFrame.

            The function do the following
            - Flatted the array in each band in the raster then mask the values if a vector
            file is given otherwise it will flatten all values.
            - put the values for each band in a column in a dataframe under the name of the raster band, if no meta
            data in the raster band, and index number will be used [1, 2, 3, ...]

        Parameters
        ----------
        src : [str/gdal Dataset]
            Path to raster file.
        vector : Optional[str]
            path to vector file. If given, it will be used to clip the raster

        Returns
        -------
        DataFrame
        """
        temp_dir = None

        # Get raster band names. open the dataset using gdal.Open
        if isinstance(src, str):
            src = Raster.openDataset(src)

        band_names = Raster.getBandNames(src)

        # Create a mask from the pixels touched by the vector.
        if vector is not None:
            # Create a temporary directory for files.
            temp_dir = tempfile.mkdtemp()
            new_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}")

            # read the vector with geopandas
            gdf = Vector.openVector(vector, geodataframe=True)
            # add a unique value for each row to use it to rasterize the vector
            gdf["burn_value"] = list(range(1, len(gdf) + 1))
            # save the new vector to disk to read it with ogr later
            gdf.to_file(new_vector_path, driver="GeoJSON")

            # rasterize the vector by burning the unique values as cell values.
            rasterized_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.tif")
            rasterized_vector = Convert.rasterize(
                new_vector_path, src, rasterized_vector_path, vector_field="burn_value"
            )

            # Loop over mask values to extract pixels.
            tile_dfs = []  # DataFrames of each tile.
            mask_arr = rasterized_vector.GetRasterBand(1).ReadAsArray()

            for arr in Raster.getTile(src):

                mask_dfs = []
                for mask_val in gdf["burn_value"].values:
                    # Extract only masked pixels.
                    flatten_masked_values = getPixels(
                        arr, mask_arr, mask_val=mask_val
                    ).transpose()
                    fid_px = np.ones(flatten_masked_values.shape[0]) * mask_val

                    # Create a DataFrame of masked flatten_masked_values and their FID.
                    mask_df = pd.DataFrame(flatten_masked_values, columns=band_names)
                    mask_df["burn_value"] = fid_px
                    mask_dfs.append(mask_df)

                # Concat the mask DataFrames.
                mask_df = pd.concat(mask_dfs)

                # Join with pixels with vector attributes using the FID.
                tile_dfs.append(mask_df.merge(gdf, how="left", on="burn_value"))

            # Merge all the tiles.
            out_df = pd.concat(tile_dfs)

        else:
            # No vector given, simply load the raster.
            tile_dfs = []  # DataFrames of each tile.
            for arr in Raster.getTile(src):

                idx = (1, 2)  # Assume multiband
                if arr.ndim == 2:
                    idx = (0, 1)  # Handle single band rasters

                mask_arr = np.ones((arr.shape[idx[0]], arr.shape[idx[1]]))
                pixels = getPixels(arr, mask_arr).transpose()
                tile_dfs.append(pd.DataFrame(pixels, columns=band_names))

            # Merge all the tiles.
            out_df = pd.concat(tile_dfs)

        # TODO mask no data values.

        # Remove temporary files.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Return dropping any extra cols.
        return out_df.drop(columns=["burn_value", "geometry"], errors="ignore")

    @staticmethod
    def ogrDataSourceToGeoDF(ds: DataSource) -> GeoDataFrame:
        """Convert ogr DataSource object to a GeoDataFrame.

        Parameters
        ----------
        ds: [ogr.DataSource]
            ogr DataSource

        Returns
        -------
        GeoDataFrame
        """
        # # TODO: not complete yet the function needs to take an ogr.DataSource and then write it to disk and then read
        # #  it using the gdal.OpenEx as below
        # # but this way if i write the vector to disk i can just read it ysing geopandas as df directly.
        # # https://gis.stackexchange.com/questions/227737/python-gdal-ogr-2-x-read-vectors-with-gdal-openex-or-ogr-open
        #
        # # read the vector using gdal not ogr
        # ds = gdal.OpenEx(path)  # , gdal.OF_READONLY
        # layer = ds.GetLayer(0)
        # layer_name = layer.GetName()
        # mempath = "/vsimem/test.geojson"
        # # convert the vector read as a gdal dataset to memory
        # # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.VectorTranslateOptions
        # gdal.VectorTranslate(mempath, ds)  # , SQLStatement=f"SELECT * FROM {layer_name}", layerName=layer_name
        # # reading the memory file using fiona
        # f = fiona.open(mempath, driver='geojson')
        # gdf = gpd.GeoDataFrame.from_features(f, crs=f.crs)

        # till i manage to do the above way just write the ogr.DataSource to disk and then read it using geopandas

        # Create a temporary directory for files.
        temp_dir = tempfile.mkdtemp()
        new_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.geojson")
        Vector.saveVector(ds, new_vector_path)
        gdf = gpd.read_file(new_vector_path)
        return gdf
