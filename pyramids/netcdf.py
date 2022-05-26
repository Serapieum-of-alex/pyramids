import os
import numpy as np
import netCDF4
import pandas as pd
from pyramids.raster import Raster

class NC:
    """Nectcdf"""
    def __init__(self):
        pass


    @staticmethod
    def ncDetails(nc, var=None):
        """
        NCGetGeotransform takes a netcdf object and return the geottansform data of
        the bottom left corner

        Parameters
        ----------
        nc : [netcdf object]
            netcdf object .
        var : [string], optional
            the variable you want to read from the netcdf file if None is given the
            last variable in the file will be read. The default is None.

        Returns
        -------
        geo : [tuple]
            geotransform data of the netcdf file
        epsg : [integer]
            epsg number
        size_X : [integer]
            number of coordinates in x direction
        size_Y : [integer]
            number of coordinates in y direction
        size_Z : [integer]
            number of coordinates in z direction
        Time : [integer]
            time varialble in the netcdf file
        """
        # list if variables
        if var is None:
            var = list(nc.variables.keys())[-1]

        data = nc.variables[var]
        # nodatavalue
        try:
            NoDataValue = data._FillValue
        except AttributeError:
            NoDataValue = data.missing_value
        # data type
        try:
            datatype = data.datatype
        except AttributeError:
            datatype = data.dtype

        size_Y, size_X = np.int_(data.shape[-2:])
        # if there is a stack of layers in the file (3d array)
        if len(data.shape) == 3 and data.shape[0] > 1:
            size_Z = np.int_(data.shape[0])
            try:
                TimeVar = nc.variables["time"]
                Time = TimeVar[:]
                # convert  time numbers to dates
                Time = netCDF4.num2date(Time[:], TimeVar.units)
            except:
                Time = nc.variables["t"][:]
                # Time = nc.variables['t'].units[11:]
        else:
            # if there is only one layer(2D array)
            size_Z = 1
            Time = -9999

        # get lats and lons
        try:
            lats = nc.variables["latitude"][:]
            # Geo6 = nc.variables['latitude'].res
        except:
            lats = nc.variables["lat"][:]
            # Geo6 = nc.variables['lat'].res

        try:
            lons = nc.variables["longitude"][:]

        except:
            lons = nc.variables["lon"][:]
            # Geo2 = nc.variables['lon'].size

        # try to get the resolutio of the file
        try:
            try:
                Geo2 = nc.variables["longitude"].res
            except:
                try:
                    Geo2 = nc.variables["lon"].res
                except:
                    Geo2 = lons[1] - lons[0]
        except:
            assert False, "the netcdf file does not hae a resolution attribute"

        # Lower left corner corner coordinates
        Geo4 = np.min(lats) + Geo2 / 2
        Geo1 = np.min(lons) - Geo2 / 2

        try:
            crso = nc.variables["crs"]
            proj = crso.projection
            epsg = Raster.getEPSG(proj, extension="GEOGCS")
        except:
            epsg = 4326

        geo = tuple([Geo1, Geo2, 0, Geo4, 0, Geo2])

        return geo, epsg, size_X, size_Y, size_Z, Time, NoDataValue, datatype


    @staticmethod
    def saveNC(
            namenc,
            DataCube,
            Var,
            Reference_filename,
            Startdate="",
            Enddate="",
            Time_steps="",
            Scaling_factor=1,
    ):
        """
        Save_as_NC(namenc, DataCube, Var, Reference_filename,  Startdate = '',
                   Enddate = '', Time_steps = '', Scaling_factor = 1)

        Parameters
        ----------
        namenc : [str]
            complete path of the output file with .nc extension.
        DataCube : [array]
            dataset of the nc file, can be a 2D or 3D array [time, lat, lon],
            must be same size as reference data.
        Var : [str]
            the name of the variable.
        Reference_filename : [str]
            complete path to the reference file name.
        Startdate : str, optional
            needs to be filled when you want to save a 3D array,'YYYY-mm-dd'
            defines the Start datum of the dataset. The default is ''.
        Enddate : str, optional
            needs to be filled when you want to save a 3D array, 'YYYY-mm-dd'
            defines the End datum of the dataset. The default is ''.
        Time_steps : str, optional
            'monthly' or 'daily', needs to be filled when you want to save a
            3D array, defines the timestep of the dataset. The default is ''.
        Scaling_factor : TYPE, optional
            number, scaling_factor of the dataset. The default is 1.

        Returns
        -------
        None.
        """
        if not os.path.exists(namenc):

            # Get raster information
            geo_out, proj, size_X, size_Y = Raster.Open_array_info(Reference_filename)

            # Create the lat/lon rasters
            lon = np.arange(size_X) * geo_out[1] + geo_out[0] - 0.5 * geo_out[1]
            lat = np.arange(size_Y) * geo_out[5] + geo_out[3] - 0.5 * geo_out[5]

            # Create the nc file
            nco = netCDF4.Dataset(namenc, "w", format="NETCDF4_CLASSIC")
            nco.description = "%s data" % Var

            # Create dimensions, variables and attributes:
            nco.createDimension("longitude", size_X)
            nco.createDimension("latitude", size_Y)

            # Create time dimension if the parameter is time dependent
            if Startdate != "":
                if Time_steps == "monthly":
                    Dates = pd.date_range(Startdate, Enddate, freq="MS")
                if Time_steps == "daily":
                    Dates = pd.date_range(Startdate, Enddate, freq="D")
                time_or = np.zeros(len(Dates))
                i = 0
                for Date in Dates:
                    time_or[i] = Date.toordinal()
                    i += 1
                nco.createDimension("time", None)
                timeo = nco.createVariable("time", "f4", ("time",))
                timeo.units = "%s" % Time_steps
                timeo.standard_name = "time"

            # Create the lon variable
            lono = nco.createVariable("longitude", "f8", ("longitude",))
            lono.standard_name = "longitude"
            lono.units = "degrees_east"
            lono.pixel_size = geo_out[1]

            # Create the lat variable
            lato = nco.createVariable("latitude", "f8", ("latitude",))
            lato.standard_name = "latitude"
            lato.units = "degrees_north"
            lato.pixel_size = geo_out[5]

            # Create container variable for CRS: lon/lat WGS84 datum
            crso = nco.createVariable("crs", "i4")
            crso.long_name = "Lon/Lat Coords in WGS84"
            crso.grid_mapping_name = "latitude_longitude"
            crso.projection = proj
            crso.longitude_of_prime_meridian = 0.0
            crso.semi_major_axis = 6378137.0
            crso.inverse_flattening = 298.257223563
            crso.geo_reference = geo_out

            # Create the data variable
            if Startdate != "":
                preco = nco.createVariable(
                    "%s" % Var,
                    "f8",
                    ("time", "latitude", "longitude"),
                    zlib=True,
                    least_significant_digit=1,
                )
                timeo[:] = time_or
            else:
                preco = nco.createVariable(
                    "%s" % Var,
                    "f8",
                    ("latitude", "longitude"),
                    zlib=True,
                    least_significant_digit=1,
                )

            # Set the data variable information
            preco.scale_factor = Scaling_factor
            preco.add_offset = 0.00
            preco.grid_mapping = "crs"
            preco.set_auto_maskandscale(False)

            # Set the lat/lon variable
            lono[:] = lon
            lato[:] = lat

            # Set the data variable
            if Startdate != "":
                for i in range(len(Dates)):
                    preco[i, :, :] = DataCube[i, :, :] * 1.0 / np.float(Scaling_factor)
            else:
                preco[:, :] = DataCube[:, :] * 1.0 / np.float(Scaling_factor)

            nco.close()
        return ()


    # def Create_NC_name(Var, Simulation, Dir_Basin, sheet_nmbr, info=""):
    #
    #     # Create the output name
    #     nameOut = "".join(
    #         ["_".join([Var, "Simulation%d" % Simulation, "_".join(info)]), ".nc"]
    #     )
    #     namePath = os.path.join(
    #         Dir_Basin,
    #         "Simulations",
    #         "Simulation_%d" % Simulation,
    #         "Sheet_%d" % sheet_nmbr,
    #     )
    #     if not os.path.exists(namePath):
    #         os.makedirs(namePath)
    #     nameTot = os.path.join(namePath, nameOut)
    #
    #     return nameTot
    #
    #
    # def Create_new_NC_file(nc_outname, Basin_Example_File, Basin):
    #
    #     # Open basin file
    #     dest = gdal.Open(Basin_Example_File)
    #     Basin_array = dest.GetRasterBand(1).ReadAsArray()
    #     Basin_array[np.isnan(Basin_array)] = -9999
    #     Basin_array[Basin_array < 0] = -9999
    #
    #     # Get Basic information
    #     Geo = dest.GetGeoTransform()
    #     size_X = dest.RasterXSize
    #     size_Y = dest.RasterYSize
    #     epsg = dest.GetProjection()
    #
    #     # Get Year and months
    #     year = int(os.path.basename(nc_outname).split(".")[0])
    #     Dates = pd.date_range("%d-01-01" % year, "%d-12-31" % year, freq="MS")
    #
    #     # Latitude and longitude
    #     lons = np.arange(size_X) * Geo[1] + Geo[0] + 0.5 * Geo[1]
    #     lats = np.arange(size_Y) * Geo[5] + Geo[3] + 0.5 * Geo[5]
    #
    #     # Create NetCDF file
    #     nco = netCDF4.Dataset(nc_outname, "w", format="NETCDF4_CLASSIC")
    #     nco.set_fill_on()
    #     nco.description = "%s" % Basin
    #
    #     # Create dimensions
    #     nco.createDimension("latitude", size_Y)
    #     nco.createDimension("longitude", size_X)
    #     nco.createDimension("time", None)
    #
    #     # Create NetCDF variables
    #     crso = nco.createVariable("crs", "i4")
    #     crso.long_name = "Lon/Lat Coords in WGS84"
    #     crso.standard_name = "crs"
    #     crso.grid_mapping_name = "latitude_longitude"
    #     crso.projection = epsg
    #     crso.longitude_of_prime_meridian = 0.0
    #     crso.semi_major_axis = 6378137.0
    #     crso.inverse_flattening = 298.257223563
    #     crso.geo_reference = Geo
    #
    #     ######################### Save Rasters in NetCDF ##############################
    #
    #     lato = nco.createVariable("latitude", "f8", ("latitude",))
    #     lato.units = "degrees_north"
    #     lato.standard_name = "latitude"
    #     lato.pixel_size = Geo[5]
    #
    #     lono = nco.createVariable("longitude", "f8", ("longitude",))
    #     lono.units = "degrees_east"
    #     lono.standard_name = "longitude"
    #     lono.pixel_size = Geo[1]
    #
    #     timeo = nco.createVariable("time", "f4", ("time",))
    #     timeo.units = "Monthly"
    #     timeo.standard_name = "time"
    #
    #     # Variables
    #     basin_var = nco.createVariable(
    #         "Landuse", "i", ("latitude", "longitude"), fill_value=-9999
    #     )
    #     basin_var.long_name = "Landuse"
    #     basin_var.grid_mapping = "crs"
    #
    #     # Create time unit
    #     i = 0
    #     time_or = np.zeros(len(Dates))
    #     for Date in Dates:
    #         time_or[i] = Date.toordinal()
    #         i += 1
    #
    #     # Load data
    #     lato[:] = lats
    #     lono[:] = lons
    #     timeo[:] = time_or
    #     basin_var[:, :] = Basin_array
    #
    #     # close the file
    #     time.sleep(1)
    #     nco.close()
    #     return ()
    #
    #
    # def Add_NC_Array_Variable(nc_outname, Array, name, unit, Scaling_factor=1):
    #
    #     # create input array
    #     Array[np.isnan(Array)] = -9999 * np.float(Scaling_factor)
    #     Array = np.int_(Array * 1.0 / np.float(Scaling_factor))
    #
    #     # Create NetCDF file
    #     nco = netCDF4.Dataset(nc_outname, "r+", format="NETCDF4_CLASSIC")
    #     nco.set_fill_on()
    #
    #     paro = nco.createVariable(
    #         "%s" % name,
    #         "i",
    #         ("time", "latitude", "longitude"),
    #         fill_value=-9999,
    #         zlib=True,
    #         least_significant_digit=0,
    #     )
    #
    #     paro.scale_factor = Scaling_factor
    #     paro.add_offset = 0.00
    #     paro.grid_mapping = "crs"
    #     paro.long_name = name
    #     paro.units = unit
    #     paro.set_auto_maskandscale(False)
    #
    #     # Set the data variable
    #     paro[:, :, :] = Array
    #
    #     # close the file
    #     time.sleep(1)
    #     nco.close()
    #
    #     return ()
    #
    #
    # def Add_NC_Array_Static(nc_outname, Array, name, unit, Scaling_factor=1):
    #
    #     # create input array
    #     Array[np.isnan(Array)] = -9999 * np.float(Scaling_factor)
    #     Array = np.int_(Array * 1.0 / np.float(Scaling_factor))
    #
    #     # Create NetCDF file
    #     nco = netCDF4.Dataset(nc_outname, "r+", format="NETCDF4_CLASSIC")
    #     nco.set_fill_on()
    #
    #     paro = nco.createVariable(
    #         "%s" % name,
    #         "i",
    #         ("latitude", "longitude"),
    #         fill_value=-9999,
    #         zlib=True,
    #         least_significant_digit=0,
    #     )
    #
    #     paro.scale_factor = Scaling_factor
    #     paro.add_offset = 0.00
    #     paro.grid_mapping = "crs"
    #     paro.long_name = name
    #     paro.units = unit
    #     paro.set_auto_maskandscale(False)
    #
    #     # Set the data variable
    #     paro[:, :] = Array
    #
    #     # close the file
    #     time.sleep(1)
    #     nco.close()


    # def Open_nc_info(NC_filename, Var=None):
    #     """
    #     Opening a nc info, for example size of array, time (ordinal), projection and transform matrix.
    #
    #     Keyword Arguments:
    #     filename -- 'C:/file/to/path/file.nc'
    #         string that defines the input nc file
    #
    #     """
    #
    #     fh = netCDF4.Dataset(NC_filename, mode="r")
    #
    #     if Var is None:
    #         Var = list(fh.variables.keys())[-1]
    #
    #     data = fh.variables[Var][:]
    #
    #     size_Y, size_X = np.int_(data.shape[-2:])
    #     if len(data.shape) == 3:
    #         size_Z = np.int_(data.shape[0])
    #         Time = fh.variables["time"][:]
    #     else:
    #         size_Z = 1
    #         Time = -9999
    #     lats = fh.variables["latitude"][:]
    #     lons = fh.variables["longitude"][:]
    #
    #     Geo6 = fh.variables["latitude"].pixel_size
    #     Geo2 = fh.variables["longitude"].pixel_size
    #     Geo4 = np.max(lats) + Geo6 / 2
    #     Geo1 = np.min(lons) - Geo2 / 2
    #
    #     crso = fh.variables["crs"]
    #     proj = crso.projection
    #     epsg = Raster.Get_epsg(proj, extension="GEOGCS")
    #     geo_out = tuple([Geo1, Geo2, 0, Geo4, 0, Geo6])
    #     fh.close()
    #
    #     return geo_out, epsg, size_X, size_Y, size_Z, Time


    # def Open_nc_array(NC_filename, Var=None, Startdate="", Enddate=""):
    #     """
    #     Opening a nc array.
    #
    #     Keyword Arguments:
    #     filename -- 'C:/file/to/path/file.nc'
    #         string that defines the input nc file
    #     Var -- string
    #         Defines the band name that must be opened.
    #     Startdate -- "yyyy-mm-dd"
    #         Defines the startdate (default is from beginning of array)
    #     Enddate -- "yyyy-mm-dd"
    #         Defines the enddate (default is from end of array)
    #     """
    #
    #     fh = netCDF4.Dataset(NC_filename, mode="r")
    #     if Var is None:
    #         Var = fh.variables.keys()[-1]
    #
    #     if Startdate != "":
    #         Time = fh.variables["time"][:]
    #         Array_check_start = np.ones(np.shape(Time))
    #         Date = pd.Timestamp(Startdate)
    #         Startdate_ord = Date.toordinal()
    #         Array_check_start[Time >= Startdate_ord] = 0
    #         Start = np.sum(Array_check_start)
    #     else:
    #         Start = 0
    #
    #     if Enddate != "":
    #         Time = fh.variables["time"][:]
    #         Array_check_end = np.zeros(np.shape(Time))
    #         Date = pd.Timestamp(Enddate)
    #         Enddate_ord = Date.toordinal()
    #         Array_check_end[Enddate_ord >= Time] = 1
    #         End = np.sum(Array_check_end)
    #     else:
    #         try:
    #             Time = fh.variables["time"][:]
    #             End = len(Time)
    #         except:
    #             End = ""
    #
    #     if Enddate != "" or Startdate != "":
    #         Data = fh.variables[Var][int(Start): int(End), :, :]
    #
    #     else:
    #         Data = fh.variables[Var][:]
    #     fh.close()
    #
    #     Data = np.array(Data)
    #     try:
    #         Data[Data == -9999] = np.nan
    #     except:
    #         pass
    #
    #     return Data


    # def Open_ncs_array(NC_Directory, Var, Startdate, Enddate):
    #     """
    #     Opening a nc array.
    #
    #     Keyword Arguments:
    #     NC_Directory -- 'C:/file/to/path'
    #         string that defines the path to all the simulation nc files
    #     Var -- string
    #         Defines the band name that must be opened.
    #     Startdate -- "yyyy-mm-dd"
    #         Defines the startdate
    #     Enddate -- "yyyy-mm-dd"
    #         Defines the enddate
    #     """
    #
    #     panda_start = pd.Timestamp(Startdate)
    #     panda_end = pd.Timestamp(Enddate)
    #
    #     years = range(int(panda_start.year), int(panda_end.year) + 1)
    #     Data_end = []
    #     for year in years:
    #
    #         NC_filename = os.path.join(NC_Directory, "%d.nc" % year)
    #
    #         if year == years[0]:
    #             Startdate_now = Startdate
    #         else:
    #             Startdate_now = "%d-01-01" % int(year)
    #
    #         if year == years[-1]:
    #             Enddate_now = Enddate
    #         else:
    #             Enddate_now = "%d-12-31" % int(year)
    #
    #         Data_now = Raster.Open_nc_array(
    #             NC_filename, Var, Startdate_now, Enddate_now
    #         )
    #
    #         if year == years[0]:
    #             Data_end = Data_now
    #         else:
    #             Data_end = np.vstack([Data_end, Data_now])
    #
    #     Data_end = np.array(Data_end)
    #
    #     return Data_end
    #
    #
    # def Open_nc_dict(input_netcdf, group_name, startdate="", enddate=""):
    #     """
    #     Opening a nc dictionary.
    #
    #     Keyword Arguments:
    #     filename -- 'C:/file/to/path/file.nc'
    #         string that defines the input nc file
    #     group_name -- string
    #         Defines the group name that must be opened.
    #     Startdate -- "yyyy-mm-dd"
    #         Defines the startdate (default is from beginning of array)
    #     Enddate -- "yyyy-mm-dd"
    #         Defines the enddate (default is from end of array)
    #     """
    #     # sort out if the dataset is static or dynamic (written in group_name)
    #     kind_of_data = group_name.split("_")[-1]
    #
    #     # if it is dynamic also collect the time parameter
    #     if kind_of_data == "dynamic":
    #         time_dates = Raster.Open_nc_array(input_netcdf, Var="time")
    #         Amount_months = len(time_dates)
    #
    #     # Open the input netcdf and the wanted group name
    #     in_nc = netCDF4.Dataset(input_netcdf)
    #     data = in_nc.groups[group_name]
    #
    #     # Convert the string into a string that can be retransformed into a dictionary
    #     string_dict = str(data)
    #     split_dict = str(string_dict.split("\n")[2:-4])
    #     split_dict = split_dict.replace("'", "")
    #     split_dict = split_dict[1:-1]
    #     dictionary = dict()
    #     split_dict_split = re.split(":|,  ", split_dict)
    #
    #     # Loop over every attribute and add the array
    #     for i in range(0, len(split_dict_split)):
    #         number_val = split_dict_split[i]
    #         if i % 2 == 0:
    #             Array_text = split_dict_split[i + 1].replace(",", "")
    #             Array_text = Array_text.replace("[", "")
    #             Array_text = Array_text.replace("]", "")
    #             # If the array is dynamic add a 2D array
    #             if kind_of_data == "dynamic":
    #                 tot_length = len(np.fromstring(Array_text, sep=" "))
    #                 dictionary[int(number_val)] = np.fromstring(
    #                     Array_text, sep=" "
    #                 ).reshape((int(Amount_months), int(tot_length / Amount_months)))
    #             # If the array is static add a 1D array
    #             else:
    #                 dictionary[int(number_val)] = np.fromstring(Array_text, sep=" ")
    #
    #     # Clip the dynamic dataset if a start and enddate is defined
    #     if kind_of_data == "dynamic":
    #
    #         if startdate != "":
    #             Array_check_start = np.ones(np.shape(time_dates))
    #             Date = pd.Timestamp(startdate)
    #             Startdate_ord = Date.toordinal()
    #             Array_check_start[time_dates >= Startdate_ord] = 0
    #             Start = np.sum(Array_check_start)
    #         else:
    #             Start = 0
    #
    #         if enddate != "":
    #             Array_check_end = np.zeros(np.shape(time_dates))
    #             Date = pd.Timestamp(enddate)
    #             Enddate_ord = Date.toordinal()
    #             Array_check_end[Enddate_ord >= time_dates] = 1
    #             End = np.sum(Array_check_end)
    #         else:
    #             try:
    #                 time_dates = in_nc.variables["time"][:]
    #                 End = len(time_dates)
    #             except:
    #                 End = ""
    #
    #         if Start != 0 or (End != len(time_dates) or ""):
    #
    #             if End == "":
    #                 End = len(time_dates)
    #
    #             for key in dictionary.iterkeys():
    #                 Array = dictionary[key][:, :]
    #                 Array_new = Array[int(Start): int(End), :]
    #                 dictionary[key] = Array_new
    #     in_nc.close()
    #
    #     return dictionary