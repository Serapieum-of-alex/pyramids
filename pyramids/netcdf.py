import os

import netCDF4
import numpy as np
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
        time : [integer]
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
                # time = nc.variables['t'].units[11:]
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
            geo_out, proj, size_X, size_Y = Raster.openArrayInfo(Reference_filename)

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
