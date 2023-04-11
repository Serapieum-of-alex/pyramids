"""Convert data from one form to another."""
import os
import netCDF4
import numpy as np
from osgeo import gdal

from pyramids.netcdf import NC
from pyramids.dataset import Dataset


class Convert:
    """Convert data from one form to another."""

    def __init__(self):
        # self.vector_catalog: Dict[str, str] = Feature.getCatalog()
        pass

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
        geo, epsg, _, _, time_len, time_var, no_data_value, datatype = NC.getNCDetails(
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

            sr = Dataset._create_sr_from_epsg(epsg=epsg)
            # set the geotransform
            dst.SetGeoTransform(geo)
            # set the projection
            dst.SetProjection(sr.ExportToWkt())
            dst = Dataset._set_no_data_value(dst)
            dst.GetRasterBand(1).WriteArray(data)
            dst.FlushCache()
            dst = None
