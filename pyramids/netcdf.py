"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""

from numbers import Number
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from osgeo import gdal
from pyramids._utils import (
    create_time_conversion_func,
    numpy_to_gdal_dtype,
)

from pyramids import _io
from pyramids.dataset import Dataset
from pyramids.abstract_dataset import DEFAULT_NO_DATA_VALUE


class NetCDF(Dataset):
    """NetCDF.

    The NetCDF class contains methods to deal with netcdf files.
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__."""
        super().__init__(src)
        # set the is_subset to false before retrieving the variables
        self._is_subset = False
        self._is_md_array = False
        # variables and variable_names
        self.variable_names = self.get_variable_names()
        self._variables = self.get_variables()

        if len(self.variable_names) > 0:
            self._time_stamp = self._get_time_variable()
            self._lat, self._lon = self._get_lat_lon()

    def __str__(self):
        """__str__."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            Variables: {self.variables}
            File: {self.file_name}
        """
        return message

    def __repr__(self):
        """__repr__."""
        message = """
            Cell size: {0}
            Dimension: {1} * {2}
            EPSG: {3}
            Variables: {4}
            projection: {5}
            Metadata: {6}
            File: {7}
        """.format(
            self.cell_size,
            self.rows,
            self.columns,
            self.epsg,
            self.variable_names,
            self.crs,
            self.meta_data,
            self.file_name,
        )
        return message

    @property
    def lon(self):
        """Longitude coordinates."""
        if not hasattr(self, "_lon"):
            pivot_x = self.top_left_corner[0]
            cell_size = self.cell_size
            x_coords = NetCDF.get_x_lon_dimension_array(
                pivot_x, cell_size, self.columns
            )
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            x_coords = self._lon
        return np.array(x_coords)

    @property
    def lat(self):
        """Latitude-coordinate."""
        if not hasattr(self, "_lat"):
            pivot_y = self.top_left_corner[1]
            cell_size = self.cell_size
            y_coords = NetCDF.get_y_lat_dimension_array(pivot_y, cell_size, self.rows)
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @property
    def x(self):
        """x-coordinate/longitude."""
        # X_coordinate = upperleft corner x + index * cell size + celsize/2
        if not hasattr(self, "_lon"):
            pivot_x = self.top_left_corner[0]
            cell_size = self.cell_size
            x_coords = NetCDF.get_x_lon_dimension_array(
                pivot_x, cell_size, self.columns
            )
        else:
            # in case the lat and lon are read from the netcdf file just read the values from the file
            x_coords = self._lon
        return np.array(x_coords)

    @property
    def y(self):
        """y-coordinate/latitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        if not hasattr(self, "_lat"):
            pivot_y = self.top_left_corner[1]
            cell_size = self.cell_size
            y_coords = NetCDF.get_y_lat_dimension_array(pivot_y, cell_size, self.rows)
        else:
            # in case the lat and lon are read from the netcdf file, just read the values from the file
            y_coords = self._lat
        return np.array(y_coords)

    @staticmethod
    def get_y_lat_dimension_array(
        pivot_y: float, cell_size: int, rows: int
    ) -> List[float]:
        """get_y_lat_dimension_array."""
        y_coords = [pivot_y - i * cell_size - cell_size / 2 for i in range(rows)]
        return y_coords

    @staticmethod
    def get_x_lon_dimension_array(
        pivot_x: float, cell_size: int, columns: int
    ) -> List[float]:
        """get_x_lon_dimension_array."""
        x_coords = [pivot_x + i * cell_size + cell_size / 2 for i in range(columns)]
        return x_coords

    @property
    def variables(self) -> Dict[str, "NetCDF"]:
        """Variables in the dataset (resembles the variables in netcdf files.)."""
        return self._variables

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, value: Union[List, Number]):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes
        -----
            - the setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - to change the values of the cells, to the new no_data_value, use the `change_no_data_value` method.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self._change_no_data_value_attr(i, val)
        else:
            self._change_no_data_value_attr(0, value)

    @property
    def file_name(self):
        """File name."""
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    def time_stamp(self):
        """Time stamp."""
        if hasattr(self, "_time_stamp"):
            val = self._time_stamp
        else:
            val = None

        return val

    @classmethod
    def read_file(
        cls, path: str, read_only=True, open_as_multi_dimensional: bool = False
    ) -> "NetCDF":
        """read_file.

        Parameters
        ----------
        path: [str]
            Path of file to open.
        read_only: [bool]
            File mode, set to False, to open in "update" mode.
        open_as_multi_dimensional: [bool]
            Default is False.

        Returns
        -------
        NetCDF
        """
        src = _io.read_file(path, read_only, open_as_multi_dimensional)
        return cls(src)

    def _get_time_variable(self):
        """_get_time_variable."""
        # time_vars = [(i, self.meta_data.get(i)) for i in self.meta_data.keys() if i.startswith("time")]
        # time_var_name = time_vars[0][0].split("#")[0]
        extra_dim = self.meta_data.get("NETCDF_DIM_EXTRA")
        if extra_dim is not None:
            time_var_name = extra_dim.replace("{", "").replace("}", "")
            units = self.meta_data.get(f"{time_var_name}#units")
            func = create_time_conversion_func(units)
            time_vals = self._read_variable(time_var_name)
            time_stamp = list(map(func, time_vals[0]))
        else:
            time_stamp = None
        return time_stamp

    def _get_lat_lon(self):
        lon = self._read_variable("lon")
        lat = self._read_variable("lat")
        return lat, lon

    def _read_variable(self, var: str) -> Union[gdal.Dataset, None]:
        """_read_variable.

        Read variables in a dataset

        Parameters
        ----------
        var: [str]
            variable name in the dataset

        Returns
        -------
        GDAL dataset/None
            if the variable exists in the dataset it will return a gdal dataset otherwise it will return None.
        """
        try:
            var_ds = gdal.Open(f"NETCDF:{self.file_name}:{var}").ReadAsArray()
        except RuntimeError:
            var_ds = None
        return var_ds

    def get_variable_names(self) -> List[str]:
        """get_variable_names."""
        rg = self._raster.GetRootGroup()
        if rg is not None:
            variable_names = rg.GetMDArrayNames()
            dims = rg.GetDimensions()
            dims = [dim.GetName() for dim in dims]
            variable_names = [var for var in variable_names if var not in dims]
        else:
            variable_names = [
                var[1].split(" ")[1] for var in self._raster.GetSubDatasets()
            ]

        return variable_names

    def _read_md_array(self, variable_name: str) -> gdal.Dataset:
        rg = self._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        dtype = md_arr.GetDataType()
        dims = md_arr.GetDimensions()
        if len(dims) == 1:
            if dtype.GetClass() == gdal.GEDTC_STRING:
                src = md_arr
            else:
                src = md_arr.AsClassicDataset(0, 1, rg)
        else:
            src = md_arr.AsClassicDataset(len(dims) - 1, len(dims) - 2, rg)

        return src

    def get_variables(self, read_only: bool = True) -> Dict[str, "NetCDF"]:
        """get_variables.

        Parameters
        ----------
        read_only: [bool]
            Default is True.

        Returns
        -------
        Dict["Dataset", "Dataset"]
            Dictionary of the netcdf variables
        """
        variables = {}
        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()
        for i, var in enumerate(self.variable_names):
            if prefix == "MEMORY" or rg is not None:
                src = self._read_md_array(var)
                if isinstance(src, gdal.Dataset):
                    variables[var] = NetCDF(src)
                    variables[var]._is_md_array = True
                else:
                    variables[var] = src
            else:
                src = gdal.Open(f"{prefix}:{self.file_name}:{var}")
                variables[var] = NetCDF(src)
                variables[var]._is_md_array = False

            variables[var]._is_subset = True

        return variables

    @property
    def is_subset(self) -> bool:
        """is_subset.

        Returns
        -------
        bool
            True if the dataset is a sub_dataset .
        """
        return self._is_subset

    @property
    def is_md_array(self):
        """is_md_array.

        Returns
        -------
        bool
            True if the dataset is a multidimensional array.
        """
        return self._is_md_array

    @staticmethod
    def create_main_dimension(
        group: gdal.Group, dim_name: str, dtype: int, values: np.ndarray
    ) -> gdal.Dimension:
        """Create NetCDF dimension.

        if the dimension name is y, lat, latitude, the dimension type will be horizontal y,
        if the dimension name is x, lon, longitude, the dimension type will be horizontal x,
        if the dimension name is bands, time, the dimension type will be temporal.

        Parameters
        ----------
        group: [gdal.Group]
            Dataset group
        dim_name: [str]
            dimension name
        dtype: [int]
            data type of the dimension
        values: [np.ndarray]
            values of the dimension

        Returns
        -------
        gdal.Dimension
        """
        if dim_name in ["y", "lat", "latitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_Y
        elif dim_name in ["x", "lon", "longitude"]:
            dim_type = gdal.DIM_TYPE_HORIZONTAL_X
        elif dim_name in ["bands", "time"]:
            dim_type = gdal.DIM_TYPE_TEMPORAL
        else:
            dim_type = None
        dim = group.CreateDimension(dim_name, dim_type, None, values.shape[0])
        x_values = group.CreateMDArray(dim_name, [dim], dtype)
        x_values.Write(values)
        dim.SetIndexingVariable(x_values)
        return dim

    @classmethod
    def create_from_array(
        cls,
        arr: np.ndarray,
        geo: Tuple[float, float, float, float, float, float],
        bands_values: List = None,
        epsg: Union[str, int] = 4326,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
        variable_name: str = None,
    ) -> "Dataset":
        """create_from_array.

            - Create_from_array method creates a `Dataset` from a given array and geotransform data.

        Parameters
        ----------
        arr: [np.ndarray]
            numpy array.
        geo : [Tuple]
            geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
        bands_values: [List]
            Name of the bands to be used in the netcdf file. Default is None,
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator)
        no_data_value : Any, optional
            no data value to mask the cells out of the domain. The default is -9999.
        driver_type: [str] optional
            driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM"
        path : [str]
            path to save the driver.
        variable_name: [str]
            name of the variable in the netcdf file. Default is None.

        Returns
        -------
        dst: [DataSet].
            Dataset object will be returned.
        """
        if arr.ndim == 2:
            bands = 1
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            bands = arr.shape[0]
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        if bands_values is None:
            bands_values = list(range(1, bands + 1))
        dst_ds = cls._create_netcdf_from_array(
            arr,
            variable_name,
            cols,
            rows,
            bands,
            bands_values,
            geo,
            epsg,
            no_data_value,
            driver_type=driver_type,
            path=path,
        )
        dst_obj = cls(dst_ds)

        return dst_obj

    @staticmethod
    def _create_netcdf_from_array(
        arr: np.ndarray,
        variable_name: str,
        cols: int,
        rows: int,
        bands: int = None,
        bands_values: List = None,
        geo: Tuple[float, float, float, float, float, float] = None,
        epsg: Union[str, int] = None,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
    ) -> gdal.Dataset:
        """_create_netcdf_from_array.

        Parameters
        ----------
        arr: [np.array]
            numpy array.
        variable_name: [str]
            variable name in the netcdf file.
        cols: [int]
            number of columns in the array.
        rows: [int]
            number of rows in the array.
        bands: [int]
            number of bands, the array is 3d and bands is the first dimension.
        bands_values: [List]
            Name of the bands to be used in the netcdf file. Default is None,
        geo : [Tuple]
            geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator)
        no_data_value : Any, optional
            no data value to mask the cells out of the domain. The default is -9999.
        driver_type: [str] optional
            driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM"
        path : [str]
            path to save the driver.

        Returns
        -------
        gdal.Dataset
        """
        if variable_name is None:
            raise ValueError("Variable_name can not be None")

        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        x_dim_values = NetCDF.get_x_lon_dimension_array(geo[0], geo[1], cols)
        y_dim_values = NetCDF.get_y_lat_dimension_array(geo[3], geo[1], rows)

        if path is None and driver_type == "netcdf":
            path = "netcdf"
            driver_type = "MEM"
        src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(path)
        rg = src.GetRootGroup()

        dim_x = NetCDF.create_main_dimension(rg, "x", dtype, np.array(x_dim_values))
        dim_y = NetCDF.create_main_dimension(rg, "y", dtype, np.array(y_dim_values))
        if arr.ndim == 3:
            dim_bands = NetCDF.create_main_dimension(
                rg, "bands", dtype, np.array(bands_values)
            )
            md_arr = rg.CreateMDArray(variable_name, [dim_bands, dim_y, dim_x], dtype)
        else:
            md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], dtype)

        md_arr.Write(arr)
        md_arr.SetNoDataValueDouble(no_data_value)
        srse = Dataset._create_sr_from_epsg(epsg=epsg)
        md_arr.SetSpatialRef(srse)

        return src

    @staticmethod
    def _add_md_array_to_group(dst_group, var_name, src_mdarray):
        src_dims = src_mdarray.GetDimensions()
        arr = src_mdarray.ReadAsArray()
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        new_md_array = dst_group.CreateMDArray(var_name, src_dims, dtype)
        new_md_array.Write(arr)
        try:
            new_md_array.SetNoDataValueDouble(src_mdarray.GetNoDataValue())
        except:
            new_md_array.SetNoDataValueDouble(-9999)

        new_md_array.SetSpatialRef(src_mdarray.GetSpatialRef())

    def add_variable(self, dataset: "Dataset", variable_name: str = None):
        """add_variable.

        Parameters
        ----------
        dataset: [Dataset]
            dataset to add to the current dataset.
        variable_name: [str], Optional, Default = None
            variable name in the netcdf file. if not given all the variable in the given dataset will be added.


        Example
        -------
        >>> dataset_1 = Dataset.read_file(
        >>>         "tests/data/netcdf/era5_land_monthly_averaged.nc", open_as_multi_dimensional=True
        >>> )
        >>> dataset_2 = Dataset.read_file("tests/data/netcdf/noah-precipitation-1979.nc")
        >>> dataset_1.add_variable(dataset_2, "temperature")
        """
        src_rg = self._raster.GetRootGroup()
        var_rg = dataset._raster.GetRootGroup()
        if variable_name is None:
            variable_name = dataset.variable_names

        for var in variable_name:
            md_arr = var_rg.OpenMDArray(var)
            # incase the variable name already exists in the destination dataset.
            if var in self.variable_names:
                var = f"{var}-new"
            self._add_md_array_to_group(src_rg, var, md_arr)
        self.__init__(self._raster)

    def remove_variable(self, variable_name: str):
        """remove_variable.

        Parameters
        ----------
        variable_name: [str]
            variable name

        Returns
        -------
        Memory Dataset:
            an updated in-memory dataset will be returned even if the original dataset was saved on desk

        Notes
        -----
        The method will not remove the variable from the disk if the dataset is saved on disk. Rather, the method will
        make a Memory driver and copy the original dataset to the memory driver. and then remove the variable from the
        memory dataset.
        """
        if self.driver_type == "memory":
            # first copy the dataset to memory
            dst = self._raster
        else:
            dst = gdal.GetDriverByName("MEM").CreateCopy("", self._raster, 0)

        rg = dst.GetRootGroup()
        rg.DeleteMDArray(variable_name)

        self.__init__(dst)
