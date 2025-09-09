"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""

from numbers import Number
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from osgeo import gdal
from pyramids.base._utils import (
    create_time_conversion_func,
    numpy_to_gdal_dtype,
)

from pyramids import _io
from pyramids.dataset import Dataset
from pyramids.abstract_dataset import DEFAULT_NO_DATA_VALUE


class NetCDF(Dataset):
    """NetCDF.

    DataCube class is a recursive data structure or self-referential object.
    The DataCube class contains methods to deal with DataCube files.

    DataCube Creation guidelines:
        https://acdguide.github.io/Governance/create/create-basics.html
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__.

        Hint:
            - The method will first look for the variables named "lat" and "lon" in the dataset.
            - If the variables are not found, the method will look for the variables named "x" and "y".
            - If the variables are not found, the method will return None.

        """
        super().__init__(src, access=access)
        # set the is_subset to false before retrieving the variables
        self._is_subset = False
        self._is_md_array = False

        if len(self.variable_names) > 0:
            self._time_stamp = self._get_time_variable()

    def __str__(self):
        """__str__."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            Variables: {self.variable_names}
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
    def lon(self) -> np.ndarray:
        """Longitude coordinates.

        Returns
        -------
        np.ndarray:
            If the longitude does not exist as a variable in the netcdf file, it will return None.

        Hint
        ----
        - The method will first look for the variables "lon" in the dataset.
        - If the variable is not found, the method will look for the variable "x".
        - If both lon/x are not found, the method will return None.
        """
        lon = self._read_variable("lon")
        if lon is None:
            lon = self._read_variable("x")
        if lon is not None:
            lon = lon.reshape(lon.size)
        return lon

    @property
    def lat(self) -> np.ndarray:
        """Latitude-coordinate.

        Returns
        -------
        np.ndarray:
            If the variables are not found in the dataset, it will return None.

        Hint
        ----
        - The method will first look for the variables "lat" in the dataset.
        - If the variable is not found, the method will look for the variable "y".
        - If the variables are not found, the method will Calculate the longitude coordinate using the
        pivot point coordinates, cell size and the number of columns.
        """
        lat = self._read_variable("lat")
        if lat is None:
            lat = self._read_variable("y")
        if lat is not None:
            lat = lat.reshape(lat.size)
        return lat

    @property
    def x(self) -> np.ndarray:
        """x-coordinate/longitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """y-coordinate/latitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lat

    @property
    def geotransform(self):
        """Geotransform."""
        if self.lon is None:
            geotransform = None
        else:
            geotransform = (
                self.lon[0] - self.cell_size / 2,
                self.cell_size,
                0,
                self.lat[0] + self.cell_size / 2,
                0,
                -self.cell_size,
            )
        return geotransform

    @property
    def variable_names(self) -> List[str]:
        """variable_names."""
        return self.get_variable_names()

    @property
    def variables(self) -> Dict[str, str]:
        """Variables in the dataset (resembles the variables in DataCube files.)."""
        vars_dict = {}
        for var in self.variable_names:
            vars_dict[var] = self.get_variable(var)
        return vars_dict

    @property
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        return self._no_data_value

    @no_data_value.setter
    def no_data_value(self, value: Union[List, Number]):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes:
            - the setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - to change the values of the cells, to the new no_data_value, use the `change_no_data_value` method.
        """
        super().no_data_value = value

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

        Args:
            path (str):
                Path of file to open.
            read_only (bool):
                File mode. Set to False to open in "update" mode. Defaults to True.
            open_as_multi_dimensional (bool):
                Open as multi-dimensional dataset. Defaults to False.

        Returns:
            NetCDF:
                Opened NetCDF dataset.
        """
        src = _io.read_file(path, read_only, open_as_multi_dimensional)
        if read_only:
            read_only = "read_only"
        else:
            read_only = "write"
        return cls(src, access=read_only)

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

    def _get_dimension_names(self) -> List[str]:
        rg = self._raster.GetRootGroup()
        if rg is not None:
            dims = rg.GetDimensions()
            dims_names = [dim.GetName() for dim in dims]
        else:
            dims_names = None
        return dims_names

    @property
    def dimension_names(self) -> List[str]:
        """dimension_names."""
        return self._get_dimension_names()

    def _get_dimension(self, name: str) -> gdal.Dimension:
        dim_names = self.dimension_names
        if dim_names is not None and name in dim_names:
            rg = self._raster.GetRootGroup()
            dims = rg.GetDimensions()
            dim = dims[dim_names.index(name)]
        else:
            dim = None
        return dim

    def _read_variable(self, var: str) -> Union[np.ndarray, None]:
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
            dim = self._get_dimension(var)
            var_ds = (
                dim.GetIndexingVariable().ReadAsArray() if dim is not None else None
            )

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

    def get_variable(self, variable_name: str) -> "NetCDF":
        """get_variables.

        Returns
        -------
        Dict["Dataset", "Dataset"]
            Dictionary of the netcdf variables
        """
        # convert the variable_name to a list if it is a string
        if variable_name not in self.variable_names:
            raise ValueError(
                f"{variable_name} is not a valid variable name in {self.variable_names}"
            )

        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()

        if prefix == "MEMORY" or rg is not None:
            src = self._read_md_array(variable_name)
            if isinstance(src, gdal.Dataset):
                cube = NetCDF(src)
                cube._is_md_array = True
            else:
                cube = src
        else:
            src = gdal.Open(f"{prefix}:{self.file_name}:{variable_name}")
            cube = NetCDF(src)
            cube._is_md_array = False

        cube._is_subset = True

        return cube

    @property
    def is_subset(self) -> bool:
        """is_subset.

        Returns:
            bool
                True if the dataset is a sub_dataset.
        """
        return self._is_subset

    @property
    def is_md_array(self):
        """is_md_array.

        Returns:
            bool
                True if the dataset is a multidimensional array.
        """
        return self._is_md_array

    def copy(self, path: str = None) -> "Dataset":
        """Deep copy.

        Parameters
        ----------
        path: str, optional
            destination path to save the copied dataset, if None is passed, the copy dataset will be created in memory

        Examples
        --------
        - First, we will create a dataset with 1 band, 3 rows and 5 columns.

            >>> import numpy as np
            >>> arr = np.random.rand(3, 5)
            >>> top_left_corner = (0, 0)
            >>> cell_size = 0.05
            >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
            >>> print(dataset)
            <BLANKLINE>
                        Cell size: 0.05
                        Dimension: 3 * 5
                        EPSG: 4326
                        Number of Bands: 1
                        Band names: ['Band_1']
                        Mask: -9999.0
                        Data type: float64
                        File:...
            <BLANKLINE>

        - Now, we will create a copy of the dataset.

            >>> copied_dataset = dataset.copy(path="copy-dataset.tif")
            >>> print(copied_dataset)
            <BLANKLINE>
                        Cell size: 0.05
                        Dimension: 3 * 5
                        EPSG: 4326
                        Number of Bands: 1
                        Band names: ['Band_1']
                        Mask: -9999.0
                        Data type: float64
                        File: copy-dataset.tif
            <BLANKLINE>

        - Now close the dataset.

            >>> copied_dataset.close()

        """
        if path is None:
            path = ""
            driver = "MEM"
        else:
            driver = "netCDF"

        src = gdal.GetDriverByName(driver).CreateCopy(path, self._raster)

        return NetCDF(src, access="write")

    @staticmethod
    def create_main_dimension(
        group: gdal.Group, dim_name: str, dtype: int, values: np.ndarray
    ) -> gdal.Dimension:
        """Create NetCDF dimension.

        If the dimension name is y, lat, or latitude, the dimension type will be horizontal y.
        If the dimension name is x, lon, or longitude, the dimension type will be horizontal x.
        If the dimension name is bands or time, the dimension type will be temporal.

        Args:
            group (gdal.Group):
                Dataset group.
            dim_name (str):
                Dimension name.
            dtype (int):
                Data type of the dimension.
            values (np.ndarray):
                Values of the dimension.

        Returns:
            gdal.Dimension

        Hint:
            - The dimension will be saved as a dimension and a mdarray in the given group.
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

        Args:
            arr (np.ndarray):
                Numpy array.
            geo (Tuple[float, float, float, float, float, float]):
                Geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
            bands_values (List | None):
                Names of the bands to be used in the netcdf file. Default is None.
            epsg (int | str):
                EPSG code (https://epsg.io/). Default 3857 (WGS84 Web Mercator).
            no_data_value (Any | list):
                No data value to mask cells out of the domain. Default is -9999.
            driver_type (str):
                Driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM".
            path (str | None):
                Path to save the driver.
            variable_name (str | None):
                Name of the variable in the netcdf file. Default is None.

        Returns:
            Dataset:
                Dataset object.
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
        bands_values: List = None,
        geo: Tuple[float, float, float, float, float, float] = None,
        epsg: Union[str, int] = None,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
    ) -> gdal.Dataset:
        """_create_netcdf_from_array.

        Args:
            arr (np.ndarray):
                Numpy array.
            variable_name (str):
                Variable name in the netcdf file.
            cols (int):
                Number of columns in the array.
            rows (int):
                Number of rows in the array.
            bands (int | None):
                Number of bands; for 3D arrays bands is the first dimension.
            bands_values (List | None):
                Names of the bands to be used in the netcdf file. Default is None.
            geo (Tuple[float, float, float, float, float, float] | None):
                Geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
            epsg (int | str | None):
                EPSG code (https://epsg.io/). Default 3857 (WGS84 Web Mercator).
            no_data_value (Any | list):
                No data value to mask cells out of the domain. Default is -9999.
            driver_type (str):
                Driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM".
            path (str | None):
                Path to save the driver.

        Returns:
            gdal.Dataset:
                The created NetCDF GDAL dataset.
        """
        if variable_name is None:
            raise ValueError("Variable_name cannot be None")

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

    def add_variable(
        self, dataset: Union["Dataset", "DataCube"], variable_name: str = None
    ):
        """add_variable.

        Args:
            dataset (Dataset):
                Dataset to add to the current dataset.
            variable_name (str | None):
                Variable name in the netcdf file. If not given, all variables in the given dataset will be added. Default is None.

        Examples:
            - Add a variable from another dataset:
              ```python
              >>> dataset_1 = Dataset.read_file(
              ...   "tests/data/netcdf/era5_land_monthly_averaged.nc", open_as_multi_dimensional=True
              ... )
              >>> dataset_2 = Dataset.read_file("tests/data/netcdf/noah-precipitation-1979.nc")
              >>> dataset_1.add_variable(dataset_2, "temperature")

              ```
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

        Args:
            variable_name (str):
                Variable name.

        Returns:
            None:
                The internal dataset is updated in memory. Even if the original dataset was saved on disk, this updates the in-memory copy.

        Notes:
            The method will not remove the variable from the disk if the dataset is saved on disk. Rather, the method will
            make a Memory driver and copy the original dataset to the memory driver, and then remove the variable from the
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
