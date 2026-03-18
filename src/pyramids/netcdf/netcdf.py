"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""
from __future__ import annotations
from numbers import Number
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
from osgeo import gdal
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.netcdf.utils import create_time_conversion_func, _to_py_scalar

from pyramids import _io
from pyramids.dataset import Dataset
from pyramids.abstract_dataset import DEFAULT_NO_DATA_VALUE
from pyramids.netcdf.metadata import get_metadata

class NetCDF(Dataset):
    """NetCDF.

    NetCDF class is a recursive data structure or self-referential object.
    The NetCDF class contains methods to deal with NetCDF files.

    NetCDF Creation guidelines:
        https://acdguide.github.io/Governance/create/create-basics.html
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only", open_as_multi_dimensional: bool = True):
        """__init__.

        Hint:
            - The method will first look for the variables named "lat" and "lon" in the dataset.
            - If the variables are not found, the method will look for the variables named "x" and "y".
            - If the variables are not found, the method will return None.

        """
        super().__init__(src, access=access)
        # set the is_subset to false before retrieving the variables
        if open_as_multi_dimensional:
            self._is_md_array = True
            self._is_subset = False
        else:
            self._is_md_array = False
            self._is_subset = False
        # Caches (invalidated by _replace_raster, add_variable, remove_variable)
        self._cached_variables = None
        self._cached_meta_data = None

    def __str__(self):
        """__str__."""
        message = f"""
            Cell size: {self.cell_size}
            Dimension: {self.rows} * {self.columns}
            EPSG: {self.epsg}
            projection: {self.crs}
            Variables: {self.variable_names}
            Metadata: {self.meta_data}
            File: {self.file_name}
        """
        return message

    def __repr__(self):
        """__repr__."""
        return super().__repr__()

    @property
    def top_left_corner(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    def lon(self) -> np.ndarray:
        """Longitude coordinates.

        Args:
            np.ndarray:
                If the longitude does not exist as a variable in the netcdf file, it will return None.

        Hint:
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

        Args:
            np.ndarray:
                If the variables are not found in the dataset, it will return None.

        Hint:
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
        """Geotransform.

        Computes from lon/lat coordinate arrays if available.
        Falls back to the parent GDAL GetGeoTransform() otherwise.
        """
        if self.lon is not None and self.lat is not None:
            return (
                self.lon[0] - self.cell_size / 2,
                self.cell_size,
                0,
                self.lat[0] + self.cell_size / 2,
                0,
                -self.cell_size,
            )
        return self._geotransform

    @property
    def variable_names(self) -> List[str]:
        """variable_names."""
        return self.get_variable_names()

    @property
    def variables(self) -> Dict[str, "NetCDF"]:
        """Variables in the dataset (resembles the variables in NetCDF files.)."""
        if self._cached_variables is None:
            self._cached_variables = {
                var: self.get_variable(var) for var in self.variable_names
            }
        return self._cached_variables

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
        return self.get_time_variable()

    def _check_not_container(self, operation: str):
        """Raise ValueError if this is a root MDIM container (not a variable subset)."""
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            raise ValueError(
                f"Spatial operations are not supported on the NetCDF container. "
                f"Use nc.get_variable('var_name').{operation}(...) instead."
            )

    def read_array(self, band: int = None, window=None) -> np.ndarray:
        """Read array from the dataset.

        Raises a clear error when called on the root MDIM container
        (which has no raster bands).
        """
        self._check_not_container("read_array")
        return super().read_array(band=band, window=window)

    def crop(self, mask, touch: bool = True, inplace: bool = False):
        """Crop dataset. Blocked on root MDIM container."""
        self._check_not_container("crop")
        return super().crop(mask=mask, touch=touch, inplace=inplace)

    def to_crs(self, to_epsg, method="nearest neighbor", maintain_alignment=False, inplace=False):
        """Reproject dataset. Blocked on root MDIM container."""
        self._check_not_container("to_crs")
        return super().to_crs(
            to_epsg=to_epsg, method=method,
            maintain_alignment=maintain_alignment, inplace=inplace,
        )

    @classmethod
    def read_file(
        cls, path: str, read_only=True, open_as_multi_dimensional: bool = True
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
        return cls(src, access=read_only, open_as_multi_dimensional=open_as_multi_dimensional)

    @property
    def meta_data(self) -> "NetCDFMetadata":
        """Structured metadata for this NetCDF.

        Uses the GDAL Multidimensional API (groups, arrays, dimensions) when
        the file was opened with ``open_as_multi_dimensional=True``.  Falls
        back to the classic ``NETCDF_DIM_*`` parser (``dimensions.py``) when
        opened in classic mode (no root group available).

        Cached on first access. Invalidated by add_variable/remove_variable.

        Returns:
            NetCDFMetadata
        """
        if self._cached_meta_data is None:
            open_options = {
                "Open Mode": "SHARED" if self.is_subset else "MULTIDIM_RASTER"
            }
            self._cached_meta_data = get_metadata(self._raster, open_options)
        return self._cached_meta_data

    def get_all_metadata(self, open_options: Dict = None) -> "NetCDFMetadata":
        """Get full MDIM metadata with dimension overview.

        Parameters
        ----------
        open_options : dict, optional
            Open options passed to get_metadata.

        Returns
        -------
        NetCDFMetadata
            Metadata with dimension_overview populated.
        """
        metadata = get_metadata(self._raster, open_options)
        metadata.dimension_overview = self._build_dimension_overview(metadata)
        return metadata

    def _build_dimension_overview(
        self, metadata: "NetCDFMetadata" = None
    ) -> Optional[Dict[str, Any]]:
        """Create a compact snapshot of dimensions.

        Parameters
        ----------
        metadata : NetCDFMetadata, optional
            Pre-built metadata to avoid re-calling self.meta_data (which would
            cause infinite recursion if called from within meta_data).
        """
        try:
            md = metadata if metadata is not None else self.meta_data
            names = list(md.names)
            sizes = {
                name: int(md.get_dimension(name).size)
                for name in names
                if md.get_dimension(name) is not None
            }
            attrs: Dict[str, Dict[str, Any]] = {}
            values: Dict[str, List[Union[int, float, str]]] = {}

            for name in names:
                dim = md.get_dimension(name)
                if dim is not None and dim.attrs:
                    attrs[name] = {
                        str(k): (list(v) if isinstance(v, list) else v)
                        for k, v in dim.attrs.items()
                    }

            for name in names:
                try:
                    arr = self._read_variable(name)
                except Exception:
                    arr = None
                if arr is None:
                    continue
                try:
                    values[name] = [
                        _to_py_scalar(v) for v in arr.reshape(-1).tolist()
                    ]
                except Exception:
                    try:
                        values[name] = [_to_py_scalar(v) for v in list(arr)]
                    except Exception:
                        pass

            return {
                "names": names,
                "sizes": sizes,
                "attrs": attrs,
                "values": values if values else None,
            }
        except Exception:
            return None

    def get_time_variable(self, var_name = "time", time_format: str = "%Y-%m-%d"):
        """_get_time_variable."""
        time_dim = self.meta_data.get_dimension(var_name)
        if time_dim:
            units = time_dim.attrs["units"]
            func = create_time_conversion_func(units, time_format)
            time_vals = self._read_variable(var_name)
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
        """Read a variable's data as a numpy array.

        Uses the MDIM root group when available (avoids opening a new GDAL
        handle). Falls back to the classic ``NETCDF:file:var`` path.

        Args:
            var (str):
                Variable name in the dataset.

        Returns:
            np.ndarray or None
        """
        rg = self._raster.GetRootGroup()
        if rg is not None:
            # Try as an MDArray first
            try:
                md_arr = rg.OpenMDArray(var)
                if md_arr is not None:
                    return md_arr.ReadAsArray()
            except Exception:
                pass
            # Fall back to dimension indexing variable
            dim = self._get_dimension(var)
            if dim is not None:
                iv = dim.GetIndexingVariable()
                if iv is not None:
                    return iv.ReadAsArray()
            return None

        # Classic mode: open via subdataset string
        try:
            return gdal.Open(
                f"NETCDF:{self.file_name}:{var}"
            ).ReadAsArray()
        except (RuntimeError, AttributeError):
            return None

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
        """Read multidimensional array. and return it as a classical dataset"""
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
        """Extract a single variable as a classic-raster NetCDF object.

        The returned object carries origin metadata so that modified data
        can be written back via ``set_variable()``.

        Parameters
        ----------
        variable_name : str
            Name of the variable to extract.

        Returns
        -------
        NetCDF
            A subset backed by a classic dataset (bands = non-spatial dims).
        """
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

        # --- RT-4: Track variable origin for round-trip ---
        cube._parent_nc = self
        cube._source_var_name = variable_name

        if rg is not None:
            md_arr = rg.OpenMDArray(variable_name)
            if md_arr is not None:
                dims = md_arr.GetDimensions()
                cube._md_array_dims = [d.GetName() for d in dims]

                # Identify which dimension became bands (all except X/Y)
                if len(dims) > 2:
                    spatial_indices = {len(dims) - 1, len(dims) - 2}
                    band_dims = [
                        d for i, d in enumerate(dims) if i not in spatial_indices
                    ]
                    if len(band_dims) == 1:
                        cube._band_dim_name = band_dims[0].GetName()
                        iv = band_dims[0].GetIndexingVariable()
                        cube._band_dim_values = (
                            iv.ReadAsArray().tolist() if iv is not None else None
                        )
                    else:
                        cube._band_dim_name = None
                        cube._band_dim_values = None
                else:
                    cube._band_dim_name = None
                    cube._band_dim_values = None

                # Copy variable attributes
                cube._variable_attrs = {}
                try:
                    for attr in md_arr.GetAttributes():
                        cube._variable_attrs[attr.GetName()] = attr.Read()
                except Exception:
                    pass
            else:
                cube._md_array_dims = []
                cube._band_dim_name = None
                cube._band_dim_values = None
                cube._variable_attrs = {}
        else:
            cube._md_array_dims = []
            cube._band_dim_name = None
            cube._band_dim_values = None
            cube._variable_attrs = {}

        return cube

    def _replace_raster(self, new_raster: gdal.Dataset):
        """Replace the internal GDAL dataset, closing the old one if different.

        Re-derives all base-class state (geotransform, CRS, band info, etc.)
        without resetting NetCDF-specific flags (_is_md_array, _is_subset).
        """
        old = self._raster
        if old is not None and old is not new_raster:
            old.FlushCache()
        # AbstractDataset state
        self._raster = new_raster
        self._geotransform = new_raster.GetGeoTransform()
        self._cell_size = self._geotransform[1]
        self._meta_data = new_raster.GetMetadata()
        self._file_name = new_raster.GetDescription()
        self._epsg = self._get_epsg()
        self._rows = new_raster.RasterYSize
        self._columns = new_raster.RasterXSize
        self._band_count = new_raster.RasterCount
        self._block_size = [
            new_raster.GetRasterBand(i).GetBlockSize()
            for i in range(1, self._band_count + 1)
        ]
        # Dataset state
        self._no_data_value = [
            new_raster.GetRasterBand(i).GetNoDataValue()
            for i in range(1, self._band_count + 1)
        ]
        self._band_names = self._get_band_names()
        self._band_units = [
            new_raster.GetRasterBand(i).GetUnitType()
            for i in range(1, self._band_count + 1)
        ]
        # Invalidate caches
        self._cached_variables = None
        self._cached_meta_data = None

    def _invalidate_caches(self):
        """Invalidate cached variables and metadata."""
        self._cached_variables = None
        self._cached_meta_data = None

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

    def to_file(self, path: str, **kwargs) -> None:
        """Save NetCDF to disk.

        For ``.nc`` files the multidimensional structure is preserved using
        ``CreateCopy`` with the netCDF driver.  For other extensions (e.g.
        ``.tif``) the parent ``Dataset.to_file`` is used.

        Parameters
        ----------
        path : str
            Destination file path.
        **kwargs
            Forwarded to ``Dataset.to_file`` for non-NetCDF extensions.
        """
        extension = path.rsplit(".", 1)[-1].lower()
        if extension in ("nc", "nc4"):
            dst = gdal.GetDriverByName("netCDF").CreateCopy(
                path, self._raster, 0
            )
            if dst is None:
                raise RuntimeError(
                    f"Failed to save NetCDF to {path}"
                )
            dst.FlushCache()
            dst = None
        else:
            if self._is_md_array and not self._is_subset:
                raise ValueError(
                    "Cannot save a multidimensional NetCDF container as "
                    f"'{extension}'. Use .nc extension or extract a "
                    "variable first with .get_variable()."
                )
            super().to_file(path, **kwargs)

    def copy(self, path: str = None) -> "NetCDF":
        """Deep copy of this NetCDF dataset.

        Parameters
        ----------
        path : str, optional
            Destination path. If None, the copy is created in memory.

        Returns
        -------
        NetCDF
        """
        if path is None:
            path = ""
            driver = "MEM"
        else:
            driver = "netCDF"

        src = gdal.GetDriverByName(driver).CreateCopy(path, self._raster)
        if src is None:
            raise RuntimeError(
                f"Failed to copy NetCDF dataset to '{path}'"
            )
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
        except Exception:
            new_md_array.SetNoDataValueDouble(-9999)

        new_md_array.SetSpatialRef(src_mdarray.GetSpatialRef())

    @staticmethod
    def _get_or_create_dimension(
        rg: gdal.Group, dim_name: str, values: np.ndarray, dtype, dim_type=None
    ) -> gdal.Dimension:
        """Reuse an existing dimension or create a new one.

        If a dimension with ``dim_name`` already exists in the root group and
        has the same size as ``values``, it is returned directly.  Otherwise a
        new dimension (with its indexing variable) is created.

        Parameters
        ----------
        rg : gdal.Group
            The root group of the multidimensional dataset.
        dim_name : str
            Name of the dimension (e.g., "x", "y", "time").
        values : np.ndarray
            Coordinate values for this dimension.
        dtype : gdal.ExtendedDataType
            Data type for the indexing variable.
        dim_type : str or None
            GDAL dimension type constant (e.g., gdal.DIM_TYPE_HORIZONTAL_X).

        Returns
        -------
        gdal.Dimension
        """
        for existing_dim in (rg.GetDimensions() or []):
            if existing_dim.GetName() == dim_name:
                if existing_dim.GetSize() == len(values):
                    return existing_dim
                # Size mismatch — need a new dimension with a unique name
                dim_name = f"{dim_name}_{len(values)}"
                break

        return NetCDF.create_main_dimension(rg, dim_name, dtype, values)

    def set_variable(
        self,
        variable_name: str,
        dataset: "Dataset",
        band_dim_name: str = None,
        band_dim_values: list = None,
        attrs: dict = None,
    ):
        """Write a classic Dataset back as an MDArray variable in this container.

        This is the reverse of ``get_variable()``.  After performing GIS
        operations (crop, reproject, …) on a variable subset, use this method
        to store the result back into the NetCDF container.

        Parameters
        ----------
        variable_name : str
            Name for the variable in this container.  If a variable with this
            name already exists it is replaced.
        dataset : Dataset
            A classic raster dataset — typically the result of a GIS operation
            on a variable obtained via ``get_variable()``.
        band_dim_name : str, optional
            Name of the dimension that maps to bands (e.g. "time", "bands").
            Auto-detected from the dataset's ``_band_dim_name`` if it was
            obtained via ``get_variable()``.
        band_dim_values : list, optional
            Coordinate values for the band dimension.  Auto-detected from
            ``_band_dim_values`` if available.
        attrs : dict, optional
            Variable attributes to set (e.g. ``{"units": "K"}``).
            Auto-detected from ``_variable_attrs`` if available.
        """
        rg = self._raster.GetRootGroup()
        if rg is None:
            raise ValueError(
                "set_variable requires a multidimensional container. "
                "Open the file with open_as_multi_dimensional=True."
            )

        # Auto-detect from tracked origin metadata (RT-4)
        if band_dim_name is None and hasattr(dataset, "_band_dim_name"):
            band_dim_name = dataset._band_dim_name
        if band_dim_values is None and hasattr(dataset, "_band_dim_values"):
            band_dim_values = dataset._band_dim_values
        if attrs is None and hasattr(dataset, "_variable_attrs"):
            attrs = dataset._variable_attrs

        # Delete existing variable if present
        if variable_name in self.variable_names:
            rg.DeleteMDArray(variable_name)

        # Read data from the classic dataset
        arr = dataset.read_array()
        gt = dataset.geotransform
        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))

        # Build spatial dimensions from the geotransform
        x_values = np.array(
            NetCDF.get_x_lon_dimension_array(gt[0], gt[1], dataset.columns)
        )
        y_values = np.array(
            NetCDF.get_y_lat_dimension_array(gt[3], abs(gt[5]), dataset.rows)
        )
        dim_x = self._get_or_create_dimension(
            rg, "x", x_values, dtype, gdal.DIM_TYPE_HORIZONTAL_X
        )
        dim_y = self._get_or_create_dimension(
            rg, "y", y_values, dtype, gdal.DIM_TYPE_HORIZONTAL_Y
        )

        # Build band dimension if the data is 3D
        if arr.ndim == 3:
            if band_dim_name is None:
                band_dim_name = "bands"
            if band_dim_values is None:
                band_dim_values = list(range(arr.shape[0]))
            dim_band = self._get_or_create_dimension(
                rg,
                band_dim_name,
                np.array(band_dim_values, dtype=np.float64),
                dtype,
                gdal.DIM_TYPE_TEMPORAL,
            )
            md_arr = rg.CreateMDArray(
                variable_name, [dim_band, dim_y, dim_x], dtype
            )
        else:
            md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], dtype)

        # Write array data
        md_arr.Write(arr)

        # Set spatial reference (RT-7: attribute copying)
        if dataset.epsg:
            srs = Dataset._create_sr_from_epsg(dataset.epsg)
            md_arr.SetSpatialRef(srs)

        # Set no-data value
        if dataset.no_data_value and dataset.no_data_value[0] is not None:
            try:
                md_arr.SetNoDataValueDouble(float(dataset.no_data_value[0]))
            except Exception:
                pass

        # Set variable attributes (RT-7)
        if attrs:
            for key, value in attrs.items():
                try:
                    if isinstance(value, str):
                        attr = md_arr.CreateAttribute(
                            key, [], gdal.ExtendedDataType.CreateString()
                        )
                    elif isinstance(value, float):
                        attr = md_arr.CreateAttribute(
                            key, [],
                            gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                        )
                    elif isinstance(value, int):
                        attr = md_arr.CreateAttribute(
                            key, [],
                            gdal.ExtendedDataType.Create(gdal.GDT_Int32),
                        )
                    else:
                        attr = md_arr.CreateAttribute(
                            key, [], gdal.ExtendedDataType.CreateString()
                        )
                        value = str(value)
                    attr.Write(value)
                except Exception:
                    pass

        self._invalidate_caches()

    def add_variable(
        self, dataset: Union["Dataset", "NetCDF"], variable_name: str = None
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
        self._invalidate_caches()

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

        self._replace_raster(dst)
