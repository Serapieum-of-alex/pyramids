"""
netcdf module.

netcdf contains python functions to handle netcdf data. gdal class: https://gdal.org/api/index.html#python-api.
"""
from __future__ import annotations
from numbers import Number
from typing import Any
import numpy as np
from osgeo import gdal
from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.netcdf.utils import create_time_conversion_func, _to_py_scalar

from pyramids import _io
from pyramids.dataset import Dataset
from pyramids.abstract_dataset import DEFAULT_NO_DATA_VALUE
from pyramids.netcdf.metadata import get_metadata
from pyramids.netcdf.models import NetCDFMetadata
from pyramids.netcdf.dimensions import DimMetaData

class NetCDF(Dataset):
    """NetCDF.

    NetCDF class is a recursive data structure or self-referential object.
    The NetCDF class contains methods to deal with NetCDF files.

    NetCDF Creation guidelines:
        https://acdguide.github.io/Governance/create/create-basics.html
    """

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        open_as_multi_dimensional: bool = True,
    ):
        """Initialize a NetCDF dataset wrapper.

        Args:
            src: A GDAL dataset handle (either classic or multidimensional).
            access: Access mode, either ``"read_only"`` or ``"write"``.
                Defaults to ``"read_only"``.
            open_as_multi_dimensional: If True the dataset was opened with
                ``gdal.OF_MULTIDIM_RASTER`` and supports groups, MDArrays,
                and dimensions.  If False it was opened in classic raster
                mode (subdatasets, bands). Defaults to True.
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
        self._cached_variables: dict[str, NetCDF] | None = None
        self._cached_meta_data: NetCDFMetadata | None = None
        # Origin-tracking attributes set by get_variable (RT-4)
        self._parent_nc: NetCDF | None = None
        self._source_var_name: str | None = None
        self._gdal_md_arr_ref: Any = None
        self._gdal_rg_ref: Any = None
        self._md_array_dims: list[str] = []
        self._band_dim_name: str | None = None
        self._band_dim_values: list[Any] | None = None
        self._variable_attrs: dict[str, Any] = {}

    def __str__(self):
        """Return a human-readable summary of the NetCDF dataset."""
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
        """Longitude / x-coordinate values as a 1D array.

        Looks for a variable named ``"lon"`` first, then ``"x"``.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither ``lon`` nor ``x`` exists in the dataset.
        """
        lon = self._read_variable("lon")
        if lon is None:
            lon = self._read_variable("x")

        result: np.ndarray
        if lon is not None:
            result = lon.reshape(lon.size)
        else:
            result = super().lon
        return result

    @property
    def lat(self) -> np.ndarray:
        """Latitude / y-coordinate values as a 1D array.

        Looks for a variable named ``"lat"`` first, then ``"y"``.

        Returns:
            np.ndarray or None: Flattened coordinate array, or None if
            neither ``lat`` nor ``y`` exists in the dataset.
        """
        lat = self._read_variable("lat")
        if lat is None:
            lat = self._read_variable("y")

        result: np.ndarray
        if lat is not None:
            result = lat.reshape(lat.size)
        else:
            result = super().lat
        return result

    @property
    def x(self) -> np.ndarray:
        """x-coordinate/longitude."""
        # X_coordinate = upper-left corner x + index * cell size + cell-size/2
        return self.lon

    @property
    def y(self) -> np.ndarray:
        """y-coordinate/latitude."""
        # Y_coordinate = upper-left corner y - index * cell size - cell-size/2
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
    def variable_names(self) -> list[str]:
        """Names of data variables (excluding dimension coordinate arrays).

        Returns:
            list[str]: Variable names. For MDIM mode these come from
            ``GetMDArrayNames()`` minus dimension names; for classic mode
            from ``GetSubDatasets()``.
        """
        return self.get_variable_names()

    @property
    def variables(self) -> dict[str, "NetCDF"]:
        """All data variables as a dict of ``{name: NetCDF}`` subsets.

        Each value is a classic-raster NetCDF obtained via
        ``get_variable()``.  Cached on first access; invalidated by
        ``add_variable`` / ``remove_variable`` / ``set_variable``.

        Returns:
            dict[str, NetCDF]: Mapping from variable name to its subset.
        """
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
    def no_data_value(self, value: list | Number):
        """Set the no-data value that marks cells outside the domain.

        The setter only changes the ``no_data_value`` attribute; it does
        **not** modify the underlying cell values.  Use this to align the
        attribute with whatever sentinel is already stored in the cells.
        To actually rewrite cell values, use ``change_no_data_value``.

        Args:
            value: New no-data value. A single number applied to all
                bands, or a list with one value per band.
        """
        if isinstance(value, list):
            for i, val in enumerate(value):
                self._change_no_data_value_attr(i, val)
        else:
            self._change_no_data_value_attr(0, value)

    @property
    def file_name(self):
        """File path, with the ``NETCDF:"path":var`` prefix stripped if present.

        Returns:
            str: Clean file path.
        """
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    def time_stamp(self):
        """Time coordinate values parsed from the CF-compliant ``time`` variable.

        Returns:
            list[str] or None: Formatted time strings, or None if no time
            dimension with a ``units`` attribute is found.
        """
        return self.get_time_variable()

    def _check_not_container(self, operation: str):
        """Raise ValueError if this is a root MDIM container (not a variable subset)."""
        if self._is_md_array and not self._is_subset and self.band_count == 0:
            raise ValueError(
                f"Spatial operations are not supported on the NetCDF container. "
                f"Use nc.get_variable('var_name').{operation}(...) instead."
            )

    def read_array(self, band: int | None = None, window=None) -> np.ndarray:
        """Read array from the dataset.

        Raises a clear error when called on the root MDIM container
        (which has no raster bands).
        """
        self._check_not_container("read_array")
        return super().read_array(band=band, window=window)

    def crop(self, mask, touch: bool = True, inplace: bool = False):
        """Crop the dataset using a polygon or raster mask.

        Blocked on root MDIM containers — extract a variable first with
        ``get_variable()``.

        Args:
            mask: GeoDataFrame with polygon geometry, or a Dataset to use
                as a spatial mask.
            touch: If True, include cells that touch the mask boundary.
                Defaults to True.
            inplace: If True, modify this object in place. Defaults to False.

        Returns:
            Dataset or None: Cropped dataset (or None if inplace=True).

        Raises:
            ValueError: If called on a root MDIM container.
        """
        self._check_not_container("crop")
        return super().crop(mask=mask, touch=touch, inplace=inplace)

    def to_crs(
        self, to_epsg, method="nearest neighbor",
        maintain_alignment=False, inplace=False,
    ):
        """Reproject the dataset to a different coordinate reference system.

        Blocked on root MDIM containers — extract a variable first with
        ``get_variable()``.

        Args:
            to_epsg: Target EPSG code (e.g., 4326, 32637).
            method: Resampling method. Defaults to ``"nearest neighbor"``.
            maintain_alignment: If True, keep the same number of rows and
                columns after reprojection. Defaults to False.
            inplace: If True, modify this object in place. Defaults to False.

        Returns:
            Dataset or None: Reprojected dataset (or None if inplace=True).

        Raises:
            ValueError: If called on a root MDIM container.
        """
        self._check_not_container("to_crs")
        return super().to_crs(
            to_epsg=to_epsg, method=method,
            maintain_alignment=maintain_alignment, inplace=inplace,
        )

    @classmethod
    def read_file(  # type: ignore[override]
        cls,
        path: str,
        read_only=True,
        open_as_multi_dimensional: bool = True,
    ) -> "NetCDF":
        """Open a NetCDF file from disk.

        Args:
            path: Path to the ``.nc`` file.
            read_only: If True, open in read-only mode. Set to False for
                write access. Defaults to True.
            open_as_multi_dimensional: If True, open with
                ``gdal.OF_MULTIDIM_RASTER`` to access the full group /
                dimension / variable hierarchy.  If False, open in classic
                raster mode where each variable is a subdataset.
                Defaults to True.

        Returns:
            NetCDF: The opened dataset.
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

    @meta_data.setter
    def meta_data(self, value: dict[str, str] | NetCDFMetadata) -> None:
        """Set metadata on this NetCDF dataset."""
        if isinstance(value, dict):
            for key, val in value.items():
                self._raster.SetMetadataItem(key, val)
        else:
            self._cached_meta_data = value

    def get_all_metadata(self, open_options: dict | None = None) -> "NetCDFMetadata":
        """Get full MDIM metadata with a dimension overview snapshot.

        Unlike ``meta_data`` (which is cached), this always re-traverses
        the GDAL multidimensional structure and populates the
        ``dimension_overview`` field with coordinate values and sizes.

        Args:
            open_options: Driver-specific open options forwarded to
                ``get_metadata()``. Defaults to None.

        Returns:
            NetCDFMetadata: Metadata with ``dimension_overview`` populated.
        """
        metadata = get_metadata(self._raster, open_options)
        metadata.dimension_overview = self._build_dimension_overview(metadata)
        return metadata

    def _build_dimension_overview(
        self, metadata: NetCDFMetadata | None = None
    ) -> dict[str, Any] | None:
        """Create a compact snapshot of dimensions.

        Args:
            metadata: Pre-built metadata to avoid re-calling
                ``self.meta_data`` (which would cause infinite recursion
                if called from within ``meta_data``). Defaults to None.

        Returns:
            dict or None: Dictionary with ``names``, ``sizes``, ``attrs``,
                and ``values`` keys, or None on failure.
        """
        md = metadata if metadata is not None else self.meta_data
        names = list(md.names)
        sizes: dict[str, int] = {}

        for name in names:
            dim_info = md.get_dimension(name)
            if dim_info is not None:
                sizes[name] = int(dim_info.size)
        attrs: dict[str, dict[str, Any]] = {}
        values: dict[str, list[int | float | str]] = {}

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
            except (RuntimeError, AttributeError) as e:
                self.logger.debug(
                    f"Could not read dimension variable '{name}': {e}"
                )
                arr = None
            if arr is None:
                continue
            try:
                values[name] = [
                    _to_py_scalar(v) for v in arr.reshape(-1).tolist()
                ]
            except (ValueError, TypeError, AttributeError) as e:
                self.logger.debug(
                    f"Could not reshape dimension '{name}': {e}"
                )
                try:
                    values[name] = [_to_py_scalar(v) for v in list(arr)]
                except (ValueError, TypeError) as e:
                    self.logger.debug(
                        f"Could not convert dimension '{name}': {e}"
                    )

        return {
            "names": names,
            "sizes": sizes,
            "attrs": attrs,
            "values": values if values else None,
        }

    def get_time_variable(
        self, var_name="time", time_format: str = "%Y-%m-%d"
    ):
        """Parse the time coordinate variable into formatted date strings.

        Reads the ``units`` attribute (e.g., ``"days since 1979-01-01"``)
        from the dimension metadata and converts raw numeric values to
        human-readable date strings.

        Args:
            var_name: Name of the time dimension / variable.
                Defaults to ``"time"``.
            time_format: strftime format for the output strings.
                Defaults to ``"%Y-%m-%d"``.

        Returns:
            list[str] or None: Formatted time strings, or None if the
            time dimension is not found or lacks a ``units`` attribute.
        """
        time_stamp = None
        time_dim = self.meta_data.get_dimension(var_name)
        if time_dim is not None:
            units = time_dim.attrs.get("units")
            if units is not None:
                time_vals = self._read_variable(var_name)
                if time_vals is not None:
                    func = create_time_conversion_func(units, time_format)
                    time_stamp = list(map(func, time_vals.reshape(-1)))
        return time_stamp

    def _get_dimension_names(self) -> list[str] | None:
        rg = self._raster.GetRootGroup()
        if rg is not None:
            dims = rg.GetDimensions()
            dims_names: list[str] | None = [dim.GetName() for dim in dims]
        else:
            dims_names = None
        return dims_names

    @property
    def dimension_names(self) -> list[str] | None:
        """Names of all dimensions in the root group (e.g., ``["x", "y", "time"]``).

        Returns:
            list[str] or None: Dimension names, or None if no root group
            is available (classic mode).
        """
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

    def _read_variable(self, var: str) -> np.ndarray | None:
        """Read a variable's data as a numpy array.

        Uses the MDIM root group when available (avoids opening a new GDAL
        handle). Falls back to the classic ``NETCDF:file:var`` path.

        Args:
            var: Variable name in the dataset.

        Returns:
            np.ndarray or None: The variable data, or None if the
                variable is not found.
        """
        result = None
        rg = self._raster.GetRootGroup()
        if rg is not None:
            # Try as an MDArray first
            try:
                md_arr = rg.OpenMDArray(var)
                if md_arr is not None:
                    result = md_arr.ReadAsArray()
            except Exception:
                pass
            # Fall back to dimension indexing variable
            if result is None:
                dim = self._get_dimension(var)
                if dim is not None:
                    iv = dim.GetIndexingVariable()
                    if iv is not None:
                        result = iv.ReadAsArray()
        else:
            # Classic mode: open via subdataset string
            try:
                ds = gdal.Open(f"NETCDF:{self.file_name}:{var}")
                if ds is not None:
                    result = ds.ReadAsArray()
                ds = None
            except (RuntimeError, AttributeError):
                pass
        return result

    def get_variable_names(self) -> list[str]:
        """Return names of data variables, excluding dimension coordinates.

        In MDIM mode, queries ``GetMDArrayNames()`` and filters out arrays
        that are also dimensions (x, y, time, etc.).  In classic mode,
        parses subdataset metadata.

        Returns:
            list[str]: Variable names (e.g., ``["temperature", "precipitation"]``).
        """
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

    def _read_md_array(self, variable_name: str):
        """Convert an MDArray to a classic GDAL dataset via AsClassicDataset.

        The last two dimensions become X (columns) and Y (rows); all
        remaining dimensions are flattened into bands.

        If the Y dimension is stored south-to-north (positive Y pixel
        size), it is reversed via ``MDArray.GetView()`` **before** the
        conversion.  This is a lazy, zero-copy operation — GDAL handles
        the reversed indexing internally without reading the whole array.

        Returns a tuple ``(classic_dataset, md_array, root_group)`` so
        callers can keep the GDAL objects alive.  ``AsClassicDataset``
        returns a **view** whose C++ backing depends on the MDArray and
        root group; if the Python SWIG wrappers for those are garbage-
        collected the view becomes a dangling pointer (segfault on
        Windows).
        """
        rg = self._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        dtype = md_arr.GetDataType()
        dims = md_arr.GetDimensions()

        if len(dims) == 1:
            if dtype.GetClass() == gdal.GEDTC_STRING:
                return md_arr, md_arr, rg
            src = md_arr.AsClassicDataset(0, 1, rg)
            return src, md_arr, rg

        iXDim = len(dims) - 1
        iYDim = len(dims) - 2

        # First pass: check if Y orientation needs flipping.
        src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        if src.GetGeoTransform()[5] > 0:
            # Positive Y pixel size = south-to-north (NetCDF convention).
            # Use GetView to reverse the Y dimension — this is lazy and
            # zero-copy; GDAL handles reversed indexing internally.
            slices = ",".join(
                "::-1" if i == iYDim else ":" for i in range(len(dims))
            )
            md_arr = md_arr.GetView(f"[{slices}]")
            src = md_arr.AsClassicDataset(iXDim, iYDim, rg)

        return src, md_arr, rg

    @staticmethod
    def _needs_y_flip(src: gdal.Dataset) -> bool:
        """Check if a classic dataset has south-to-north Y orientation."""
        return bool(src.GetGeoTransform()[5] > 0)

    def get_variable(self, variable_name: str) -> "NetCDF":
        """Extract a single variable as a classic-raster NetCDF object.

        The returned object carries origin metadata so that modified data
        can be written back via ``set_variable()``.

        Args:
            variable_name: Name of the variable to extract.

        Returns:
            NetCDF: A subset backed by a classic dataset where
                non-spatial dimensions are mapped to bands.

        Raises:
            ValueError: If ``variable_name`` is not present in the dataset.
        """
        if variable_name not in self.variable_names:
            raise ValueError(
                f"{variable_name} is not a valid variable name in {self.variable_names}"
            )

        prefix = self.driver_type.upper()
        rg = self._raster.GetRootGroup()
        md_arr_ref = None
        rg_ref = None

        if prefix == "MEMORY" or rg is not None:
            src, md_arr_ref, rg_ref = self._read_md_array(variable_name)
            if isinstance(src, gdal.Dataset):
                cube = NetCDF(src)
                cube._is_md_array = True
                # _read_md_array uses GetView to flip the data lazily,
                # and GDAL usually corrects the geotransform.  But when
                # the Y dimension has no indexing variable (e.g. WRF
                # "south_north"), the geotransform may still be wrong.
                # Fix it on the wrapper object (no data copy).
                gt = cube._geotransform
                if gt[5] > 0:
                    cube._geotransform = (
                        gt[0], gt[1], gt[2],
                        gt[3] + gt[5] * cube._rows,
                        gt[4], -gt[5],
                    )
                    cube._cell_size = abs(gt[1])
            else:
                cube = src
            # Keep GDAL SWIG references alive — AsClassicDataset returns a
            # view whose C++ backing is owned by the MDArray/root group.
            # Without these the view becomes a dangling pointer on Windows.
            cube._gdal_md_arr_ref = md_arr_ref
            cube._gdal_rg_ref = rg_ref
        else:
            src = gdal.Open(f"{prefix}:{self.file_name}:{variable_name}")
            if src is None:
                raise ValueError(
                    f"Could not open variable '{variable_name}' via "
                    f"'{prefix}:{self.file_name}:{variable_name}'"
                )
            cube = NetCDF(src)
            cube._is_md_array = False

        cube._is_subset = True

        # --- RT-4: Track variable origin for round-trip ---
        cube._parent_nc = self
        cube._source_var_name = variable_name

        md_arr = md_arr_ref if rg is not None else None
        if rg is not None:
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
                        try:
                            cube._band_dim_values = (
                                iv.ReadAsArray().tolist()
                                if iv is not None
                                else None
                            )
                        except RuntimeError:
                            # String-typed indexing variables (e.g. WRF
                            # "Times") can't be read via ReadAsArray in
                            # GDAL SWIG bindings — fall back to indices.
                            cube._band_dim_values = list(
                                range(band_dims[0].GetSize())
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
        """Whether this object represents a single-variable subset.

        Returns:
            bool: True if the dataset is a variable subset extracted
                via ``get_variable()``.
        """
        return self._is_subset

    @property
    def is_md_array(self):
        """Whether this dataset was opened in multidimensional mode.

        Returns:
            bool: True if the dataset was opened with
                ``gdal.OF_MULTIDIM_RASTER`` and supports groups,
                MDArrays, and dimensions.
        """
        return self._is_md_array

    def to_file(  # type: ignore[override]
        self,
        path: str,
        **kwargs,
    ) -> None:
        """Save the dataset to disk.

        For ``.nc`` / ``.nc4`` files the full multidimensional structure
        (groups, dimensions, variables, attributes) is preserved via
        ``CreateCopy`` with the netCDF driver.  For other extensions
        (e.g. ``.tif``), the parent ``Dataset.to_file`` is used — but only
        on variable subsets, not on root MDIM containers.

        Args:
            path: Destination file path. The extension determines the
                output driver (``.nc`` → netCDF, ``.tif`` → GeoTIFF, etc.).
            **kwargs: Forwarded to ``Dataset.to_file`` for non-NetCDF
                extensions (e.g. ``tile_length``, ``creation_options``).

        Raises:
            RuntimeError: If the netCDF ``CreateCopy`` call fails.
            ValueError: If a root MDIM container is saved to a non-NC
                extension (use ``.nc`` or extract a variable first).
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

    def copy(self, path: str | None = None) -> "NetCDF":
        """Create a deep copy of this NetCDF dataset.

        Args:
            path: Destination file path. If None, the copy is created
                in memory using the MEM driver. Defaults to None.

        Returns:
            NetCDF: A new NetCDF object with copied data.

        Raises:
            RuntimeError: If ``CreateCopy`` fails.
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
        """Create a NetCDF dimension with an indexing variable.

        The dimension type is inferred from ``dim_name``:
        ``y``/``lat``/``latitude`` → horizontal Y,
        ``x``/``lon``/``longitude`` → horizontal X,
        ``bands``/``time`` → temporal.

        The dimension is registered in the group together with a
        matching MDArray that stores the coordinate values.

        Args:
            group: Root group (or sub-group) of the multidimensional
                dataset.
            dim_name: Name of the dimension to create.
            dtype: GDAL ``ExtendedDataType`` for the indexing variable.
            values: Coordinate values for the dimension.

        Returns:
            gdal.Dimension: The newly created dimension.
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
    def create_from_array(  # type: ignore[override]
        cls,
        arr: np.ndarray,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int = 4326,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | None = None,
        variable_name: str | None = None,
        extra_dim_name: str = "time",
        extra_dim_values: list | None = None,
        top_left_corner: tuple[float, float] | None = None,
        cell_size: int | float | None = None,
    ) -> NetCDF:
        """Create a NetCDF dataset from a NumPy array and geotransform.

        For 3-D arrays the first axis is treated as a non-spatial
        dimension (time, level, depth, etc.) whose name and coordinate
        values are controlled by ``extra_dim_name`` and
        ``extra_dim_values``.

        The driver is inferred from ``path``: if ``path`` is ``None``
        the dataset is created in memory (MEM driver); if a path is
        provided the netCDF driver writes to disk.

        Args:
            arr: 2-D ``(rows, cols)`` or 3-D
                ``(extra_dim, rows, cols)`` NumPy array.
            geo: Geotransform tuple ``(x_min, pixel_size, rotation,
                y_max, rotation, pixel_size)``.
            epsg: EPSG code for the spatial reference.
                Defaults to 4326.
            no_data_value: Sentinel value for cells outside the
                domain. Defaults to DEFAULT_NO_DATA_VALUE.
            path: Output file path. If ``None``, the dataset is
                created in memory. Defaults to None.
            variable_name: Name of the data variable in the NetCDF
                file. Defaults to ``"data"``.
            extra_dim_name: Name of the non-spatial dimension for 3-D
                arrays (e.g. ``"time"``, ``"level"``, ``"depth"``).
                Ignored for 2-D arrays. Defaults to ``"time"``.
            extra_dim_values: Coordinate values for the non-spatial
                dimension. Must have length ``arr.shape[0]`` for 3-D
                arrays. Defaults to ``[0, 1, 2, ..., N-1]``.
            top_left_corner: ``(x, y)`` of the top-left corner. Used
                with ``cell_size`` to build ``geo`` when ``geo`` is
                not provided. Defaults to None.
            cell_size: Pixel size. Used with ``top_left_corner`` to
                build ``geo``. Defaults to None.

        Returns:
            NetCDF: The newly created NetCDF dataset.
        """
        if geo is None and top_left_corner is not None and cell_size is not None:
            geo = (
                top_left_corner[0], cell_size, 0,
                top_left_corner[1], 0, -cell_size,
            )
        if geo is None:
            raise ValueError(
                "Either 'geo' or both 'top_left_corner' and "
                "'cell_size' must be provided."
            )

        if arr.ndim == 2:
            rows = int(arr.shape[0])
            cols = int(arr.shape[1])
        else:
            rows = int(arr.shape[1])
            cols = int(arr.shape[2])

        if extra_dim_values is None and arr.ndim == 3:
            extra_dim_values = list(range(arr.shape[0]))

        if arr.ndim == 3:
            DimMetaData(
                name=extra_dim_name,
                size=arr.shape[0],
                values=extra_dim_values,
            )

        if variable_name is None:
            variable_name = "data"

        dst_ds = cls._create_netcdf_from_array(
            arr,
            variable_name,
            cols,
            rows,
            extra_dim_name,
            extra_dim_values,
            geo,
            epsg,
            no_data_value,
            path=path,
        )
        result = cls(dst_ds)

        return result

    @staticmethod
    def _create_netcdf_from_array(
        arr: np.ndarray,
        variable_name: str,
        cols: int,
        rows: int,
        extra_dim_name: str = "time",
        extra_dim_values: list | None = None,
        geo: tuple[float, float, float, float, float, float] | None = None,
        epsg: str | int | None = None,
        no_data_value: Any | list = DEFAULT_NO_DATA_VALUE,
        path: str | None = None,
    ) -> gdal.Dataset:
        """Build a multidimensional GDAL dataset from an array.

        The driver is inferred from ``path``: ``None`` → MEM (in-memory),
        otherwise the netCDF driver writes to disk.

        Args:
            arr: 2-D ``(rows, cols)`` or 3-D
                ``(extra_dim, rows, cols)`` NumPy array.
            variable_name: Name of the data variable.
            cols: Number of columns.
            rows: Number of rows.
            extra_dim_name: Name of the non-spatial dimension
                (e.g. ``"time"``, ``"level"``). Defaults to ``"time"``.
            extra_dim_values: Coordinate values for the non-spatial
                dimension. Defaults to None.
            geo: Geotransform tuple. Defaults to None.
            epsg: EPSG code. Defaults to None.
            no_data_value: No-data sentinel. Defaults to
                DEFAULT_NO_DATA_VALUE.
            path: Output file path. If None, created in memory.
                Defaults to None.

        Returns:
            gdal.Dataset: The created multidimensional GDAL dataset.
        """
        if variable_name is None:
            raise ValueError("Variable_name cannot be None")
        if geo is None:
            raise ValueError("geo cannot be None")

        dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        x_dim_values = NetCDF.get_x_lon_dimension_array(geo[0], geo[1], cols)
        y_dim_values = NetCDF.get_y_lat_dimension_array(geo[3], geo[1], rows)

        if path is not None:
            driver_type = "netCDF"
        else:
            driver_type = "MEM"
            path = "netcdf"
        src = gdal.GetDriverByName(driver_type).CreateMultiDimensional(path)
        rg = src.GetRootGroup()

        dim_x = NetCDF.create_main_dimension(rg, "x", dtype, np.array(x_dim_values))
        dim_y = NetCDF.create_main_dimension(rg, "y", dtype, np.array(y_dim_values))
        if arr.ndim == 3:
            extra_dim = NetCDF.create_main_dimension(
                rg, extra_dim_name, dtype, np.array(extra_dim_values)
            )
            md_arr = rg.CreateMDArray(
                variable_name, [extra_dim, dim_y, dim_x], dtype
            )
        else:
            md_arr = rg.CreateMDArray(variable_name, [dim_y, dim_x], dtype)

        md_arr.Write(arr)
        md_arr.SetNoDataValueDouble(no_data_value)
        if epsg is None:
            raise ValueError("epsg cannot be None")
        srse = Dataset._create_sr_from_epsg(epsg=int(epsg))
        md_arr.SetSpatialRef(srse)

        return src

    @staticmethod
    def _add_md_array_to_group(dst_group, var_name, src_mdarray):
        """Copy an MDArray from one group to another, preserving data and metadata."""
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

        If a dimension with ``dim_name`` already exists in the root group
        and has the same size as ``values``, it is returned directly.
        On size mismatch, a new dimension with a ``_{size}`` suffix is
        created to avoid conflicts.

        Args:
            rg: The root group of the multidimensional dataset.
            dim_name: Name of the dimension (e.g., ``"x"``, ``"time"``).
            values: Coordinate values for this dimension.
            dtype: GDAL ``ExtendedDataType`` for the indexing variable.
            dim_type: GDAL dimension type constant (e.g.,
                ``gdal.DIM_TYPE_HORIZONTAL_X``). Defaults to None.

        Returns:
            gdal.Dimension: The reused or newly created dimension.
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
        band_dim_name: str | None = None,
        band_dim_values: list | None = None,
        attrs: dict | None = None,
    ):
        """Write a classic Dataset back as an MDArray variable in this container.

        This is the reverse of ``get_variable()``.  After performing GIS
        operations (crop, reproject, etc.) on a variable subset, use this
        method to store the result back into the NetCDF container.

        Args:
            variable_name: Name for the variable in this container.  If a
                variable with this name already exists it is replaced.
            dataset: A classic raster dataset, typically the result of a
                GIS operation on a variable obtained via ``get_variable()``.
            band_dim_name: Name of the dimension that maps to bands
                (e.g. ``"time"``, ``"bands"``).  Auto-detected from the
                dataset's ``_band_dim_name`` attribute when available.
                Defaults to None.
            band_dim_values: Coordinate values for the band dimension.
                Auto-detected from ``_band_dim_values`` when available.
                Defaults to None.
            attrs: Variable attributes to set (e.g. ``{"units": "K"}``).
                Auto-detected from ``_variable_attrs`` when available.
                Defaults to None.

        Raises:
            ValueError: If called on a dataset without a root group
                (not opened in multidimensional mode).
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
        gt: tuple[float, float, float, float, float, float] = (
            dataset.geotransform
        )
        data_dtype = gdal.ExtendedDataType.Create(numpy_to_gdal_dtype(arr))
        # Coordinate dimensions must always be float64 to avoid truncation
        # when the data array is integer (e.g., classified rasters).
        coord_dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Build spatial dimensions from the geotransform
        x_values = np.array(
            NetCDF.get_x_lon_dimension_array(gt[0], gt[1], dataset.columns)
        )
        y_values = np.array(
            NetCDF.get_y_lat_dimension_array(gt[3], abs(gt[5]), dataset.rows)
        )
        dim_x = self._get_or_create_dimension(
            rg, "x", x_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_X
        )
        dim_y = self._get_or_create_dimension(
            rg, "y", y_values, coord_dtype, gdal.DIM_TYPE_HORIZONTAL_Y
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
                coord_dtype,
                gdal.DIM_TYPE_TEMPORAL,
            )
            md_arr = rg.CreateMDArray(
                variable_name, [dim_band, dim_y, dim_x], data_dtype
            )
        else:
            md_arr = rg.CreateMDArray(
                variable_name, [dim_y, dim_x], data_dtype
            )

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
        self, dataset: Dataset | NetCDF, variable_name: str | None = None
    ):
        """Copy MDArray variables from another NetCDF into this container.

        Args:
            dataset: Source NetCDF dataset whose variables will be copied.
                Must have a root group (opened in MDIM mode).
            variable_name: Specific variable name(s) to copy. If None, all
                variables from the source are copied. If a variable with
                the same name already exists, it is renamed with a
                ``"-new"`` suffix.
        """
        src_rg = self._raster.GetRootGroup()
        var_rg = dataset._raster.GetRootGroup()
        names_to_copy: list[str]
        if variable_name is not None:
            names_to_copy = [variable_name]
        elif isinstance(dataset, NetCDF):
            names_to_copy = dataset.variable_names
        else:
            names_to_copy = []

        for var in names_to_copy:
            md_arr = var_rg.OpenMDArray(var)
            # If the variable name already exists in the destination dataset,
            # use a suffixed name to avoid overwriting the original.
            target_name = f"{var}-new" if var in self.variable_names else var
            self._add_md_array_to_group(src_rg, target_name, md_arr)
        self._invalidate_caches()

    def remove_variable(self, variable_name: str):
        """Delete a variable from this container.

        If the dataset is backed by a file on disk, a MEM copy is made first
        so that the on-disk file is not modified.  The internal raster
        reference is replaced with the modified copy.

        Args:
            variable_name: Name of the variable to remove.
        """
        if self.driver_type == "memory":
            # first copy the dataset to memory
            dst = self._raster
        else:
            dst = gdal.GetDriverByName("MEM").CreateCopy("", self._raster, 0)

        rg = dst.GetRootGroup()
        rg.DeleteMDArray(variable_name)

        self._replace_raster(dst)
