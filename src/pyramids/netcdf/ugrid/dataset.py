"""UgridDataset — top-level container for UGRID NetCDF mesh data.

Combines mesh topology, data variables, metadata, and provides
the user-facing API for reading, inspecting, and operating on
unstructured mesh data.
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

if TYPE_CHECKING:
    from cleopatra.glyphs.gridded.array_glyph import PointOverlay
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling
from osgeo import gdal
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box

from pyramids.base._errors import AlignmentError
from pyramids.base._reductions import (
    COUNTING_REDUCERS,
    FLAG_NO_DATA,
    INTERP_METHODS,
    REDUCERS,
    WEIGHTED_HOWS,
    gaps_as_nan,
    interpolated,
    pushed,
    reduce_axis,
    reduce_by_label,
    shifted,
    weighted_statistic,
    window_members,
)
from pyramids.base._summary import DEFAULT_METRICS, variable_summary
from pyramids.base.crs import crs_from_user_input, crs_spec, sr_from_epsg
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset._plot_helpers import mesh_render as _mesh_render
from pyramids.dataset._plot_helpers import nonnull_group_kwargs as _nonnull_group_kwargs
from pyramids.feature import FeatureCollection
from pyramids.netcdf._mdim import open_mdarray
from pyramids.netcdf.cf import write_global_attributes
from pyramids.netcdf.ugrid import construct as _construct
from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid._facades import _UgridFacades
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.fill import _location_coords
from pyramids.netcdf.ugrid.interpolation import mesh_to_grid
from pyramids.netcdf.ugrid.io import (
    parse_ugrid_topology,
    write_ugrid_data_variable,
    write_ugrid_topology,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import (
    DEFAULT_MESH_NAME,
    MeshTopologyInfo,
    MeshVariable,
    UgridMetadata,
    read_mesh_variable,
)
from pyramids.netcdf.ugrid.spatial import (
    MeshSpatialIndex,
    clip_mesh,
    subset_by_bounds,
    subset_mesh_by_faces,
)
from pyramids.netcdf.utils import (
    _dtype_to_str,
    _read_attributes,
    read_cf_attributes,
)


class _joins_meshes:  # noqa: N801
    """Descriptor making a join callable on the class and on a dataset.

    ``UgridDataset.concat([a, b])`` is the canonical spelling, but ``a.concat([b])`` reads
    naturally as "join this dataset with that one". A plain method only supports the second and
    a ``classmethod`` only the first — and as a classmethod the instance it was reached through
    would be invisible, silently dropping ``a``. This threads the receiver in instead: an
    instance call passes the receiver as ``self``, a class call promotes the first element of
    the list to ``self`` and joins the rest onto it, so both spellings answer the same dataset.
    """

    def __init__(self, function: Any) -> None:
        """Wrap the join.

        Args:
            function: The join, taking ``(self, objs, ...)``.
        """
        self._function = function
        self.__doc__ = function.__doc__

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """Bind the join to the class or to a dataset.

        Args:
            instance: The dataset it was reached through, or ``None`` for a class call.
            owner: The owning class.

        Returns:
            The join, with the receiver threaded in.
        """
        function = self._function
        if instance is not None:
            # A real bound method, so the join keeps the signature tooling reads (`compat=`
            # stays visible) and the equality an ordinary method has -- `a.concat ==
            # a.concat` holds, which a freshly built closure could not give.
            return MethodType(function, instance)

        target = owner if owner is not None else UgridDataset

        @functools.wraps(function)
        def call(objs: Any, *args: Any, **kwargs: Any) -> Any:
            # A bare dataset is iterable over its variable names, so `list(objs)` would turn
            # `UgridDataset.concat(a)` into a list of strings. Promote a single dataset to a
            # one-element list, exactly as the instance path already accepts `a.concat(b)`.
            parts = [objs] if isinstance(objs, target) else list(objs)
            if not parts:
                raise ValueError(
                    f"{function.__name__}: needs at least one dataset to join."
                )
            return function(parts[0], parts[1:], *args, **kwargs)

        return call


class UgridDataset(_UgridFacades):
    """Container for UGRID NetCDF mesh data.

    Combines mesh topology, data variables, and global attributes
    into a single object with GIS-aware operations. Does NOT inherit
    from Dataset or RasterBase — the raster paradigm does not
    apply to unstructured meshes.

    Attributes:
        _mesh: Mesh2d topology instance.
        _data_variables: Mapping of variable name to MeshVariable.
        _global_attributes: File-level NetCDF attributes.
        _topology_info: Parsed UGRID topology metadata.
        _crs_wkt: CRS in WKT format.
        _file_name: Source file path, if read from disk.
    """

    def __init__(
        self,
        mesh: Mesh2d,
        data_variables: dict[str, MeshVariable],
        global_attributes: dict[str, Any],
        topology_info: MeshTopologyInfo | None = None,
        crs_wkt: str | None = None,
        file_name: str | None = None,
    ):
        self._mesh = mesh
        self._data_variables = data_variables
        self._global_attributes = global_attributes
        self._topology_info = topology_info
        self._crs_wkt = crs_wkt
        self._file_name = file_name
        self._cached_crs: Any = None

    @classmethod
    def read_file(cls, path: str | Path, mesh_name: str | None = None) -> UgridDataset:
        """Open a UGRID NetCDF file.

        Automatically detects mesh topology, separates data variables
        from topology/coordinate variables, and builds the mesh.

        Args:
            path: Path to the .nc file. A GDAL virtual-filesystem path
                (``/vsimem/``, ``/vsizip/``, ``/vsicurl/``, …) is passed straight to GDAL
                rather than checked on disk.
            mesh_name: Which topology to read, for a file declaring more than one (see
                :meth:`mesh_names`). ``None`` (the default) takes the first.

        Returns:
            UgridDataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: GDAL cannot open the file. GDAL exceptions are enabled
                package-wide, so an unreadable file surfaces as GDAL's own error rather
                than a `ValueError` from here.
            ValueError: If no UGRID topology is found in the file, or ``mesh_name`` names
                a topology the file does not declare.
        """
        # A GDAL virtual-filesystem path (/vsimem/, /vsizip/, /vsicurl/, …) is not an OS path:
        # `Path` would mangle the separators and `exists()` would always be False, so hand
        # those straight to GDAL and let it report. `from_bytes` reads through /vsimem/.
        target = str(path)
        if target.startswith("/vsi"):
            opened_name = target
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            opened_name = str(path)

        ds = gdal.OpenEx(
            opened_name,
            gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
        )
        if ds is None:
            raise ValueError(f"GDAL cannot open file: {path}")

        rg = ds.GetRootGroup()
        if rg is None:
            raise ValueError(f"Cannot get root group from: {path}")

        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise ValueError(f"No UGRID mesh topology found in: {path}")

        # A file may declare several meshes. The default stays the first one — changing that
        # would silently alter every existing caller — but `mesh_name` picks any of them, and
        # `read_all` reads them all. Without either, a multi-mesh file quietly dropped the rest.
        if mesh_name is None:
            topo_info = topologies[0]
        else:
            available = [topo.mesh_name for topo in topologies]
            chosen = [topo for topo in topologies if topo.mesh_name == mesh_name]
            if not chosen:
                raise ValueError(
                    f"No UGRID topology named {mesh_name!r} in {path}; it declares "
                    f"{available}."
                )
            topo_info = chosen[0]
        mesh = Mesh2d.from_gdal_group(rg, topo_info)

        # Resolve to an absolute path before threading it into the lazy variable loaders:
        # data reads are deferred to first `.data` access (PERF-3), which re-opens the file.
        # A relative path would break that deferred open if the process changed directory in
        # the meantime; the old eager read was immune because it read while still in `read_file`.
        # A virtual-filesystem name is already absolute and has no OS path to resolve.
        source_name = (
            opened_name if opened_name.startswith("/vsi") else str(Path(path).resolve())
        )
        data_variables = _read_data_variables(rg, topo_info, source_name)

        global_attrs = _read_attributes(rg)

        ds = None

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=global_attrs,
            topology_info=topo_info,
            crs_wkt=topo_info.crs_wkt,
            file_name=str(path),
        )
        return result

    @property
    def mesh(self) -> Mesh2d:
        """The mesh topology."""
        return self._mesh

    @property
    def mesh_name(self) -> str:
        """Name of the mesh topology variable."""
        result = (
            self._topology_info.mesh_name if self._topology_info else DEFAULT_MESH_NAME
        )
        return result

    @property
    def data_variable_names(self) -> list[str]:
        """Names of all data variables."""
        result = list(self._data_variables.keys())
        return result

    @property
    def crs(self) -> CRS | None:
        """CRS as a pyproj.CRS object, or None. Cached after first access."""
        if self._cached_crs is None and self._crs_wkt is not None:
            try:
                self._cached_crs = CRS.from_wkt(self._crs_wkt)
            except Exception:  # nosec B110 - best-effort CRS parse; falls back to None
                pass
        return cast("CRS | None", self._cached_crs)

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or None."""
        crs = self.crs
        result = crs.to_epsg() if crs is not None else None
        return result

    @property
    def crs_wkt(self) -> str | None:
        """The mesh CRS as WKT, or `None` when there is none that parses.

        Gated on a successful parse rather than returning `_crs_wkt` verbatim.
        `_crs_wkt` is copied straight out of the file and :attr:`crs` tolerates
        an unparseable one on purpose, swallowing the error and reporting no
        CRS. Handing the raw string to a caller would move that parse out from
        under the handler, so a malformed WKT would start raising downstream
        instead of being reported as "no CRS" here.
        """
        return self._crs_wkt if self.crs is not None else None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Mesh bounding box as (xmin, ymin, xmax, ymax)."""
        return self._mesh.bounds

    @property
    def global_attributes(self) -> dict[str, Any]:
        """File-level NetCDF attributes."""
        return self._global_attributes

    @property
    def n_node(self) -> int:
        """Number of mesh nodes."""
        return self._mesh.n_node

    @property
    def n_face(self) -> int:
        """Number of mesh faces."""
        return self._mesh.n_face

    @property
    def n_edge(self) -> int:
        """Number of mesh edges."""
        return self._mesh.n_edge

    def get_data(self, variable_name: str) -> MeshVariable:
        """Get a data variable by name.

        Args:
            variable_name: Name of the data variable.

        Returns:
            MeshVariable instance.

        Raises:
            KeyError: If the variable name is not found.
        """
        if variable_name not in self._data_variables:
            raise KeyError(
                f"Variable '{variable_name}' not found. "
                f"Available: {self.data_variable_names}"
            )
        result = self._data_variables[variable_name]
        return result

    def __getitem__(self, key: str) -> MeshVariable:
        """Get a data variable by name using bracket notation."""
        return self.get_data(key)

    def __contains__(self, key: str) -> bool:
        """True when ``key`` is a data variable of this dataset."""
        return key in self._data_variables

    def __len__(self) -> int:
        """The number of data variables."""
        return len(self._data_variables)

    def __iter__(self) -> Any:
        """Iterate over the data variable names."""
        return iter(self._data_variables)

    def keys(self) -> list[str]:
        """The data variable names (read-only mapping surface)."""
        return list(self._data_variables.keys())

    def values(self) -> list[MeshVariable]:
        """The data variables (read-only mapping surface)."""
        return list(self._data_variables.values())

    def items(self) -> list[tuple[str, MeshVariable]]:
        """The ``(name, variable)`` pairs (read-only mapping surface)."""
        return list(self._data_variables.items())

    def get(self, key: str, default: Any = None) -> MeshVariable | Any:
        """The variable ``key``, or ``default`` when it is absent.

        Args:
            key: The variable name to look up.
            default: What to answer when no variable is named ``key``. Defaults to ``None``.

        Returns:
            MeshVariable | Any: The variable, or ``default``.
        """
        return self._data_variables.get(key, default)

    @property
    def data_vars(self) -> dict[str, MeshVariable]:
        """A copy of the ``{name: variable}`` mapping.

        A fresh dict, so mutating it never changes the dataset — variable edits go through
        :meth:`with_variable` / :meth:`drop_variables` / :meth:`rename_variable`, which each
        return a new dataset.
        """
        return dict(self._data_variables)

    def with_variable(
        self,
        name: str,
        data: np.ndarray,
        *,
        location: str = "face",
        nodata: float | None = None,
    ) -> UgridDataset:
        """Return a new dataset with ``name`` added or replaced.

        UGRID derivations are immutable — unlike ``NetCDF.set_variable`` this does not mutate
        in place but returns a fresh dataset. A 2-D ``data`` array is treated as temporal
        (leading axis = time); a 1-D array is a static per-element variable.

        Args:
            name: Variable name to add or replace.
            data: The values; ``(n_elements,)`` static or ``(n_time, n_elements)`` temporal.
            location: Mesh location — ``"face"``, ``"node"`` or ``"edge"``. Defaults to
                ``"face"``.
            nodata: Optional no-data value for the new variable.

        Returns:
            UgridDataset: A new dataset carrying the variable.
        """
        arr = np.asarray(data)
        variable = MeshVariable(
            name=name,
            location=location,
            mesh_name=self.mesh_name,
            shape=arr.shape,
            nodata=nodata,
            _data=arr,
        )
        new_vars = dict(self._data_variables)
        new_vars[name] = variable
        return self._rebuild(new_vars)

    def drop_variables(self, names: str | list[str]) -> UgridDataset:
        """Return a new dataset without the named variable(s).

        Args:
            names: A variable name, or a list of them, to drop.

        Returns:
            UgridDataset: A new dataset without those variables.

        Raises:
            KeyError: A named variable is not present.
        """
        drop = [names] if isinstance(names, str) else list(names)
        new_vars = dict(self._data_variables)
        for name in drop:
            if name not in new_vars:
                raise KeyError(
                    f"Variable {name!r} not found. Available: {self.data_variable_names}"
                )
            del new_vars[name]
        return self._rebuild(new_vars)

    def rename_variable(self, old: str, new: str) -> UgridDataset:
        """Return a new dataset with variable ``old`` renamed to ``new``.

        Args:
            old: The current variable name.
            new: The new name.

        Returns:
            UgridDataset: A new dataset with the variable renamed.

        Raises:
            KeyError: ``old`` is not present.
            ValueError: ``new`` already names a different variable.
        """
        if old not in self._data_variables:
            raise KeyError(
                f"Variable {old!r} not found. Available: {self.data_variable_names}"
            )
        if new in self._data_variables and new != old:
            raise ValueError(f"Variable {new!r} already exists.")
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if name == old:
                # `replace` keeps the lazy loader and every other field, only re-labelling
                # the variable, so a rename never forces a read.
                new_vars[new] = replace(var, name=new)
            else:
                new_vars[name] = var
        return self._rebuild(new_vars)

    @property
    def metadata(self) -> UgridMetadata:
        """Full metadata summary for this dataset."""
        topo_tuple = (self._topology_info,) if self._topology_info else ()
        data_vars = {name: var.location for name, var in self._data_variables.items()}
        conventions = self._global_attributes.get("Conventions")
        result = UgridMetadata(
            mesh_topologies=topo_tuple,
            data_variables=data_vars,
            global_attributes=self._global_attributes,
            conventions=conventions,
            n_nodes=self.n_node,
            n_faces=self.n_face,
            n_edges=self.n_edge,
        )
        return result

    def to_dataset(
        self,
        variable_name: str,
        cell_size: float,
        method: str = "nearest",
        bounds: tuple[float, float, float, float] | None = None,
        epsg: int | None = None,
        nodata: float = -9999.0,
    ) -> Dataset:
        """Convert a mesh variable to a regular-grid Dataset.

        Interpolates mesh data onto a regular grid and returns a
        standard pyramids Dataset. This is the bridge between
        unstructured (UGRID) and structured (raster) worlds.

        Args:
            variable_name: Name of the data variable to rasterize.
            cell_size: Target grid cell size in coordinate units.
            method: Interpolation method ("nearest" or "linear").
            bounds: Target (xmin, ymin, xmax, ymax). Defaults to mesh bounds.
            epsg: Target EPSG code. Defaults to mesh CRS.
            nodata: No-data value for the output raster.

        Returns:
            pyramids Dataset with the interpolated data.
        """
        var = self.get_data(variable_name)
        data = var.data
        if data is None:
            raise ValueError(f"Variable '{variable_name}' has no data loaded.")
        if var.has_time:
            # First step along the variable's own time axis (not blindly axis 0, which for
            # a trailing-time variable would take the first face) — see `_element_values`.
            data = np.take(np.asarray(data), 0, axis=cast("int", var.time_index))

        grid_array, geotransform = mesh_to_grid(
            mesh=self._mesh,
            data=data,
            location=var.location,
            cell_size=cell_size,
            method=method,
            bounds=bounds,
            nodata=nodata,
        )

        # `crs_spec` rather than `self.epsg or 4326`: a mesh carrying a projected
        # WKT with no EPSG code has `epsg is None`, and the old default stamped
        # EPSG:4326 onto metre coordinates. That is the case this resolves.
        #
        # The `or 4326` behind it keeps the *other* case as it was. A mesh with
        # no CRS at all -- no code and no WKT -- has always rasterised to WGS 84
        # here, and `to_dataset(...).to_file("out.tif")` writing a GeoTIFF with
        # no CRS instead would be a silent behaviour change for every caller
        # holding such a mesh. Narrowing that default is a decision for its own
        # change, not a side effect of teaching this line about WKT.
        target_epsg = (
            epsg if epsg is not None else crs_spec(self.epsg, self.crs_wkt) or 4326
        )
        result = Dataset.from_array(
            grid_array,
            no_data_value=nodata,
            geo_ref=GeoReference(
                geo=cast(
                    "tuple[float, float, float, float, float, float]", geotransform
                ),
                epsg=target_epsg,
            ),
        )
        return result

    def crop(
        self,
        mask: Any = None,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: int | None = None,
    ) -> UgridDataset:
        """Crop the mesh to a polygon mask or a bbox — the unstructured-mesh analogue of crop.

        The mesh equivalent of :meth:`pyramids.dataset.Dataset.crop` / :meth:`NetCDF.crop`. Rather
        than warping a raster, it selects the **faces** that fall inside the region (renumbering the
        node connectivity for the resulting sub-mesh) and keeps the data on the surviving elements.
        Delegates to :meth:`clip` for a polygon and :meth:`subset_by_bounds` for a bbox; this method
        exists so the spatial-subset call is named ``crop`` across the raster and mesh classes alike.

        Args:
            mask (Any):
                Polygon mask — a shapely geometry, ``GeoDataFrame``, or ``FeatureCollection``.
                Mutually exclusive with ``bbox``.
            touch (bool):
                Applies only to a polygon ``mask``: if ``True`` (default), keep faces that touch the
                mask boundary; if ``False``, keep only faces fully inside it. Ignored for ``bbox``,
                which always selects faces by its axis-aligned envelope (``subset_by_bounds``).
                Defaults to True.
            bbox (tuple or list of 4 floats, keyword-only):
                ``(west, south, east, north)`` in the mesh CRS, or in ``epsg`` when supplied. Accepts
                a tuple or a list. Selects faces by the axis-aligned envelope; not affected by
                ``touch``. Mutually exclusive with ``mask``.
            epsg (int, keyword-only):
                CRS of ``bbox``. When it differs from the mesh CRS the box is reprojected to the mesh
                CRS and subset by its envelope; when it equals the mesh CRS it is a no-op. Defaults to
                the mesh CRS.

        Returns:
            UgridDataset: A new sub-mesh — faces inside the region, connectivity renumbered, and data
                variables subset to the surviving elements.

        Raises:
            ValueError: If both ``mask`` and ``bbox`` are supplied, if ``bbox`` is not a 4-tuple, or
                if ``epsg`` is given for a ``bbox`` but the mesh has no CRS to reproject into.
            TypeError: If neither ``mask`` nor ``bbox`` is supplied.

        Examples:
            - Crop a mesh to a polygon (faces intersecting it survive):
                ```python
                >>> from shapely.geometry import Polygon
                >>> from pyramids.netcdf import UgridDataset
                >>> ug = UgridDataset.read_file("mesh.nc")                        # doctest: +SKIP
                >>> sub = ug.crop(Polygon([(-1, -1), (0, -1), (0, 1), (-1, 1)]))  # doctest: +SKIP
                >>> sub.n_face <= ug.n_face                                       # doctest: +SKIP
                True

                ```
            - Crop to a bounding box in the mesh's own CRS:
                ```python
                >>> sub = ug.crop(bbox=(-1.0, -1.0, 0.0, 1.0))                    # doctest: +SKIP

                ```

        See Also:
            clip: Polygon-mask subsetting that ``crop`` delegates to when ``mask`` is given.
            subset_by_bounds: Bounding-box subsetting that ``crop`` delegates to when ``bbox`` is
                given.
            pyramids.dataset.Dataset.crop: The raster equivalent on a gridded dataset.
        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            if len(bbox) != 4:
                raise ValueError(
                    "bbox must be a 4-tuple of (west, south, east, north), "
                    f"got {len(bbox)} value(s)"
                )
            west, south, east, north = bbox
            if epsg is not None and (self.epsg is None or int(epsg) != int(self.epsg)):
                if self.epsg is None:
                    raise ValueError(
                        f"cannot reproject a bbox given in EPSG:{int(epsg)} into a mesh that has no "
                        "CRS; drop epsg to treat the bbox as native coordinates"
                    )
                # Reproject the bbox to the mesh CRS and subset by its envelope, so the bbox path
                # selects faces with the same rule (subset_by_bounds) regardless of the source CRS.
                west, south, east, north = (
                    gpd.GeoSeries([box(west, south, east, north)], crs=epsg)
                    .to_crs(self.epsg)
                    .total_bounds
                )
            result = self.subset_by_bounds(west, south, east, north)
        elif mask is not None:
            result = self.clip(mask, touch=touch)
        else:
            raise TypeError(
                "crop requires a `mask` (polygon) or a `bbox` (west, south, east, north) tuple"
            )
        return result

    def _wrap_subset(
        self, mesh: Mesh2d, data_variables: dict[str, MeshVariable]
    ) -> UgridDataset:
        """Wrap a ``(mesh, data_variables)`` pair from the spatial subsetters into a dataset.

        The spatial subsetting helpers (:func:`clip_mesh` / :func:`subset_by_bounds`) return
        the rebuilt mesh and data variables rather than a dataset (STR-3 — keeps
        ``ugrid.spatial`` independent of this module). This carries the source dataset's
        global attributes / topology info / CRS onto the subset.

        Args:
            mesh: The subset mesh.
            data_variables: The sliced data variables.

        Returns:
            UgridDataset: The wrapped subset.
        """
        return type(self)(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def clip(self, mask: Any, touch: bool = True) -> UgridDataset:
        """Clip the mesh to a polygon mask.

        Selects faces that intersect (touch=True) or are fully
        contained within (touch=False) the mask polygon.

        Args:
            mask: Polygon mask (GeoDataFrame, FeatureCollection,
                or Shapely geometry).
            touch: If True, include faces touching the boundary.

        Returns:
            New UgridDataset with clipped mesh and data.
        """
        mesh, data_variables = clip_mesh(self, mask, touch=touch)
        return self._wrap_subset(mesh, data_variables)

    def subset_by_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> UgridDataset:
        """Subset mesh to faces within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            New UgridDataset with subset mesh and data.
        """
        mesh, data_variables = subset_by_bounds(self, xmin, ymin, xmax, ymax)
        return self._wrap_subset(mesh, data_variables)

    def to_crs(self, to_epsg: int) -> UgridDataset:
        """Reproject all node coordinates to a new CRS.

        Uses pyproj.Transformer to reproject node coordinates.
        Face/edge center coordinates are recomputed after reprojection.
        Data values are preserved — only coordinates change.

        Args:
            to_epsg: Target EPSG code.

        Returns:
            New UgridDataset with reprojected coordinates.
        """
        # The mesh may describe its CRS by WKT alone, with no EPSG code to
        # resolve; `crs_spec` returns whichever of the two is usable, so a
        # projected mesh stops being refused for a CRS it plainly has.
        source_crs = crs_spec(self.epsg, self.crs_wkt)
        if source_crs is None:
            raise ValueError(
                "Cannot reproject: source CRS is unknown. "
                "Set CRS before calling to_crs()."
            )

        # Through `crs_from_user_input` so a mesh in a CRS whose code only GDAL's
        # PROJ database carries still reprojects (issue #943). The source goes in
        # as-is rather than as `EPSG:{...}`: `crs_spec` yields the mesh's WKT when
        # there is no usable code, and prefixing that builds the nonsense
        # `EPSG:PROJCS[...]`. The target is always a bare integer code.
        transformer = Transformer.from_crs(
            crs_from_user_input(source_crs),
            crs_from_user_input(f"EPSG:{to_epsg}"),
            always_xy=True,
        )
        new_node_x, new_node_y = transformer.transform(
            self._mesh.node_x,
            self._mesh.node_y,
        )

        new_face_x = None
        new_face_y = None
        if self._mesh.has_face_coords:
            new_face_x, new_face_y = transformer.transform(
                self._mesh.face_x,
                self._mesh.face_y,
            )

        new_edge_x = None
        new_edge_y = None
        if self._mesh.has_edge_coords:
            new_edge_x, new_edge_y = transformer.transform(
                self._mesh.edge_x,
                self._mesh.edge_y,
            )

        new_mesh = Mesh2d(
            node_x=new_node_x,
            node_y=new_node_y,
            face_node_connectivity=self._mesh.face_node_connectivity,
            edge_node_connectivity=self._mesh.edge_node_connectivity,
            face_edge_connectivity=self._mesh.face_edge_connectivity,
            face_face_connectivity=self._mesh.face_face_connectivity,
            edge_face_connectivity=self._mesh.edge_face_connectivity,
            face_x=new_face_x,
            face_y=new_face_y,
            edge_x=new_edge_x,
            edge_y=new_edge_y,
        )

        srs = sr_from_epsg(to_epsg)
        new_crs_wkt = srs.ExportToWkt()

        new_topo_info = None
        if self._topology_info is not None:
            new_topo_info = replace(self._topology_info, crs_wkt=new_crs_wkt)

        result = UgridDataset(
            mesh=new_mesh,
            data_variables=self._data_variables,
            global_attributes=self._global_attributes,
            topology_info=new_topo_info,
            crs_wkt=new_crs_wkt,
        )
        return result

    @property
    def time_values(self) -> list | None:
        """Parsed time coordinate values from the first temporal variable.

        Returns None if no variables have a time dimension.
        """
        result = None
        for var in self._data_variables.values():
            if var.has_time:
                time_attr = var.attributes.get("time_values")
                if time_attr is not None:
                    result = list(time_attr)
                else:
                    result = list(range(var.n_time_steps))
                break
        return result

    def sel_time(self, index: int) -> UgridDataset:
        """Select a single time step from all temporal variables.

        Non-temporal variables are kept unchanged.

        Args:
            index: Time step index.

        Returns:
            New UgridDataset with single time step data.
        """
        new_data_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.has_time:
                new_data_vars[name] = var.with_data(var.sel_time(index))
            else:
                new_data_vars[name] = var

        result = UgridDataset(
            mesh=self._mesh,
            data_variables=new_data_vars,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
        )
        return result

    def sel_time_range(self, start: int, stop: int) -> UgridDataset:
        """Select a time range from all temporal variables.

        Args:
            start: Start index (inclusive).
            stop: Stop index (exclusive).

        Returns:
            New UgridDataset with the selected time range.
        """
        new_data_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if var.has_time:
                new_data_vars[name] = var.sel_time_range(start, stop)
            else:
                new_data_vars[name] = var

        result = UgridDataset(
            mesh=self._mesh,
            data_variables=new_data_vars,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
        )
        return result

    def to_file(self, path: str | Path) -> None:
        """Write to a UGRID-compliant NetCDF file.

        Creates a NetCDF file with topology variable, node coordinates,
        connectivity arrays, face/edge centers, data variables, and
        global attributes following the UGRID convention.

        Args:
            path: Output file path.
        """
        path = Path(path)
        drv = gdal.GetDriverByName("netCDF")
        ds = drv.CreateMultiDimensional(str(path))
        rg = ds.GetRootGroup()

        mesh_name = self.mesh_name
        dims = write_ugrid_topology(rg, self._mesh, mesh_name, self._crs_wkt)

        for var in self._data_variables.values():
            if var.has_time and "time" not in dims:
                time_dim = rg.CreateDimension("time", None, None, var.n_time_steps)
                dims["time"] = time_dim
            # `load_array` reads each variable without memoising it on the shared dataset, so the
            # write streams one variable at a time instead of holding the whole cube resident (#982).
            write_ugrid_data_variable(
                rg, var.with_data(var.load_array()), mesh_name, dims
            )

        global_attrs = dict(self._global_attributes)
        if "Conventions" not in global_attrs:
            global_attrs["Conventions"] = "CF-1.8 UGRID-1.0"
        write_global_attributes(rg, global_attrs)

        ds = None

    def to_geodataframe(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> gpd.GeoDataFrame:
        """Convert mesh to a GeoDataFrame.

        For faces: each row is a Polygon with data columns.
        For nodes: each row is a Point.
        For edges: each row is a LineString.

        Args:
            variable_name: Optional data variable to include as a column.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            geopandas GeoDataFrame.
        """
        geometries = self._build_geometries(location)

        data_dict: dict[str, Any] = {}
        if variable_name is not None:
            var = self.get_data(variable_name)
            if var.location == location:
                # For a temporal variable only the first step is tabulated; `sel_time(0)` reads just
                # that slab instead of loading every step to slice `[0]` (#982). `has_data_source`
                # avoids a temporal-specific `sel_time` error for a variable with no readable data
                # (checked without forcing a load — review L3).
                if var.has_time:
                    var_data = var.sel_time(0) if var.has_data_source else None
                else:
                    var_data = var.data
                # A variable with no readable data becomes a length-correct null column rather than
                # raising — pandas rejects a scalar `None` column ("must pass an index") (review L3).
                if var_data is None:
                    var_data = np.full(len(geometries), np.nan)
                data_dict[variable_name] = var_data

        gdf = gpd.GeoDataFrame(data_dict, geometry=geometries)
        if self.crs is not None:
            gdf = gdf.set_crs(self.crs)

        result = gdf
        return result

    def _build_geometries(self, location: str) -> list:
        """Build the geometry list for a mesh location.

        Args:
            location: Mesh location ("face", "node", or "edge").

        Returns:
            List of shapely geometries — Polygons for "face", Points for
            "node", LineStrings for "edge".

        Raises:
            ValueError: If `location` is unknown, or edge connectivity is
                unavailable for an edge conversion.
        """
        if location == "face":
            geometries = MeshSpatialIndex(self._mesh).face_polygons
        elif location == "node":
            # Vectorized point construction — meshes routinely have 1e5-1e7 nodes, so a per-node
            # Python `Point(...)` loop is a hot spot (ARC-59). `list(...)` keeps the return type
            # consistent with the face/edge branches (review N2).
            geometries = list(shapely.points(self._mesh.node_x, self._mesh.node_y))
        elif location == "edge":
            if self._mesh.edge_node_connectivity is None:
                raise ValueError("Edge connectivity not available.")
            geometries = self._edge_linestrings(self._mesh.edge_node_connectivity)
        else:
            raise ValueError(f"Unknown location: {location}")
        return geometries

    def _edge_linestrings(self, enc: Connectivity) -> Any:
        """Build one LineString per edge, vectorized for standard 2-node edges (ARC-59)."""
        node_idx = np.asarray(enc.data)
        # A `None` fill means no sentinels are present, so the fast path is valid without an
        # elementwise `node_idx != None` compare (which NumPy deprecates) (review N3).
        no_fill = enc.fill_value is None or bool(np.all(node_idx != enc.fill_value))
        if node_idx.ndim == 2 and node_idx.shape[1] == 2 and no_fill:
            xs = self._mesh.node_x[node_idx]
            ys = self._mesh.node_y[node_idx]
            return list(shapely.linestrings(np.stack([xs, ys], axis=-1)))
        # Rare ragged / filled edge connectivity: fall back to a per-edge build.
        return [
            LineString(
                [
                    (self._mesh.node_x[n], self._mesh.node_y[n])
                    for n in enc.get_element(i)
                ]
            )
            for i in range(enc.n_elements)
        ]

    def to_feature_collection(
        self,
        variable_name: str | None = None,
        location: str = "face",
    ) -> FeatureCollection:
        """Convert mesh to a pyramids FeatureCollection.

        Args:
            variable_name: Optional data variable to include.
            location: Mesh location ("face", "node", or "edge").

        Returns:
            pyramids FeatureCollection.
        """
        gdf = self.to_geodataframe(variable_name, location)
        result = FeatureCollection(gdf)
        return result

    @classmethod
    def from_arrays(
        cls,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: np.ndarray,
        data: dict[str, np.ndarray] | None = None,
        data_locations: dict[str, str] | None = None,
        epsg: int = 4326,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> UgridDataset:
        """Create a UgridDataset programmatically from arrays.

        Plural because an unstructured mesh is not one array: the topology
        needs node coordinates *and* a face-node connectivity table before any
        data can be attached. It therefore takes a flat `epsg` rather than the
        :class:`~pyramids.base.georeference.GeoReference` the gridded
        constructors take — a mesh carries its own coordinates, so there is no
        affine transform to describe.

        Args:
            node_x: Node x-coordinates.
            node_y: Node y-coordinates.
            face_node_connectivity: (n_faces, max_nodes) array of node
                indices. Use -1 as fill value for mixed meshes.
            data: Optional dict mapping variable name to data array.
            data_locations: Optional dict mapping variable name to
                location ("face", "node", "edge"). Defaults to "face".
            epsg: EPSG code for the CRS.
            mesh_name: Name for the topology variable.

        Returns:
            UgridDataset instance.

        Examples:
            - Build the smallest possible mesh — two triangles — and inspect
              its topology:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ... )
                >>> (mesh.n_node, mesh.n_face)
                (4, 2)
                >>> mesh.bounds
                (0.0, 0.0, 1.0, 1.0)

                ```
            - Attach a per-face variable and read it back:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([1.5, 2.5])},
                ... )
                >>> mesh.data_variable_names
                ['depth']
                >>> mesh["depth"].location
                'face'
                >>> mesh["depth"].data.tolist()
                [1.5, 2.5]

                ```
        """
        fnc = Connectivity(
            data=np.asarray(face_node_connectivity, dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        mesh = Mesh2d(
            node_x=np.asarray(node_x, dtype=np.float64),
            node_y=np.asarray(node_y, dtype=np.float64),
            face_node_connectivity=fnc,
        )

        data_variables: dict[str, MeshVariable] = {}
        topo_data_vars: dict[str, str] = {}
        if data is not None:
            if data_locations is None:
                data_locations = {}
            for name, arr in data.items():
                loc = data_locations.get(name, "face")
                topo_data_vars[name] = loc
                data_variables[name] = MeshVariable(
                    name=name,
                    location=loc,
                    mesh_name=mesh_name,
                    shape=arr.shape,
                    _data=arr,
                )

        srs = sr_from_epsg(epsg)
        crs_wkt = srs.ExportToWkt()

        topo_info = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=2,
            node_x_var=f"{mesh_name}_node_x",
            node_y_var=f"{mesh_name}_node_y",
            face_node_var=f"{mesh_name}_face_nodes",
            data_variables=topo_data_vars,
            crs_wkt=crs_wkt,
        )

        result = cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=topo_info,
            crs_wkt=crs_wkt,
        )
        return result

    @classmethod
    def from_structured(
        cls,
        dataset: Any,
        band_names: list[str] | None = None,
        mesh_name: str = DEFAULT_MESH_NAME,
    ) -> UgridDataset:
        """Build a mesh from a raster, one quad face per cell.

        The bridge from the structured world: each cell becomes a quad over the corner
        lattice (shared corners deduplicated) and each band becomes a per-face variable, so a
        raster can be fed to the mesh operations.

        Args:
            dataset: The source raster (a pyramids ``Dataset`` / ``NetCDF`` variable).
            band_names: Names for the per-face variables, one per band. Defaults to
                ``band_1``, ``band_2``, ….
            mesh_name: The topology name to stamp. Defaults to the package default.

        Returns:
            UgridDataset: The quad mesh carrying the band values.
        """
        node_x, node_y, faces, data = _construct.from_structured(dataset, band_names)
        return cls.from_arrays(
            node_x,
            node_y,
            faces,
            data=data,
            epsg=dataset.epsg,
            mesh_name=mesh_name,
        )

    @classmethod
    def from_geodataframe(
        cls, gdf: Any, mesh_name: str = DEFAULT_MESH_NAME
    ) -> UgridDataset:
        """Build a mesh from polygon geometries, one face per polygon.

        Vertices are deduplicated into the node set and each ring is oriented
        counter-clockwise; the frame's attribute columns become per-face variables.

        Args:
            gdf: A GeoDataFrame of polygons. Its ``crs`` is carried onto the mesh verbatim,
                so a projected CRS stays projected even when it has no EPSG code.
            mesh_name: The topology name to stamp. Defaults to the package default.

        Returns:
            UgridDataset: The mesh built from the polygons.

        Raises:
            ValueError: The frame declares no CRS, so the mesh would carry no
                georeferencing.
        """
        node_x, node_y, faces, data = _construct.from_geodataframe(gdf)
        # Carry the frame's own CRS. Reading an EPSG code and defaulting to 4326 stamped degrees
        # on a projected frame whose CRS has no EPSG code, and on a frame with no CRS at all --
        # silent georeferencing corruption, since everything downstream trusts `epsg`.
        crs = getattr(gdf, "crs", None)
        if crs is None:
            raise ValueError(
                "from_geodataframe: the GeoDataFrame declares no CRS, so the mesh would have no "
                "georeferencing. Set `gdf.crs` (e.g. gdf.set_crs(4326)) before converting."
            )
        epsg = crs.to_epsg()
        built = cls.from_arrays(
            node_x,
            node_y,
            faces,
            data=data,
            epsg=epsg if epsg is not None else 4326,
            mesh_name=mesh_name,
        )
        if epsg is not None:
            return built
        # A CRS with no EPSG code (a projected one given as WKT, say) would otherwise be
        # reduced to the 4326 placeholder `from_arrays` needs, stamping degrees on metres.
        # Keep the frame's WKT verbatim instead.
        wkt = crs.to_wkt()
        return cls(
            mesh=built.mesh,
            data_variables=built.data_vars,
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=_topology_info_named(mesh_name, crs_wkt=wkt),
            crs_wkt=wkt,
            file_name=None,
        )

    @classmethod
    def from_xarray(cls, ds: Any) -> UgridDataset:
        """Read a mesh back from an ``xarray.Dataset`` in the UGRID-CF encoding.

        The inverse of :meth:`to_xarray`. xarray is a peer dependency, imported only inside
        the bridge, so nothing in this module imports it at module scope.

        Args:
            ds: The ``xarray.Dataset`` to read.

        Returns:
            UgridDataset: The reconstructed mesh and variables.

        Raises:
            OptionalPackageDoesNotExist: xarray is not installed.
        """
        mesh, data_variables = _interop.from_xarray(ds)
        return cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={},
            # Keep the topology's own name: without it a mesh called anything other than
            # "mesh2d" came back renamed, and a re-export then wrote the wrong dimensions.
            topology_info=_topology_info_named(_interop.read_mesh_name(ds)),
            crs_wkt=_interop.read_crs_wkt(ds),
            file_name=None,
        )

    @classmethod
    def from_zarr(cls, store: Any, **kwargs: Any) -> UgridDataset:
        """Read a mesh from a zarr store written by :meth:`to_zarr`.

        Args:
            store: The zarr store (a path or a mapping).
            **kwargs: Passed to ``xarray.open_zarr``.

        Returns:
            UgridDataset: The reconstructed mesh and variables.

        Raises:
            OptionalPackageDoesNotExist: xarray or zarr is not installed.
        """
        # One open, read twice: the mesh and the CRS both come off the same dataset, so the
        # store is not opened again just to recover the grid mapping.
        opened = _interop.open_zarr_dataset(store, **kwargs)
        mesh, data_variables = _interop.from_xarray(opened)
        return cls(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes={},
            topology_info=_topology_info_named(_interop.read_mesh_name(opened)),
            crs_wkt=_interop.read_crs_wkt(opened),
            file_name=None,
        )

    @staticmethod
    def mesh_names(path: str | Path) -> list[str]:
        """The names of every UGRID topology a file declares.

        A UGRID file may carry several meshes; :meth:`read_file` takes the first unless told
        otherwise, so this is how a caller discovers what else is in there.

        Args:
            path: Path to the ``.nc`` file.

        Returns:
            list[str]: One name per declared topology, in file order.

        Raises:
            FileNotFoundError: The file does not exist.
            RuntimeError: GDAL cannot open the file (GDAL exceptions are enabled
                package-wide, so the open failure surfaces as GDAL's own error).
            ValueError: The file declares no UGRID topology.
        """
        target = str(path)
        if not target.startswith("/vsi"):
            resolved = Path(path)
            if not resolved.exists():
                raise FileNotFoundError(f"File not found: {resolved}")
            target = str(resolved)
        dataset = gdal.OpenEx(target, gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR)
        if dataset is None:
            raise ValueError(f"GDAL cannot open file: {path}")
        rg = dataset.GetRootGroup()
        if rg is None:
            raise ValueError(f"Cannot get root group from: {path}")
        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise ValueError(f"No UGRID mesh topology found in: {path}")
        return [topo.mesh_name for topo in topologies]

    @classmethod
    def read_all(cls, path: str | Path) -> dict[str, UgridDataset]:
        """Read **every** UGRID topology in a file, one dataset each.

        :meth:`read_file` answers a single mesh (the first, or the one named). This answers
        all of them, so nothing in a multi-mesh file is silently dropped.

        Args:
            path: Path to the ``.nc`` file.

        Returns:
            dict[str, UgridDataset]: Each topology name mapped to its dataset.

        Raises:
            FileNotFoundError: The file does not exist.
            ValueError: GDAL cannot open the file, or it declares no UGRID topology.
        """
        return {
            name: cls.read_file(path, mesh_name=name) for name in cls.mesh_names(path)
        }

    def _layer_axis(self, var: MeshVariable) -> int | None:
        """The axis holding the layer dimension, or ``None`` when the variable has none.

        A mesh variable's axes read ``[time?, layer?, element]`` — the element axis is the
        last and time is whichever :attr:`MeshVariable.time_index` names, so a remaining
        middle axis is the layer. A 2-D variable whose dimension names mark it non-temporal
        is therefore ``(layer, element)`` with layer axis 0.

        Two cases answer ``None`` rather than guessing. A variable carrying no dimension
        names at all (an array-built one) falls back to "the leading axis of a >1-D array is
        time", so its 2-D form reads as ``(time, element)``; name the dimensions to have it
        read as layers. And a variable whose time dimension is *last* contradicts the layout
        above — the element axis is supposed to be last — so no axis can be claimed as the
        layer axis without mistaking the element axis for it.

        Args:
            var: The variable to inspect.

        Returns:
            int | None: The layer axis, or ``None``.
        """
        if not var.has_data_source:
            return None
        # The declared shape, not `var.data`: touching the data fires the lazy loader and
        # pulls every variable off disk just to learn how many axes it has -- the same defect
        # `info()` carried.
        rank = len(var.shape)
        if rank < 2:
            return None
        if var.time_index == rank - 1:
            # Time sits where the element axis belongs, so the layout is not one we can read.
            return None
        axes = [axis for axis in range(rank - 1) if axis != var.time_index]
        return axes[0] if axes else None

    @property
    def n_layers(self) -> int | None:
        """The number of layers, or ``None`` when no variable is layered.

        A 3-D mesh output puts some variables on the ``n`` layer centres and others on the
        ``n + 1`` layer interfaces, so two counts differing by one is normal rather than a
        contradiction; this reports ``n``, the layer count proper. Any other disagreement is
        a real inconsistency and raises. :meth:`isel_layer` is unaffected either way — it
        indexes each variable on its own layer axis.

        Raises:
            ValueError: The layered variables carry counts that are not a layer-centre /
                layer-interface pair.
        """
        counts = set()
        for var in self._data_variables.values():
            axis = self._layer_axis(var)
            if axis is not None:
                counts.add(int(var.shape[axis]))
        if not counts:
            return None
        smallest = min(counts)
        if counts - {smallest, smallest + 1}:
            raise ValueError(
                f"layered variables disagree on their layer count: {sorted(counts)}."
            )
        return smallest

    @property
    def layer_values(self) -> list | None:
        """The layer coordinate values, when a layered variable carries them.

        Read from a layered variable's ``layer_values`` attribute. The UGRID reader does not
        populate that attribute -- as with :attr:`time_values`, the coordinate is supplied by
        whoever builds or transforms the dataset -- so this is normally ``None`` for a mesh
        read straight from a file, and :meth:`sel_layer` then selects by layer position.
        """
        for var in self._data_variables.values():
            if self._layer_axis(var) is not None:
                stamps = var.attributes.get("layer_values")
                if stamps is not None:
                    return list(stamps)
        return None

    def isel_layer(self, index: int | slice | list[int] | np.ndarray) -> UgridDataset:
        """Select along the layer axis by position.

        A layered variable — ``(layer, element)``, or ``(time, layer, element)`` — is indexed
        on its layer axis. An integer collapses that axis (the variable loses its layer
        dimension); a slice or sequence keeps it, trimmed. Variables with no layer axis are
        carried through untouched, as the time selectors carry static variables.

        Args:
            index: An ``int`` layer, a ``slice``, or a sequence of layer positions.

        Returns:
            UgridDataset: A new dataset with the layer selection applied.

        Raises:
            ValueError: No variable has a layer dimension.
        """
        new_vars: dict[str, MeshVariable] = {}
        layered = False
        for name, var in self._data_variables.items():
            axis = self._layer_axis(var)
            if axis is None:
                new_vars[name] = var
                continue
            layered = True
            data = np.asarray(var.data)
            if isinstance(index, (int, np.integer)):
                picked = np.take(data, int(index), axis=axis)
                moved = var.with_data(picked)
                moved.dimensions = tuple(
                    dim
                    for position, dim in enumerate(var.dimensions)
                    if position != axis
                )
            else:
                positions = (
                    np.arange(data.shape[axis])[index]
                    if isinstance(index, slice)
                    else np.asarray(index, dtype=np.intp)
                )
                moved = var.with_data(np.take(data, positions, axis=axis))
            new_vars[name] = moved
        if not layered:
            raise ValueError(
                "isel_layer needs at least one variable with a layer dimension; this "
                "dataset has none."
            )
        return self._rebuild(new_vars)

    def sel_layer(self, value: Any) -> UgridDataset:
        """Select along the layer axis by coordinate value.

        Matches ``value`` against :attr:`layer_values`, falling back to layer positions when
        no layer coordinate is present -- which is the usual case, since the reader does not
        populate `layer_values` (see :attr:`layer_values`). So on a mesh read from a file,
        ``sel_layer(1)`` selects layer position 1.

        Args:
            value: The layer coordinate value to select.

        Returns:
            UgridDataset: A new dataset with that layer selected.

        Raises:
            ValueError: No layer dimension, or the value is not among the layer coordinates.
        """
        count = self.n_layers
        if count is None:
            raise ValueError(
                "sel_layer needs at least one variable with a layer dimension; this "
                "dataset has none."
            )
        # Read the property once: it walks every variable, so evaluating it twice walked
        # them twice.
        stored = self.layer_values
        coords = stored if stored is not None else list(range(count))
        if value not in coords:
            raise ValueError(
                f"sel_layer: layer value {value!r} not found in the layer coordinate "
                f"{coords!r}."
            )
        return self.isel_layer(coords.index(value))

    def to_dataframe(
        self, variables: Sequence[str] | None = None, *, dropna: bool = False
    ) -> pd.DataFrame:
        """A tidy frame of the mesh values, one row per element (and step).

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.to_dataframe`, in the **tidy**
        shape rather than the raster ``(y, x)`` MultiIndex one: an unstructured element has no
        row/column address, so each row carries the element index, its centroid coordinates,
        the time step where the variable has one, and the value.

        Args:
            variables: Which data variables to include. ``None`` (the default) takes them all.
            dropna: Drop rows whose value is no-data. Defaults to False.

        Returns:
            pandas.DataFrame: Columns ``variable``, ``element``, ``x``, ``y``, ``layer``,
            ``time`` and ``value``. ``element`` indexes the facet the variable sits on and
            ``x``/``y`` are that facet's coordinates, so a node variable is addressed by node.
            ``layer`` is the layer coordinate, or None for an unlayered variable, and ``time``
            the step coordinate, or None for a static one.

        Raises:
            KeyError: A requested variable is not a data variable of this dataset.
        """
        names = list(self._data_variables) if variables is None else list(variables)
        frames: list[pd.DataFrame] = []
        for name in names:
            if name not in self._data_variables:
                raise KeyError(name)
            var = self._data_variables[name]
            if var.data is None:
                continue
            frames.extend(self._variable_frames(name, var))
        if not frames:
            # Match the populated frame's dtypes, so a caller that concatenates or
            # dtype-checks sees one schema either way.
            return pd.DataFrame(
                {
                    "variable": pd.Series(dtype="str"),
                    "element": pd.Series(dtype="int64"),
                    "x": pd.Series(dtype="float64"),
                    "y": pd.Series(dtype="float64"),
                    "layer": pd.Series(dtype="object"),
                    "time": pd.Series(dtype="object"),
                    "value": pd.Series(dtype="float64"),
                }
            )
        frame = pd.concat(frames, ignore_index=True)
        if dropna:
            frame = frame[~frame["value"].isna()].reset_index(drop=True)
        return frame

    def _variable_frames(self, name: str, var: MeshVariable) -> list[pd.DataFrame]:
        """One tidy block per (time step, layer) of a single variable.

        Args:
            name: The variable's name, carried into the ``variable`` column.
            var: The variable to lay out.

        Returns:
            list: The blocks, in time-then-layer order.
        """
        raw = np.asarray(var.data)
        # Gap-masking casts to float64, which a classified / label field cannot survive --
        # and such a field carries no no-data value to mask anyway.
        values = (
            gaps_as_nan(raw, var.nodata) if np.issubdtype(raw.dtype, np.number) else raw
        )
        # Address the rows with the coordinates of the facet the variable actually sits on. Using
        # face centroids for every variable left a node- or edge-located one with a NaN
        # coordinate on every row, since its value count is n_node / n_edge.
        coords = self._element_coords(var.location)
        arranged, steps, layer_labels = self._arrange_for_rows(var, values)
        temporal = var.has_time
        layered = self._layer_axis(var) is not None
        blocks: list[pd.DataFrame] = []
        for step_position, step in enumerate(steps):
            per_step = arranged[step_position] if temporal else arranged
            for layer_position, layer in enumerate(layer_labels):
                slab = per_step[layer_position] if layered else per_step
                blocks.append(
                    self._tidy_rows(name, np.asarray(slab), coords, step, layer)
                )
        return blocks

    def _arrange_for_rows(
        self, var: MeshVariable, values: np.typing.NDArray
    ) -> tuple[np.typing.NDArray, list[Any], list[Any]]:
        """Reorder a variable's axes to ``(time?, layer?, element)`` and label the leading ones.

        The tidy frame walks one row per element, so the axes above the element axis have to be
        walked in a known order with each position labelled. Flattening instead ran the element
        index past the element count and lost which layer a row came from.

        A missing axis is reported as the single label ``[None]``, so the caller's loop runs once
        over it and the frame still carries the column.

        Args:
            var: The variable whose axes are being arranged.
            values: Its values, already gap-masked.

        Returns:
            tuple: ``(arranged, steps, layer_labels)`` -- the transposed array, one label per time
            step, and one per layer.
        """
        element_axis = values.ndim - 1
        layer_axis = self._layer_axis(var)
        time_axis = cast("int | None", var.time_index) if var.has_time else None
        if time_axis == element_axis:
            # Time sitting in the element slot contradicts the [time?, layer?, element] layout,
            # so there is no step axis to walk; the values are laid out as they come.
            time_axis = None
        # `order` has to be a real permutation of every axis. Listing only time, layer and the
        # element axis repeated an axis when time was last, and dropped axes entirely on a
        # variable of rank 4 -- numpy answered "repeated axis in transpose" and "axes don't
        # match array". Any axis that is neither time, layer nor the element axis keeps its
        # relative order between them.
        leading = [axis for axis in (time_axis, layer_axis) if axis is not None]
        middle = [
            axis
            for axis in range(values.ndim)
            if axis != element_axis and axis not in leading
        ]
        order = [*leading, *middle, element_axis]
        arranged = np.transpose(values, order) if values.ndim > 1 else values

        steps: list[Any] = [None]
        if time_axis is not None:
            stamps = var.attributes.get("time_values")
            steps = (
                list(stamps) if stamps is not None else list(range(arranged.shape[0]))
            )

        layer_labels: list[Any] = [None]
        if layer_axis is not None:
            depth = arranged.shape[1 if time_axis is not None else 0]
            stored = var.attributes.get("layer_values")
            layer_labels = list(stored) if stored is not None else list(range(depth))

        return arranged, steps, layer_labels

    def _element_coords(self, location: str) -> np.typing.NDArray:
        """The ``(n_elements, 2)`` coordinate table of a mesh facet.

        Args:
            location: ``"face"``, ``"node"`` or ``"edge"``.

        Returns:
            np.ndarray: One ``(x, y)`` row per element, or an empty table when the facet's
            coordinates are not available (an edge-located variable on a mesh carrying no edge
            coordinates), which leaves those rows with a NaN coordinate.
        """
        try:
            table = _location_coords(self._mesh, location)
        except ValueError:
            table = np.empty((0, 2), dtype="float64")
        return np.asarray(table, dtype="float64")

    def _tidy_rows(
        self,
        name: str,
        values: np.typing.NDArray,
        element_coords: np.typing.NDArray,
        step: Any,
        layer: Any = None,
    ) -> pd.DataFrame:
        """One tidy block: every element of ``values`` as a row."""
        flat = np.asarray(values).ravel()
        count = flat.shape[0]
        coords = (
            element_coords
            if element_coords.shape[0] == count
            else np.full((count, 2), np.nan, dtype="float64")
        )
        return pd.DataFrame(
            {
                "variable": np.repeat(name, count),
                "element": np.arange(count),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "layer": np.repeat(layer, count),
                "time": np.repeat(step, count),
                # Keep a non-numeric variable as it is. Forcing float64 raised
                # "could not convert string to float" on exactly the classified /
                # label field `polygonize` recommends.
                "value": (
                    flat.astype("float64")
                    if np.issubdtype(flat.dtype, np.number)
                    else flat
                ),
            }
        )

    def copy(self) -> UgridDataset:
        """An independent deep copy of this dataset.

        Every variable's values are copied, so writing into the copy never shows in the
        original — the counterpart of :meth:`pyramids.netcdf.NetCDF.copy`.

        Returns:
            UgridDataset: The copy.
        """
        copied: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            copied[name] = (
                var
                if var.data is None
                else var.with_data(np.array(var.data, copy=True))
            )
        return type(self)(
            mesh=self._mesh,
            data_variables=copied,
            global_attributes=dict(self._global_attributes),
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    @classmethod
    def from_bytes(cls, data: bytes, *, suffix: str = ".nc") -> UgridDataset:
        """Read a UGRID mesh from an in-memory buffer.

        Writes ``data`` into GDAL's in-memory filesystem and reads it as a file, mirroring
        :meth:`pyramids.netcdf.NetCDF.from_bytes` — for a mesh fetched over the network or
        held in a database blob. The values are materialised before the buffer is unlinked, so
        the result does not depend on it.

        Args:
            data: The bytes of a UGRID NetCDF file.
            suffix: The extension to present to GDAL. Defaults to ``".nc"``.

        Returns:
            UgridDataset: The mesh read from the buffer.

        Raises:
            ValueError: GDAL cannot read the buffer as a UGRID file.
        """
        name = f"/vsimem/pyramids_ugrid_{uuid4().hex}{suffix}"
        gdal.FileFromMemBuffer(name, data)
        try:
            result = cls.read_file(name).copy()
        finally:
            gdal.Unlink(name)
        return result

    def info(self, buf: Any = None) -> None:
        """Print a summary of the mesh, its variables and its attributes.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.info`: one readable block naming
        the topology, the element counts, the CRS, and every data variable with its location,
        dtype and shape.

        Args:
            buf: Where to write. Defaults to ``sys.stdout``.
        """
        stream = sys.stdout if buf is None else buf
        lines = [
            f"UgridDataset: {self.mesh_name}",
            f"  nodes: {self.n_node}  faces: {self.n_face}  edges: {self.n_edge}",
            f"  epsg: {self.epsg}",
            f"  variables ({len(self._data_variables)}):",
        ]
        for name, var in self._data_variables.items():
            # Metadata only: `var.data` would fire the lazy loader and pull every variable's
            # full array off disk, which is the opposite of what a summary is for. The declared
            # shape and `has_data_source` answer the same questions without a read.
            shape = str(tuple(var.shape)) if var.has_data_source else "unloaded"
            temporal = "temporal" if var.has_time else "static"
            lines.append(
                f"    {name}: location={var.location} dtype={var.dtype} "
                f"shape={shape} ({temporal})"
            )
        if self._global_attributes:
            lines.append(f"  global attributes ({len(self._global_attributes)}):")
            lines.extend(
                f"    {key}: {value}" for key, value in self._global_attributes.items()
            )
        stream.write("\n".join(lines) + "\n")

    def set_global_attribute(self, name: str, value: Any) -> UgridDataset:
        """Return a new dataset carrying one more global attribute.

        A UGRID derivation is immutable here (as :meth:`with_variable` is), so this answers a
        new dataset rather than writing in place; :meth:`to_file` then persists the attributes
        through the shared CF writer.

        Args:
            name: The attribute name.
            value: The attribute value.

        Returns:
            UgridDataset: A new dataset whose global attributes include ``name``.
        """
        return type(self)(
            mesh=self._mesh,
            data_variables=dict(self._data_variables),
            global_attributes={**self._global_attributes, name: value},
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def delete_global_attribute(self, name: str) -> UgridDataset:
        """Return a new dataset without the global attribute ``name``.

        Args:
            name: The attribute to drop. Dropping an absent attribute is not an error,
                mirroring the raster ``delete_global_attribute``.

        Returns:
            UgridDataset: A new dataset without that attribute.
        """
        remaining = {
            key: value for key, value in self._global_attributes.items() if key != name
        }
        return type(self)(
            mesh=self._mesh,
            data_variables=dict(self._data_variables),
            global_attributes=remaining,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def _element_values(
        self, variable_name: str, time_index: int = 0
    ) -> tuple[MeshVariable, np.typing.NDArray]:
        """The per-element values of a variable, no-data blanked to NaN.

        Shared by the analysis members (:meth:`sample`, :meth:`weighted`,
        :meth:`zonal_stats`). A temporal variable is reduced to one
        step (``time_index``, the first by default) so the returned array is
        ``(n_elements,)``, matching the single-step contract :meth:`plot` and
        :meth:`to_dataset` already use. The variable's declared no-data value is turned
        into NaN through the shared :func:`pyramids.base._reductions.gaps_as_nan`, so
        every statistic leaves gaps out.

        Args:
            variable_name: Name of the data variable.
            time_index: Which step to take for a temporal variable. Defaults to 0.

        Returns:
            tuple: The :class:`MeshVariable` and its ``(n_elements,)`` float64 values.

        Raises:
            ValueError: The variable has no loaded data, or is not per-element after the
                time collapse (e.g. a layered ``(n_layers, n_face)`` non-temporal variable).
        """
        var = self.get_data(variable_name)
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data.")
        if var.has_time:
            # Take the step along the variable's own time axis, not blindly axis 0 — a
            # mesh variable may store time as a trailing axis (e.g. (nFaces, time)), and
            # the rest of this module keys off var.time_index for exactly that reason.
            data = np.take(
                np.asarray(data), time_index, axis=cast("int", var.time_index)
            )
        result = gaps_as_nan(np.asarray(data), var.nodata)
        if result.ndim != 1:
            # A non-temporal multi-dimensional variable (e.g. layered (n_layers, n_face))
            # has no single per-element vector; reducing it over an arbitrary axis would
            # silently return a per-column scalar. Refuse rather than mislead.
            raise ValueError(
                f"Variable {variable_name!r} is not per-element after the time collapse "
                f"(shape {result.shape}); the analysis members need a 1-D variable. A "
                "layered/multi-dimensional non-temporal variable is not supported here."
            )
        return var, result

    def summary(
        self,
        variables: Sequence[str] | None = None,
        *,
        metrics: Sequence[str] = DEFAULT_METRICS,
        skipna: bool = True,
        ddof: int = 0,
    ) -> pd.DataFrame:
        """A per-variable summary table over the whole mesh.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.summary`, built on the same shared
        assembly (:func:`pyramids.base._summary.variable_summary`) so the two classes answer in an
        identical shape. Each variable is reduced over **all** of its samples — every element and,
        for a temporal variable, every step — with the declared no-data value left out.

        Args:
            variables: Which data variables to summarise, in output-row order. ``None`` (the
                default) takes every data variable. An unknown name raises ``KeyError``.
            metrics: Which statistics to report, in column order. Defaults to
                ``("count", "min", "max", "mean", "std")``; ``median`` / ``var`` / ``sum`` /
                ``prod`` are available too.
            skipna: Leave no-data out of the statistics (the default) or let it propagate.
            ddof: Delta degrees of freedom for ``std`` / ``var``. Defaults to 0.

        Returns:
            pandas.DataFrame: One row per variable (index name ``"variable"``), one column per
            metric. ``count`` is ``int64``; the rest are ``float64``. A variable with no valid
            element yields NaN statistics and a ``count`` of 0.

        Raises:
            KeyError: A requested variable is not a data variable of this dataset.
            TypeError: A requested variable is non-numeric.
            ValueError: A requested variable has no loaded data, or an unknown metric.

        Examples:
            - Per-variable summary of a two-triangle mesh:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([2.0, 4.0])},
                ... )
                >>> mesh.summary().loc["depth", ["count", "min", "max", "mean"]].tolist()
                [2.0, 2.0, 4.0, 3.0]

                ```
        """
        names = list(self._data_variables) if variables is None else list(variables)
        arrays: dict[str, np.typing.NDArray] = {}
        for name in names:
            if name not in self._data_variables:
                raise KeyError(name)
            var = self._data_variables[name]
            if var.data is None:
                raise ValueError(f"Variable {name!r} has no loaded data.")
            data = np.asarray(var.data)
            if not np.issubdtype(data.dtype, np.number):
                if variables is None:
                    warnings.warn(
                        f"skipping non-numeric variable {name!r} in summary",
                        stacklevel=2,
                    )
                    continue
                raise TypeError(f"variable {name!r} is non-numeric; cannot summarise")
            arrays[name] = gaps_as_nan(data, var.nodata)
        return variable_summary(arrays, metrics=metrics, skipna=skipna, ddof=ddof)

    def sample(
        self,
        variable_name: str,
        x: float | np.ndarray,
        y: float | np.ndarray,
        *,
        method: str = "contains",
        time_index: int = 0,
    ) -> np.typing.NDArray:
        """Sample a mesh variable at point locations.

        The mesh counterpart of :meth:`pyramids.dataset.Dataset.sample`. Where the raster
        version inverts an affine geotransform to find the pixel under each point, a mesh
        resolves the point to an element through :class:`MeshSpatialIndex`: a face-located
        variable by which face contains the point (``method="contains"``) or the nearest
        face centroid (``method="nearest"``); a node-located variable by nearest node.

        Args:
            variable_name: Name of the data variable.
            x: Query x-coordinate(s), in the mesh CRS.
            y: Query y-coordinate(s), in the mesh CRS.
            method: ``"contains"`` (point-in-face, face variables only) or ``"nearest"``.
                Defaults to ``"contains"``.
            time_index: Step for a temporal variable. Defaults to 0.

        Returns:
            numpy.ndarray: One value per query point; NaN where a ``"contains"`` query
            falls outside every face.

        Raises:
            ValueError: An unknown ``method``, ``"contains"`` on a non-face variable, or a
                variable located on edges (unsupported).
        """
        var, arr = self._element_values(variable_name, time_index=time_index)
        xs = np.atleast_1d(np.asarray(x, dtype="float64"))
        ys = np.atleast_1d(np.asarray(y, dtype="float64"))
        index = MeshSpatialIndex(self._mesh)
        if var.location == "face":
            if method == "contains":
                idx = index.locate_faces(xs, ys)
            elif method == "nearest":
                idx = np.atleast_1d(index.locate_nearest_face(xs, ys)).ravel()
            else:
                raise ValueError(
                    f"sample method must be 'contains' or 'nearest', got {method!r}"
                )
        elif var.location == "node":
            # A node is a point, so "contains" has no meaning here — only "nearest"
            # applies. Reject "contains" rather than silently doing a nearest lookup, as
            # the docstring promises.
            if method == "contains":
                raise ValueError(
                    "sample method='contains' is only valid for a face-located variable; "
                    f"{variable_name!r} is on 'node' — use method='nearest'."
                )
            if method != "nearest":
                raise ValueError(
                    f"sample method must be 'contains' or 'nearest', got {method!r}"
                )
            idx = np.atleast_1d(index.locate_nearest_node(xs, ys)).ravel()
        else:
            raise ValueError(
                f"sample supports 'face' and 'node' variables; {variable_name!r} is on "
                f"{var.location!r}."
            )
        out = np.full(len(xs), np.nan, dtype="float64")
        inside = idx >= 0
        out[inside] = arr[idx[inside]]
        return out

    def weighted(
        self,
        variable_name: str,
        *,
        how: str = "mean",
        time_index: int = 0,
        q: float | None = None,
    ) -> float:
        """Area-weighted statistic of a face variable over the mesh.

        An irregular mesh has faces of different sizes, so the meaningful aggregate weights
        each face by its area. Delegates to the shared
        :func:`pyramids.base._reductions.weighted_statistic` with :attr:`Mesh2d.face_areas`
        as the weights — the same kernel the raster ``weighted`` reduction uses.

        Args:
            variable_name: Name of a **face-located** data variable.
            how: One of ``"mean"``, ``"sum"``, ``"sum_of_weights"``, ``"std"``, ``"var"``,
                ``"quantile"`` (which needs ``q``). Defaults to ``"mean"``.
            time_index: Step for a temporal variable. Defaults to 0.
            q: The quantile in ``[0, 1]`` when ``how="quantile"`` (the area-weighted quantile
                uses the Hazen convention and does not match ``reduce(how="quantile")``); must be
                ``None`` otherwise.

        Returns:
            float: The area-weighted statistic over all faces.

        Raises:
            ValueError: ``how`` is not one of ``WEIGHTED_HOWS``; the variable is not
                face-located; or its length does not match the number of faces.

        Examples:
            - Area-weighted mean: the larger face (area 4, value 10) outweighs the
              smaller (area 2, value 20), pulling the mean below the unweighted 15.0:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 4.0, 4.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 2.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"v": np.array([10.0, 20.0])},
                ... )
                >>> round(mesh.weighted("v"), 3)
                13.333

                ```
            - The area-weighted quantile clamps to the data range at the extremes:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 4.0, 4.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 2.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"v": np.array([10.0, 20.0])},
                ... )
                >>> mesh.weighted("v", how="quantile", q=0.0)
                10.0
                >>> mesh.weighted("v", how="quantile", q=1.0)
                20.0

                ```
        """
        if how not in WEIGHTED_HOWS:
            # Validate up front, mirroring the raster weighted path: the kernel's catch-all
            # else-branch would otherwise return the standard deviation for any unsupported
            # `how`, a silent wrong answer.
            raise ValueError(
                f"weighted: how must be one of {sorted(WEIGHTED_HOWS)}, got {how!r}."
            )
        if how == "quantile":
            if q is None or isinstance(q, bool) or not 0.0 <= float(q) <= 1.0:
                raise ValueError(
                    f"weighted(how='quantile') needs q in [0, 1], got {q!r}."
                )
        elif q is not None:
            raise ValueError(
                f"weighted(q=...) is only valid with how='quantile', not {how!r}."
            )
        var, arr = self._element_values(variable_name, time_index=time_index)
        if var.location != "face":
            raise ValueError(
                "area-weighted statistics need a face-located variable; "
                f"{variable_name!r} is on {var.location!r}."
            )
        areas = self._mesh.face_areas
        if arr.shape[-1] != areas.shape[0]:
            raise ValueError(
                f"variable {variable_name!r} has {arr.shape[-1]} face value(s) but the "
                f"mesh has {areas.shape[0]} face(s)."
            )
        # `arr` already carries NaN for gaps, so `ndv=None` and `skipna=True` reduce over
        # the single face axis; the kernel keeps the axis as length 1, squeezed off here.
        result = weighted_statistic(arr, areas, (0,), how, None, True, q)
        return float(np.asarray(result).ravel()[0])

    def zonal_stats(
        self,
        zones: Any,
        *,
        variable_name: str,
        stats: tuple[str, ...] = ("mean",),
        weighted: bool = True,
        time_index: int = 0,
    ) -> pd.DataFrame:
        """Aggregate a face variable within polygons, weighted by face area.

        The mesh counterpart of :func:`pyramids.dataset.ops._zonal.zonal_stats`. Each face
        is assigned to the zone whose polygon contains its centroid (the mesh analogue of a
        raster pixel's cell centre), then the shared
        :func:`pyramids.base._reductions.reduce_by_label` reduces the face values per zone.
        With ``weighted`` (the default) the **averaging** statistics (``mean`` / ``std`` /
        ``var``) are area-weighted, so a zone's mean is area-exact — a capability the raster
        path does not yet have. ``sum`` is always the plain Σ value (never an area integral),
        and ``count`` / ``min`` / ``max`` are weight-invariant.

        Args:
            zones: Polygons, as a :class:`~pyramids.feature.FeatureCollection` (or any
                object exposing ``geometry``, ``index`` and ``len``). CRS must match the
                mesh; reproject first if not.
            variable_name: Name of a **face-located** data variable.
            stats: Statistics per zone, any of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
                ``"std"``, ``"var"``, ``"count"``. Defaults to ``("mean",)``.
            weighted: Area-weight the averaging statistics (``mean`` / ``std`` / ``var``).
                Defaults to True. ``sum`` / ``count`` / ``min`` / ``max`` are unaffected.
            time_index: Step for a temporal variable. Defaults to 0.

        Returns:
            pandas.DataFrame: Indexed like ``zones``; one column per statistic. A zone
            containing no face centroid is NaN (``count`` ``0.0``).

        Raises:
            ValueError: The variable is not face-located, or the zones' CRS disagrees with
                the mesh CRS.
        """
        var, arr = self._element_values(variable_name, time_index=time_index)
        if var.location != "face":
            raise ValueError(
                "mesh zonal_stats needs a face-located variable; "
                f"{variable_name!r} is on {var.location!r}."
            )
        zone_crs = getattr(zones, "crs", None)
        if zone_crs is not None and self.epsg is not None:
            zone_epsg = zone_crs.to_epsg()
            if zone_epsg is not None and zone_epsg != self.epsg:
                raise ValueError(
                    f"zonal_stats: zones CRS (EPSG:{zone_epsg}) does not match mesh CRS "
                    f"(EPSG:{self.epsg}). Reproject the zones to the mesh CRS first."
                )
        geometries = list(zones.geometry)
        cx, cy = self._mesh.face_centroids
        centroids = shapely.points(cx, cy)
        tree = shapely.STRtree(geometries)
        # Face centroid within a zone polygon -> that zone. A centroid in more than one
        # (overlapping zones) keeps the first, mirroring a raster pixel landing in one cell.
        centroid_idx, zone_idx = tree.query(centroids, predicate="within")
        labels = np.full(self._mesh.n_face, -1, dtype=np.intp)
        for face_i, zone_i in zip(centroid_idx, zone_idx):
            if labels[face_i] == -1:
                labels[face_i] = zone_i
        n_zones = len(geometries)
        columns: dict[str, np.typing.NDArray] = {}
        # Area weighting applies only to the averaging statistics. `sum` stays the plain
        # Σ value (a weighted sum would be an area integral — a different quantity, and a
        # silent surprise); `count` / `min` / `max` are weight-invariant anyway.
        averaging = [s for s in stats if s in ("mean", "std", "var")]
        plain = [s for s in stats if s not in ("mean", "std", "var")]
        if weighted and averaging:
            columns.update(
                reduce_by_label(
                    arr, labels, n_zones, averaging, weights=self._mesh.face_areas
                )
            )
            if plain:
                columns.update(reduce_by_label(arr, labels, n_zones, plain))
        else:
            columns.update(reduce_by_label(arr, labels, n_zones, list(stats)))
        return pd.DataFrame({stat: columns[stat] for stat in stats}, index=zones.index)

    def _temporal_names(self) -> list[str]:
        """Names of the variables that carry a real time dimension."""
        return [name for name, var in self._data_variables.items() if var.has_time]

    def _rebuild(self, data_variables: dict[str, MeshVariable]) -> UgridDataset:
        """A new dataset with the same mesh/metadata but different data variables.

        Every along-time member below is immutable — it derives a fresh dataset rather than
        mutating in place — mirroring :meth:`sel_time` / :meth:`crop`. The mesh topology,
        global attributes, topology info, CRS and source file name all carry over unchanged,
        and the result is of the receiver's own class so a subclass is not downgraded.
        """
        return type(self)(
            mesh=self._mesh,
            data_variables=data_variables,
            global_attributes=self._global_attributes,
            topology_info=self._topology_info,
            crs_wkt=self._crs_wkt,
            file_name=self._file_name,
        )

    def _require_temporal(self, operation: str) -> list[str]:
        """The temporal variable names, or a clear error when there are none."""
        temporal = self._temporal_names()
        if not temporal:
            raise ValueError(
                f"{operation} needs at least one variable with a time dimension; this "
                f"dataset has none."
            )
        return temporal

    def _check_time_dim(self, operation: str, dim: str | None) -> None:
        """Refuse a `dim` that is not the time axis.

        The along-time members take `dim` for signature parity with the raster side, where a
        cube has several non-spatial dimensions to choose from. A mesh has exactly one: the
        element axis is the mesh's own, and carrying a value along it is meaningless. Accept
        ``"time"`` (or ``None``) and refuse anything else rather than silently using time.

        Args:
            operation: The calling method's name, for the message.
            dim: The dimension the caller named, or ``None``.

        Raises:
            ValueError: ``dim`` is neither ``None`` nor ``"time"``.
        """
        if dim is not None and dim != "time":
            raise ValueError(
                f"{operation}: a mesh has one along-able dimension, 'time'; got dim={dim!r}. "
                "The element axis is the mesh's own axis, which carries no ordering to run "
                "along."
            )

    def reduce(
        self,
        how: str,
        *,
        skipna: bool = True,
        q: float | None = None,
        groupby: Any = None,
    ) -> UgridDataset:
        """Reduce the time dimension of every temporal variable, whole or in groups.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.reduce` over time. Each
        temporal variable is reduced along its time axis through the shared
        :func:`pyramids.base._reductions.reduce_axis` — the very kernel the raster reductions
        use. Non-temporal variables are kept unchanged.

        Without ``groupby`` the whole axis collapses to one statistic and the variable becomes
        static. With ``groupby`` each group is reduced on its own slab and the results are
        stacked back along the time axis, so the variable stays temporal with one step per
        group — the same shape contract the raster grouped reduce has. For a ``(4, 2)``
        variable over four monthly steps, ``reduce("mean")`` answers shape ``(2,)`` with
        ``has_time`` False, while ``reduce("mean", groupby="MS")`` answers shape ``(2, 2)``
        with ``has_time`` still True.

        Args:
            how: One of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``, ``"var"``,
                ``"median"``, ``"prod"``, ``"quantile"``, ``"count"``, ``"all"`` or ``"any"``.
            skipna: Skip the declared no-data value and NaN. Defaults to True.
            q: The quantile for ``how="quantile"``; must stay ``None`` otherwise.
            groupby: Reduce within groups of steps instead of over the whole axis. Two
                spellings, matching the raster grouped reduce: a **sequence of per-step
                labels** the same length as the time axis (grouped in first-appearance order,
                so the caller controls the order), or a **pandas offset alias** (``"MS"``,
                ``"YS"``, …) binned against the dataset's date-like time coordinates. Each
                group's label becomes the result's ``time_values`` entry, and empty bins are
                dropped. ``None`` reduces the whole axis. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset. Without ``groupby`` each temporal variable is
            collapsed over time into a static per-element variable; with ``groupby`` each one
            keeps its time dimension, at one step per group.

        Raises:
            ValueError: ``how`` is not a known reduction; ``q`` is given for a non-quantile
                ``how`` or missing for ``how="quantile"``; no variable has a time dimension;
                a temporal variable has no loaded data; a ``groupby`` label sequence's length
                is not the time length; or a ``groupby`` offset alias is asked of time
                coordinates that are step positions, or that pandas cannot read as dates.

        Examples:
            - Mean over two time steps of a per-face variable:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"d": np.array([[1.0, 2.0], [3.0, 4.0]])},
                ...     data_locations={"d": "face"},
                ... )
                >>> mesh["d"].has_time
                True
                >>> reduced = mesh.reduce("mean")
                >>> reduced["d"].data.tolist()
                [2.0, 3.0]
                >>> reduced["d"].has_time
                False

                ```
        """
        _check_reduce_how(how, q)
        self._require_temporal("reduce")
        # all/any collapse to a uint8 flag band whose all-gap value is FLAG_NO_DATA (255);
        # the collapsed variable must declare that so a consumer does not read 255 as data.
        result_nodata = FLAG_NO_DATA if how in ("all", "any") else None
        grouped = (
            _group_positions(self._time_coords("reduce"), groupby)
            if groupby is not None
            else None
        )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to reduce.")
            arr = np.asarray(data)
            nodata = result_nodata if result_nodata is not None else var.nodata
            if grouped is None:
                reduced = reduce_axis(arr, axis, how, skipna, var.nodata, q)
                new_vars[name] = _static_from(
                    var, np.asarray(reduced), axis, nodata=nodata
                )
                continue
            labels, positions = grouped
            # Each group is reduced on its own slab and the results stacked back along the
            # time axis, so a grouped reduce keeps the dimension (shorter) instead of
            # collapsing it — the same shape contract the raster grouped reduce has.
            parts = [
                np.asarray(
                    reduce_axis(
                        np.take(arr, group, axis=axis), axis, how, skipna, var.nodata, q
                    )
                )
                for group in positions
            ]
            new_var = var.with_data(np.stack(parts, axis=axis))
            new_var.nodata = nodata
            new_var.attributes = {**var.attributes, "time_values": list(labels)}
            new_vars[name] = new_var
        return self._rebuild(new_vars)

    def resample(
        self,
        freq: str,
        how: str = "mean",
        *,
        skipna: bool = True,
        q: float | None = None,
    ) -> UgridDataset:
        """Bin the time axis by a calendar frequency and reduce each bin.

        Sugar over ``reduce(how, groupby=freq)``: the mesh counterpart of the raster
        ``resample``. The time coordinates must be date-like, since the bins come from a pandas
        offset alias (``"1MS"``, ``"YS"``, ``"7D"``, …). The time dimension is kept, one step
        per non-empty bin, stamped with **pandas' own label for that bin** — which is the bin's
        start for a start-anchored alias and its end for an end-anchored one. Over steps in
        January, February and March 2020, ``"MS"`` labels the bins 01-01 / 02-01 / 03-01 while
        ``"ME"`` labels them 01-31 / 02-29 / 03-31, and ``"W"`` labels each week by its Sunday.

        Args:
            freq: A pandas offset alias naming the bin width.
            how: The statistic each bin is reduced with. Defaults to ``"mean"``.
            skipna: Skip the declared no-data value and NaN. Defaults to True.
            q: The quantile, required when ``how="quantile"``.

        Returns:
            UgridDataset: A new dataset with one time step per bin.

        Raises:
            ValueError: No variable has a time dimension, the coordinates are not date-like,
                or ``how``/``q`` is invalid.
        """
        return self.reduce(how, skipna=skipna, q=q, groupby=freq)

    def mean(self, *, skipna: bool = True) -> UgridDataset:
        """Mean over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time mean.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("mean", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> UgridDataset:
        """Sum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time sum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("sum", skipna=skipna)

    def min(self, *, skipna: bool = True) -> UgridDataset:
        """Minimum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time minimum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> UgridDataset:
        """Maximum over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time maximum.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("max", skipna=skipna)

    def std(self, *, skipna: bool = True) -> UgridDataset:
        """Std-dev over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time std.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("std", skipna=skipna)

    def var(self, *, skipna: bool = True) -> UgridDataset:
        """Variance over time of every temporal variable (thin wrapper over :meth:`reduce`).

        Args:
            skipna: Skip the declared no-data value and NaN. Defaults to True.

        Returns:
            UgridDataset: A new dataset; each temporal variable collapsed to its time variance.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self.reduce("var", skipna=skipna)

    def _transform_time(self, operation: str, fn: Any) -> UgridDataset:
        """Apply a shape-preserving ``fn(float_data, axis)`` to each temporal variable.

        ``fn`` receives the variable's values as float64 with no-data blanked to NaN, and the
        integer time axis, and returns an array of the same shape. Static variables are kept
        unchanged. Shared by the cumulative, shift, fill and rolling members.
        """
        self._require_temporal(operation)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(
                    f"Variable {name!r} has no loaded data for {operation}."
                )
            transformed = fn(gaps_as_nan(np.asarray(data), var.nodata), axis)
            new_vars[name] = var.with_data(np.asarray(transformed))
        return self._rebuild(new_vars)

    def cumsum(self) -> UgridDataset:
        """Cumulative sum along time for every temporal variable.

        Returns:
            UgridDataset: A new dataset in which each temporal variable holds the running
            sum of its steps along the time axis; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.

        Examples:
            - Running sum over three time steps of a per-face variable:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"d": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
                ...     data_locations={"d": "face"},
                ... )
                >>> mesh.cumsum()["d"].data.tolist()
                [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]

                ```
        """
        return self._transform_time("cumsum", lambda d, axis: np.cumsum(d, axis=axis))

    def cumprod(self) -> UgridDataset:
        """Cumulative product along time for every temporal variable.

        Returns:
            UgridDataset: A new dataset in which each temporal variable holds the running
            product of its steps along the time axis; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        return self._transform_time("cumprod", lambda d, axis: np.cumprod(d, axis=axis))

    def shift(self, periods: int = 1) -> UgridDataset:
        """Shift each temporal variable ``periods`` steps along time, vacated steps NaN.

        Args:
            periods: Steps to shift; negative shifts towards the start. Defaults to 1.

        Returns:
            UgridDataset: A new dataset with each temporal variable shifted along time, the
            vacated steps filled with NaN; static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        return self._transform_time(
            "shift", lambda d, axis: shifted(d, axis, periods, np.nan)
        )

    def ffill(
        self, *, limit: int | None = None, dim: str | None = None
    ) -> UgridDataset:
        """Forward-fill gaps along time from the last valid step.

        Args:
            limit: Maximum consecutive gaps one valid step may fill, or ``None`` for no limit.
            dim: The dimension to fill along. A mesh fills along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset with each temporal variable's gaps carried forward
            from the previous valid step; static variables are unchanged.

        Raises:
            ValueError: ``dim`` names anything but ``"time"``, no variable has a time
                dimension, or a temporal variable has no loaded data.
        """
        self._check_time_dim("ffill", dim)
        return self._transform_time(
            "ffill", lambda d, axis: pushed(d, axis, limit, False)
        )

    def bfill(
        self, *, limit: int | None = None, dim: str | None = None
    ) -> UgridDataset:
        """Back-fill gaps along time from the next valid step.

        Args:
            limit: Maximum consecutive gaps one valid step may fill, or ``None`` for no limit.
            dim: The dimension to fill along. A mesh fills along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset with each temporal variable's gaps carried backward
            from the next valid step; static variables are unchanged.

        Raises:
            ValueError: ``dim`` names anything but ``"time"``, no variable has a time
                dimension, or a temporal variable has no loaded data.
        """
        self._check_time_dim("bfill", dim)
        return self._transform_time(
            "bfill", lambda d, axis: pushed(d, axis, limit, True)
        )

    def interpolate_na(
        self,
        *,
        method: str = "linear",
        limit: int | None = None,
        use_coordinate: bool = True,
        dim: str | None = None,
    ) -> UgridDataset:
        """Fill each interior time gap by interpolating between the steps on either side.

        Distance is measured along the time coordinate by default, so unevenly spaced steps
        weight their neighbours accordingly; pass ``use_coordinate=False`` to measure by step
        position instead. A leading or trailing gap has only one neighbour and is left alone.

        Args:
            method: One of ``"linear"`` (distance-weighted), ``"nearest"``, or the scipy spline
                kinds ``"slinear"`` / ``"quadratic"`` / ``"cubic"`` (each falling back to linear on
                a variable with too few valid steps). Defaults to ``"linear"``.
            limit: Maximum consecutive gaps a run may fill, or ``None`` for no limit.
            use_coordinate: Measure distance along the time coordinate rather than by step
                position. With time values ``[0, 1, 10]`` and a gap at the middle step, the
                coordinate reading fills ``1.0`` and the positional reading ``5.0``. Date-like
                coordinates count as distances too (converted to nanoseconds), so unevenly
                spaced dates weight their neighbours; a coordinate with no measurable spacing
                (plain labels) falls back to step positions. Defaults to ``True``.
            dim: The dimension to interpolate along. A mesh interpolates along ``"time"``
                only, so any other name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset with each temporal variable's interior gaps
            interpolated along time; static variables are unchanged.

        Raises:
            ValueError: ``method`` is not one of the accepted interpolations; no variable has a
                time dimension; or a temporal variable has no loaded data.
        """
        if method not in INTERP_METHODS:
            # Validate up front, before the time check, so a bogus method is reported as such
            # even on a mesh with no temporal variable — matching the raster interpolate_na.
            raise ValueError(
                f"interpolate_na() takes method="
                f"{' or '.join(repr(one) for one in INTERP_METHODS)}, got {method!r}."
            )

        self._check_time_dim("interpolate_na", dim)
        stamps = self.time_values if use_coordinate else None
        axis_values = None
        if stamps is not None:
            axis_values = _axis_distances(stamps)

        def _fill(data: np.ndarray, axis: int) -> np.ndarray:
            size = data.shape[axis]
            if axis_values is not None and axis_values.shape[0] == size:
                positions = axis_values
            else:
                positions = np.arange(size, dtype="float64")
            return np.asarray(interpolated(data, axis, positions, method, limit))

        return self._transform_time("interpolate_na", _fill)

    def rolling(
        self,
        window: int,
        how: str = "mean",
        *,
        center: bool = False,
        min_periods: int | None = None,
        q: float | None = None,
        dim: str | None = None,
    ) -> UgridDataset:
        """Rolling-window reduction along time for every temporal variable.

        Each time step becomes a statistic of the window of steps it owns (ending at it, or
        centred on it), cut where the window runs off the axis. Gaps are skipped; a step whose
        window holds fewer than ``min_periods`` valid cells is NaN. The time dimension keeps
        its length.

        Args:
            window: Steps per window.
            how: The reduction over each window (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
                ``"std"``, ``"var"``, ``"median"``, ``"quantile"``). Defaults to ``"mean"``.
            center: Centre each window on its step rather than ending at it. Defaults to False.
            min_periods: Valid cells a window needs before its step holds a value. ``None``
                means the full ``window``, matching the raster `rolling`, so a step whose
                window runs off the axis is NaN rather than a partial statistic. Defaults to
                ``None``.
            q: The quantile in ``[0, 1]``, required when ``how="quantile"`` and rejected
                otherwise. Defaults to ``None``.
            dim: The dimension to roll along. A mesh rolls along ``"time"`` only, so any
                other name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset with each temporal variable replaced by its rolling
            statistic along time; static variables are unchanged.

        Raises:
            ValueError: ``how`` is not a known reduction; ``q`` is given for a non-quantile
                ``how`` or missing for ``how="quantile"``; ``window`` is below 1;
                ``min_periods`` falls outside ``[1, window]``; ``dim`` names anything but
                ``"time"``; no variable has a time dimension; or a temporal variable has no
                loaded data.
        """
        if how not in _ROLLING_HOWS:
            # Restrict to the documented statistics. The counting reductions
            # (count/all/any) are excluded on purpose: a window's all/any flag carries the
            # 255 FLAG_NO_DATA sentinel, which would surface as a literal data value here.
            raise ValueError(
                f"rolling how must be one of {sorted(_ROLLING_HOWS)}, got {how!r}."
            )
        # Same guard as every other along-time member, through the shared check rather than
        # a second copy of it with its own wording.
        self._check_time_dim("rolling", dim)
        if how == "quantile" and q is None:
            raise ValueError("rolling(how='quantile') needs q in [0, 1].")
        if how != "quantile" and q is not None:
            raise ValueError(
                f"rolling(q=...) is only valid with how='quantile', not {how!r}."
            )
        if window < 1:
            raise ValueError(f"rolling: window must be >= 1, got {window!r}.")
        # `None` means the full window, as the raster rolling defaults — a step whose window
        # runs off the axis then has no statistic rather than a partial one.
        needed = window if min_periods is None else int(min_periods)
        if not 1 <= needed <= window:
            raise ValueError(
                f"rolling: min_periods must be between 1 and window ({window}), got "
                f"{min_periods!r}."
            )

        def _roll(data: np.ndarray, axis: int) -> np.ndarray:
            moved = np.moveaxis(data, axis, 0)
            size = moved.shape[0]
            out = np.full_like(moved, np.nan, dtype="float64")
            for step in range(size):
                members = window_members(step, size, window, center)
                block = moved[members]
                valid = np.sum(~np.isnan(block), axis=0)
                reduced = reduce_axis(block, 0, how, True, None, q)
                out[step] = np.where(valid >= needed, reduced, np.nan)
            return np.moveaxis(out, 0, axis)

        return self._transform_time("rolling", _roll)

    def _time_length(self, operation: str) -> int:
        """The shared time-axis length across all temporal variables.

        The positional/label selectors build one index set and apply it to every temporal
        variable, so they require a single shared length. Validate that here rather than let
        a mismatch leak a raw numpy ``IndexError`` (or silently trim against the wrong axis).

        Raises:
            ValueError: No variable has a time dimension, or temporal variables disagree on
                their time length.
        """
        temporal = self._require_temporal(operation)
        lengths = {name: self._data_variables[name].n_time_steps for name in temporal}
        distinct = set(lengths.values())
        if len(distinct) > 1:
            raise ValueError(
                f"{operation} needs every temporal variable to share one time length, but "
                f"they differ: {lengths}. Select per variable, or align them first."
            )
        return distinct.pop()

    def _select_steps(self, indices: np.typing.NDArray) -> UgridDataset:
        """Keep the time steps at ``indices`` (in order) for every temporal variable.

        Trims each temporal variable's data along its time axis and, when the variable
        carries a ``time_values`` coordinate attribute, trims that to match. Static variables
        are kept unchanged. The time dimension is retained (shorter), so the result stays
        temporal.
        """
        indices = np.asarray(indices, dtype=np.intp)
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to select.")
            new_var = var.with_data(np.take(np.asarray(data), indices, axis=axis))
            times = var.attributes.get("time_values")
            if times is not None:
                new_var.attributes = {
                    **var.attributes,
                    "time_values": [times[int(i)] for i in indices],
                }
            new_vars[name] = new_var
        return self._rebuild(new_vars)

    def _select_elements(
        self, element: int | slice | list[int] | np.ndarray
    ) -> UgridDataset:
        """Keep the faces at ``element`` positions, renumbering the mesh around them.

        The element axis is the mesh's own axis, so selecting along it is a *topology* change,
        not a slice: the kept faces' nodes are renumbered compactly and the edges filtered, the
        way :meth:`clip` and :meth:`subset_by_bounds` already do. This shares their machinery
        (:func:`pyramids.netcdf.ugrid.spatial.subset_mesh_by_faces`) rather than re-deriving it.

        Args:
            element: An ``int`` face index, a ``slice``, or a sequence of face indices.

        Returns:
            UgridDataset: A new dataset whose mesh holds only the selected faces.

        Raises:
            IndexError: A face index is out of range for this mesh. A negative index counts
                from the end, so ``-1`` is the last face.
        """
        count = self._mesh.n_face
        if isinstance(element, slice):
            faces = [int(i) for i in np.arange(count)[element]]
        else:
            asked = [int(i) for i in np.atleast_1d(np.asarray(element, dtype=np.intp))]
            # A negative index counts from the end, as `isel(time=-1)` already does through
            # np.take. Refusing it here left one method with two index conventions.
            faces = [i + count if -count <= i < 0 else i for i in asked]
        out_of_range = [i for i in faces if i < 0 or i >= count]
        if out_of_range:
            raise IndexError(
                f"isel: face index/indices {out_of_range} out of range for a mesh with "
                f"{count} face(s)."
            )
        mesh, data_variables = subset_mesh_by_faces(self, faces)
        return self._wrap_subset(mesh, data_variables)

    def isel(
        self,
        time: int | slice | list[int] | np.ndarray | None = None,
        *,
        element: int | slice | list[int] | np.ndarray | None = None,
    ) -> UgridDataset:
        """Select by position along the time axis, the element axis, or both.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.isel`. Over **time** an integer
        selects a single step and collapses the time dimension (the variables become static);
        a slice or sequence keeps the time dimension, trimmed to the chosen steps. Over
        **element** the selection is a topology change: the chosen faces are kept and the mesh
        is renumbered around them (as :meth:`clip` does), so the result is a smaller mesh.
        Giving both applies the element selection first, then the time selection.

        Args:
            time: An ``int`` step, a ``slice``, or a sequence of integer step positions.
                ``None`` (the default) leaves the time axis alone.
            element: An ``int`` face index, a ``slice``, or a sequence of face indices.
                ``None`` (the default) leaves the mesh alone.

        Returns:
            UgridDataset: A new dataset with the selection applied.

        Raises:
            ValueError: Neither ``time`` nor ``element`` was given, or ``time`` was asked of a
                dataset with no time dimension.
            IndexError: A face index is out of range for this mesh.

        Examples:
            - Keep one face, renumbering the mesh around it:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import UgridDataset
                >>> mesh = UgridDataset.from_arrays(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
                ...     data={"depth": np.array([2.0, 4.0])},
                ... )
                >>> one = mesh.isel(element=0)
                >>> one.n_face, one.n_node
                (1, 3)
                >>> one["depth"].data.tolist()
                [2.0]

                ```
        """
        if time is None and element is None:
            raise ValueError("isel needs `time`, `element`, or both.")
        if element is not None:
            selected = self._select_elements(element)
            return selected if time is None else selected.isel(time)
        n = self._time_length("isel")
        if isinstance(time, (int, np.integer)):
            return self._collapse_to_step(int(time))
        if isinstance(time, slice):
            indices = np.arange(n)[time]
        else:
            indices = np.asarray(time, dtype=np.intp)
        return self._select_steps(indices)

    def _collapse_to_step(self, index: int) -> UgridDataset:
        """Take one time step from every temporal variable, dropping the time dimension.

        The single-step half of :meth:`isel`: each temporal variable becomes static, as
        :meth:`reduce` produces, while a static variable is carried through untouched.

        Args:
            index: The time position to take.

        Returns:
            UgridDataset: A new dataset whose temporal variables hold that step.

        Raises:
            ValueError: A temporal variable has no loaded data.
        """
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to select.")
            step = np.take(np.asarray(data), index, axis=axis)
            new_vars[name] = _static_from(var, np.asarray(step), axis)
        return self._rebuild(new_vars)

    def head(self, n: int = 5) -> UgridDataset:
        """Keep the first ``n`` time steps (fewer if the axis is shorter).

        Args:
            n: Number of leading steps to keep. Defaults to 5.

        Returns:
            UgridDataset: A new dataset trimmed to the first ``n`` steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._time_length("head")
        return self._select_steps(np.arange(min(n, length)))

    def tail(self, n: int = 5) -> UgridDataset:
        """Keep the last ``n`` time steps (fewer if the axis is shorter).

        Args:
            n: Number of trailing steps to keep. Defaults to 5.

        Returns:
            UgridDataset: A new dataset trimmed to the last ``n`` steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._time_length("tail")
        return self._select_steps(np.arange(max(length - n, 0), length))

    def thin(self, step: int) -> UgridDataset:
        """Keep every ``step``-th time step.

        Args:
            step: Stride; must be >= 1.

        Returns:
            UgridDataset: A new dataset keeping every ``step``-th step.

        Raises:
            ValueError: ``step`` is less than 1, or no variable is temporal.
        """
        if step < 1:
            raise ValueError(f"thin step must be >= 1, got {step}.")
        length = self._time_length("thin")
        return self._select_steps(np.arange(0, length, step))

    def drop_isel(self, indices: int | list[int] | np.ndarray) -> UgridDataset:
        """Drop the time steps at ``indices`` by position, keeping the rest.

        Args:
            indices: A step position, or a sequence of them, to drop (negative indices
                count from the end).

        Returns:
            UgridDataset: A new dataset without the dropped steps.

        Raises:
            ValueError: No variable has a time dimension.
        """
        length = self._time_length("drop_isel")
        drop = {int(i) % length for i in np.atleast_1d(np.asarray(indices))}
        keep = np.array([i for i in range(length) if i not in drop], dtype=np.intp)
        return self._select_steps(keep)

    def squeeze(self) -> UgridDataset:
        """Drop the time dimension when it has length 1; otherwise return unchanged.

        Mirrors :meth:`pyramids.netcdf.NetCDF.squeeze` for the time axis: a single-step
        temporal variable becomes static, a multi-step one is left alone.

        Returns:
            UgridDataset: A new dataset with single-step temporal variables collapsed to
            static, or ``self`` when there is nothing to squeeze.

        Raises:
            ValueError: A single-step temporal variable has no loaded data.
        """
        # Decide per variable, not from the first temporal one: with mixed time lengths a
        # later single-step variable must still be squeezed even if the first has many steps.
        if not any(
            var.has_time and var.n_time_steps == 1
            for var in self._data_variables.values()
        ):
            return self
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time or var.n_time_steps != 1:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to squeeze.")
            new_vars[name] = _static_from(
                var, np.squeeze(np.asarray(data), axis=axis), axis
            )
        return self._rebuild(new_vars)

    def diff(self, n: int = 1) -> UgridDataset:
        """Discrete difference along time (``out[i] = x[i] - x[i-1]``), ``n`` times.

        The time dimension shortens by ``n``; a ``time_values`` coordinate, if present, drops
        its first ``n`` entries to stay aligned with the result.

        Args:
            n: The number of successive differences. Defaults to 1.

        Returns:
            UgridDataset: A new dataset with each temporal variable differenced along time
            (its time axis shorter by ``n``); static variables are unchanged.

        Raises:
            ValueError: No variable has a time dimension, or a temporal variable has no
                loaded data.
        """
        self._require_temporal("diff")
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to diff.")
            diffed = np.diff(gaps_as_nan(np.asarray(data), var.nodata), n=n, axis=axis)
            new_var = var.with_data(diffed)
            times = var.attributes.get("time_values")
            if times is not None:
                new_var.attributes = {**var.attributes, "time_values": list(times[n:])}
            new_vars[name] = new_var
        return self._rebuild(new_vars)

    def _extremum_along_time(
        self, operation: str, *, largest: bool, as_coordinate: bool
    ) -> UgridDataset:
        """Locate each element's extreme time step, as a position or as a coordinate.

        The time axis is dropped — the answer is one value per element — so the result is a
        static variable, as :meth:`reduce` produces. Gaps are left out: the search runs over the
        no-data-masked values, and an element with no valid step at all has no answer, which the
        position form declares as ``-1`` and the coordinate form as NaN (mirroring the raster
        ``argmin`` / ``idxmin``).

        Args:
            operation: The calling method's name, for error messages.
            largest: Search for the maximum rather than the minimum.
            as_coordinate: Answer with the time coordinate value instead of the step position.

        Returns:
            UgridDataset: A new dataset whose temporal variables became static; static
            variables are carried over unchanged.

        Raises:
            ValueError: No variable has a time dimension, a temporal variable has no loaded
                data, or the coordinate form was asked of date-like time coordinates, which
                cannot be expressed as the float64 the answer carries.
        """
        self._require_temporal(operation)
        coords = self._time_coords(operation) if as_coordinate else None
        if coords is not None and not all(
            isinstance(stamp, (int, float, np.integer, np.floating))
            and not isinstance(stamp, bool)
            for stamp in coords
        ):
            # The coordinate form answers float64, so a date-like coordinate cannot be
            # expressed. Say so, rather than letting the float() cast surface a raw
            # TypeError about Timestamp from inside numpy.
            raise ValueError(
                f"{operation}() answers the time coordinate as a float, so it needs numeric "
                "time coordinates; this dataset's are date-like. Use argmin()/argmax() for "
                "the step position instead."
            )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                new_vars[name] = var
                continue
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data.")
            values, nodata = _extremum_values(
                gaps_as_nan(np.asarray(data), var.nodata),
                axis,
                largest=largest,
                coords=coords if as_coordinate else None,
            )
            new_vars[name] = _static_from(var, values, axis, nodata=nodata)
        return self._rebuild(new_vars)

    def argmin(self) -> UgridDataset:
        """The time-step position of each element's minimum.

        Returns:
            UgridDataset: Static ``int64`` variables holding the step index of the minimum,
            declaring ``-1`` for an element with no valid step.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._extremum_along_time("argmin", largest=False, as_coordinate=False)

    def argmax(self) -> UgridDataset:
        """The time-step position of each element's maximum.

        Returns:
            UgridDataset: Static ``int64`` variables holding the step index of the maximum,
            declaring ``-1`` for an element with no valid step.

        Raises:
            ValueError: No variable has a time dimension.
        """
        return self._extremum_along_time("argmax", largest=True, as_coordinate=False)

    def idxmin(self) -> UgridDataset:
        """The time **coordinate** of each element's minimum.

        Where :meth:`argmin` answers a step position, this answers the value of the time
        coordinate at that step, so it needs numeric time coordinates.

        Returns:
            UgridDataset: Static ``float64`` variables holding the time coordinate of the
            minimum, NaN where an element has no valid step.

        Raises:
            ValueError: No variable has a time dimension, or the time coordinates are
                date-like rather than numeric (use `argmin` / `argmax` for the step
                position instead).
        """
        return self._extremum_along_time("idxmin", largest=False, as_coordinate=True)

    def idxmax(self) -> UgridDataset:
        """The time **coordinate** of each element's maximum.

        Returns:
            UgridDataset: Static ``float64`` variables holding the time coordinate of the
            maximum, NaN where an element has no valid step.

        Raises:
            ValueError: No variable has a time dimension, or the time coordinates are
                date-like rather than numeric (use `argmin` / `argmax` for the step
                position instead).
        """
        return self._extremum_along_time("idxmax", largest=True, as_coordinate=True)

    def _time_coords(self, operation: str) -> list:
        """The time coordinate values, or a clear error when the dataset has no time."""
        coords = self.time_values
        if coords is None:
            raise ValueError(
                f"{operation} needs a time dimension; this dataset has none."
            )
        return list(coords)

    def sel(
        self,
        time: Any = None,
        *,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> UgridDataset:
        """Select time steps by coordinate value, exactly or by nearest.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.sel` over time. A scalar selects
        one step and collapses the time dimension; a sequence keeps it. Values are matched
        against the dataset's :attr:`time_values` (which fall back to step positions when the
        file carries no explicit time coordinate). ``method="nearest"`` snaps each value to the
        closest coordinate instead of requiring an exact hit, and ``tolerance`` caps how far it
        may snap — the same contract the raster ``sel`` uses, including raising ``KeyError`` (not
        ``ValueError``) when a snap would exceed ``tolerance``.

        Args:
            time: A coordinate value, or a sequence of them.
            method: ``None`` (the default) matches exactly; ``"nearest"`` snaps to the closest
                coordinate, which needs numeric coordinates.
            tolerance: The furthest a ``method="nearest"`` snap may travel. Only valid with
                ``"nearest"``, and must not be negative.

        Returns:
            UgridDataset: A new dataset with the selected steps.

        Raises:
            ValueError: No time dimension; an unknown ``method``; ``tolerance`` without
                ``method="nearest"`` or negative; a value is not among the coordinates (exact
                match); or ``"nearest"`` was asked of non-numeric coordinates.
            KeyError: A ``"nearest"`` snap would exceed ``tolerance``.
        """
        if time is None:
            # `time` is optional in the signature for parity with the raster `sel`, but a
            # missing value has no meaning here. Say so, as `isel` does, instead of
            # reporting that the coordinate [None] was not found.
            raise ValueError(
                "sel needs a time value to select; pass one, or use isel() to select by "
                "position."
            )
        coords = self._time_coords("sel")
        if method not in (None, "nearest"):
            raise ValueError(f"sel: method must be None or 'nearest', got {method!r}.")
        if tolerance is not None:
            if method != "nearest":
                raise ValueError("sel: tolerance is only valid with method='nearest'.")
            if float(tolerance) < 0:
                raise ValueError(
                    f"sel: tolerance must not be negative, got {tolerance!r}."
                )
        wanted = list(time) if isinstance(time, (list, tuple, np.ndarray)) else [time]
        positions = _resolve_time_positions(coords, wanted, method, tolerance)
        if isinstance(time, (list, tuple, np.ndarray)):
            result = self._select_steps(np.asarray(positions, dtype=np.intp))
        else:
            result = self.isel(positions[0])
        return result

    def drop_sel(self, values: Any) -> UgridDataset:
        """Drop the time steps whose coordinate is in ``values``, keeping the rest.

        Args:
            values: A coordinate value, or a sequence of them, to drop.

        Returns:
            UgridDataset: A new dataset without the steps whose coordinate is in ``values``.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._time_coords("drop_sel")
        drop = (
            set(values)
            if isinstance(values, (list, tuple, set))
            else {*np.atleast_1d(np.asarray(values)).tolist()}
        )
        keep = [i for i, coord in enumerate(coords) if coord not in drop]
        return self._select_steps(np.asarray(keep, dtype=np.intp))

    def sortby(self) -> UgridDataset:
        """Sort the time steps by their coordinate value (stable).

        Returns:
            UgridDataset: A new dataset with steps ordered by ascending time coordinate.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._time_coords("sortby")
        order = np.argsort(np.asarray(coords), kind="stable")
        return self._select_steps(order)

    def drop_duplicates(self) -> UgridDataset:
        """Keep the first step of each distinct time coordinate value.

        Returns:
            UgridDataset: A new dataset with duplicate-coordinate steps removed, first kept.

        Raises:
            ValueError: No variable has a time dimension.
        """
        coords = self._time_coords("drop_duplicates")
        seen: set = set()
        keep: list[int] = []
        for i, coord in enumerate(coords):
            if coord not in seen:
                seen.add(coord)
                keep.append(i)
        return self._select_steps(np.asarray(keep, dtype=np.intp))

    def dropna(
        self, how: str = "any", thresh: int | None = None, *, dim: str | None = None
    ) -> UgridDataset:
        """Drop time steps that do not hold enough valid cells.

        A step is kept only when **every** temporal variable meets the threshold at that
        step, so the variables stay time-aligned. ``how="any"`` requires every cell of a
        variable to be valid (a single gap drops the step); ``how="all"`` requires at least
        one; ``thresh`` overrides ``how`` with an explicit valid-cell count per variable.

        Args:
            how: ``"any"`` or ``"all"``. Defaults to ``"any"``.
            thresh: Explicit minimum valid-cell count per variable, overriding ``how``.
            dim: The dimension to drop along. A mesh drops along ``"time"`` only, so any other
                name is refused; ``None`` means time. Defaults to ``None``.

        Returns:
            UgridDataset: A new dataset with the surviving steps.

        Raises:
            ValueError: ``how`` is neither ``"any"`` nor ``"all"``; ``dim`` names anything but
                ``"time"``; no variable has a time dimension; a temporal variable has no
                loaded data; or the threshold would drop **every** step, which would leave a
                temporal dataset with no steps for anything downstream to read.
        """
        if how not in ("any", "all"):
            raise ValueError(f"dropna how must be 'any' or 'all', got {how!r}.")
        self._check_time_dim("dropna", dim)
        temporal = self._require_temporal("dropna")
        length = self._time_length("dropna")
        keep = np.ones(length, dtype=bool)
        for name in temporal:
            var = self._data_variables[name]
            axis = cast("int", var.time_index)
            data = var.data
            if data is None:
                raise ValueError(f"Variable {name!r} has no loaded data to dropna.")
            moved = np.moveaxis(gaps_as_nan(np.asarray(data), var.nodata), axis, 0)
            flat = moved.reshape(moved.shape[0], -1)
            per_step_valid = np.sum(~np.isnan(flat), axis=1)
            needed = flat.shape[1] if how == "any" else 1
            if thresh is not None:
                needed = thresh
            keep &= per_step_valid >= needed
        if not keep.any():
            # Returning a zero-length time axis would leave a "temporal" dataset with no
            # steps, which nothing downstream can read; the raster dropna refuses too.
            raise ValueError(
                f"dropna(how={how!r}, thresh={thresh!r}) would drop every one of the "
                f"{length} time step(s); nothing would be left."
            )
        return self._select_steps(np.flatnonzero(keep))

    def _assert_same_topology(self, other: UgridDataset, operation: str) -> None:
        """Refuse ``operation`` unless ``other`` sits on the same mesh as ``self``.

        Same mesh means the same node/face/edge counts and the same node coordinates and
        face-node connectivity — combining data across different meshes is meaningless.
        """
        mine, theirs = self._mesh, other._mesh
        same = (
            mine.n_node == theirs.n_node
            and mine.n_face == theirs.n_face
            and mine.n_edge == theirs.n_edge
            and np.array_equal(mine.node_x, theirs.node_x)
            and np.array_equal(mine.node_y, theirs.node_y)
            and np.array_equal(
                np.asarray(mine.face_node_connectivity.data),
                np.asarray(theirs.face_node_connectivity.data),
            )
        )
        if not same:
            raise AlignmentError(
                f"{operation} requires datasets on the same mesh topology "
                f"(matching node/face/edge counts, coordinates and connectivity)."
            )

    @_joins_meshes
    def concat(self, others: UgridDataset | list[UgridDataset]) -> UgridDataset:
        """Concatenate same-topology datasets along time.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.concat`. Every dataset must sit
        on the same mesh. Each temporal variable's data is concatenated along its time axis
        (and its ``time_values`` coordinate, when present); a static variable is taken from
        ``self`` unchanged. The variable set must match across datasets.

        Args:
            others: One dataset, or a list of them, to append after ``self`` in order.

        Returns:
            UgridDataset: A new dataset spanning the concatenated time axis.

        Raises:
            AlignmentError: A dataset sits on a different mesh. Note this is **not** a
                `ValueError` subclass, so a caller catching `ValueError` around `concat` has
                to catch `AlignmentError` instead.
            ValueError: A dataset carries a different variable set, or a variable has no
                loaded data.
        """
        parts = [
            self,
            *([others] if isinstance(others, UgridDataset) else list(others)),
        ]
        for other in parts[1:]:
            self._assert_same_topology(other, "concat")
            if set(other._data_variables) != set(self._data_variables):
                raise ValueError(
                    "concat requires the same variables in every dataset; got "
                    f"{sorted(self._data_variables)} vs {sorted(other._data_variables)}."
                )
        new_vars: dict[str, MeshVariable] = {}
        for name, var in self._data_variables.items():
            if not var.has_time:
                # Static in self: refuse if any other part has it temporal, rather than
                # silently drop that part's time series (the mirror of the temporal-in-self
                # / static-in-other case _concat_temporal_variable rejects).
                if any(part._data_variables[name].has_time for part in parts[1:]):
                    raise ValueError(
                        f"concat: variable {name!r} is static in one dataset but temporal "
                        "in another; they cannot be joined along time."
                    )
                new_vars[name] = var  # a static variable is taken from self, as-is
            else:
                new_vars[name] = _concat_temporal_variable(name, var, parts)
        return self._rebuild(new_vars)

    @_joins_meshes
    def merge(
        self,
        others: UgridDataset | list[UgridDataset],
        *,
        compat: str = "no_conflicts",
    ) -> UgridDataset:
        """Merge the variables of same-topology datasets into one.

        The mesh counterpart of :meth:`pyramids.netcdf.NetCDF.merge`. Every dataset must sit
        on the same mesh; the union of their variables is returned. A variable name present
        in more than one dataset is a conflict and is refused.

        Args:
            others: One dataset, or a list of them, whose variables join ``self``'s.
            compat: How to handle a variable name present in more than one dataset --
                ``"no_conflicts"`` refuses the collision, ``"override"`` keeps ``self``'s.
                Defaults to ``"no_conflicts"``.

        Returns:
            UgridDataset: A new dataset carrying every variable.

        Raises:
            AlignmentError: A dataset sits on a different mesh.
            ValueError: An unknown ``compat``, or a variable name collides under
                ``compat="no_conflicts"``.
        """
        if compat not in ("no_conflicts", "override"):
            raise ValueError(
                f"merge: compat must be 'no_conflicts' or 'override', got {compat!r}."
            )
        parts = [others] if isinstance(others, UgridDataset) else list(others)
        new_vars: dict[str, MeshVariable] = dict(self._data_variables)
        for other in parts:
            self._assert_same_topology(other, "merge")
            for name, var in other._data_variables.items():
                if name in new_vars:
                    if compat == "no_conflicts":
                        raise ValueError(
                            f"merge: variable {name!r} is present in more than one dataset; "
                            "rename it first, or pass compat='override'."
                        )
                    # compat="override": the first dataset that carries the name wins, which
                    # is what already sits in new_vars, so the duplicate is simply skipped.
                    continue
                new_vars[name] = var
        return self._rebuild(new_vars)

    def plot(
        self,
        variable_name: str,
        ax: Any = None,
        cmap: str = "viridis",
        title: str | None = None,
        basemap: bool | str | None = None,
        colorbar: bool | ColorBar | None = None,
        points: np.ndarray | PointOverlay | None = None,
        kind: str = "auto",
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        data_style: DataStyle | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a mesh data variable.

        N-6 — this facade now goes through the same module-level
        helper as the raster path. The mesh-specific dispatch lives in
        :func:`pyramids.dataset._plot_helpers.mesh_render`; both
        ``Dataset.plot``/``NetCDF.plot`` and ``UgridDataset.plot`` share
        the "resolve data, hand to a single helper" contract so the
        two formats no longer maintain independent plotting code paths.

        Args:
            variable_name: Name of the data variable to plot.
            ax: matplotlib Axes. Created if None.
            cmap: Colormap name.
            title: Plot title. Defaults to variable name.
            basemap: If True, add an OpenStreetMap basemap. If a string,
                use it as the tile provider name (e.g. "CartoDB.Positron").
                Default is None (no basemap). Requires the [viz] extra.
            colorbar (bool or ColorBar, optional): Colour-bar spec, part of the
                shared plot signature. A ``pyramids.plot.ColorBar(label=…, …)``
                configures the bar; ``False`` hides it and ``None`` (default) uses
                cleopatra's default (a bar is drawn). Only forwarded when set.
            points (np.ndarray or PointOverlay, optional): Accepted for signature
                symmetry with the raster plot family, but a **no-op here** — a mesh
                has no point-overlay layer (the mesh geometry is the data). Ignored.
            kind (str, optional): Accepted for signature symmetry with the raster
                plot family, but a **no-op here** — ``kind`` selects a raster
                renderer (``imshow``/``pcolormesh``); the mesh always renders via
                ``tripcolor``/``tricontour``. Ignored.
            color (ColorScaling, optional): Colour-scale spec
                ``pyramids.plot.ColorScaling`` (linear / power / sym-log / boundary /
                midpoint norm), e.g. ``ColorScaling.power(gamma=0.7)``. Default ``None``.
            contour (Contour, optional): Contour-line spec
                ``pyramids.plot.Contour(levels=…, label_kw=…)``. Default ``None``.
            data_style (DataStyle, optional): Data-style / relief spec
                ``pyramids.plot.DataStyle(style=…, hillshade=…)``. (A mesh has no
                cell-value overlay, so there is no ``cells`` param here.) Default ``None``.
            **kwargs: Additional arguments passed to mesh_render
                (forwarded to plot_mesh_data). Notably ``colorbar``
                (``bool``, default ``True``): pass ``colorbar=False`` to
                suppress the per-mesh colorbar when you want to attach a
                custom or shared one to ``glyph.ax``. Also ``style`` (name of
                a cleopatra ``DATA_STYLES`` preset, e.g. ``"flow_accumulation"``)
                and ``hillshade`` (``True`` or a params dict) to colour / relief-
                shade the mesh; both require cleopatra >= 0.24 (``hillshade``
                needs node-centered data). Distinct from
                :meth:`pyramids.dataset.Dataset.hillshade`, which *returns* a
                shaded-relief array.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the plot
                rendered. Use the returned object to access the matplotlib
                handles and the mappable:

                - ``glyph.fig`` / ``glyph.ax`` — Figure and Axes.
                - ``glyph.im`` — the mesh mappable (the
                  ``tripcolor``/``tricontour(f)`` artist) set by ``plot()``;
                  use it for a custom colorbar or ``glyph.im.set_clim(...)``.
                  It is ``None`` after :meth:`plot_outline` (an outline
                  carries no scalar mapping).
                - ``glyph.apply_style(style)`` (cleopatra >= 0.25) — re-apply a
                  ``DATA_STYLES`` preset by name in place, without re-plotting.

        Raises:
            ValueError: If the selected variable has no loaded data, or if
                `basemap` is requested while the dataset has no CRS (`epsg`).
        """
        var = self.get_data(variable_name)
        data = var.data
        if data is None:
            raise ValueError(f"Variable {variable_name!r} has no loaded data to plot.")
        if var.has_time:
            # First step along the variable's own time axis (see `_element_values`).
            data = np.take(np.asarray(data), 0, axis=cast("int", var.time_index))
        if title is None:
            title = variable_name
        if basemap and self.epsg is None:
            raise ValueError("UgridDataset must have a CRS (epsg) to use basemap.")
        # ``points`` / ``kind`` are part of the shared raster-family plot signature
        # but have no meaning for a mesh (no point overlay; the renderer is fixed to
        # tripcolor/tricontour), so they are accepted and ignored. ``colorbar`` and the
        # typed render groups (``color`` / ``contour`` / ``data_style``) map onto the mesh
        # backend and are forwarded only when set (so cleopatra's backend defaults are
        # preserved otherwise).
        if colorbar is not None:
            kwargs["colorbar"] = colorbar
        kwargs.update(
            _nonnull_group_kwargs(color=color, contour=contour, data_style=data_style)
        )
        result = _mesh_render(
            mesh=self._mesh,
            data=data,
            location=var.location,
            ax=ax,
            cmap=cmap,
            title=title,
            basemap=basemap,
            basemap_epsg=self.epsg,
            **kwargs,
        )
        return result

    def plot_outline(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot mesh wireframe.

        Args:
            ax: matplotlib Axes. Created if None.
            **kwargs: Additional arguments passed to plot_mesh_outline.

        Returns:
            cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph instance with the wireframe
                rendered. ``glyph.fig`` / ``glyph.ax`` are the matplotlib
                handles; ``glyph.im`` is ``None`` (an outline carries no
                scalar mapping, so no mappable is produced).
        """
        from pyramids.netcdf.ugrid.plot import plot_mesh_outline

        result = plot_mesh_outline(self._mesh, ax=ax, **kwargs)
        return result

    def __str__(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"UgridDataset: {self._file_name or '(in-memory)'}",
            f"  Mesh: {self.mesh_name}",
            f"  Nodes: {self.n_node}, Faces: {self.n_face}, Edges: {self.n_edge}",
            f"  Bounds: {self.bounds}",
            # A mesh may carry a projected WKT with no EPSG code; reporting
            # 'unknown' for a CRS it plainly has was misleading.
            f"  CRS: {self.epsg or (self.crs.name if self.crs is not None else 'unknown')}",
            f"  Data variables ({len(self._data_variables)}):",
        ]
        for name, var in self._data_variables.items():
            lines.append(f"    {name}: location={var.location}, shape={var.shape}")
        result = "\n".join(lines)
        return result

    def __repr__(self) -> str:
        """Repr string for the dataset."""
        result = (
            f"UgridDataset(mesh='{self.mesh_name}', "
            f"n_node={self.n_node}, n_face={self.n_face}, n_edge={self.n_edge}, "
            f"variables={self.data_variable_names})"
        )
        return result


#: The statistics :meth:`UgridDataset.rolling` supports — the collapsing reducers that make
#: sense per window. The counting reducers (``count``/``all``/``any``) are excluded: their
#: flag bands carry the ``FLAG_NO_DATA`` sentinel, which must not surface as a data value.
_ROLLING_HOWS: frozenset[str] = frozenset(
    {"mean", "sum", "min", "max", "std", "var", "median", "quantile"}
)


def _check_reduce_how(
    how: str, q: float | None, *, allow_quantile: bool = True
) -> None:
    """Validate a reduction ``how`` / ``q`` pair, mirroring the raster path's guards.

    The raster ``Selection.reduce`` routes ``how`` through ``_check_how`` and ``q`` through
    ``_check_quantile``; the mesh members call this to get the same clear ``ValueError``
    instead of a raw numpy ``KeyError`` / ``TypeError``.

    Args:
        how: The requested reduction.
        q: The quantile, required for ``how="quantile"`` and rejected otherwise.
        allow_quantile: Whether ``"quantile"`` is a permitted ``how`` (``rolling`` has no
            ``q`` argument, so it forbids it). Defaults to True.

    Raises:
        ValueError: ``how`` is not a known reduction, ``q`` is given for a non-quantile
            ``how``, or ``how="quantile"`` is given without ``q``.
    """
    valid = set(REDUCERS) | set(COUNTING_REDUCERS)
    if not allow_quantile:
        valid.discard("quantile")
    if how not in valid:
        raise ValueError(f"how must be one of {sorted(valid)}, got {how!r}.")
    if how == "quantile" and q is None:
        raise ValueError("reduce(how='quantile') requires q, a float in [0, 1].")
    if how != "quantile" and q is not None:
        raise ValueError(f"q is only valid for how='quantile', not how={how!r}.")


def _extremum_values(
    masked: np.typing.NDArray,
    axis: int,
    *,
    largest: bool,
    coords: list | None,
) -> tuple[np.typing.NDArray, float]:
    """Locate the extreme step along `axis`, as a coordinate value or a step position.

    Gaps are left out of the search: they are parked at the losing end so the plain arg-search
    skips them without `nanargmin`'s all-NaN exception, and an element with no valid step at all
    is then blanked -- which the coordinate form declares as NaN and the position form as ``-1``,
    mirroring the raster `argmin` / `idxmin`.

    Args:
        masked: The values, with gaps already turned into NaN.
        axis: The time axis to search along.
        largest: Search for the maximum rather than the minimum.
        coords: The time coordinate values to answer with, or ``None`` for step positions.

    Returns:
        tuple: ``(values, nodata)`` -- one value per element, and the no-data value it declares.
    """
    valid = ~np.isnan(masked)
    anything = valid.any(axis=axis)
    filler = -np.inf if largest else np.inf
    picked = (np.argmax if largest else np.argmin)(
        np.where(valid, masked, filler), axis=axis
    )
    if coords is None:
        return np.where(anything, picked, -1).astype("int64"), -1
    stamps = np.asarray(coords, dtype="float64")
    return np.where(anything, stamps[picked], np.nan), float("nan")


def _bin_by_frequency(coords: list, freq: str) -> tuple[list, list[np.typing.NDArray]]:
    """Bin date-like time coordinates by a pandas offset alias, dropping empty bins.

    Args:
        coords: The time coordinate values, in step order.
        freq: A pandas offset alias naming the bin width (``"1MS"``, ``"YS"``, ...).

    Returns:
        tuple: ``(labels, positions)`` -- each non-empty bin's pandas label and its step
        positions. The label is the bin's start only for a start-anchored alias; an
        end-anchored one (``"ME"``, ``"YE"``, ``"QE"``, ``"W"``) labels the bin by its end.

    Raises:
        ValueError: The coordinates are step positions, or pandas cannot read them as dates.
    """
    # A mesh with no decoded time coordinate falls back to step positions. pandas reads plain
    # numbers as nanoseconds since the epoch, so `[0, 1, 2]` parses happily and every step lands
    # in the same 1970 month -- the axis collapses to one bin instead of raising. Numbers carry
    # no calendar, so refuse them here, as the docstring says.
    numeric = bool(coords) and all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        for value in coords
    )
    if numeric:
        raise ValueError(
            f"reduce(groupby={freq!r}) needs date-like time coordinates; this dataset's are "
            "step positions, which carry no calendar. Pass a per-step label sequence instead."
        )
    try:
        with warnings.catch_warnings():
            # pandas warns about falling back to dateutil per element before it decides the
            # values are not dates at all; the ValueError below is the real answer.
            warnings.simplefilter("ignore", UserWarning)
            index = pd.DatetimeIndex(pd.to_datetime(list(coords)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reduce(groupby={freq!r}) needs date-like time coordinates; this dataset's cannot "
            "be read as dates. Pass a per-step label sequence instead."
        ) from exc

    steps = pd.Series(np.arange(len(index)), index=index)
    labels: list = []
    positions: list[np.typing.NDArray] = []
    for label, group in steps.groupby(pd.Grouper(freq=freq)):
        if len(group) == 0:
            continue
        labels.append(label)
        positions.append(np.asarray(group.to_numpy(), dtype=np.intp))
    return labels, positions


def _axis_distances(stamps: list) -> np.typing.NDArray | None:
    """Time coordinates as distances along the axis, or ``None`` when they carry none.

    Numeric coordinates are already distances. Date-like ones are not, but they do carry a real
    spacing, so they are converted to nanoseconds since the epoch -- without that step a
    `float64` cast raises on a `Timestamp` and the whole coordinate reading silently degraded to
    step positions, which is the one case `use_coordinate=True` exists for.

    Args:
        stamps: The time coordinate values, in step order.

    Returns:
        np.ndarray: One float distance per step, or ``None`` for coordinates with no measurable
        spacing (plain labels, say), where an evenly spaced axis is the sensible reading.
    """
    try:
        return np.asarray(stamps, dtype="float64")
    except (TypeError, ValueError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.DatetimeIndex(pd.to_datetime(list(stamps)))
    except (TypeError, ValueError):
        return None
    return np.asarray(converted.view("int64"), dtype="float64")


def _topology_info_named(
    mesh_name: str, crs_wkt: str | None = None
) -> MeshTopologyInfo:
    """A minimal topology record carrying just the mesh name.

    The bridges rebuild a mesh from arrays rather than from a file's variables, so there are
    no variable names to record -- but the topology's name still has to survive, because it
    is what :attr:`UgridDataset.mesh_name` answers and what a re-export names its dimensions
    after.

    Args:
        mesh_name: The topology variable's name.
        crs_wkt: The CRS to record, or ``None`` for a record carrying none.

    Returns:
        MeshTopologyInfo: The record, with the 2-D topology dimension and the conventional
        node-coordinate variable names.
    """
    return MeshTopologyInfo(
        mesh_name=mesh_name,
        topology_dimension=2,
        node_x_var=f"{mesh_name}_node_x",
        node_y_var=f"{mesh_name}_node_y",
        crs_wkt=crs_wkt,
    )


def _group_positions(
    coords: list, groupby: Any
) -> tuple[list, list[np.typing.NDArray]]:
    """The time-step positions of each group, and each group's label.

    Two spellings, matching the raster grouped reduce: a **sequence of per-step labels** the
    same length as the time axis (grouped in first-appearance order, so the caller controls the
    order), or a **pandas offset alias** (``"1MS"``, ``"YS"``, …) binned against date-like
    coordinates. Empty bins are dropped, so a sparse year leaves no all-gap step behind.

    Args:
        coords: The time coordinate values, in step order.
        groupby: A per-step label sequence, or a pandas offset alias string.

    Returns:
        tuple: ``(labels, positions)`` — one label and one array of step positions per group.

    Raises:
        ValueError: A label sequence whose length is not the time length, or an offset alias
            asked of coordinates pandas cannot read as dates.
    """
    if isinstance(groupby, str):
        labels, positions = _bin_by_frequency(coords, groupby)
    else:
        given = list(groupby)
        if len(given) != len(coords):
            raise ValueError(
                f"reduce(groupby=...) needs one label per time step: got {len(given)} "
                f"label(s) for {len(coords)} step(s)."
            )
        buckets: dict = {}
        labels = []
        for position, label in enumerate(given):
            if label not in buckets:
                buckets[label] = []
                labels.append(label)
            buckets[label].append(position)
        positions = [np.asarray(buckets[label], dtype=np.intp) for label in labels]
    return labels, positions


def _resolve_time_positions(
    coords: list,
    wanted: list,
    method: str | None,
    tolerance: float | None,
) -> list[int]:
    """The step positions ``wanted`` names in ``coords``, exactly or by nearest.

    Exact matching uses ``list.index`` so a date-like or string coordinate works unchanged.
    ``"nearest"`` needs numeric coordinates, since snapping is a distance: it takes the closest
    coordinate and, when ``tolerance`` is set, refuses a snap further than that — raising
    ``KeyError`` for the breach, as the raster ``sel`` does, so a caller can tell "no coordinate
    is close enough" apart from "that value is not a coordinate at all" (``ValueError``).

    Args:
        coords: The time coordinate values, in step order.
        wanted: The requested coordinate values.
        method: ``None`` for exact, ``"nearest"`` to snap.
        tolerance: The furthest a snap may travel, or ``None`` for no cap.

    Returns:
        list[int]: One step position per requested value, in order.

    Raises:
        ValueError: A value is absent (exact), or ``"nearest"`` was asked of non-numeric
            coordinates.
        KeyError: A snap would exceed ``tolerance``.
    """
    if method is None:
        try:
            positions = [coords.index(value) for value in wanted]
        except ValueError as exc:
            raise ValueError(
                f"sel: time value(s) {wanted!r} not found in the time coordinate."
            ) from exc
        return positions
    try:
        axis = np.asarray(coords, dtype="float64")
        targets = np.asarray(wanted, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sel: method='nearest' needs numeric time coordinates; this dataset's are not."
        ) from exc
    positions = []
    for target in targets:
        gaps = np.abs(axis - target)
        index = int(np.argmin(gaps))
        if tolerance is not None and float(gaps[index]) > float(tolerance):
            raise KeyError(
                f"sel: no time coordinate within tolerance {tolerance!r} of {target!r}; "
                f"the nearest is {coords[index]!r}, {float(gaps[index])!r} away."
            )
        positions.append(index)
    return positions


def _static_from(
    var: MeshVariable,
    data: np.typing.NDArray,
    time_axis: int,
    nodata: float | None = None,
) -> MeshVariable:
    """A static (time-collapsed) copy of ``var`` carrying ``data``.

    Drops the time dimension (at ``time_axis``) from the variable's ``dimensions`` so the
    result reports ``has_time == False``, and takes its shape from ``data``. Used by
    :meth:`UgridDataset.reduce` when a reduction collapses the time axis.

    Args:
        var: The source temporal variable.
        data: The reduced values, with the time axis already gone.
        time_axis: The index of the time dimension in ``var.dimensions`` that was collapsed.
        nodata: The no-data value the result declares. Defaults to ``None``, which keeps the
            source's — a reduction that introduces its own sentinel (``all`` / ``any`` emit
            the ``255`` flag) passes it so the result does not hide that value as data.

    Returns:
        MeshVariable: The static result.
    """
    new_dims = (
        tuple(d for i, d in enumerate(var.dimensions) if i != time_axis)
        if var.dimensions
        else ()
    )
    # Copy the attributes (never alias — the same invariant MeshVariable.with_data keeps),
    # and drop `time_values`: the time axis is gone, so a coordinate for it would be a
    # stale, wrong-length list on a now-static variable.
    attributes = {
        key: value for key, value in var.attributes.items() if key != "time_values"
    }
    return MeshVariable(
        name=var.name,
        location=var.location,
        mesh_name=var.mesh_name,
        shape=data.shape,
        attributes=attributes,
        nodata=var.nodata if nodata is None else nodata,
        units=var.units,
        standard_name=var.standard_name,
        dimensions=new_dims,
        _data=data,
    )


def _concat_temporal_variable(
    name: str, var: MeshVariable, parts: list[UgridDataset]
) -> MeshVariable:
    """Concatenate one temporal variable's data across ``parts`` along its time axis.

    The per-variable body of :meth:`UgridDataset.concat`. Joins each part's array along the
    variable's time axis and its ``time_values`` coordinate when every part has one; if any
    part lacks it, the result carries no (stale) coordinate.

    Args:
        name: The variable name (present in every part — concat validated that).
        var: ``self``'s copy of the variable, whose time axis and attributes seed the result.
        parts: The datasets to join, ``self`` first.

    Returns:
        MeshVariable: The concatenated temporal variable.

    Raises:
        ValueError: A part's same-named variable is static, or has no loaded data.
    """
    axis = cast("int", var.time_index)
    arrays: list[np.typing.NDArray] = []
    times: list[Any] = []
    has_times = True
    for part in parts:
        pv = part._data_variables[name]
        if not pv.has_time:
            # `var` is temporal, so a static same-named variable in another part cannot join
            # along time. Say so with a domain message, not a raw np.concatenate shape error.
            raise ValueError(
                f"concat: variable {name!r} is temporal in one dataset but static in "
                "another; they cannot be joined along time."
            )
        data = pv.data
        if data is None:
            raise ValueError(f"Variable {name!r} has no loaded data to concat.")
        arrays.append(np.asarray(data))
        part_times = pv.attributes.get("time_values")
        if part_times is None:
            has_times = False
        else:
            times.extend(list(part_times))
    new_var = var.with_data(np.concatenate(arrays, axis=axis))
    if has_times:
        new_var.attributes = {**var.attributes, "time_values": times}
    elif "time_values" in new_var.attributes:
        # Some part had no time coordinate, so the joined axis has none; drop the stale,
        # too-short `time_values` carried over by with_data.
        new_var.attributes = {
            key: value
            for key, value in new_var.attributes.items()
            if key != "time_values"
        }
    return new_var


def _make_variable_loader(path: str, var_name: str):
    """Build a zero-arg loader that reads one variable's array on first access.

    The store opened in :meth:`UgridDataset.read_file` is closed before any
    :class:`MeshVariable` data is touched, so a lazy loader cannot capture the live
    root group — it re-opens ``path`` and reads ``var_name`` on demand instead. This
    keeps ``read_file`` metadata-only: variables the caller never touches are never read.

    Args:
        path: File path to re-open for the read.
        var_name: Name of the MDArray to read.

    Returns:
        Callable[[], np.ndarray | None]: A loader returning the variable's array (or
        ``None`` when it has no readable values).
    """

    def _load() -> np.typing.NDArray | None:
        # The whole-array variant of the shared mesh reader: re-open, resolve, read the
        # full array (no window). `ReadAsArray` already returns a fresh, numpy-owned array,
        # so no extra `.copy()` is needed (#982).
        return read_mesh_variable(path, var_name, context="lazy variable read")

    return _load


def _read_data_variables(
    rg: gdal.Group,
    topo_info: MeshTopologyInfo,
    path: str,
) -> dict[str, MeshVariable]:
    """Read every mesh data variable's metadata, deferring the array read.

    Creates a :class:`MeshVariable` per variable that references the mesh topology.
    Only metadata (attributes, shape, dtype, nodata, units, standard name) is read
    eagerly; the array itself loads lazily on first ``.data`` access via a re-opening
    loader, so ``read_file`` does not pull every variable into memory.

    Args:
        rg: GDAL root group (used for metadata only).
        topo_info: Parsed topology info with data variable names and locations.
        path: File path threaded into each variable's lazy loader.

    Returns:
        Dictionary mapping variable name to MeshVariable.
    """
    variables: dict[str, MeshVariable] = {}

    for var_name, location in topo_info.data_variables.items():
        md_arr = open_mdarray(rg, var_name)
        if md_arr is None:
            continue
        attrs = read_cf_attributes(md_arr)
        dims = md_arr.GetDimensions()
        shape = tuple(d.GetSize() for d in dims) if dims else ()
        dim_names = tuple(d.GetName() for d in dims) if dims else ()

        nodata = attrs.get("_FillValue")
        if nodata is not None:
            nodata = float(cast("float", nodata))
        units = cast("str | None", attrs.get("units"))
        standard_name = cast("str | None", attrs.get("standard_name"))
        try:
            dtype = np.dtype(_dtype_to_str(md_arr.GetDataType()))
        except (RuntimeError, TypeError):
            dtype = None

        variables[var_name] = MeshVariable(
            name=var_name,
            location=location,
            mesh_name=topo_info.mesh_name,
            shape=shape,
            attributes=attrs,
            nodata=nodata,
            units=units,
            standard_name=standard_name,
            dimensions=dim_names,
            _loader=_make_variable_loader(path, var_name),
            _dtype=dtype,
            _source_path=path,
        )

    return variables
