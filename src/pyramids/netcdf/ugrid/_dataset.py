"""UgridDataset — top-level container for UGRID NetCDF mesh data.

Combines mesh topology, data variables, metadata, and provides
the user-facing API for reading, inspecting, and operating on
unstructured mesh data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from pyramids.netcdf.ugrid._connectivity import Connectivity
from pyramids.netcdf.ugrid._io import parse_ugrid_topology
from pyramids.netcdf.ugrid._mesh import Mesh2d
from pyramids.netcdf.ugrid._models import MeshTopologyInfo, MeshVariable, UgridMetadata
from pyramids.netcdf.utils import _read_attributes


class UgridDataset:
    """Container for UGRID NetCDF mesh data.

    Combines mesh topology, data variables, and global attributes
    into a single object with GIS-aware operations. Does NOT inherit
    from Dataset or AbstractDataset — the raster paradigm does not
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

    @classmethod
    def read_file(cls, path: str | Path) -> UgridDataset:
        """Open a UGRID NetCDF file.

        Automatically detects mesh topology, separates data variables
        from topology/coordinate variables, and builds the mesh.

        Args:
            path: Path to the .nc file.

        Returns:
            UgridDataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no UGRID topology is found in the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ds = gdal.OpenEx(
            str(path),
            gdal.OF_MULTIDIM_RASTER | gdal.OF_VERBOSE_ERROR,
        )
        if ds is None:
            raise ValueError(f"GDAL cannot open file: {path}")

        rg = ds.GetRootGroup()
        if rg is None:
            raise ValueError(f"Cannot get root group from: {path}")

        topologies = parse_ugrid_topology(rg)
        if not topologies:
            raise ValueError(
                f"No UGRID mesh topology found in: {path}"
            )

        topo_info = topologies[0]
        mesh = Mesh2d.from_gdal_group(rg, topo_info)

        data_variables = _read_data_variables(rg, topo_info)

        global_attrs = _read_global_attributes(rg)

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
        result = self._topology_info.mesh_name if self._topology_info else "mesh2d"
        return result

    @property
    def data_variable_names(self) -> list[str]:
        """Names of all data variables."""
        result = list(self._data_variables.keys())
        return result

    @property
    def crs(self) -> Any:
        """CRS as a pyproj.CRS object, or None."""
        if self._crs_wkt is None:
            return None
        try:
            from pyproj import CRS
            result = CRS.from_wkt(self._crs_wkt)
        except Exception:
            result = None
        return result

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or None."""
        crs = self.crs
        if crs is None:
            return None
        result = crs.to_epsg()
        return result

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

    @property
    def metadata(self) -> UgridMetadata:
        """Full metadata summary for this dataset."""
        topo_tuple = (self._topology_info,) if self._topology_info else ()
        data_vars = {
            name: var.location for name, var in self._data_variables.items()
        }
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

    def clip(self, mask: Any, touch: bool = True) -> "UgridDataset":
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
        from pyramids.netcdf.ugrid._spatial import clip_mesh
        result = clip_mesh(self, mask, touch=touch)
        return result

    def subset_by_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> "UgridDataset":
        """Subset mesh to faces within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            New UgridDataset with subset mesh and data.
        """
        from pyramids.netcdf.ugrid._spatial import subset_by_bounds
        result = subset_by_bounds(self, xmin, ymin, xmax, ymax)
        return result

    def __str__(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"UgridDataset: {self._file_name or '(in-memory)'}",
            f"  Mesh: {self.mesh_name}",
            f"  Nodes: {self.n_node}, Faces: {self.n_face}, Edges: {self.n_edge}",
            f"  Bounds: {self.bounds}",
            f"  CRS: {self.epsg or 'unknown'}",
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


def _read_data_variables(
    rg: gdal.Group,
    topo_info: MeshTopologyInfo,
) -> dict[str, MeshVariable]:
    """Read all data variables from a GDAL root group.

    Creates MeshVariable instances with lazy loading for each
    variable that references the mesh topology.

    Args:
        rg: GDAL root group.
        topo_info: Parsed topology info with data variable names and locations.

    Returns:
        Dictionary mapping variable name to MeshVariable.
    """
    variables: dict[str, MeshVariable] = {}

    for var_name, location in topo_info.data_variables.items():
        md_arr = rg.OpenMDArray(var_name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        dims = md_arr.GetDimensions()
        shape = tuple(d.GetSize() for d in dims) if dims else ()

        nodata = attrs.get("_FillValue")
        if nodata is not None:
            nodata = float(nodata)
        units = attrs.get("units")
        standard_name = attrs.get("standard_name")

        data = md_arr.ReadAsArray()
        if data is not None:
            data = data.copy()

        variables[var_name] = MeshVariable(
            name=var_name,
            location=location,
            mesh_name=topo_info.mesh_name,
            shape=shape,
            attributes=attrs,
            nodata=nodata,
            units=units,
            standard_name=standard_name,
            _data=data,
        )

    return variables


def _read_global_attributes(rg: gdal.Group) -> dict[str, Any]:
    """Read global attributes from a GDAL root group.

    Args:
        rg: GDAL root group.

    Returns:
        Dictionary of global attribute name-value pairs.
    """
    result = _read_attributes(rg)
    return result
