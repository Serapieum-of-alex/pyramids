from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from osgeo import gdal
from pyramids.netcdf.utils import (_dtype_to_str, AttributeValue, _read_attributes, _get_array_nodata,
                                   _get_array_scale_offset, _export_srs, _get_block_size, _get_coord_variable_names)

@dataclass(frozen=True)
class GroupInfo:
    """Information about an MDIM Group.

    Fields are JSON-serializable and use full names for stable cross-references.
    """

    name: str
    full_name: str
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # full names of child groups
    arrays: List[str] = field(default_factory=list)  # full names of arrays

    @classmethod
    def from_group(
        cls,
        group: gdal.Group,
        *,
        arrays: List[str],
        children: List[str],
        name: Optional[str] = None,
        full_name: Optional[str] = None,
        attributes: Optional[Dict[str, AttributeValue]] = None,
    ) -> "GroupInfo":
        """Build a GroupInfo from a GDAL Group.

        Parameters:
            group: gdal.Group
                The GDAL group object.
            arrays: List[str]
                Full names of arrays that belong to this group.
            children: List[str]
                Full names of child groups.
            name: Optional[str]
                Precomputed short name; if not provided, computed from the group.
            full_name: Optional[str]
                Precomputed full name; if not provided, computed with fallback.
            attributes: Optional[Dict[str, AttributeValue]]
                Pre-read attributes; if not provided, they are read from the group.

        Returns:
            GroupInfo: Constructed metadata instance for the group.
        """
        # Name
        if name is None:
            try:
                name = group.GetName()
            except Exception:
                name = ""
        # Full name with fallback
        if full_name is None:
            try:
                full_name = group.GetFullName()
            except Exception:
                full_name = "/" if not name else f"/{name}"
        # Attributes
        if attributes is None:
            attributes = _read_attributes(group)
        return cls(
            name=name,
            full_name=full_name,
            attributes=attributes or {},
            children=list(children) if children else [],
            arrays=list(arrays) if arrays else [],
        )



@dataclass(frozen=True)
class DimensionInfo:
    """Information about an MDarray Dimension."""

    name: str
    full_name: str
    size: int
    type: Optional[str] = None
    direction: Optional[str] = None
    indexing_variable: Optional[str] = None  # full name or short name

    @classmethod
    def from_gdal_dim(cls, d: gdal.Dimension, group_full_name: str) -> "DimensionInfo":
        try:
            dim_name = d.GetName()
        except Exception:
            dim_name = ""

        try:
            dim_full_name = d.GetFullName()
        except Exception:
            dim_full_name = f"{group_full_name}/{dim_name}" if group_full_name != "/" else f"/{dim_name}"

        try:
            dim_size = int(d.GetSize())
        except Exception:
            dim_size = 0

        try:
            dtype = str(d.GetType())  # type: ignore[attr-defined]
        except Exception:
            dtype = None

        try:
            dim_dir = str(d.GetDirection())  # type: ignore[attr-defined]
        except Exception:
            dim_dir = None

        try:
            iv = d.GetIndexingVariable()
            if iv is not None:
                ivname = iv.GetFullName() if hasattr(iv, "GetFullName") else iv.GetName()
            else:
                ivname = None
        except Exception:
            ivname = None

        return cls(
            name=dim_name,
            full_name=dim_full_name,
            size=dim_size,
            type=dtype,
            direction=dim_dir,
            indexing_variable=ivname,
        )


@dataclass(frozen=True)
class ArrayInfo:
    """Information about an MDIM Array (variable)."""

    name: str
    full_name: str
    dtype: str
    shape: List[int]
    dimensions: List[str]
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)
    unit: Optional[str] = None
    nodata: Optional[Union[int, float, str]] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    srs_wkt: Optional[str] = None
    srs_projjson: Optional[str] = None
    coordinate_variables: List[str] = field(default_factory=list)
    structural_info: Optional[Dict[str, str]] = None
    block_size: Optional[List[int]] = None

    @classmethod
    def from_md_array(cls, md_arr: gdal.MDArray, md_arr_name:str, group_full_name: str) -> "ArrayInfo":
        try:
            md_arr_name = md_arr.GetName()
        except Exception:
            md_arr_name = md_arr_name

        try:
            md_arr_full_name = md_arr.GetFullName()
        except Exception:
            md_arr_full_name = f"{group_full_name}/{md_arr_name}" if group_full_name != "/" else f"/{md_arr_name}"

        # dtype
        try:
            dt = _dtype_to_str(md_arr.GetDataType())
        except Exception:
            dt = "unknown"

            # shape
        try:
            shape = list(md_arr.GetShape())
        except Exception:
            try:
                shape = [int(d.GetSize()) for d in (md_arr.GetDimensions() or [])]
            except Exception:
                shape = []

            # dimension names
        try:
            dims2 = md_arr.GetDimensions() or []
            dim_names: List[str] = []
            for d in dims2:
                try:
                    dn = d.GetFullName()
                except Exception:
                    dn = d.GetName()
                dim_names.append(dn)
        except Exception:
            dim_names = []

        a_attrs = _read_attributes(md_arr)
        try:
            unit = md_arr.GetUnit()
        except Exception:
            unit = None
        nodata = _get_array_nodata(md_arr, a_attrs)
        scale, offset = _get_array_scale_offset(md_arr, a_attrs)
        try:
            srs = md_arr.GetSpatialRef()
        except Exception:
            srs = None

        wkt, proj_json = _export_srs(srs)
        try:
            sinfo = md_arr.GetStructuralInfo()
            if sinfo is not None:
                sinfo = {str(k): str(v) for k, v in dict(sinfo).items()}
        except Exception:
            sinfo = None
        block_size = _get_block_size(md_arr)
        coord_vars = _get_coord_variable_names(md_arr)

        return cls(
            name=md_arr_name,
            full_name=md_arr_full_name,
            dtype=dt,
            shape=[int(x) for x in (shape or [])],
            dimensions=dim_names,
            attributes=a_attrs,
            unit=unit,
            nodata=nodata,
            scale=scale,
            offset=offset,
            srs_wkt=wkt,
            srs_projjson=proj_json,
            coordinate_variables=coord_vars,
            structural_info=sinfo,
            block_size=block_size,
        )


@dataclass(frozen=True)
class StructuralInfo:
    driver_name: str
    driver_metadata: Optional[Dict[str, str]] = None

    @classmethod
    def from_dataset(cls, dataset: gdal.Dataset, driver_name: str) -> StructuralInfo:
        try:
            dmd = dataset.GetDriver().GetMetadata_Dict()
            dmd = {str(k): str(v) for k, v in dmd.items()} if dmd else None
        except Exception:
            dmd = None
        return cls(driver_name=driver_name, driver_metadata=dmd)

@dataclass(frozen=True)
class NetCDFMetadata:
    """Top-level metadata model for NetCDF MDIM content."""

    driver: str
    root_group: Optional[str]
    groups: Dict[str, GroupInfo]
    arrays: Dict[str, ArrayInfo]
    dimensions: Dict[str, DimensionInfo]
    global_attributes: Dict[str, AttributeValue]
    structural: Optional[StructuralInfo]
    open_options_used: Optional[Dict[str, str]]
    created_with: Dict[str, str]
    dimension_overview: Optional[Dict[str, Any]] = None
