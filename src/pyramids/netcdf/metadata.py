from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import json
from osgeo import gdal
from collections import deque
from pyramids.netcdf.models import (
    ArrayInfo,
    DimensionInfo,
    GroupInfo,
    NetCDFMetadata,
    StructuralInfo,
)
from pyramids.netcdf.utils import (
    _read_attributes,
    _get_driver_name,
    _get_root_group,
    _safe_group_names,
    _safe_array_names,
)

from pyramids.netcdf.dimensions import MetaData as SharedMetaData

class MetadataBuilder:
    """Class-based builder that constructs NetCDFMetadata from a NetCDF source.

    This refactors the previous function-centric approach into a small, cohesive
    object with clearer responsibilities and smaller methods. Public module
    functions delegate to this builder to preserve API compatibility.
    """

    def __init__(self, src: gdal.Dataset, open_options: Optional[Dict[str, Any]] = None) -> None:
        self.gdal_dataset = src
        self.open_options = open_options or None

    def build(self) -> NetCDFMetadata:
        ds = self.gdal_dataset

        # Driver name and root group
        driver_name = _get_driver_name(ds)
        root_group = _get_root_group(ds)

        groups_map: Dict[str, GroupInfo] = {}
        arrays_map: Dict[str, ArrayInfo] = {}
        dimensions_map: Dict[str, DimensionInfo] = {}

        if root_group is not None:
            traverser = GroupTraverser(groups_map, arrays_map, dimensions_map)
            traverser.walk(root_group)

            try:
                root_name = root_group.GetFullName()
            except Exception:
                root_name = "/"
            global_attrs = _read_attributes(root_group)
        else:
            root_name = None
            # Fallback: use classic flattened metadata exposed by MetaData
            try:
                md = SharedMetaData.from_metadata(ds.GetMetadata())
                global_attrs = {str(k): str(v) for k, v in (md.to_metadata() if hasattr(md, 'to_metadata') else {}).items()}  # type: ignore[arg-type]
            except Exception:
                global_attrs = {}

        structural_info = StructuralInfo.from_dataset(ds, driver_name)
        created_with = {"library": "GDAL", "version": getattr(gdal, "__version__", "unknown")}

        return NetCDFMetadata(
            driver=driver_name,
            root_group=root_name,
            groups=groups_map,
            arrays=arrays_map,
            dimensions=dimensions_map,
            global_attributes=global_attrs,
            structural=structural_info,
            created_with=created_with,
            open_options_used=self.open_options,
        )


class GroupTraverser:
    """Iterative (BFS) traverser that collects MDIM groups, arrays, and dimensions.

    Uses a deque for predictable, stack-safe traversal and sorts names for
    deterministic ordering. Results are written into the provided maps.
    """

    def __init__(
        self,
        groups: Dict[str, GroupInfo],
        arrays: Dict[str, ArrayInfo],
        dimensions: Dict[str, DimensionInfo],
    ) -> None:
        self.groups = groups
        self.arrays = arrays
        self.dimensions = dimensions

    def _collect_dimensions(self, group: gdal.Group, group_full_name: str) -> None:

        try:
            dims = group.GetDimensions() or []
        except Exception:
            dims = []

        # Sort by name for determinism
        def _dim_name(d: Any) -> str:
            try:
                return d.GetName()
            except Exception:
                return ""

        dims_sorted = sorted(list(dims), key=_dim_name)

        for d in dims_sorted:
            dim = DimensionInfo.from_gdal_dim(d, group_full_name)
            self.dimensions[dim.full_name] = dim

    def _collect_arrays(self, group: gdal.Group, group_full_name: str) -> List[str]:
        array_full_names: List[str] = []
        for md_arr_name in _safe_array_names(group):
            try:
                md_arr = group.OpenMDArray(md_arr_name)
            except Exception:
                md_arr = None

            if md_arr is None:
                continue

            array_info = ArrayInfo.from_md_array(md_arr, md_arr_name, group_full_name)

            self.arrays[array_info.full_name] = array_info
            array_full_names.append(array_info.full_name)
        return array_full_names

    def walk(self, root: gdal.Group) -> None:
        q = deque([root])

        while q:
            group = q.popleft()

            # Compute group identity (name/full_name) via GroupInfo for separation of concerns
            base_group = GroupInfo.from_group(group, arrays=[], children=[], attributes={})
            group_full_name = base_group.full_name

            # Dimensions and arrays for this group
            self._collect_dimensions(group, group_full_name)
            group_arrays = self._collect_arrays(group, group_full_name)

            # Children
            children_full: List[str] = []
            for cn in _safe_group_names(group):
                try:
                    current_group = group.OpenGroup(cn)
                except Exception:
                    current_group = None

                if current_group is None:
                    continue

                # Delegate child full-name resolution to GroupInfo
                try:
                    child_info = GroupInfo.from_group(current_group, arrays=[], children=[], attributes={})
                    current_group_full_name = child_info.full_name
                except Exception:
                    # As a last resort, fall back to simple path concatenation
                    current_group_full_name = f"{group_full_name}/{cn}" if group_full_name != "/" else f"/{cn}"

                children_full.append(current_group_full_name)
                q.append(current_group)

            # Record this group entry via GroupInfo factory
            group_info = GroupInfo.from_group(
                group,
                arrays=group_arrays,
                children=children_full,
            )
            self.groups[group_info.full_name] = group_info


def get_metadata(source: gdal.Dataset, open_options: Optional[Dict[str, Any]] = None) -> NetCDFMetadata:
    """Read and normalize all NetCDF MDIM metadata.

    This is a thin wrapper that delegates to the class-based builder
    MetadataBuilder for clarity and maintainability.
    """
    builder = MetadataBuilder(source, open_options)
    return builder.build()


def to_dict(metadata: NetCDFMetadata) -> Dict[str, Any]:
    """Convert NetCDFMetadata dataclasses to plain dicts suitable for JSON."""
    def convert(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(metadata)


def to_json(metadata: NetCDFMetadata) -> str:
    """Serialize NetCDFMetadata to JSON string."""
    return json.dumps(to_dict(metadata), ensure_ascii=False, separators=(",", ":"))


def from_json(s: str) -> NetCDFMetadata:
    """Deserialize NetCDFMetadata from JSON string created by to_json().

    Note: We rebuild dataclasses manually from dicts; only supports the schema produced by to_dict().
    """
    d = json.loads(s)

    def build_group(gd: Dict[str, Any]) -> GroupInfo:
        return GroupInfo(
            name=gd["name"],
            full_name=gd["full_name"],
            attributes=gd.get("attributes", {}),
            children=gd.get("children", []),
            arrays=gd.get("arrays", []),
        )

    def build_dim(dd: Dict[str, Any]) -> DimensionInfo:
        return DimensionInfo(
            name=dd["name"],
            full_name=dd["full_name"],
            size=int(dd["size"]),
            type=dd.get("type"),
            direction=dd.get("direction"),
            indexing_variable=dd.get("indexing_variable"),
        )

    def build_array(ad: Dict[str, Any]) -> ArrayInfo:
        return ArrayInfo(
            name=ad["name"],
            full_name=ad["full_name"],
            dtype=ad.get("dtype", "unknown"),
            shape=[int(x) for x in ad.get("shape", [])],
            dimensions=[str(x) for x in ad.get("dimensions", [])],
            attributes=ad.get("attributes", {}),
            unit=ad.get("unit"),
            nodata=ad.get("nodata"),
            scale=ad.get("scale"),
            offset=ad.get("offset"),
            srs_wkt=ad.get("srs_wkt"),
            srs_projjson=ad.get("srs_projjson"),
            coordinate_variables=[str(x) for x in ad.get("coordinate_variables", [])],
            structural_info=ad.get("structural_info"),
            block_size=[int(x) for x in ad.get("block_size", [])] if ad.get("block_size") is not None else None,
        )

    groups = {k: build_group(v) for k, v in d.get("groups", {}).items()}
    arrays = {k: build_array(v) for k, v in d.get("arrays", {}).items()}
    dims = {k: build_dim(v) for k, v in d.get("dimensions", {}).items()}

    structural = d.get("structural")
    structural_obj = (
        StructuralInfo(
            driver_name=structural.get("driver_name", "UNKNOWN"),
            driver_metadata=structural.get("driver_metadata"),
        )
        if structural is not None
        else None
    )

    return NetCDFMetadata(
        driver=d.get("driver", "UNKNOWN"),
        root_group=d.get("root_group"),
        groups=groups,
        arrays=arrays,
        dimensions=dims,
        global_attributes=d.get("global_attributes", {}),
        structural=structural_obj,
        open_options_used=d.get("open_options_used"),
        created_with=d.get("created_with", {}),
        dimension_overview=d.get("dimension_overview"),
    )


def flatten_for_index(metadata: NetCDFMetadata) -> Dict[str, Any]:
    """Return a flat dict of key properties suitable for indexing/search."""
    d: Dict[str, Any] = {
        "driver": metadata.driver,
        "root_group": metadata.root_group,
        "group_count": len(metadata.groups),
        "array_count": len(metadata.arrays),
        "dimension_count": len(metadata.dimensions),
    }
    # include some global attrs
    for k, v in list(metadata.global_attributes.items())[:20]:
        d[f"global.{k}"] = v
    # include names of arrays and dims
    d["arrays"] = sorted([a for a in metadata.arrays.keys()])
    d["dimensions"] = sorted([dname for dname in metadata.dimensions.keys()])
    return d
