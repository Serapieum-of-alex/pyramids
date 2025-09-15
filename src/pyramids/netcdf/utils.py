from typing import Any, Dict, Union, List, TypeAlias, Optional, Tuple
import re
from datetime import datetime, timedelta
from osgeo import gdal, osr

# Keep simple, JSON-serializable attribute values only
AttributeScalar: TypeAlias = Union[bool, int, float, str]
AttributeVector: TypeAlias = List[AttributeScalar]
AttributeValue: TypeAlias = Union[AttributeScalar, AttributeVector]
_ORIGIN_RE = re.compile(r'^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$', re.IGNORECASE)



def _full_name_with_fallback(group: gdal.Group, default_name: Optional[str] = None) -> str:
    try:
        return group.GetFullName()
    except Exception:
        # Root or fallback to "/<name>"
        try:
            gname = group.GetName()
        except Exception:
            gname = default_name or ""
        return "/" if not gname else f"/{gname}"


def _get_group_name(group: gdal.Group) -> str:
    # Names
    try:
        gname = group.GetName()
    except Exception:
        gname = ""
    return gname


def _safe_array_names(group: gdal.Group) -> List[str]:
    try:
        names = group.GetMDArrayNames() or []
    except Exception:
        names = []
    return sorted(list(names))


def _safe_group_names(group: gdal.Group) -> List[str]:
    try:
        names = group.GetGroupNames() or []
    except Exception:
        names = []
    return sorted(list(names))


def _get_root_group(dataset: gdal.Dataset) -> Optional[gdal.Group]:
    try:
        return dataset.GetRootGroup()
    except Exception:
        return None

def _get_driver_name(dataset: gdal.Dataset) -> str:
    try:
        return dataset.GetDriver().ShortName
    except Exception:
        return "UNKNOWN"


def _export_srs(srs: Optional[osr.SpatialReference]) -> Tuple[Optional[str], Optional[str]]:
    if not srs:
        return None, None
    wkt = None
    projjson = None
    try:
        wkt = srs.ExportToWkt()
    except Exception:
        pass
    try:
        projjson = srs.ExportToJSON()
    except Exception:
        pass
    return wkt, projjson


def _get_array_nodata(mdarr: gdal.MDArray, attrs: Dict[str, AttributeValue]) -> Optional[Union[int, float, str]]:
    # Precedence: CF _FillValue, then missing_value, then driver API
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            v = attrs[key]
            if isinstance(v, list):
                return v[0] if v else None
            return v  # type: ignore[return-value]
    # Try driver API
    for meth in ("GetNoDataValueAsDouble", "GetNoDataValueAsInt64", "GetNoDataValueAsString"):
        if hasattr(mdarr, meth):
            try:
                v = getattr(mdarr, meth)()
                # Some GDAL versions return (value, hasval)
                if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], (bool, int)):
                    if v[1]:
                        return _to_py_scalar(v[0])
                    continue
                return _to_py_scalar(v)
            except Exception:
                continue
    return None


def _get_array_scale_offset(mdarr: gdal.MDArray, attrs: Dict[str, AttributeValue]) -> Tuple[Optional[float], Optional[float]]:
    scale = None
    offset = None
    # CF attributes first
    if isinstance(attrs.get("scale_factor"), (int, float)):
        scale = float(attrs["scale_factor"])  # type: ignore[index]
    if isinstance(attrs.get("add_offset"), (int, float)):
        offset = float(attrs["add_offset"])  # type: ignore[index]
    # GDAL API may also expose
    if hasattr(mdarr, "GetScale"):
        try:
            s = mdarr.GetScale()
            if s is not None:
                scale = float(s)
        except Exception:
            pass
    if hasattr(mdarr, "GetOffset"):
        try:
            o = mdarr.GetOffset()
            if o is not None:
                offset = float(o)
        except Exception:
            pass
    return scale, offset


def _get_block_size(mdarr: gdal.MDArray) -> Optional[List[int]]:
    try:
        bs = mdarr.GetBlockSize()
        if bs:
            return [int(b) for b in bs]
    except Exception:
        pass
    return None


def _get_coord_variable_names(mdarr: gdal.MDArray) -> List[str]:
    names: List[str] = []
    try:
        cvs = mdarr.GetCoordinateVariables()
    except Exception:
        cvs = None
    if not cvs:
        return names
    for cv in cvs:
        try:
            # Some GDAL versions return MDArray objects, others names
            if hasattr(cv, "GetFullName"):
                names.append(cv.GetFullName())  # type: ignore[attr-defined]
            elif hasattr(cv, "GetName"):
                names.append(cv.GetName())
            else:
                names.append(str(cv))
        except Exception:
            # Fallback
            names.append(str(cv))
    return names

def _normalize_origin_string(origin: str) -> str:
    """
    Normalize CF time origin like '1-1-1 0:0:0' or '1-1-1T0:0:0'
    into zero-padded '0001-01-01 00:00:00'.
    """
    origin = origin.strip().replace("T", " ")
    if " " in origin:
        date_part, time_part = origin.split(" ", 1)
    else:
        date_part, time_part = origin, "0:0:0"

    ymd = date_part.strip().split("-")
    while len(ymd) < 3:
        ymd.append("1")
    y, m, d = ymd[:3]
    y = y.zfill(4)
    m = m.zfill(2)
    d = d.zfill(2)

    hms = time_part.strip().split(":")
    while len(hms) < 3:
        hms.append("0")
    H, M, S = (hms[0].zfill(2), hms[1].zfill(2), hms[2].zfill(2))

    # Support fractional seconds
    if "." in S:
        # Keep fractional seconds as-is; datetime.fromisoformat can handle it
        return f"{y}-{m}-{d} {H}:{M}:{S}"
    else:
        return f"{y}-{m}-{d} {H}:{M}:{S}"

def _parse_units_origin(units: str) -> tuple[str, datetime]:
    """
    Parse CF-like time units, returning (unit, origin_datetime).
    Accepts origins like 'days since 1-1-1 0:0:0'.
    """
    m = _ORIGIN_RE.match(units)
    if not m:
        raise ValueError(f"Unrecognized time units: {units!r}")

    unit, origin_raw = m.groups()
    origin_norm = _normalize_origin_string(origin_raw)

    # Try ISO-style parsing
    try:
        origin_dt = datetime.fromisoformat(origin_norm)
    except ValueError:
        # Fallback to explicit format if needed
        origin_dt = datetime.strptime(origin_norm, "%Y-%m-%d %H:%M:%S")

    return unit.lower(), origin_dt

def create_time_conversion_func(units: str, out_format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Create a converter that maps numeric CF times to formatted strings.
    Supports units: days, hours, minutes, seconds since <origin>.
    """
    unit, origin = _parse_units_origin(units)

    if unit.startswith("day"):
        scale = timedelta(days=1)
    elif unit.startswith("hour"):
        scale = timedelta(hours=1)
    elif unit.startswith("min"):
        scale = timedelta(minutes=1)
    elif unit.startswith("sec"):
        scale = timedelta(seconds=1)
    else:
        raise ValueError(f"Unsupported time unit: {unit!r}")

    def convert(value):
        # value can be int/float; CF allows fractional units
        dt = origin + value * scale
        return dt.strftime(out_format)

    return convert


def _dtype_to_str(dt: Any) -> str:
    try:
        # gdal.ExtendedDataType in MDIM
        name = dt.GetName()
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    try:
        # As a fallback, class name
        return str(dt)
    except Exception:
        return "unknown"


def _to_py_scalar(x: Any) -> Any:
    """Convert numpy scalars/bytes to native JSON-serializable Python types."""
    try:
        # numpy scalar
        if hasattr(x, "item") and callable(x.item):
            return x.item()
    except Exception:
        pass

    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode("utf-8", errors="ignore")

    # Already a JSON-friendly scalar
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x

    # Fallback to string representation to avoid breaking JSON dump
    return str(x)


def _normalize_attr_value(val: Any) -> AttributeValue:
    # Vector
    if isinstance(val, (list, tuple)):
        return [ _to_py_scalar(v) for v in val ]

    # Scalar
    return _to_py_scalar(val)  # type: ignore[return-value]


def _read_attribute_value(attr: gdal.Attribute) -> AttributeValue:
    # Try the generic Read() first; it often returns appropriate Python types
    val: Any
    try:
        val = attr.Read()
    except Exception:
        # try type-specifics
        for meth in (
                "ReadAsInt64",
                "ReadAsInt64Array",
                "ReadAsDouble",
                "ReadAsDoubleArray",
                "ReadAsString",
                "ReadAsStringArray",
        ):
            if hasattr(attr, meth):
                try:
                    val = getattr(attr, meth)()
                    break
                except Exception:
                    continue
        else:
            val = None
    return _normalize_attr_value(val)


def _read_attributes(obj: Any) -> Dict[str, AttributeValue]:
    attrs: Dict[str, AttributeValue] = {}
    try:
        att_list = obj.GetAttributes()
    except Exception:
        att_list = None
    if not att_list:
        return attrs
    for att in att_list:
        try:
            name = att.GetName()
        except Exception:
            continue
        try:
            attrs[name] = _read_attribute_value(att)
        except Exception:
            # Be robust; don't crash on odd attribute types
            attrs[name] = _normalize_attr_value(None)
    return attrs