import re
from datetime import datetime, timedelta

_ORIGIN_RE = re.compile(r'^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$', re.IGNORECASE)


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