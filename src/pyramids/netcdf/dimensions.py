from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union
import re

Number = Union[int, float]

_DIM_KEY_RE = re.compile(r"^NETCDF_DIM_([A-Za-z0-9_]+?)(?:_(DEF|VALUES))?$")


def _strip_braces(value: str) -> str:
    """Return the content between outer curly braces if present.

    Args:
        value(str) :
            Input string which may look like ``"{1,2,3}"`` or have surrounding whitespace.

    Returns:
        str
            The substring without the outer braces and surrounding whitespace.
    """
    s = value.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1].strip()
    return s


def _smart_split_csv(text: str) -> List[str]:
    """Split a comma-separated list that may contain spaces.

    Notes:
        - This helper is intentionally simple because GDAL's netCDF metadata values
          for *DEF* and *VALUES* are typically simple comma-separated scalars.
        - Empty strings and whitespace-only items are ignored.
    """
    core = _strip_braces(text)
    if core:
        vals = [item.strip() for item in core.split(",") if item.strip()]
    else:
        vals = []
    return vals


def _coerce_scalar(token: str) -> Union[str, Number]:
    """Coerce a token to ``int``/``float`` where possible.

    The function prefers ``int`` if the token represents an integer, otherwise
    falls back to ``float`` if conversion succeeds, and finally the original
    string.
    """
    # Try int
    try:
        # Account for +/-, hex/float-like strings should fail here
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
    except ValueError:
        pass
    # Try float
    try:
        return float(token)
    except ValueError:
        return token


def _parse_values_list(text: str) -> List[Union[str, Number]]:
    return [_coerce_scalar(t) for t in _smart_split_csv(text)]


@dataclass(frozen=True)
class Dimension:
    """Represents a single netCDF dimension discovered in GDAL metadata.

    Args:
        name (str):
            Dimension name (e.g., ``"time"``, ``"level0"``).
        size (int | None):
            Dimension length when known. If not present in metadata it may be
            ``None``.
        values (list[int]|float|str|None):
            Parsed numeric (or string) values from the ``*_VALUES`` entry, if
            provided by the GDAL netCDF driver.
        def_fields (tuple[int], optional):
            Parsed integers from the ``*_DEF`` entry. The meaning is driver-
            specific; commonly the first value corresponds to the dimension size.
        raw (dict):
            Raw string values pulled from metadata for this dimension.

    Notes:
        This class does not assume semantics for ``*_DEF`` beyond exposing it. See
        the GDAL netCDF driver docs for details.

    Examples:
        ```python
        >>> md = {
        ...     'NETCDF_DIM_EXTRA': '{time,level0}',
        ...     'NETCDF_DIM_level0_DEF': '{3,6}',
        ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
        ...     'NETCDF_DIM_time_DEF': '{2,6}',
        ...     'NETCDF_DIM_time_VALUES': '{0,31}',
        ... }
        >>> idx = DimensionsIndex.from_metadata(md)
        >>> sorted(idx.names)
        ['level0', 'time']
        >>> idx['time'].size
        2
        >>> idx['level0'].values
        [1, 2, 3]
        ```
    """

    name: str
    size: Optional[int] = None
    values: Optional[List[Union[str, Number]]] = None
    def_fields: Optional[Tuple[int, ...]] = None
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class DimensionsIndex:
    """Index of :class:`Dimension` parsed from GDAL netCDF metadata.

    Use :meth:`from_metadata` to construct from a GDAL metadata mapping (e.g.,
    the result of ``gdal.Dataset.GetMetadata()``). This parser is intentionally
    permissive and will:

        - accept dimensions listed under ``NETCDF_DIM_EXTRA``;
        - also pick up any ``NETCDF_DIM_<name>_DEF`` or ``NETCDF_DIM_<name>_VALUES``
          even if ``<name>`` is not listed in ``EXTRA``;
        - coerce numeric tokens to ``int`` or ``float`` where possible.

    The parsed dimensions are available via mapping-style access (``idx[name]``)
    and iteration.
    """

    _dims: Dict[str, Dimension] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        prefix: str = "NETCDF_DIM_",
    ) -> "DimensionsIndex":
        """Parse dimensions from a GDAL metadata dictionary.

        Args:
            metadata : Mapping[str, str]
                GDAL metadata mapping (e.g., from ``GetMetadata()``).
            prefix : str, optional
                Key prefix to filter on (default ``'NETCDF_DIM_'``).

        Returns:
            DimensionsIndex
                Parsed index of dimensions.
        """
        # Gather candidate keys first
        dim_names: set[str] = set()
        buckets: Dict[str, Dict[str, str]] = {}

        for key, value in metadata.items():
            if not key.startswith(prefix):
                continue
            m = _DIM_KEY_RE.match(key)
            if not m:
                # Keep unknown but prefixed keys under a synthetic "_misc" bucket
                buckets.setdefault("_misc", {})[key] = value
                continue
            name, suffix = m.groups()

            # Special case: EXTRA carries a comma-separated list of dim names
            if name.upper() == "EXTRA" and suffix is None:
                for nm in _smart_split_csv(value):
                    if nm:
                        dim_names.add(nm)
                continue

            # Normal dimension attributes (DEF/VALUES)
            buckets.setdefault(name, {})[suffix or "_root"] = value
            dim_names.add(name)

        dims: Dict[str, Dimension] = {}
        for name in sorted(dim_names):
            raw_bucket = buckets.get(name, {})
            # DEF may contain integers, often first item is size
            def_fields: Optional[Tuple[int, ...]] = None
            size: Optional[int] = None
            if "DEF" in raw_bucket:
                def_list = _parse_values_list(raw_bucket["DEF"])
                # Only keep integers in def_fields; ignore non-ints gracefully
                ints = tuple(int(x) for x in def_list if isinstance(x, int))
                def_fields = ints or None
                if ints:
                    size = ints[0]

            # VALUES: keep as numbers/strings
            values: Optional[List[Union[str, Number]]] = None
            if "VALUES" in raw_bucket:
                values = _parse_values_list(raw_bucket["VALUES"]) or None
                # If size wasn't in DEF, infer from VALUES length
                if size is None and values is not None:
                    size = len(values)

            dims[name] = Dimension(
                name=name,
                size=size,
                values=values,
                def_fields=def_fields,
                raw={
                    k if k in ("DEF", "VALUES") else "OTHER": v
                    for k, v in raw_bucket.items()
                },
            )

        return cls(dims)

    @property
    def names(self) -> List[str]:
        """List of dimension names in this index."""
        return list(self._dims.keys())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._dims)

    def __iter__(self) -> Iterable[Dimension]:  # pragma: no cover - trivial
        return iter(self._dims.values())

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._dims

    def __getitem__(self, name: str) -> Dimension:
        return self._dims[name]

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        """Serialize to a plain dictionary for easy debugging/IO.

        Returns
        -------
        dict
            Mapping from dimension name to a structure with ``size``,
            ``values`` and ``def_fields`` fields.
        """
        out: Dict[str, Dict[str, object]] = {}
        for k, d in self._dims.items():
            out[k] = {
                "size": d.size,
                "values": d.values,
                "def_fields": d.def_fields,
            }
        return out


def parse_gdal_netcdf_dimensions(metadata: Mapping[str, str]) -> DimensionsIndex:
    """Convenience wrapper for :meth:`DimensionsIndex.from_metadata`.

    Args:
        metadata : Mapping[str, str]
            GDAL metadata mapping (e.g., from ``GetMetadata()``).

    Returns:
        DimensionsIndex
            Parsed index of dimensions.

    Examples:
        ```python
        >>> md = {
        ...     'NETCDF_DIM_EXTRA': '{time,level0}',
        ...     'NETCDF_DIM_level0_DEF': '{3,6}',
        ...     'NETCDF_DIM_level0_VALUES': '{1,2,3}',
        ...     'NETCDF_DIM_time_DEF': '{2,6}',
        ...     'NETCDF_DIM_time_VALUES': '{0,31}',
        ... }
        >>> idx = parse_gdal_netcdf_dimensions(md)
        >>> idx.to_dict()['time']['size']
        2
        >>> idx.to_dict()['level0']['values']
        [1, 2, 3]
        ```
    """
    return DimensionsIndex.from_metadata(metadata)
