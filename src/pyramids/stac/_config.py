"""Caller-supplied asset metadata, band aliases, and the read-side materialiser.

Two STAC read features meet here, because both end in the same place — a
writable copy of the opened asset:

* **`rescale`** (STAC-04) turns stored counts into physical units using the
  asset's `raster:bands` `scale` / `offset`.
* **`cfg`** (STAC-07) supplies per-asset metadata a thin catalog omits
  (`data_type`, `nodata`, `unit`) and maps **band aliases** (`"rededge"` ->
  `"B05"`) to real asset keys, mirroring odc-stac's `stac_cfg`.

Why they share code: :func:`pyramids.stac.load_asset` opens an asset
**read-only** (a remote `/vsicurl` COG cannot be opened for write), and the
`Dataset.scale` / `Dataset.offset` / `Dataset.no_data_value` /
`Dataset.band_units` setters all raise
:class:`~pyramids.base._errors.ReadOnlyError` on a read-only on-disk handle. So
neither feature can stamp anything onto the handle it was given; both have to
build a new, writable in-memory dataset instead. :func:`materialise` is that one
rebuild, and :class:`AssetOverrides` is the single description of what to apply.

**The config schema** is a plain nested dict, keyed by collection id:

```python
cfg = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0, "unit": "1"},
            "SCL": {"data_type": "uint8", "nodata": 0},
        },
        "aliases": {"red": "B04", "rededge": "B05"},
        "warnings": "ignore",
    },
}
```

A `"*"` collection key supplies defaults for every collection (and is the only
section that applies to a bare asset dict, which carries no collection id).
Precedence, lowest first: `cfg["*"]["assets"]["*"]`, `cfg["*"]["assets"][key]`,
`cfg[collection]["assets"]["*"]`, `cfg[collection]["assets"][key]`.

There is deliberately **no `scale` / `offset`** in `cfg` — packing comes from the
item's own `raster:bands` via `rescale=True`, which keeps the schema
odc-compatible. That is one reason the schema is checked: a key nothing reads
(a typo'd `no_data`, or an odc `scale` carried over) warns with
:class:`AssetMetadataWarning` instead of resolving to nothing in silence.

**Overrides fill gaps; they do not replace what the asset already declares.** A
configured `nodata` is applied only to bands the opened raster reports no
no-data value for, and a configured `unit` only to bands carrying no unit
label; a skipped override warns with :class:`AssetMetadataWarning` unless the
collection sets `warnings: "ignore"`. `data_type` is the exception: a raster
always has a dtype, so a configured one is applied whenever it differs from the
native dtype (an explicit cast of the stored counts).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyramids.base._utils import _is_identity_packing, apply_unpack
from pyramids.base.georeference import GeoReference
from pyramids.dataset.dataset import Dataset
from pyramids.stac._extensions import parse_number, read_extension_metadata

_WILDCARD = "*"

OVERRIDE_KEYS = ("data_type", "nodata", "unit")
"""The per-asset keys :func:`resolve_asset_metadata` results are read for.

Mirrors odc-stac's `ConversionConfig` minus `scale` / `offset`, which belong to
the item's `raster:bands` (see `rescale`) rather than to the caller's config.
"""

SECTION_KEYS = ("assets", "aliases", "warnings")
"""The keys a `cfg` collection section is read for.

Every one is optional — a section may configure only aliases, or only a
`warnings` policy. A key that is *present* and not one of these is a typo
nothing will ever read, which :func:`resolve_asset_metadata` reports.
"""

_STORED_VALUE_TAGS = frozenset(
    {
        "scale_factor",
        "add_offset",
        "_fillvalue",
        "missing_value",
        "valid_min",
        "valid_max",
        "valid_range",
        "statistics_minimum",
        "statistics_maximum",
        "statistics_mean",
        "statistics_stddev",
        "statistics_valid_percent",
        "statistics_approximate",
        "statistics_excluded_values",
        "statistics_skipfactorx",
        "statistics_skipfactory",
    }
)
"""Band-metadata tags that describe a band's *stored* values, matched casefolded.

:func:`_carry_identity` drops these rather than copying them onto a rebuilt
raster, because the rebuild changes exactly what they describe:

* `scale_factor` / `add_offset` say how to decode raw counts, and the rebuilt
  raster's pixels are already decoded (its own GDAL scale/offset are identity,
  so keeping the CF pair would leave the result contradicting itself);
* `_FillValue` / `missing_value` name the stored sentinel, which the rebuild
  replaces with the resolved no-data value (`NaN` once a packing is applied);
* CF `valid_min` / `valid_max` / `valid_range` are expressed in the stored
  data's own units and type;
* GDAL's `STATISTICS_*` are computed over the stored counts, so a cast, a
  packing, or a new no-data value all invalidate them.

Everything else — `units` (which names the unit the packing decodes *to*),
`WAVELENGTH`, `BANDWIDTH`, `DESCRIPTION`, a driver's own band tags — identifies
what the band *is* and stays true, so it is carried verbatim.
"""


class AssetMetadataWarning(UserWarning):
    """A configured override was not applied because the asset already declares it.

    Emitted as a warning rather than raised because the config is a gap-filler:
    an item that turns out to carry the metadata is the good case, and the read
    should still succeed. Silence it per collection with `warnings: "ignore"`.
    """


def item_collection_id(item: Any) -> str | None:
    """Return a STAC Item's collection id, or `None` when it declares none.

    Duck-typed like the rest of the STAC readers: `item.collection_id`
    (pystac) then `item["collection"]` (raw STAC JSON). A bare asset carries
    neither and answers `None`, which is why `cfg` supports a `"*"` collection
    key.

    Args:
        item: A STAC Item (pystac object or raw dict), or an Asset.

    Returns:
        The collection id as a string, or `None`.

    Examples:
        - Raw STAC JSON carries it as the `collection` member:
            ```python
            >>> from pyramids.stac._config import item_collection_id
            >>> item_collection_id({"collection": "sentinel-2-l2a", "assets": {}})
            'sentinel-2-l2a'

            ```
        - A bare asset dict declares none:
            ```python
            >>> item_collection_id({"href": "a.tif"}) is None
            True

            ```
    """
    value = getattr(item, "collection_id", None)
    if value is None and isinstance(item, Mapping):
        value = item.get("collection")
    return None if value is None else str(value)


def _sections(cfg: Any, collection_id: str | None) -> list[Mapping[str, Any]]:
    """Return the `cfg` sections that apply, least specific first.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.

    Returns:
        The `"*"` section followed by the collection's own section, skipping
        either when absent or not a mapping.
    """
    keys: list[str] = [_WILDCARD]
    if collection_id is not None and collection_id != _WILDCARD:
        keys.append(collection_id)
    sections: list[Mapping[str, Any]] = []
    if isinstance(cfg, Mapping):
        for key in keys:
            section = cfg.get(key)
            if isinstance(section, Mapping):
                sections.append(section)
    return sections


def _unknown(keys: Any, known: tuple[str, ...], where: str, quiet: bool) -> None:
    """Warn that a `cfg` key is spelled in a way nothing reads.

    The counterpart of :func:`_skipped`: that one reports an override the read
    *could* not apply, this one an override the read never even looked for. A
    key is only reported when it is **present** and unrecognised — every key in
    the schema is individually optional, so an omitted one is not a typo.

    Args:
        keys: The keys actually present in the `cfg` mapping being read.
        known: The keys the mapping is read for.
        where: Where the keys were found, for the message (e.g.
            `"assets['*']"`).
        quiet: Say nothing (the collection's `warnings: "ignore"`).
    """
    if not quiet:
        for key in keys:
            if str(key) not in known:
                warnings.warn(
                    f"cfg {where} key {str(key)!r} is not understood and was "
                    f"ignored: nothing reads it. Expected one of "
                    f"{', '.join(known)}.",
                    AssetMetadataWarning,
                    stacklevel=3,
                )


def resolve_asset_metadata(
    cfg: Any, collection_id: str | None, asset_key: str | None
) -> dict[str, Any]:
    """Merge the per-collection `'*'` asset defaults with a per-asset override.

    Precedence (highest last): the `"*"` collection's `"*"` asset defaults, its
    named-asset entry, the collection's own `"*"` defaults, the collection's
    named-asset entry. The result **fills gaps** in the STAC metadata — it is
    :func:`materialise` that decides whether a value is actually missing.

    This is also where the schema is checked, because it is the one funnel every
    read passes through exactly once: a key present in a section but not in
    :data:`SECTION_KEYS`, or in an asset entry but not in
    :data:`OVERRIDE_KEYS`, is reported with an :class:`AssetMetadataWarning`
    rather than dropped in silence — a typo would otherwise be the one failure
    mode of `cfg` that gives no signal at all. The collection ids and asset
    keys themselves are caller-chosen names and cannot be checked.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None` to read the `"*"`
            section only.
        asset_key: The asset key being read, or `None` for a bare asset (only
            the `"*"` asset defaults then apply).

    Returns:
        The merged mapping, empty when nothing is configured. Keys outside
        :data:`OVERRIDE_KEYS` are kept verbatim but never applied.

    Warns:
        AssetMetadataWarning: A section or asset-entry key is spelled in a way
            nothing reads. Silenced by the collection's `warnings: "ignore"`.

    Examples:
        - A named asset's entry wins over the `'*'` defaults:
            ```python
            >>> from pyramids.stac._config import resolve_asset_metadata
            >>> cfg = {"c": {"assets": {"*": {"nodata": 0, "data_type": "uint16"},
            ...                         "SCL": {"data_type": "uint8"}}}}
            >>> resolve_asset_metadata(cfg, "c", "SCL") == {
            ...     "nodata": 0, "data_type": "uint8"
            ... }
            True
            >>> resolve_asset_metadata(cfg, "c", "B02") == {
            ...     "nodata": 0, "data_type": "uint16"
            ... }
            True

            ```
        - An unconfigured collection resolves to nothing:
            ```python
            >>> resolve_asset_metadata(cfg, "other", "x")
            {}
            >>> resolve_asset_metadata(None, "c", "SCL")
            {}

            ```
    """
    merged: dict[str, Any] = {}
    quiet = warnings_ignored(cfg, collection_id)
    for section in _sections(cfg, collection_id):
        _unknown(section.keys(), SECTION_KEYS, "collection section", quiet)
        assets = section.get("assets")
        if not isinstance(assets, Mapping):
            continue
        for key in (_WILDCARD, asset_key):
            entry = assets.get(key) if key is not None else None
            if isinstance(entry, Mapping):
                _unknown(entry.keys(), OVERRIDE_KEYS, f"assets[{key!r}]", quiet)
                merged.update(entry)
    return merged


def resolve_alias(cfg: Any, collection_id: str | None, name: str) -> str:
    """Map a band alias to its real asset key, or return `name` unchanged.

    Reads `cfg[collection]["aliases"]`, falling back to `cfg["*"]["aliases"]`,
    so a caller can ask for `"rededge"` and read the asset the catalog calls
    `"B05"`. Must run **before** href resolution, or the real asset key is
    never looked up.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.
        name: The asset key or alias the caller asked for.

    Returns:
        The aliased asset key when one is configured, else `name`.

    Examples:
        - A configured alias resolves; anything else passes through:
            ```python
            >>> from pyramids.stac._config import resolve_alias
            >>> cfg = {"c": {"aliases": {"red": "B04"}}}
            >>> resolve_alias(cfg, "c", "red")
            'B04'
            >>> resolve_alias(cfg, "c", "green")
            'green'
            >>> resolve_alias(None, "c", "red")
            'red'

            ```
    """
    resolved = name
    for section in _sections(cfg, collection_id):
        aliases = section.get("aliases")
        if isinstance(aliases, Mapping) and name in aliases:
            resolved = str(aliases[name])
    return resolved


def warnings_ignored(cfg: Any, collection_id: str | None) -> bool:
    """Return whether this collection asks for metadata warnings to be silenced.

    Args:
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The item's collection id, or `None`.

    Returns:
        `True` when the collection (or the `"*"` section) sets
        `warnings: "ignore"`.

    Examples:
        - Only the configured collection is silenced:
            ```python
            >>> from pyramids.stac._config import warnings_ignored
            >>> cfg = {"c": {"warnings": "ignore"}}
            >>> warnings_ignored(cfg, "c"), warnings_ignored(cfg, "d")
            (True, False)

            ```
    """
    return any(
        section.get("warnings") == "ignore" for section in _sections(cfg, collection_id)
    )


def band_packing(raster_bands: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read per-band `scale` / `offset` out of a STAC `raster:bands` list.

    Values go through :func:`pyramids.stac._extensions.parse_number`, so the
    `raster` extension's `"nan"` / `"inf"` spellings and numeric strings are
    honoured and an unparseable entry falls back to the identity.

    Args:
        raster_bands: The verbatim `raster:bands` list, or `None`.

    Returns:
        A `(scales, offsets)` pair of equal-length tuples — empty when
        `raster_bands` is missing or empty.

    Examples:
        - A one-band packed asset:
            ```python
            >>> from pyramids.stac._config import band_packing
            >>> band_packing([{"scale": 0.01, "offset": 5}])
            ((0.01,), (5.0,))

            ```
        - A band declaring neither is the identity:
            ```python
            >>> band_packing([{"nodata": 0}])
            ((1.0,), (0.0,))
            >>> band_packing(None)
            ((), ())

            ```
    """
    scales: list[float] = []
    offsets: list[float] = []
    for band in raster_bands or ():
        entry = band if isinstance(band, Mapping) else {}
        scales.append(float(parse_number(entry.get("scale"), 1.0)))
        offsets.append(float(parse_number(entry.get("offset"), 0.0)))
    return tuple(scales), tuple(offsets)


@dataclass(frozen=True)
class AssetOverrides:
    """What a STAC read should apply to an asset after opening it.

    The single currency between `rescale` (which fills `scales` / `offsets`
    from the item's `raster:bands`) and `cfg` (which fills the rest), so
    :func:`materialise` has one thing to apply and one place to decide whether
    a rebuild is needed at all.

    Attributes:
        scales: Per-band multiplicative packing factors, in band order.
        offsets: Per-band additive packing offsets, in band order.
        no_data_value: A configured no-data value, or `None`.
        data_type: A configured numpy dtype name, or `None`.
        unit: A configured unit label, or `None`.
        quiet: Suppress :class:`AssetMetadataWarning` (the collection's
            `warnings: "ignore"`).
    """

    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    no_data_value: float | None = None
    data_type: str | None = None
    unit: str | None = None
    quiet: bool = False

    def packing(self, band: int) -> tuple[float, float]:
        """Return the `(scale, offset)` pair for one band index.

        A band beyond the declared `raster:bands` gets the identity, so a
        catalog that documents only the first band of a multi-band asset leaves
        the rest untouched instead of borrowing band 0's factor.

        Args:
            band: Zero-based band index.

        Returns:
            The `(scale, offset)` pair to apply to that band.
        """
        scale = self.scales[band] if band < len(self.scales) else 1.0
        offset = self.offsets[band] if band < len(self.offsets) else 0.0
        return scale, offset

    @property
    def rescales(self) -> bool:
        """Whether any band declares a non-identity `scale` / `offset`.

        Returns:
            `True` when at least one band index covered by `scales` / `offsets`
            carries a packing other than `(1.0, 0.0)`, `False` when every
            declared band is the identity (and when nothing is declared at all).
        """
        # Not a zip over the two tuples: a declaration carrying only one of the
        # pair leaves the other empty, and zip would truncate it away.
        return any(
            not _is_identity_packing(*self.packing(index))
            for index in range(max(len(self.scales), len(self.offsets)))
        )

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to apply, so the opened handle can be kept.

        Returns:
            `True` when no band rescales and no `no_data_value`, `data_type` or
            `unit` is configured — :func:`materialise` then hands the dataset
            back untouched instead of rebuilding it. `False` when any one of
            them is set.
        """
        return not (
            self.rescales
            or self.no_data_value is not None
            or self.data_type is not None
            or self.unit is not None
        )


def _configured_dtype(configured: Mapping[str, Any]) -> str | None:
    """Validate and return a configured `data_type`, or `None`.

    Args:
        configured: The merged per-asset config.

    Returns:
        The dtype name, or `None` when none is configured.

    Raises:
        ValueError: `data_type` is not a name numpy understands.
    """
    name = configured.get("data_type")
    if name is not None:
        try:
            np.dtype(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cfg data_type {name!r} is not a numpy dtype name; use e.g. "
                "'uint8', 'int16', 'uint16', 'float32'."
            ) from exc
    return None if name is None else str(name)


def resolve_overrides(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    rescale: bool = False,
    cfg: Any = None,
    collection_id: str | None = None,
) -> AssetOverrides:
    """Collect everything a read should apply to one asset.

    Reads the item's `raster:bands` packing (only when `rescale` is asked for)
    and the caller's `cfg` entry for the asset. No file is opened: this is pure
    metadata, and :func:`materialise` is what touches pixels.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict), or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset. Must
            already be alias-resolved (see :func:`resolve_alias`).
        rescale: Read the asset's `raster:bands` `scale` / `offset`.
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The collection the item belongs to; `None` reads only
            the `"*"` section of `cfg`.

    Returns:
        The resolved :class:`AssetOverrides` (`is_empty` when nothing applies).

    Raises:
        ValueError: `cfg` configures a `data_type` numpy does not know.
        StacAssetError: `asset_key` is given but absent from the item.

    Examples:
        - `rescale` picks the packing out of `raster:bands`:
            ```python
            >>> from pyramids.stac._config import resolve_overrides
            >>> asset = {"href": "a.tif", "raster:bands": [{"scale": 0.01}]}
            >>> overrides = resolve_overrides(asset, rescale=True)
            >>> overrides.packing(0), overrides.rescales
            ((0.01, 0.0), True)

            ```
        - Without `rescale`, and without `cfg`, there is nothing to apply:
            ```python
            >>> resolve_overrides(asset).is_empty
            True

            ```
    """
    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    if rescale:
        metadata = read_extension_metadata(item_or_asset, asset_key)
        scales, offsets = band_packing(metadata.get("raster_bands"))
    configured = resolve_asset_metadata(cfg, collection_id, asset_key)
    unit = configured.get("unit")
    return AssetOverrides(
        scales=scales,
        offsets=offsets,
        no_data_value=parse_number(configured.get("nodata")),
        data_type=_configured_dtype(configured),
        unit=None if unit is None else str(unit),
        quiet=warnings_ignored(cfg, collection_id),
    )


def _skipped(name: str, value: Any, declared: Any, quiet: bool) -> None:
    """Warn that a configured override was already declared by the asset.

    Args:
        name: The override key (`"nodata"` / `"unit"`).
        value: The configured value that was not applied.
        declared: What the opened raster declares instead.
        quiet: Say nothing (the collection's `warnings: "ignore"`).
    """
    if not quiet:
        warnings.warn(
            f"cfg {name}={value!r} was not applied: the asset already declares "
            f"{declared!r}. Overrides fill gaps in the STAC metadata rather than "
            "replacing a value the asset carries.",
            AssetMetadataWarning,
            stacklevel=3,
        )


def _nodata_targets(dataset: Any, overrides: AssetOverrides) -> list[Any]:
    """Resolve the no-data value each band should end up declaring.

    The configured value fills only the bands that report none; a band that
    already declares one keeps it (and the skip is reported once).

    Args:
        dataset: The opened raster.
        overrides: The resolved overrides.

    Returns:
        One no-data value per band (possibly `None`), in band order.
    """
    native = list(dataset.no_data_value)
    configured = overrides.no_data_value
    resolved = list(native)
    if configured is not None:
        present = [value for value in native if value is not None]
        if present:
            _skipped("nodata", configured, present[0], overrides.quiet)
        resolved = [configured if value is None else value for value in native]
    return resolved


def _unit_targets(dataset: Any, overrides: AssetOverrides) -> list[str] | None:
    """Resolve the unit label each band should end up carrying, or `None`.

    Args:
        dataset: The opened raster.
        overrides: The resolved overrides.

    Returns:
        One unit label per band when a configured unit fills at least one gap,
        else `None` (nothing to stamp).
    """
    targets: list[str] | None = None
    if overrides.unit is not None:
        native = [str(unit or "") for unit in dataset.band_units]
        labelled = [unit for unit in native if unit]
        if labelled:
            _skipped("unit", overrides.unit, labelled[0], overrides.quiet)
        resolved = [unit or overrides.unit for unit in native]
        targets = None if resolved == native else resolved
    return targets


def _recast(dataset: Any, overrides: AssetOverrides) -> str | None:
    """Resolve the dtype the stored counts must be cast to, or `None`.

    A raster always has a dtype, so a configured `data_type` is not "filling a
    gap" the way `nodata` and `unit` are: it applies whenever it differs from
    what the raster already stores, and is dropped when it agrees.

    Args:
        dataset: The opened raster.
        overrides: The resolved overrides.

    Returns:
        The configured `data_type` when it differs from the dtype the first band
        already stores, else `None` (nothing to cast).
    """
    native = next(iter(dataset.dtype)) if dataset.band_count else None
    return overrides.data_type if overrides.data_type != native else None


def _band_first(dataset: Any) -> Any:
    """Read a raster's stored counts as a band-first masked array.

    Reads with `unpack=False` so the counts are raw (any packing the *file*
    declares is deliberately left for the caller to decide about) and
    `masked=True` so the declared no-data sentinels are masked **before** any
    arithmetic touches them.

    Args:
        dataset: The opened raster.

    Returns:
        A `(bands, rows, columns)` masked array of stored values.
    """
    stored = np.ma.asanyarray(dataset.read_array(unpack=False, masked=True))
    # np.ma.atleast_3d would append the new axis, not prepend it.
    return stored[np.newaxis, :, :] if stored.ndim == 2 else stored


def _collapse(values: list[Any]) -> Any:
    """Return a single value when every band agrees, else the per-band list.

    `Dataset.from_array` takes either, and a scalar keeps the common
    single-band case readable in tracebacks and tests.

    Args:
        values: One value per band.

    Returns:
        The shared value, or `values` unchanged.
    """
    return values[0] if len(set(map(repr, values))) == 1 else values


@dataclass(frozen=True)
class _BandPlane:
    """One band's stored counts together with the no-data value it will declare.

    The two travel as a pair through the rebuild because unpacking changes both
    at once: applying a `scale` / `offset` turns the counts into physical values
    *and* invalidates the stored sentinel, so the no-data value becomes `NaN`.
    Keeping them in one object is what stops the two from drifting apart.

    Attributes:
        values: The band's stored counts as a masked array.
        fill: The no-data value the rebuilt band should declare, or `None` when
            the band declares none.
    """

    values: Any
    fill: Any

    def unpacked(self, packing: tuple[float, float] | None) -> _BandPlane:
        """Return the plane with a band packing applied, or itself unchanged.

        Args:
            packing: The band's `(scale, offset)` pair, or `None` to leave the
                stored counts alone.

        Returns:
            A new plane holding `float32` physical values and a `NaN` no-data
            value, or `self` when `packing` is `None`.
        """
        unpacked = self
        if packing is not None:
            scale, offset = packing
            physical = np.ma.asanyarray(
                apply_unpack(self.values, scale, offset)
            ).astype("float32")
            unpacked = _BandPlane(physical, np.nan)
        return unpacked

    @property
    def filled(self) -> np.ndarray:
        """The plane as a plain array with its masked cells written out.

        Returns:
            The values with every masked cell replaced by :attr:`fill`, or by
            `0` when the band declares no no-data value at all.
        """
        return np.ma.filled(self.values, 0 if self.fill is None else self.fill)


def _band_identity(metadata: Mapping[str, str]) -> dict[str, str]:
    """Strip one band's stored-value tags, keeping the tags that identify it.

    Args:
        metadata: One band's default-domain metadata, as the source declares it.

    Returns:
        The same mapping without any key in :data:`_STORED_VALUE_TAGS` (matched
        without regard to case, since drivers spell CF tags inconsistently).

    Examples:
        - The decode tags go, the band's own identity stays:
            ```python
            >>> from pyramids.stac._config import _band_identity
            >>> _band_identity(
            ...     {"scale_factor": "0.01", "units": "m", "WAVELENGTH": "665"}
            ... ) == {"units": "m", "WAVELENGTH": "665"}
            True

            ```
    """
    return {
        key: value
        for key, value in metadata.items()
        if key.lower() not in _STORED_VALUE_TAGS
    }


def _carry_identity(source: Any, target: Any, units: list[str] | None) -> None:
    """Copy the source raster's identity onto the rebuilt one.

    `Dataset.from_array` builds from a bare numpy stack, so everything that is
    not a pixel has to be put back by hand. Three things are carried:

    1. **The projection, as WKT.** `GeoReference` only takes an `epsg` code, and
       `AbstractDataset.epsg` is `None` for any CRS with no EPSG authority
       (MODIS sinusoidal, a geostationary product, a custom LCC) — so rebuilding
       from the code alone leaves such a result with no CRS at all, which
       silently breaks every later `to_crs`, `crop(bbox)`, `align` and write.
       The same WKT carry is what `Spatial._correct_wrap_cutline_error` does
       for the same reason. A source with no projection is skipped: assigning
       its empty WKT would wipe the `from_array` default.
    2. **Band names and per-band metadata**, which identify the bands of a
       multi-asset or `eo:bands`-named raster; without them the result's bands
       come back as `Band_1`, `Band_2`, with no wavelength or description. Only
       the tags that describe what a band **is** travel: every key in
       :data:`_STORED_VALUE_TAGS` describes the *source's stored values*
       instead — how to decode them (`scale_factor` / `add_offset`), which one
       means "missing" (`_FillValue` / `missing_value`), what range they span
       (`valid_*`) and how they are distributed (`STATISTICS_*`) — and the
       rebuild has just changed all four, so copying them would publish a
       description that contradicts the result's own pixels and no-data value.
    3. **Band units** — the resolved `units` when an override supplied them,
       else the source's own labels. Unlike the packing, a unit label survives
       unpacking: GDAL's unit type (like CF's `units`) names the *physical*
       unit a `scale` / `offset` decodes to, which is exactly what the rebuilt
       raster now stores.

    Args:
        source: The opened raster the rebuild came from.
        target: The freshly built raster, which must be writable.
        units: The unit labels resolved by :func:`_unit_targets`, or `None` when
            no override applies and the source's own labels should be kept.
    """
    if source.crs:
        target.crs = source.crs
    names = list(source.band_names)
    if names != list(target.band_names):
        target.bands.apply_names(names, source="the materialised asset")
    metadata = [_band_identity(band) for band in source.bands.metadata]
    if any(metadata):
        target.bands.metadata = metadata
    labels = units or [str(unit or "") for unit in source.band_units]
    if any(labels):
        target.band_units = labels


@dataclass(frozen=True)
class _Rebuild:
    """What applying one asset's overrides changes, and how to carry it out.

    The three resolved targets are born together from the same `(dataset,
    overrides)` pair, are read together to decide whether a rebuild is needed at
    all, and are then consumed together by the rebuild itself — so they live
    here rather than as loose locals threaded through a helper.

    Attributes:
        dataset: The opened raster the rebuild reads from.
        overrides: What the caller asked to apply.
        nodata: The no-data value each band should end up declaring, in band
            order (from :func:`_nodata_targets`).
        units: The unit label each band should end up carrying, or `None` when
            no override applies (from :func:`_unit_targets`).
        data_type: The dtype the stored counts must be cast to, or `None` (from
            :func:`_recast`).
    """

    dataset: Any
    overrides: AssetOverrides
    nodata: list[Any]
    units: list[str] | None
    data_type: str | None

    @property
    def needed(self) -> bool:
        """Whether anything would actually change, so a rebuild is warranted.

        Returns:
            `True` when a band rescales, the counts need casting, a band's
            no-data value changes or a unit label is stamped; `False` when the
            opened handle can be handed back untouched.
        """
        return bool(
            self.overrides.rescales
            or self.data_type is not None
            or self.nodata != list(self.dataset.no_data_value)
            or self.units
        )

    def run(self, path: str | None) -> Any:
        """Build the writable raster with every resolved target applied.

        Args:
            path: Optional destination for the rebuilt raster; `None` keeps it
                in memory.

        Returns:
            The new :class:`~pyramids.dataset.Dataset`.
        """
        stored = self._stored()
        planes = [self._plane(stored, index) for index in range(stored.shape[0])]
        result = Dataset.from_array(
            np.stack([plane.filled for plane in planes]),
            no_data_value=_collapse([plane.fill for plane in planes]),
            # Reconstructed from the geotransform rather than a corner + cell
            # size, so a rotated or anisotropic grid survives the rebuild.
            geo_ref=GeoReference(
                geo=tuple(self.dataset.geotransform), epsg=self.dataset.epsg
            ),
            path=path,
        )
        # `epsg` above is `None` for a WKT-only CRS, and the array carries no
        # band identity at all, so both are put back here.
        _carry_identity(self.dataset, result, self.units)
        return result

    def _stored(self) -> Any:
        """Read the source's stored counts, cast when a `data_type` applies.

        Returns:
            A `(bands, rows, columns)` masked array of stored values.
        """
        stored = _band_first(self.dataset)
        if self.data_type is not None:
            stored = stored.astype(self.data_type)
        return stored

    def _plane(self, stored: Any, index: int) -> _BandPlane:
        """Pair one band's counts with its no-data value and unpack them.

        Args:
            stored: The band-first stack of stored counts.
            index: Zero-based band index.

        Returns:
            The band's resolved :class:`_BandPlane`.
        """
        fill = self.nodata[index] if index < len(self.nodata) else None
        packing = self.overrides.packing(index) if self.overrides.rescales else None
        return _BandPlane(stored[index], fill).unpacked(packing)


def materialise(
    dataset: Any, overrides: AssetOverrides, *, path: str | None = None
) -> Any:
    """Return a writable raster with `overrides` applied, or `dataset` untouched.

    The one rebuild shared by `rescale` (STAC-04) and `cfg` (STAC-07), because
    both are blocked by the same thing: the asset is open read-only, so nothing
    can be stamped onto it. The order is **mask -> scale -> fill**, matching how
    pyramids unpacks CF-packed netCDF data:

    1. the stored counts are read with the declared no-data masked
       (`unpack=False, masked=True`), so a sentinel is never scaled into a
       plausible-looking physical value;
    2. a configured `data_type` casts those counts;
    3. each band's `scale` / `offset` is applied to the valid pixels only,
       promoting the result to `float32`;
    4. the mask is filled — with `NaN` when rescaling (physical no-data has no
       meaningful stored sentinel), else with the band's resolved no-data value.

    The result is built by :meth:`Dataset.from_array`, so it declares **identity
    packing** (`scale == 1`, `offset == 0`): a later `read_array(unpack=True)`
    cannot apply the STAC scale a second time, even when the source *file*
    declared the same packing.

    Building from a bare array also means nothing but the pixels comes across on
    its own, so :func:`_carry_identity` puts the source's projection (as WKT,
    since a CRS with no EPSG code cannot travel as `epsg`), band names, per-band
    metadata and band units back onto the result. The per-band metadata is
    filtered first: the tags that describe the source's *stored* values rather
    than the band's identity (:data:`_STORED_VALUE_TAGS` — the CF
    `scale_factor` / `add_offset` pair, `_FillValue` / `missing_value`,
    `valid_*` and GDAL's `STATISTICS_*`) are dropped, so the result never
    re-publishes a decode instruction for pixels it has already decoded.

    Args:
        dataset: The opened raster (read-only is fine; it is never mutated).
        overrides: What to apply, from :func:`resolve_overrides`.
        path: Optional destination for the rebuilt raster; `None` keeps it in
            memory.

    Returns:
        A new :class:`~pyramids.dataset.Dataset` when something was applied, or
        `dataset` itself when nothing was — so a caller can test identity
        (`result is dataset`) to find out whether it still holds the lazy
        handle.

    Examples:
        - A packed asset's counts become physical values, no-data preserved:
            ```python
            >>> import numpy as np
            >>> from pyramids.base.georeference import GeoReference
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.stac._config import AssetOverrides, materialise
            >>> packed = Dataset.from_array(
            ...     np.array([[0, 100], [200, 300]], dtype="int16"),
            ...     no_data_value=0,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0),
            ... )
            >>> physical = materialise(packed, AssetOverrides(scales=(0.01,)))
            >>> physical.read_array().tolist()
            [[nan, 1.0], [2.0, 3.0]]
            >>> physical.scale, physical.offset
            ([1.0], [0])

            ```
        - Nothing configured means nothing is rebuilt:
            ```python
            >>> materialise(packed, AssetOverrides()) is packed
            True

            ```
    """
    # The two `_targets` calls stay here rather than inside `_Rebuild`, so the
    # `stacklevel=3` of the skipped-override warning still lands on this frame.
    rebuild = _Rebuild(
        dataset=dataset,
        overrides=overrides,
        nodata=_nodata_targets(dataset, overrides),
        units=_unit_targets(dataset, overrides),
        data_type=_recast(dataset, overrides),
    )
    return rebuild.run(path) if rebuild.needed else dataset


__all__ = [
    "OVERRIDE_KEYS",
    "AssetMetadataWarning",
    "AssetOverrides",
    "band_packing",
    "item_collection_id",
    "materialise",
    "resolve_alias",
    "resolve_asset_metadata",
    "resolve_overrides",
    "warnings_ignored",
]
