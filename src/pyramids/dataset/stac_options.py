"""Grouped option objects for the two public STAC entry points.

:meth:`pyramids.dataset.Dataset.to_stac_item` and
:meth:`pyramids.dataset.DatasetCollection.from_stac` each grew past twenty
keyword arguments, most of which are only meaningful next to two or three of
their neighbours. The options that travel together are gathered here, one
frozen dataclass per group, so each entry point takes one object per concern
instead of a flat list of flags.

Every dataclass defaults every field, so each is optional at the call site and
`None` means "all defaults for that group": `to_stac_item(ds, "id",
asset_href=...)` and `from_stac(items, "red")` are unchanged.

The module is a **leaf**: it imports nothing from `pyramids` at run time (only
`numpy` and the standard library), which is what lets
:mod:`pyramids.dataset._stac` import it at module load without reviving the
``dataset`` ↔ ``stac`` import cycle. It lives under :mod:`pyramids.dataset` for
the same reason :mod:`pyramids.dataset._stac` does — importing
``pyramids.stac.anything`` initialises the :mod:`pyramids.stac` package, whose
``__init__`` imports :mod:`pyramids.dataset` — and every name here is
re-exported from :mod:`pyramids.stac`, which is where callers should import it
from.

Group behaviour that has to touch GDAL (measuring a band, resolving a
footprint) stays in :mod:`pyramids.dataset._stac` beside the helpers it calls;
what lives here is each group's data and the validation that data can answer on
its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyramids.dataset.grid import Grid

__all__ = [
    "Alignment",
    "AssetRead",
    "BandMetadata",
    "CubeProvenance",
    "Footprint",
    "GroupFusion",
    "ReadTolerance",
    "StacExtensions",
]


@dataclass(frozen=True)
class BandMetadata:
    """How much per-band metadata an Item's asset pays to publish.

    `with_stats`, `with_histogram`, `histogram_bins` and `stats_approx_ok` are
    born together, are forwarded as a block and are read at a single point per
    band; `with_eo` joins them because it is the third opt-in band block. They
    share one property no other :func:`to_stac_item` option has — each asks
    GDAL to look at the raster's bands, and each degrades by omitting its field
    with a warning rather than failing the Item — so they are one concept: how
    much band measuring this Item pays for.

    Attributes:
        with_stats: Add a per-band `statistics` object (`minimum`, `maximum`,
            `mean`, `stddev`, in physical units) to `raster:bands`. A band with
            no valid pixels is skipped with a warning.
        with_histogram: Add a per-band `histogram` object (`count`, `min`,
            `max`, `buckets`) to `raster:bands`. A band with no value range
            (constant or all-nodata) is skipped with a warning.
        histogram_bins: Number of histogram buckets when `with_histogram`.
        with_eo: Add `eo:bands` (band names) to the asset and the `eo` schema to
            `stac_extensions`.
        stats_approx_ok: Let GDAL answer `with_stats` from overviews or a
            subsample (fast); pass `False` for exact figures.
    """

    with_stats: bool = False
    with_histogram: bool = False
    histogram_bins: int = 10
    with_eo: bool = False
    stats_approx_ok: bool = True


@dataclass(frozen=True)
class Footprint:
    """How an Item's geometry and bbox are derived from a dataset's extent.

    The six `footprint*` / geometry keywords of :func:`to_stac_item` are born
    together, are never read apart, and are consumed at a single point — the
    one step that turns a dataset into `(geometry, bbox)`. The two readings of
    the mode, the no-EPSG fallback and the antimeridian split all read nothing
    but this clump plus the dataset's own extent. `precision` belongs here
    because the reprojected footprint's coordinates are the only thing it ever
    rounds.

    Attributes:
        mode: `"bbox"` (default) for the dataset's bounding rectangle, or
            `"data"` for the polygonised extent of its valid (non-nodata)
            pixels. Both paths need an EPSG code to reach EPSG:4326, so a
            dataset whose CRS has no EPSG authority code cannot be reprojected:
            `"data"` is then declined with a `RuntimeWarning` and the bbox
            path's world-extent fallback applies.
        band: Zero-based band to footprint in `"data"` mode.
        max_samples: Approximate pixel budget for the valid-pixel mask —
            accuracy traded for speed — or `None` for an exact read.
        simplify_tolerance: Douglas-Peucker tolerance **in degrees**, applied to
            the data footprint after reprojection, or `None` to keep every
            vertex.
        densify: Maximum segment length **in native CRS units**, applied to the
            data footprint before reprojection so long edges follow the
            projection's curvature; `None`/`0` leaves the edges alone.
        precision: Decimal places for the reprojected footprint coordinates.
    """

    mode: str = "bbox"
    band: int = 0
    max_samples: int | None = None
    simplify_tolerance: float | None = None
    densify: float | None = None
    precision: int = 6

    def __post_init__(self) -> None:
        """Reject a mode that is neither `"bbox"` nor `"data"`.

        Raises:
            ValueError: `mode` is not one of the two supported footprints.
        """
        if self.mode not in ("bbox", "data"):
            raise ValueError(f"footprint must be 'bbox' or 'data', got {self.mode!r}.")


@dataclass(frozen=True)
class StacExtensions:
    """Which STAC extension blocks :func:`to_stac_item` stamps onto the Item.

    Both flags answer the same question — which extension's fields are
    populated and whose schema is appended to `stac_extensions` — and neither
    reads anything the other does not, so they are one switchboard rather than
    two unrelated booleans.

    Attributes:
        with_proj: Populate the `proj` extension (epsg/code/shape/transform/
            bbox) from the dataset grid.
        with_raster: Populate `raster:bands` (per-band `data_type` + `nodata`,
            plus `scale`/`offset` for a CF-packed band) on the asset.
    """

    with_proj: bool = True
    with_raster: bool = True


@dataclass(frozen=True)
class Alignment:
    """How :func:`from_stac` puts an Item's assets on a common grid.

    All three describe one decision — which grid the pixels land on and how
    they get there. `grid` names the cube's output grid, `align` says whether a
    mixed-resolution item may be resampled onto its first asset's grid at all,
    and `resampling` picks the algorithm that resample uses; `resampling`
    without `align` is meaningless, which is why they are one object.

    Attributes:
        grid: Optional :class:`~pyramids.dataset.Grid` describing the target
            output grid; every timestep of the built cube is reprojected /
            resampled onto it, guaranteeing pixel co-registration. `None`
            (default) or an empty `Grid()` keeps each timestep's native grid.
        align: Multi-asset only. When `True` (default), assets at differing
            native resolutions are resampled onto the first requested asset's
            grid. When `False`, a grid/CRS mismatch among an item's assets
            raises :class:`~pyramids.base._errors.AlignmentError`. Ignored in
            single-asset mode.
        resampling: Multi-asset only. Which algorithm aligns mixed-resolution
            assets onto the first requested asset's grid. `None` (default)
            keeps nearest neighbour for every band. A method name
            (`"bilinear"`, `"cubic"`, `"average"`, …) applies to every band; a
            `{asset key: method}` mapping sets it per band. Requires
            `align=True`; passing it in single-asset or grouped mode raises.
    """

    grid: Grid | None = None
    align: bool = True
    resampling: str | dict[str, str] | None = None


@dataclass(frozen=True)
class ReadTolerance:
    """What :func:`from_stac` does with an Item whose asset is not usable.

    The two failures a build can forgive, and the pair the documentation
    contrasts: an asset the Item never declared, and one it declares but that
    will not open. Both answer "drop it or stop", both are independent of the
    other's answer, and both are read at the same points, so they belong
    together — while staying two fields, because the metadata gap and the IO
    failure are genuinely different events.

    Attributes:
        skip_missing: When `True`, items **lacking the asset key** are dropped
            instead of raising. When `False` (default), a missing asset raises
            :class:`~pyramids.base._errors.StacAssetError`. This is about the
            item's metadata only — an asset that *is* declared but cannot be
            read is a different failure, covered by `errors_as_nodata`.
        errors_as_nodata: Tolerate an asset that is present on the item but
            **cannot be opened** (404, expired signed URL, unreachable host, a
            file truncated badly enough that its header will not parse) by
            substituting a no-data-filled (`NaN`) plane for that timestep
            instead of failing the whole cube. Only IO/open errors (`OSError`,
            GDAL's `RuntimeError`) are converted; a missing asset key, a bad
            argument or any other programming error still raises — that is
            `skip_missing`'s job, and the two are independent.

            **Only the open is probed.** A raster whose header parses but whose
            tiles are truncated opens cleanly and then fails later, on the lazy
            pixel read, by which point the cube is built and no plane can be
            substituted for it. So partial corruption is *not* covered, however
            the failure is spelled; a header that does not parse is.

            **The plane needs a grid.** The reference grid is
            :attr:`Alignment.grid` when one is given, otherwise the grid of the
            first timestep that *does* open — one grid for the whole build, not
            one per group. A grouped build fills an **empty group** (every
            source unreadable) from that same reference, so in a collection
            spanning several UTM zones or resolutions the plane necessarily
            lands on another group's grid and the cube then mixes grids. The
            substitution warns when it borrows a grid this way, but
            `Alignment(grid=...)` is the only way to pin it, so treat it as
            effectively required for a heterogeneous collection. When nothing
            opens and no grid was given there is nothing to fill, so a
            `ValueError` is raised. Each substitution emits a warning with the
            href redacted (credentials stripped).

            In single-asset mode this **forces an eager open probe** of every
            href (the hrefs themselves stay the collection's backing files, so
            reads remain lazy); a lazy build cannot discover an unreadable URL
            until it is too late to substitute anything.
    """

    skip_missing: bool = False
    errors_as_nodata: bool = False


@dataclass(frozen=True)
class GroupFusion:
    """How a group's overlapping pixels resolve into one timestep.

    `method` and `fuse_func` are two spellings of a single decision — the
    overlap-resolution rule for a mosaicked group — which is why they are
    mutually exclusive and why neither means anything without a `groupby`. One
    object makes the exclusivity visible at the call site.

    Attributes:
        method: The overlap-resolution rule handed to
            :func:`~pyramids.dataset.merge.merge_rasters` when a group's items
            are mosaicked: `"first"` (default, first-valid pixel wins),
            `"last"`, `"min"`, `"max"`, `"sum"`, `"count"` or `"mean"`. Passing
            anything but `"first"` without a `groupby` raises.
        fuse_func: An **odc-style in-place fuser** `(dst, src) -> None` that
            replaces the built-in mosaic for every group: the group's sources
            are read onto the first source's grid, no-data is normalised to
            `NaN`, and the callback is applied pairwise in item order,
            mutating the accumulator. Its return value is ignored. Mutually
            exclusive with a non-default `method`, and rejected when `groupby`
            is `None`. `None` (default) keeps the built-in merge.
    """

    method: str = "first"
    fuse_func: Callable[[np.ndarray, np.ndarray], None] | None = None

    def __post_init__(self) -> None:
        """Reject a `fuse_func` that cannot be called.

        Raises:
            TypeError: `fuse_func` is neither `None` nor callable.
        """
        if self.fuse_func is not None and not callable(self.fuse_func):
            raise TypeError(
                "fuse_func must be a callable (dst, src) -> None, "
                f"got {type(self.fuse_func).__name__}."
            )


@dataclass(frozen=True)
class AssetRead:
    """How each asset's values and metadata are normalised as they are read.

    `rescale` and `cfg` are the only two reasons :func:`from_stac` has to
    rewrite a source rather than read it as it stands, they decide *together*
    whether a timestep is materialised instead of staying a lazy URL, and they
    are threaded to exactly the same places. They are one concept: what the
    reader fixes up on the way in.

    Attributes:
        rescale: Return **physical units** rather than stored counts, applying
            `real = stored * scale + offset` from each asset's `raster:bands`
            metadata, with no-data masked before the scaling and `NaN` as the
            result's no-data. Off by default. Only GDAL assets
            (GeoTIFF/COG/JPEG2000) are rescaled; netCDF / Zarr / GRIB already
            unpack the CF packing their own metadata declares.

            **It changes how the cube is backed.** Single-asset mode normally
            backs each timestep with the asset URL itself and stays lazy over
            `/vsicurl`; a scale cannot be stamped onto a read-only remote
            handle, so `rescale=True` instead **materialises every timestep**
            that declares a non-identity packing into a local raster under the
            process artefact root. An asset that declares no `raster:bands`, or
            only the identity, keeps its lazy URL. The grouped and multi-asset
            modes already materialise, so they only change in value.

            Two consequences worth planning for. **Units can be
            inhomogeneous**: an item that declares no `raster:bands` keeps its
            stored counts while a sibling in the same collection is rescaled,
            and nothing marks the cube as mixed, so a reduction over the time
            axis is wrong by the packing factor for that subset. **Disk grows
            per call**: the materialised copies that back the returned
            collection live under the process artefact root until the
            interpreter exits (see :mod:`pyramids.base._artifacts`), so a
            service or notebook rescaling in a loop accumulates one
            full-resolution raster per timestep per call. Materialising also
            ends laziness: a rescaled timestep is a local raster, so a later
            `read_part` / `preview` / `read_tile` can no longer be served from a
            remote COG's overviews.
        cfg: Optional `stac_cfg`-style mapping (see
            :mod:`pyramids.stac._config`) supplying per-asset metadata the items
            omit (`data_type`, `nodata`, `unit`) and **band aliases**, so
            `asset="rededge"` can read the asset a catalog calls `"B05"`. Keyed
            by collection id — resolved per item from `item["collection"]` /
            `item.collection_id` — with a `"*"` section for cross-collection
            defaults. Overrides fill gaps only: a value an asset already
            declares is kept, and the skip warns unless the collection sets
            `warnings: "ignore"`. Like `rescale`, applying one
            materialises the affected timestep; `None` (default) changes
            nothing.
    """

    rescale: bool = False
    cfg: Any = None


@dataclass(frozen=True)
class CubeProvenance:
    """What STAC metadata rides along on the cube :func:`from_stac` builds.

    Both fields carry Item metadata onto the output rather than changing a
    single pixel: `properties` as the per-timestep attribute table, and
    `eo_band_names` as the band axis' labels. Grouping them keeps the
    "describe the cube" options apart from the ones that decide what it
    contains.

    Attributes:
        properties: Which STAC Item `properties` to carry onto the built cube as
            per-timestep attributes, readable afterwards through
            :attr:`~pyramids.dataset.DatasetCollection.time_attrs`. `False`
            (default) attaches nothing. `True` attaches every property of every
            item; a `str` or a sequence of strings attaches only those keys. The
            result is always one plain dict per **emitted timestep**. Under a
            `groupby` the group's **first item in item order** represents the
            group, which only describes the emitted pixels for
            `GroupFusion(method="first")` — any other reduction warns.
        eo_band_names: Multi-asset only. When `True`, name the output bands from
            each asset's `eo:bands` metadata (its `name`, else its
            `common_name`) instead of the raw asset keys, falling back to the
            asset key for any asset that declares no usable `eo:bands`. `False`
            (default) keeps the asset keys, unchanged. Note that an
            `Alignment(resampling=...)` mapping is still keyed by **asset key**,
            which is resolved before any renaming.
    """

    properties: bool | str | Sequence[str] = False
    eo_band_names: bool = False
