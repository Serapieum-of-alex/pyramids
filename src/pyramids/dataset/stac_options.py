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

    Examples:
        - The default group measures nothing, so an Item costs no band reads:
            ```python
            >>> from pyramids.stac import BandMetadata
            >>> bands = BandMetadata()
            >>> bands.with_stats, bands.with_histogram, bands.with_eo
            (False, False, False)
            >>> bands.histogram_bins
            10

            ```
        - Opt into statistics and a 32-bucket histogram, and ask for exact figures:
            ```python
            >>> from pyramids.stac import BandMetadata
            >>> bands = BandMetadata(
            ...     with_stats=True,
            ...     with_histogram=True,
            ...     histogram_bins=32,
            ...     stats_approx_ok=False,
            ... )
            >>> bands.histogram_bins
            32
            >>> bands.stats_approx_ok
            False

            ```
        - Hand the group to `to_stac_item` and read the blocks it published:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.stac import BandMetadata
            >>> ds = Dataset.from_array(
            ...     np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... )
            >>> item = ds.to_stac_item(
            ...     "scene-1",
            ...     asset_href="scene.tif",
            ...     datetime="2023-06-01T00:00:00Z",
            ...     bands=BandMetadata(with_stats=True, with_eo=True),
            ... )
            >>> stats = item["assets"]["data"]["raster:bands"][0]["statistics"]
            >>> stats["minimum"], stats["maximum"], stats["mean"]
            (1.0, 4.0, 2.5)
            >>> item["assets"]["data"]["eo:bands"]
            [{'name': 'Band_1'}]

            ```

    See Also:
        - StacExtensions: which extension blocks the Item declares at all.
        - pyramids.dataset.Dataset.to_stac_item: the entry point that consumes this group.
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

    Examples:
        - The default group is the bounding rectangle at six decimals:
            ```python
            >>> from pyramids.stac import Footprint
            >>> footprint = Footprint()
            >>> footprint.mode
            'bbox'
            >>> footprint.precision
            6
            >>> footprint.max_samples, footprint.simplify_tolerance, footprint.densify
            (None, None, None)

            ```
        - Ask for the valid-pixel footprint, rounded to two decimals, and read the
          geometry it produced — only the one valid pixel is covered, not the full grid:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.stac import Footprint
            >>> ds = Dataset.from_array(
            ...     np.array([[1.0, np.nan], [np.nan, np.nan]], dtype="float32"),
            ...     no_data_value=np.nan,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... )
            >>> item = ds.to_stac_item(
            ...     "scene-1",
            ...     asset_href="scene.tif",
            ...     datetime="2023-06-01T00:00:00Z",
            ...     footprint=Footprint(mode="data", precision=2),
            ... )
            >>> item["bbox"]
            [0.0, 1.0, 1.0, 2.0]
            >>> item["geometry"]["coordinates"]
            [[[0.0, 2.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]]

            ```
        - Any other mode is rejected on construction, before a dataset is touched:
            ```python
            >>> from pyramids.stac import Footprint
            >>> Footprint(mode="outline")
            Traceback (most recent call last):
                ...
            ValueError: footprint must be 'bbox' or 'data', got 'outline'.

            ```

    See Also:
        - pyramids.dataset.Dataset.to_stac_item: the entry point that consumes this group.
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

    Examples:
        - Both extensions are on by default:
            ```python
            >>> from pyramids.stac import StacExtensions
            >>> extensions = StacExtensions()
            >>> extensions.with_proj, extensions.with_raster
            (True, True)

            ```
        - Dropping `raster` leaves the asset without a `raster:bands` block while the
          `proj` fields stay:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.stac import StacExtensions
            >>> ds = Dataset.from_array(
            ...     np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... )
            >>> item = ds.to_stac_item(
            ...     "scene-1",
            ...     asset_href="scene.tif",
            ...     datetime="2023-06-01T00:00:00Z",
            ...     extensions=StacExtensions(with_raster=False),
            ... )
            >>> item["stac_extensions"]
            ['https://stac-extensions.github.io/projection/v1.1.0/schema.json']
            >>> "raster:bands" in item["assets"]["data"]
            False
            >>> item["properties"]["proj:epsg"]
            4326

            ```
        - With both off the Item declares no extension schemas at all:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids.stac import StacExtensions
            >>> ds = Dataset.from_array(
            ...     np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... )
            >>> item = ds.to_stac_item(
            ...     "scene-1",
            ...     asset_href="scene.tif",
            ...     datetime="2023-06-01T00:00:00Z",
            ...     extensions=StacExtensions(with_proj=False, with_raster=False),
            ... )
            >>> item["stac_extensions"]
            []
            >>> sorted(item["properties"])
            ['datetime']

            ```

    See Also:
        - BandMetadata: the opt-in band blocks that ride inside `raster:bands`.
        - pyramids.dataset.Dataset.to_stac_item: the entry point that consumes this group.
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

    Examples:
        - The default group names no output grid and allows nearest-neighbour
          alignment of a multi-asset item:
            ```python
            >>> from pyramids.stac import Alignment
            >>> alignment = Alignment()
            >>> alignment.grid is None
            True
            >>> alignment.align
            True
            >>> alignment.resampling is None
            True

            ```
        - Pin the cube to an explicit output grid and read the grid back:
            ```python
            >>> from pyramids.dataset import Grid
            >>> from pyramids.stac import Alignment
            >>> alignment = Alignment(grid=Grid(crs=4326, resolution=1.0, bounds=(0.0, 0.0, 2.0, 2.0)))
            >>> alignment.grid.crs, alignment.grid.resolution
            (4326, 1.0)
            >>> alignment.grid.bounds
            (0.0, 0.0, 2.0, 2.0)
            >>> alignment.grid.is_empty
            False

            ```
        - Resample per asset, and refuse alignment entirely:
            ```python
            >>> from pyramids.stac import Alignment
            >>> per_asset = Alignment(resampling={"B04": "bilinear", "B08": "average"})
            >>> per_asset.resampling["B08"]
            'average'
            >>> strict = Alignment(align=False)
            >>> strict.align
            False

            ```

    See Also:
        - ReadTolerance: whose substituted no-data plane borrows `grid` when one is given.
        - pyramids.dataset.DatasetCollection.from_stac: the entry point that consumes this group.
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

    Examples:
        - The default group forgives nothing, so either failure stops the build:
            ```python
            >>> from pyramids.stac import ReadTolerance
            >>> tolerance = ReadTolerance()
            >>> tolerance.skip_missing, tolerance.errors_as_nodata
            (False, False)

            ```
        - Forgive both failures at once — the two fields stay independent:
            ```python
            >>> from pyramids.stac import ReadTolerance
            >>> tolerance = ReadTolerance(skip_missing=True, errors_as_nodata=True)
            >>> tolerance.skip_missing
            True
            >>> tolerance.errors_as_nodata
            True

            ```
        - With `skip_missing`, an item that never declares the asset key is dropped
          instead of raising, so the cube is one timestep shorter:
            ```python
            >>> import os, tempfile
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
            >>> from pyramids.stac import ReadTolerance
            >>> folder = tempfile.mkdtemp()
            >>> items = []
            >>> for index, key in enumerate(("data", "other")):
            ...     path = os.path.join(folder, f"t{index}.tif")
            ...     Dataset.from_array(
            ...         np.full((2, 2), float(index + 1), dtype="float32"),
            ...         geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ...     ).to_file(path)
            ...     items.append(
            ...         {
            ...             "id": f"t{index}",
            ...             "bbox": [0.0, 0.0, 2.0, 2.0],
            ...             "properties": {"datetime": f"2023-06-0{index + 1}T00:00:00Z"},
            ...             "assets": {key: {"href": path}},
            ...         }
            ...     )
            >>> cube = DatasetCollection.from_stac(
            ...     items, "data", tolerance=ReadTolerance(skip_missing=True)
            ... )
            >>> cube.time_length
            1
            >>> cube.values[:, 0, 0].tolist()
            [1.0]

            ```

    See Also:
        - Alignment: supplies the reference grid a substituted no-data plane lands on.
        - pyramids.dataset.DatasetCollection.from_stac: the entry point that consumes this group.
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

    Examples:
        - The default group is the first-valid-pixel mosaic, with no custom fuser:
            ```python
            >>> from pyramids.stac import GroupFusion
            >>> fusion = GroupFusion()
            >>> fusion.method
            'first'
            >>> fusion.fuse_func is None
            True

            ```
        - A custom in-place fuser is kept as given and mutates its accumulator:
            ```python
            >>> import numpy as np
            >>> from pyramids.stac import GroupFusion
            >>> def keep_max(dst, src):
            ...     np.fmax(dst, src, out=dst)
            >>> fusion = GroupFusion(fuse_func=keep_max)
            >>> fusion.method
            'first'
            >>> accumulator = np.array([1.0, 5.0])
            >>> fusion.fuse_func(accumulator, np.array([3.0, 2.0]))
            >>> accumulator.tolist()
            [3.0, 5.0]

            ```
        - Something that is not callable is rejected on construction:
            ```python
            >>> from pyramids.stac import GroupFusion
            >>> GroupFusion(fuse_func="max")
            Traceback (most recent call last):
                ...
            TypeError: fuse_func must be a callable (dst, src) -> None, got str.

            ```

    See Also:
        - pyramids.dataset.merge.merge_rasters: the mosaic `method` is handed to.
        - pyramids.dataset.DatasetCollection.from_stac: the entry point that consumes this group.
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

    Examples:
        - The default group rewrites nothing on the way in:
            ```python
            >>> from pyramids.stac import AssetRead
            >>> read = AssetRead()
            >>> read.rescale
            False
            >>> read.cfg is None
            True

            ```
        - A `cfg` mapping is carried as given, keyed by collection id:
            ```python
            >>> from pyramids.stac import AssetRead
            >>> read = AssetRead(cfg={"sentinel-2-l2a": {"assets": {"rededge": {"alias": "B05"}}}})
            >>> read.cfg["sentinel-2-l2a"]["assets"]["rededge"]["alias"]
            'B05'
            >>> read.rescale
            False

            ```
        - With `rescale`, a packed int16 asset is read in physical units and its stored
          no-data becomes `NaN`, where the default read returns the stored counts:
            ```python
            >>> import os, tempfile
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
            >>> from pyramids.stac import AssetRead
            >>> path = os.path.join(tempfile.mkdtemp(), "packed.tif")
            >>> Dataset.from_array(
            ...     np.array([[0, 100], [200, 300]], dtype="int16"),
            ...     no_data_value=0,
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... ).to_file(path)
            >>> item = {
            ...     "id": "packed",
            ...     "bbox": [0.0, 0.0, 2.0, 2.0],
            ...     "properties": {"datetime": "2023-06-01T00:00:00Z"},
            ...     "assets": {
            ...         "data": {
            ...             "href": path,
            ...             "raster:bands": [{"scale": 0.01, "offset": 0.0, "nodata": 0}],
            ...         }
            ...     },
            ... }
            >>> DatasetCollection.from_stac([item], "data").first().tolist()
            [[0, 100], [200, 300]]
            >>> DatasetCollection.from_stac(
            ...     [item], "data", read=AssetRead(rescale=True)
            ... ).first().tolist()
            [[nan, 1.0], [2.0, 3.0]]

            ```

    See Also:
        - pyramids.stac._config: the `stac_cfg` mapping shape `cfg` accepts.
        - pyramids.dataset.DatasetCollection.from_stac: the entry point that consumes this group.
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

    Examples:
        - The default group carries nothing onto the cube:
            ```python
            >>> from pyramids.stac import CubeProvenance
            >>> provenance = CubeProvenance()
            >>> provenance.properties
            False
            >>> provenance.eo_band_names
            False

            ```
        - Naming the keys to carry, or asking for every property:
            ```python
            >>> from pyramids.stac import CubeProvenance
            >>> CubeProvenance(properties=["datetime", "eo:cloud_cover"]).properties
            ['datetime', 'eo:cloud_cover']
            >>> CubeProvenance(properties="eo:cloud_cover").properties
            'eo:cloud_cover'
            >>> CubeProvenance(properties=True, eo_band_names=True).eo_band_names
            True

            ```
        - The selected properties come back as one dict per timestep through
          :attr:`~pyramids.dataset.DatasetCollection.time_attrs`:
            ```python
            >>> import os, tempfile
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
            >>> from pyramids.stac import CubeProvenance
            >>> folder = tempfile.mkdtemp()
            >>> items = []
            >>> for index in range(2):
            ...     path = os.path.join(folder, f"t{index}.tif")
            ...     Dataset.from_array(
            ...         np.full((2, 2), float(index + 1), dtype="float32"),
            ...         geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ...     ).to_file(path)
            ...     items.append(
            ...         {
            ...             "id": f"t{index}",
            ...             "bbox": [0.0, 0.0, 2.0, 2.0],
            ...             "properties": {
            ...                 "datetime": f"2023-06-0{index + 1}T00:00:00Z",
            ...                 "eo:cloud_cover": 10 * index,
            ...             },
            ...             "assets": {"data": {"href": path}},
            ...         }
            ...     )
            >>> cube = DatasetCollection.from_stac(
            ...     items, "data", provenance=CubeProvenance(properties=["eo:cloud_cover"])
            ... )
            >>> cube.time_attrs
            [{'eo:cloud_cover': 0}, {'eo:cloud_cover': 10}]

            ```

    See Also:
        - GroupFusion: whose `method` decides whether a group's first item really
          describes the emitted pixels.
        - pyramids.dataset.DatasetCollection.from_stac: the entry point that consumes this group.
    """

    properties: bool | str | Sequence[str] = False
    eo_band_names: bool = False
