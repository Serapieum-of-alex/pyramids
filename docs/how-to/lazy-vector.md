# Lazy vector reads — `LazyFeatureCollection`

`FeatureCollection.read_file(path, backend="dask")` and
`FeatureCollection.read_parquet(path, backend="dask")` return a
`LazyFeatureCollection` — a subclass of `dask_geopandas.GeoDataFrame` that
satisfies pyramids' `SpatialObject` protocol. Every partition-aware
operation inherited from dask-geopandas (`to_crs`, `clip`, `sjoin`,
`spatial_shuffle`) runs lazily; materialise with `.compute()` when you need
eager rows.

## ⚠ Before you benchmark: set `scheduler="processes"`

Shapely and GEOS hold the Python GIL. Under dask's **default threaded**
scheduler, `map_partitions` on geometry ops runs single-core. If your
benchmark shows the dask backend *slower* than pandas, that is almost
always why.

```python
import dask
dask.config.set(scheduler="processes")   # process-based, no GIL contention

# or spin up an explicit cluster:
from dask.distributed import LocalCluster, Client
cluster = LocalCluster(n_workers=4)
client = Client(cluster)
```

Set one of these **before** issuing the first `.compute()` call. Without
it, the lazy API still works, but the speedup you expect won't appear.

## When to use `backend="dask"`

Rule of thumb:

- **> 500 MB GeoDataFrame**, or **> 10 million rows** — the partitioned
  read wins.
- Smaller than that, the eager path is faster because it avoids the
  partitioning cost.
- Anything that materialises rows (`.iloc`, `.loc`, `len`, `.plot`,
  `.to_file` for OGR drivers) forces a compute. Don't go lazy just to
  immediately materialise.

## Which ops stay lazy vs need `.compute()`

| Runs lazily on `LazyFeatureCollection`                                   | Requires `.compute()` first                   |
|--------------------------------------------------------------------------|-----------------------------------------------|
| `to_crs`, `clip`, `sjoin` (partition-pruned if `spatial_partitions` set) | `.iloc`, `.loc`, `len`                        |
| `spatial_shuffle`                                                        | `.plot` (no lazy plotting path)               |
| `.compute`, `.persist` (compute barriers)                                | `.to_file` (no lazy OGR write path)           |
| Any inherited `dask_geopandas.GeoDataFrame` method                       | pyramids-specific methods (`extract_vertices`, `rasterize_with_col`, `with_coordinates`, `with_centroid`, `center_points`) |

The rule: if dask-geopandas has a lazy implementation, so does
`LazyFeatureCollection`. Everything else — including every pyramids-specific
method that walks geometries one by one — raises `AttributeError` on a
lazy FC; call `.compute()` first.

## `spatial_shuffle` → `sjoin` pruning workflow

The biggest speedup from going lazy comes from partition-pruned
`sjoin` — each partition has a bounding box, and dask drops partition
pairs that can't intersect before dispatching work. This requires the
bounding boxes to actually exist:

```python
from pyramids.feature import FeatureCollection

# 1. Load both frames lazily.
left = FeatureCollection.read_parquet("census_blocks.parquet", backend="dask")
right = FeatureCollection.read_parquet("roads.parquet", backend="dask")

# 2. Shuffle to populate spatial_partitions. This is a one-time cost
#    amortised across subsequent sjoins with the same frame.
left_shuffled = left.spatial_shuffle(by="hilbert")
right_shuffled = right.spatial_shuffle(by="hilbert")

# 3. Join. Dask drops partition pairs whose bboxes don't overlap.
joined = left_shuffled.sjoin(right_shuffled, how="inner", predicate="intersects")

# 4. Materialise.
result = joined.compute()   # a FeatureCollection, not a bare GeoDataFrame
```

`.spatial_shuffle` is partially eager — it computes Hilbert distances for
every row before rebuilding partitions. On 10 GB+ inputs, `.persist()` the
result so subsequent reads don't recompute the shuffle graph.

## `.persist()` for interactive workflows

`.compute()` materialises rows and returns an eager `FeatureCollection` —
you leave the lazy domain. `.persist()` materialises the task graph into
worker memory but keeps the lazy wrapper, so subsequent ops build on the
persisted partitions without recomputing:

```python
from pyramids.feature import FeatureCollection

lazy = FeatureCollection.read_parquet("large.parquet", backend="dask")
shuffled = lazy.spatial_shuffle().persist()   # still lazy, but graph is warm

# subsequent sjoin / clip calls benefit from the persisted shuffle:
hits = shuffled.sjoin(query_fc, predicate="intersects").compute()
```

## `total_bounds` is lazy

`LazyFeatureCollection` inherits `total_bounds` from
`dask_geopandas.GeoDataFrame`. That attribute returns a **dask Scalar**, not
a numpy array of four floats. Indexing it returns another lazy object:

```python
lfc = FeatureCollection.read_parquet("points.parquet", backend="dask")
lfc.total_bounds           # <dask.Scalar ...> — NOT a tuple
lfc.total_bounds.compute() # → array([xmin, ymin, xmax, ymax])
```

The `top_left_corner` property on `LazyFeatureCollection` does the
`.compute()` for you (it's cheap, O(partitions), no materialisation of
the full frame) so `SpatialObject` consumers get concrete numbers:

```python
lfc.top_left_corner   # → [xmin, ymax]  as list[float]
```

## Minimal-install sentinel

On installs without the `[parquet-lazy]` extra (i.e. no `dask-geopandas`),
`from pyramids.feature import LazyFeatureCollection` returns `None`. Code
that does `isinstance(x, LazyFeatureCollection)` must guard:

```python
from pyramids.feature import LazyFeatureCollection

if LazyFeatureCollection is not None and isinstance(obj, LazyFeatureCollection):
    ...
```

Typical user code doesn't need this guard — if you're working with lazy
frames at all, you already have the extra installed. It's for library
authors writing dispatch code that runs on both minimal and full installs.

## Interop with `Dataset.zonal_stats`

The raster-side `Dataset.zonal_stats(fc)` currently expects an eager
`FeatureCollection`. Until a lazy path lands, call `.compute()` first:

```python
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

ds = Dataset.read_file("dem.tif")
polys = FeatureCollection.read_parquet("zones.parquet", backend="dask")

# must materialise for now:
stats = ds.zonal_stats(polys.compute(), stats=("mean", "sum"))
```

A lazy `zonal_stats` that accepts a `LazyFeatureCollection` is tracked as a
follow-up against DASK-25.

## Install

`LazyFeatureCollection` requires the `[parquet-lazy]` extra:

```bash
pip install 'pyramids-gis[parquet-lazy]'
```

That pulls `pyarrow`, `dask`, `dask-geopandas`, `zarr`, and `fsspec`.
On minimal installs, `LazyFeatureCollection` is `None` and the dask
branches of `read_file` / `read_parquet` raise `ImportError` with an
actionable install hint.
