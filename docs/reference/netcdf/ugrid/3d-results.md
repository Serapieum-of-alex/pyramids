# 3-D results fidelity: active masks and ragged layers

Hydrodynamic mesh results carry two kinds of structure a plain `(time, layer, element)` block
cannot represent faithfully. `UgridDataset` models both, additively — a 2-D mesh or a
uniformly-layered one behaves exactly as before, and every accessor below is inert until the
relevant structure is present.

## Per-timestep active (wet/dry) mask

A cell in a flood or coastal model can be **wet** at one step and **dry** at the next. Folding
"dry" into `nodata`/`NaN` is lossy and irreversible: a consumer can no longer tell a dry cell
from one whose value is genuinely missing, and derived quantities such as a wetted-area time
series are destroyed. pyramids keeps the mask as a first-class field on
[`MeshVariable`](models.md), separate from `nodata`.

```python
import numpy as np
from pyramids.netcdf.ugrid import UgridDataset

ds = UgridDataset.read_file("flood_map.nc")

# The mask as read — True = active/wet, never applied to the values on its own.
mask = ds.active_mask("water_depth")        # (n_element,) or (n_time, n_element), or None

# Opt in to treat dry as missing: dry cells become NaN on top of the nodata blanking.
wet_only = ds.masked_values("water_depth")  # dry AND missing are NaN; still distinguishable
```

Key properties:

- `MeshVariable.active` is `None` by default. A static mask is `(n_element,)`; a per-step mask is
  `(n_time, n_element)`. `True` means active/wet.
- The mask **rides through the shape-preserving selections** — `sel_time`, `sel_time_range`,
  `isel(time=…)`, `isel(element=…)`, and the element-subsetting spatial ops — sliced to match the
  data. A time **reduction** (`mean`, `sum`, …) drops a per-step mask (it no longer aligns) while
  keeping a static one.
- It is applied to the values **only on demand**, through `masked_values()` or the
  `apply_active=` flag on the internal per-element reader — never by default. "Dry" stays
  observably different from "missing".

## Ragged (per-column) vertical layers

A genuinely 3-D mesh usually has a **different number of active layers per column**: a deep
channel column may carry ten layers, a shallow one two. That data is ragged — a flat
`(time?, n_volume)` array indexed through CSR-style offsets — not a rectangular
`(layer, face)` block. [`VolumeTopology`](models.md) carries the map from faces to that flat
volume axis, and a variable on the new `location="volume"` is indexed through it.

```python
from pyramids.netcdf.ugrid import VolumeTopology

ds = UgridDataset.read_file("estuary_3d.nc")
ds.volume_topology          # VolumeTopology, or None for a 2-D / uniform mesh
ds.n_volume                 # total volumes, or None
ds.n_layers                 # the MAX layer count over the columns (documented), for a ragged mesh

# The k-th layer per column as a 2-D face field; columns shallower than k+1 layers are NaN,
# never silently padded.
surface = ds.isel_layer(0)              # -> location="face", shape (time?, n_face)
bed_slice = ds.collapse_layer(2)        # the explicit 2-D collapse (alias of isel_layer for faces)
```

`VolumeTopology` is the CSR layout the finite-volume codes use:

- `layer_count` — `(n_face,)` int, active layers per column.
- `volume_offset` — `(n_face + 1,)` int, offsets into the flat volume axis, so face `f` owns
  volumes `volume_offset[f] : volume_offset[f + 1]`, bottom layer first.
- `layer_z` — optional `(n_volume,)` float, each volume's vertical coordinate.
- `interface_z` — optional `(n_volume + n_face,)` float, each layer interface's coordinate.

Its invariants (`volume_offset` starts at 0, is non-decreasing, and its gaps equal
`layer_count`) are validated at construction, so a malformed topology fails early rather than at
use. `to_dataframe()` lays a ragged variable out one row per volume, carrying the column
(`element`) and the 0-based `layer` within it — no row is fabricated for an absent layer. The
uniform-layer path (`n_layers`, `isel_layer`, `sel_layer`, `layer_values`) is unchanged whenever
no `VolumeTopology` is set.

## On-disk layout

Both features round-trip through the built-in NetCDF writer/reader and the xarray/Zarr bridge,
using companion variables so an older reader still sees a valid UGRID file and simply ignores
them:

| Companion | Role attribute (`pyramids_role`) | Shape | Purpose |
|---|---|---|---|
| `<var>_active` | `active_mask` | `(n_element,)` or `(n_time, n_element)` | a variable's wet/dry mask; `Int8`, with `ancillary_for = <var>` naming the variable it masks |
| `<mesh>_layer_count` | `layer_count` | `(n_face,)` | active layers per column |
| `<mesh>_volume_offset` | `volume_offset` | `(n_face + 1,)` | CSR offsets into the volume axis |
| `<mesh>_layer_z` | `layer_z` | `(n_volume,)` | per-volume vertical coordinate (optional) |
| `<mesh>_interface_z` | `interface_z` | `(n_volume + n_face,)` | per-interface vertical coordinate (optional) |

The mask companion carries **no** `location`/`mesh` attribute, so the topology parser never
mistakes it for a data variable; it is re-attached to its variable by the `ancillary_for`
attribute. The volume arrays are found by name and `pyramids_role`. The main variable also gains
a CF `ancillary_variables` attribute pointing at its mask companion.

## Downstream

These are the two 3-D results-fidelity models that unblock true-3D reading in the `pyramids-mesh`
driver plugins (TUFLOW FV, XMDF, DFSU). Once the `pyramids-mesh` floor is bumped, those drivers
populate `MeshVariable.active` directly instead of the interim `<var>_active` sibling variable,
and build a `VolumeTopology` for genuinely 3-D files instead of reading only the 2-D slice.
