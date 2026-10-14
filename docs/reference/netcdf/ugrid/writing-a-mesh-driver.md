# Writing a mesh driver

pyramids reads and writes UGRID NetCDF itself. Every other unstructured-mesh format — 2DM,
Selafin, DFSU, XMDF, PLY and the rest — is read by a separate distribution that registers a
**driver** with pyramids. This page is the contract such a distribution implements.

The dependency runs one way: your package imports pyramids' mesh types and builds them, and
pyramids never imports your package. It knows only the entry-point group name.

## Registering

Declare one entry point per driver in the `pyramids.mesh_drivers` group. The value is a **class**,
which pyramids instantiates with no arguments:

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.mesh_drivers"]
selafin = "your_package.drivers.telemac:SelafinDriver"
2dm     = "your_package.drivers.sms:TwoDmDriver"
```

A class rather than a function, because `can_open` matters: several of these formats share
`.dat`, `.mesh` and `.out`, so a suffix alone cannot identify one.

## The contract

```python
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

class SelafinDriver:
    name = "selafin"                      # unique; the value accepted by `driver=`
    long_name = "TELEMAC Selafin/Serafin"
    suffixes = (".slf", ".ser", ".res")   # lowercase, dot-prefixed
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def can_open(self, path: str) -> bool: ...
    def read(self, path, *, mesh_name=None, lazy=True): ...
    def read_data(self, path, dataset, *, lazy=True): ...   # only for "needs_mesh";
                                                           # returns dict[str, MeshVariable]
    def mesh_names(self, path: str) -> list[str]: ...
    def write(self, dataset, path, **options) -> None:      # only for "write_mesh"
        ...
```

`can_open` must **never raise** — an empty file, a directory, or a file of another format all
answer `False` — and must not read the whole file.

### Capabilities

Four tokens **gate behaviour**, and each requires the method it promises:

| Token | Effect | Method required |
|---|---|---|
| `read_mesh` | the file carries a topology and `read` builds it | `read` |
| `needs_mesh` | the file carries **values only**; `read_data` supplies them | `read_data` |
| `write_mesh` | `to_file(driver=…)` will call this driver | `write` |
| `write_data` | `write` persists variables; without it a write warns they were dropped | — |

The other three are **declarative** — they describe your driver in `info(drivers=True)` and in
your documentation, and pyramids does not branch on them:

| Token | Meaning |
|---|---|
| `read_data` | the driver produces data variables, not just a frame |
| `lazy_data` | `lazy` is honoured with a real deferred loader. pyramids forwards `lazy` to every driver regardless, so declare this only if you act on it |
| `multi_mesh` | `mesh_names` can return more than one entry. Every driver is asked, not only these — a single-mesh format returns a one-item list |

A driver declares **exactly one** of `read_mesh` or `needs_mesh`. Declaring neither, or both, is a
registration error.

### Results-only formats

Not every mesh file contains a mesh. An SMS `.dat`, an XMDF and an XDMF carry values that must be
paired with a topology loaded from somewhere else. Such a driver declares `needs_mesh`, leaves
`read` raising, and implements `read_data`:

```python
mesh = UgridDataset.read_file("domain.2dm")
result = mesh.attach_data("velocity.dat")
```

`read_data` returns **only the variables it read**, as a `dict[str, MeshVariable]` —
`attach_data` merges them onto the mesh. Do not return the merged dataset: that division is what
makes a name the mesh already defines an unambiguous collision, and it means your driver cannot
drop the mesh's own variables by rebuilding the mapping.

`read_data` must also **check the file's declared element count against the supplied mesh**, and
raise `ValueError` on a mismatch. That check is the only safeguard there is — a bare results file
carries nothing identifying the mesh it belongs to, so attaching it to the wrong one otherwise
produces a plausible and entirely wrong dataset. If the file declares no count, refuse it rather
than hoping.

## What a driver builds

Construct pyramids' own types and return them:

```python
from pyramids.netcdf.ugrid import (
    Connectivity, Mesh2d, MeshVariable, Network1d, UgridDataset,
)

UgridDataset(
    mesh=Mesh2d(
        node_x, node_y,
        face_node_connectivity=Connectivity(
            data=faces,                      # ALWAYS 0-indexed internally
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=1,          # what the FILE used, so a write restores it
        ),
        edge_node_connectivity=...,          # needed for edge-located data
        crs=wkt_or_epsg,
    ),
    data_variables={"water_level": MeshVariable(...)},
    global_attributes={"Conventions": "UGRID-1.0", "source": "TELEMAC Selafin"},
    topology_info=None,                      # see below
    crs_wkt=wkt,
    file_name=str(path),
)
```

A format with no NetCDF variables to name — a 2DM or a Selafin — passes `topology_info=None`,
which `UgridDataset` documents as the value for a mesh built from arrays. Synthesise
UGRID-conventional names only if you want `to_file()` to write a tidy NetCDF afterwards, and then
make them match what the writer actually emits.

Return a [`Network1d`][pyramids.netcdf.ugrid.Network1d] for a file whose topology dimension is 1 —
2DM's `E2L` line elements, or FLO-2D's and 3Di's 1D domains. `UgridDataset.read_file` refuses one
rather than hand back the wrong type; `Network1d.read_file` is where it is read:

```python
from pyramids.netcdf.ugrid import Network1d

network = Network1d.read_file("channels.2dm")
```

A driver may serve both: return a `Network1d` when the file declares `topology_dimension == 1` and
a `UgridDataset` otherwise. Each entry point checks what it received and names the other one if it
is the wrong shape.

### Five invariants — each fails silently

These produce **wrong numbers rather than an exception**, so each needs its own test in your
package:

| # | Invariant | Why |
|---|---|---|
| 1 | `Connectivity.data` is 0-indexed | `original_start_index` only *records* the file's base |
| 2 | Faces are fill-padded, never ragged | mixed triangles and quads is normal; rows are `max_nodes` wide |
| 3 | `fill_value` must not collide with a valid index | `-1` is the safe choice; never `0` on a 0-based array |
| 4 | Face winding is counter-clockwise | area, centroid, `polygonize` and `voronoi` all assume it |
| 5 | `MeshVariable.shape` is correct before any lazy load | the whole along-time API trusts it without forcing a read |

### Lazy reads

`MeshVariable._loader` is a zero-argument callable, resolved once by `.data` and cached. Set
`shape` from the file header and defer the values:

```python
MeshVariable(
    name="water_level", location="face", mesh_name="mesh2d",
    shape=(n_time, n_face),              # correct BEFORE the loader runs
    _loader=lambda: _read_slab(path, variable_index),
)
```

Close over the **path**, not an open file handle: the loader may run long after your `read`
returned, and on Windows a captured handle also keeps the file locked.

## How pyramids dispatches

`UgridDataset.read_file` resolves in this order:

1. `driver=` given — that driver, or `DriverNotExistError`. Never falls back.
2. A `/vsi` path or a NetCDF suffix (`.nc`, `.nc4`, `.nc3`, `.cdf`, `.netcdf`) — the built-in
   reader. **No driver discovery runs**, so a NetCDF read costs nothing.
3. Otherwise, the drivers claiming the suffix, each asked `can_open`; the first match wins, ties
   broken by registration order.
4. Nothing claimed it — the built-in reader anyway, so a NetCDF file under an unusual suffix still
   opens as it always did.
5. That failed too — `UnsupportedMeshFormatError`, naming what is installed and how to install
   more.

`to_file` never infers a driver from the suffix: `to_file("mesh.slf")` writes NetCDF unless you
pass `driver="selafin"`. `**options` reaches your `write` uninterpreted.

## Failure handling

Every registration problem is a **warning, not an exception**: a driver that raises on import,
declares a different `api_version`, collides on a name, or names an unknown capability token is
skipped. One broken third-party package cannot stop a NetCDF file from opening.

That is also why a driver whose optional dependency is missing should import it at **driver-module
scope** through pyramids' `require_optional` helper — the registry then skips that one driver and
every other keeps working:

```python
from pyramids.base._utils import require_optional

mikeio = require_optional(
    "mikeio",
    "The DFSU driver needs mikeio: pip install 'your-package[dhi]'",
    return_module=True,
)
```

## Compatibility

`MESH_DRIVER_API_VERSION` is bumped only on a breaking change to the member names above or to the
constructors a driver builds. A driver declaring a different value is skipped with a warning, so an
old plugin never breaks a pyramids upgrade — it stops being discovered until it is updated.

## Checking what is installed

```python
from pyramids.netcdf.ugrid import mesh_drivers

mesh_drivers()                  # {name: driver}, discovered once and cached
dataset.info(drivers=True)      # the same list, in the dataset summary
```

::: pyramids.netcdf.ugrid._drivers
    options:
      members:
        - MeshDriver
        - mesh_drivers
        - UnsupportedMeshFormatError
