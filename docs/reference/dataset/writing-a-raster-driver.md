# Writing a raster driver

pyramids reads rasters through GDAL. Some sources GDAL cannot open on its own — an Earth Engine
asset id, a Sentinel `.SAFE` product directory, an EUMETSAT FCI `.nat` granule — are read by a
separate distribution that registers a driver into the `pyramids.raster_drivers` entry-point group.
`Dataset.read_file` then routes those sources into the plugin with no import coupling: pyramids
never imports a plugin, and a bare install reads through GDAL exactly as before.

It is the raster sibling of the [mesh driver hook](../netcdf/ugrid/writing-a-mesh-driver.md), with
two differences for the raster/EO case: a driver's claims are **class attributes** (read without
constructing the driver, so discovery never imports a heavy backend), and `can_open` judges an
**arbitrary source** (a path, a directory, or a scheme such as `ee://`), not just a file suffix.

## Registering

```toml
# your-package/pyproject.toml
[project.entry-points."pyramids.raster_drivers"]
earthengine = "your_package.drivers:EarthEngineDriver"
```

The dependency runs one way: your package imports pyramids' raster types and builds them; pyramids
knows only the group name.

## The contract

A driver is a **class** whose claims are class-level, so the registry reads them without
constructing it:

```python
from pyramids.dataset import Dataset

class EarthEngineDriver:
    name = "earthengine"                     # unique; the value accepted by driver=
    long_name = "Google Earth Engine"
    api_version = 1                          # must equal RASTER_DRIVER_API_VERSION
    suffixes = ()                            # file suffixes, lowercase + dot-prefixed
    schemes = ("ee",)                        # URL schemes, lowercase, no "://"
    capabilities = frozenset({"read", "needs_auth"})
    extra = "ee"                             # import name of an optional dep, or None

    @classmethod
    def can_open(cls, source) -> bool:       # cheap; no heavy import, no network, never raises
        return isinstance(source, str) and source.startswith("ee://")

    def read(self, source, *, bbox=None, bands=None, crs=None, options=None, auth=None) -> Dataset:
        ...                                  # the heavy backend is imported HERE, not at module top
```

Declare at least one of `suffixes` / `schemes` (something has to route to the driver) and the
`"read"` capability. The driver module must import cleanly without its heavy backend — put that
import inside `read`, and name the backend in `extra` so discovery skips the driver when the
backend is not installed (its `can_open` is then never reached).

### Capabilities

- `read` — required; gates `read`, the method `Dataset.read_file` dispatches into.
- `read_collection` — gates `read_collection(source, *, time=None, **kwargs) -> DatasetCollection`
  (a temporal stack); the method must exist when declared.
- `needs_auth` — declarative; tells `describe_drivers` and your docs the source needs credentials.

## How pyramids dispatches

`Dataset.read_file(source, driver=None)` resolves in this order:

1. an explicit `driver="earthengine"` routes straight to that registered driver, bypassing GDAL;
2. otherwise pyramids tries GDAL first — and **only if GDAL cannot open the source** does it scan
   the registry, handing the source to the first driver whose `can_open` claims it;
3. if neither GDAL nor any driver reads it, GDAL's own error is raised.

Trying GDAL first is deliberate: a bare install (no plugins) performs no entry-point discovery on a
normal GDAL read, and the discovery cache stays cold. Discovery is lazy and cached — it runs at
most once per process, on the first source GDAL rejects (or the first explicit `driver=`).

## Failure handling

Every discovery failure is a **warning, never an exception**: a driver that raises on import,
declares a future `api_version`, is missing its `extra`, collides on a name, or whose `can_open`
raises during a scan is skipped. One broken third-party package cannot stop a raster from opening
through GDAL.

## Checking what is installed

```python
from pyramids.dataset._drivers import describe_drivers, raster_drivers

raster_drivers()        # {name: driver class}, discovered once and cached
describe_drivers()      # [{name, long_name, suffixes, schemes, capabilities}, ...]
```

## Conformance kit

Run the reusable kit against your driver class in your own test suite — it needs no pytest and
checks the same contract discovery enforces, plus the behavioural promises (that `can_open` is a
classmethod, returns a `bool`, and never raises):

```python
from pyramids.dataset.driver_conformance import assert_raster_driver_conforms
from your_package.drivers import EarthEngineDriver

def test_driver_conforms():
    assert_raster_driver_conforms(EarthEngineDriver)
```

## Compatibility

`api_version` is bumped only on a breaking change to the driver contract. A driver declaring a
different version is skipped with a warning, so an old plugin never breaks a pyramids upgrade and a
new plugin never half-loads on an old pyramids. This hook is additive: a bare pyramids install
behaves exactly as it did before it existed.
