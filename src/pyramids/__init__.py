"""pyramids - GIS utility package"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from pathlib import Path

# The vendor bootstrap MUST run before any `from osgeo import …`
# anywhere downstream. `pyramids.base._configure` does that import at
# module load (line below), and `_configure` would fail to import the
# SWIG ext unless `_vendor/` is on sys.path. Bootstrap mechanism +
# rationale live in `pyramids.base._bootstrap`.
from pyramids.base._bootstrap import activate_vendored_osgeo

activate_vendored_osgeo(Path(__path__[0]))

# These import `from osgeo import …` transitively, so they must run *after*
# activate_vendored_osgeo() puts `_vendor/` on sys.path — hence E402 is expected.
from pyramids.base._configure import configure, configure_lazy_vector  # noqa: E402
from pyramids.base.config import Config  # noqa: E402

try:
    __version__ = _get_version("pyramids-core")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


def _warn_if_legacy_distribution() -> None:
    """Warn when the deprecated ``pyramids-gis`` alias distribution is installed.

    The distribution was renamed ``pyramids-gis`` -> ``pyramids-core``.
    ``pyramids-gis`` lives on as a thin alias that only depends on
    ``pyramids-core``, so its presence in the environment means the package was
    installed under the old name. ``FutureWarning`` (not ``DeprecationWarning``)
    is used deliberately: Python silences ``DeprecationWarning`` by default
    outside ``__main__``, so most end users would never see the notice.
    """
    legacy_installed = True
    try:
        _get_version("pyramids-gis")
    except PackageNotFoundError:
        legacy_installed = False
    if legacy_installed:
        warnings.warn(
            "The 'pyramids-gis' distribution is deprecated and will be discontinued; "
            "install 'pyramids-core' instead (the import name stays 'pyramids').",
            FutureWarning,
            stacklevel=2,
        )


_warn_if_legacy_distribution()

Config()

# `read_resource` / `sniff_kind` live in `pyramids._resource`, which imports the
# Dataset and FeatureCollection readers (and, transitively, geopandas / dask).
# Expose them lazily via PEP 562 module `__getattr__` so a bare `import pyramids`
# stays light — that heavy stack is only pulled in when these symbols are first
# accessed, not at package import time.
_LAZY_RESOURCE_EXPORTS = frozenset({"read_resource", "sniff_kind", "ResourceKind"})
# `register_dataset_accessor` lives in the heavy Dataset stack; expose it lazily too
# so a bare `import pyramids` does not pull that stack in just to reach the hook.
_LAZY_DATASET_EXPORTS = frozenset({"register_dataset_accessor"})


def __getattr__(name: str):
    """Lazily import the resource-reader / accessor exports on first access (PEP 562)."""
    if name in _LAZY_RESOURCE_EXPORTS:
        from pyramids._resource import ResourceKind, read_resource, sniff_kind

        globals().update(
            read_resource=read_resource,
            sniff_kind=sniff_kind,
            ResourceKind=ResourceKind,
        )
        return globals()[name]
    if name in _LAZY_DATASET_EXPORTS:
        from pyramids.dataset import register_dataset_accessor

        globals()["register_dataset_accessor"] = register_dataset_accessor
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(pyramids)``."""
    return sorted(set(globals()) | _LAZY_RESOURCE_EXPORTS | _LAZY_DATASET_EXPORTS)


__all__ = [
    "configure",
    "configure_lazy_vector",
    "read_resource",
    "register_dataset_accessor",
    "sniff_kind",
    "ResourceKind",
    "__version__",
]
