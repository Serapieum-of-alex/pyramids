"""pyramids - GIS utility package"""

from pyramids.base.config import Config

__all__ = ["dataset", "multidataset", "netcdf", "featurecollection"]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

config = Config()
