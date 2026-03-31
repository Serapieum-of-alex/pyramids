"""pyramids - GIS utility package"""

from __future__ import annotations

from pyramids.base.config import Config


from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

config = Config()
