"""Spatial operations mixin for Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr

from pyramids.base._domain import inside_domain, is_no_data
from pyramids.base._utils import INTERPOLATION_METHODS
from pyramids.base.crs import (
    epsg_from_wkt,
    reproject_coordinates,
    sr_from_epsg,
    sr_from_wkt,
)
from pyramids.dataset._collaborators import Vectorize as _Vectorize
from pyramids.dataset.abstract_dataset import AbstractDataset
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset
