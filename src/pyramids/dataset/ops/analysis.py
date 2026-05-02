"""Analysis, statistics, and plot mixin for Dataset."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, get_pixels2
from pandas import DataFrame

from pyramids.base._domain import inside_domain, is_no_data
from pyramids.base._errors import AlignmentError
from pyramids.base._utils import import_cleopatra
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from cleopatra.array_glyph import ArrayGlyph

    from pyramids.dataset.dataset import Dataset
