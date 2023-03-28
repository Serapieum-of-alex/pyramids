import numpy as np
from osgeo.gdal import Dataset
from pandas import DataFrame

from pyramids.dem import DEM


def test_nearest_cell(
    points: DataFrame,
    src: Dataset,
    points_location_in_array: DataFrame,
):
    points["row"] = np.nan
    points["col"] = np.nan
    loc = DEM.nearestCell(src, points[["x", "y"]][:])
    assert ["cell_row", "cell_col"] == loc.columns.to_list()
    assert (
        loc.loc[:, "cell_row"].to_list()
        == points_location_in_array.loc[:, "rows"].to_list()
    )
    assert (
        loc.loc[:, "cell_col"].to_list()
        == points_location_in_array.loc[:, "cols"].to_list()
    )
