"""A regridded variable must declare the no-data value it actually uses.

The regridder leaves NaN wherever the source covers none of a target element. Without a declared
no-data value `to_file` writes those NaNs with no `_FillValue`, so a reader cannot tell a gap from
a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def source() -> UgridDataset:
    """A unit square meshed as two triangles."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"v": np.array([1.0, 2.0])},
    )


@pytest.fixture
def partly_covered() -> UgridDataset:
    """A target whose second face lies far outside the source."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 5.0, 6.0, 6.0]),
        node_y=np.array([0.0, 0.0, 1.0, 5.0, 5.0, 6.0]),
        face_node_connectivity=np.array([[0, 1, 2], [3, 4, 5]]),
        data={"t": np.array([0.0, 0.0])},
    )


class TestRegridNodata:
    def test_an_uncovered_element_is_nan(self, source, partly_covered):
        values = np.asarray(source.regrid(partly_covered)["v"].data)
        assert np.isnan(values[1])

    def test_the_variable_declares_nan_as_nodata(self, source, partly_covered):
        regridded = source.regrid(partly_covered)
        assert np.isnan(regridded["v"].nodata)

    def test_the_covered_element_keeps_its_value(self, source, partly_covered):
        values = np.asarray(source.regrid(partly_covered)["v"].data)
        assert values[0] == pytest.approx(1.0)

    def test_a_fully_covered_target_also_declares_it(self, source):
        target = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"t": np.array([0.0, 0.0])},
        )
        assert np.isnan(source.regrid(target)["v"].nodata)
