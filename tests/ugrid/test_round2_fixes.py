"""Regressions the second review round found, pinned.

Three separate defects, all introduced or missed by the first round's fixes: the tidy frame's axis
reordering was not a permutation and crashed on two real variable shapes; the layer API read
`var.data` and so fired the lazy loader that `info()` had just been fixed not to fire; and the
element-axis subset path still hardcoded the class, so it downgraded a subclass while the
time-axis path (fixed in round 1) did not.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.models import MeshVariable

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


class _CountingLoader:
    """A lazy loader that records how many times it was asked for the data."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.calls = 0

    def __call__(self) -> np.ndarray:
        self.calls += 1
        return self.values


class _Derived(UgridDataset):
    """A subclass, to check the subset paths do not downgrade it."""


@pytest.fixture
def base() -> UgridDataset:
    """A two-triangle mesh with one face variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array([1.0, 2.0])},
    )


def _with_dimensions(data: np.ndarray, dimensions: tuple[str, ...]) -> UgridDataset:
    """A mesh whose single variable carries the given shape and dimension names."""
    dataset = UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": data},
        data_locations={"v": "face"},
    )
    dataset["v"].dimensions = dimensions
    return dataset


class TestTidyFrameAxisOrder:
    def test_a_time_last_variable_does_not_crash(self):
        frame = _with_dimensions(
            np.zeros((2, 3)), ("mesh2d_nFaces", "time")
        ).to_dataframe()
        assert len(frame) > 0

    def test_a_rank_four_variable_does_not_crash(self):
        frame = _with_dimensions(
            np.zeros((2, 2, 3, 2)),
            ("time", "mesh2d_nLayers", "extra", "mesh2d_nFaces"),
        ).to_dataframe()
        assert len(frame) == 24

    def test_the_ordinary_temporal_shape_is_unchanged(self):
        frame = _with_dimensions(
            np.array([[1.0, 2.0], [3.0, 4.0]]), ("time", "mesh2d_nFaces")
        ).to_dataframe()
        assert len(frame) == 4
        assert sorted(set(frame["element"].tolist())) == [0, 1]

    def test_the_layered_shape_is_unchanged(self):
        frame = _with_dimensions(
            np.arange(6.0).reshape(3, 2), ("mesh2d_nLayers", "mesh2d_nFaces")
        ).to_dataframe()
        assert len(frame) == 6
        assert sorted(set(frame["layer"].tolist())) == [0, 1, 2]


class TestLayerApiIsMetadataOnly:
    @pytest.fixture
    def lazy(self, base) -> tuple[UgridDataset, _CountingLoader]:
        """A layered variable read through a counting loader."""
        loader = _CountingLoader(np.arange(6.0).reshape(3, 2))
        variable = MeshVariable(
            name="s",
            location="face",
            mesh_name="mesh2d",
            shape=(3, 2),
            dimensions=("mesh2d_nLayers", "mesh2d_nFaces"),
            _loader=loader,
        )
        dataset = UgridDataset(
            mesh=base.mesh,
            data_variables={"s": variable},
            global_attributes={},
            topology_info=None,
            crs_wkt=None,
            file_name=None,
        )
        return dataset, loader

    def test_n_layers_reads_the_declared_shape(self, lazy):
        dataset, loader = lazy
        assert dataset.n_layers == 3
        assert loader.calls == 0

    def test_layer_values_does_not_load(self, lazy):
        dataset, loader = lazy
        assert dataset.layer_values is None
        assert loader.calls == 0

    def test_info_still_does_not_load(self, lazy):
        dataset, loader = lazy
        dataset.info(buf=io.StringIO())
        assert loader.calls == 0


class TestElementSubsetKeepsTheSubclass:
    @pytest.fixture
    def derived(self, base) -> _Derived:
        """The dataset as a subclass instance, carrying a file name."""
        return _Derived(
            mesh=base.mesh,
            data_variables=base.data_vars,
            global_attributes={},
            topology_info=None,
            crs_wkt=None,
            file_name="source.nc",
        )

    def test_isel_element_keeps_the_class(self, derived):
        assert type(derived.isel(element=[0])) is _Derived

    def test_isel_element_keeps_the_file_name(self, derived):
        assert derived.isel(element=[0])._file_name == "source.nc"

    def test_isel_time_still_keeps_the_class(self, base):
        temporal = _Derived(
            mesh=base.mesh,
            data_variables=_with_dimensions(
                np.array([[1.0, 2.0], [3.0, 4.0]]), ("time", "mesh2d_nFaces")
            ).data_vars,
            global_attributes={},
            topology_info=None,
            crs_wkt=None,
            file_name="source.nc",
        )
        assert type(temporal.isel(time=0)) is _Derived
