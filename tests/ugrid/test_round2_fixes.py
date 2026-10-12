"""Regressions the second review round found, pinned.

Three separate defects, all introduced or missed by the first round's fixes: the tidy frame's axis
reordering was not a permutation and crashed on two real variable shapes; the layer API read
`var.data` and so fired the lazy loader that `info()` had just been fixed not to fire; and the
element-axis subset path still hardcoded the class, so it downgraded a subclass while the
time-axis path (fixed in round 1) did not.
"""

from __future__ import annotations

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import CRS

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
    """The axis layouts the tidy frame has to get right.

    The first two tests here asserted only "does not crash" (`len(frame) > 0`) and
    `len(frame) == 24`, and both passed on a frame that was silently wrong: the time-last case
    emitted 3 rows for 6 values with every coordinate NaN, and the rank-four case emitted 24
    rows whose `element` ran 0-5 on a two-face mesh. Not crashing was never the contract. The
    full set of assertions now lives in `test_tidy_frame_axes.py`; these keep the shapes pinned
    here beside the other round 2 fixes.
    """

    def test_a_time_last_variable_emits_every_value_against_a_real_element(self):
        """A `(n_faces, time)` variable gives one row per element per step."""
        frame = _with_dimensions(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ("mesh2d_nFaces", "time")
        ).to_dataframe()

        assert len(frame) == 6, f"2 faces x 3 steps, got {len(frame)}"
        assert sorted(set(frame["element"].tolist())) == [0, 1], (
            f"a two-face mesh has elements 0 and 1, got {sorted(set(frame['element']))}"
        )
        assert not frame["x"].isna().any(), "every row should carry its face's centroid"

    def test_a_rank_four_variable_is_refused_by_name(self):
        """An axis the frame has no column for raises instead of being folded in."""
        dataset = _with_dimensions(
            np.zeros((2, 2, 3, 2)),
            ("time", "mesh2d_nLayers", "extra", "mesh2d_nFaces"),
        )

        with pytest.raises(ValueError, match="extra"):
            dataset.to_dataframe()

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


class TestGeoDataFrameCrsIsNotInvented:
    @pytest.fixture
    def polygons(self) -> list:
        """Two adjacent unit squares."""
        return [shapely.box(0.0, 0.0, 1.0, 1.0), shapely.box(1.0, 0.0, 2.0, 1.0)]

    def test_a_frame_without_a_crs_is_refused(self, polygons):
        frame = gpd.GeoDataFrame(geometry=polygons)
        with pytest.raises(ValueError, match="declares no CRS"):
            UgridDataset.from_geodataframe(frame)

    def test_an_epsg_crs_is_carried_through(self, polygons):
        frame = gpd.GeoDataFrame(geometry=polygons, crs=4326)
        assert UgridDataset.from_geodataframe(frame).epsg == 4326

    def test_a_projected_crs_stays_projected(self):
        # Previously any CRS whose to_epsg() was None landed on 4326 — metres stamped as
        # degrees, which every downstream reprojection then trusted.
        projected = CRS.from_epsg(3857)
        frame = gpd.GeoDataFrame(
            geometry=[
                shapely.box(0.0, 0.0, 1000.0, 1000.0),
                shapely.box(1000.0, 0.0, 2000.0, 1000.0),
            ],
            crs=projected,
        )
        built = UgridDataset.from_geodataframe(frame)
        assert built.crs.is_projected


class TestInterpolateNaHonoursDates:
    @pytest.fixture
    def dated(self) -> UgridDataset:
        """A gap at the middle step, with the steps a day and ten days apart."""
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0]])},
            data_locations={"v": "face"},
        )
        dataset["v"].attributes["time_values"] = [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-11"),
        ]
        return dataset

    def test_the_coordinate_reading_uses_the_dates(self, dated):
        # Day 2 of a day-1 to day-11 span is one tenth of the way, so 1.0 — not the 5.0 an
        # evenly spaced reading gives. The default silently gave the positional answer.
        filled = np.asarray(dated.interpolate_na()["v"].data)[1]
        assert filled.tolist() == pytest.approx([1.0, 1.0])

    def test_the_positional_reading_still_differs(self, dated):
        filled = np.asarray(dated.interpolate_na(use_coordinate=False)["v"].data)[1]
        assert filled.tolist() == pytest.approx([5.0, 5.0])

    def test_numeric_coordinates_are_unchanged(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0]])},
            data_locations={"v": "face"},
        )
        dataset["v"].attributes["time_values"] = [0.0, 1.0, 10.0]
        filled = np.asarray(dataset.interpolate_na()["v"].data)[1]
        assert filled.tolist() == pytest.approx([1.0, 1.0])

    def test_unmeasurable_labels_fall_back_to_positions(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0]])},
            data_locations={"v": "face"},
        )
        dataset["v"].attributes["time_values"] = ["early", "middle", "late"]
        filled = np.asarray(dataset.interpolate_na()["v"].data)[1]
        assert filled.tolist() == pytest.approx([5.0, 5.0])


class TestSelectionConventions:
    @pytest.fixture
    def faces(self) -> UgridDataset:
        """Two faces carrying distinguishable values."""
        return UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([10.0, 20.0])},
        )

    def test_a_negative_element_index_counts_from_the_end(self, faces):
        # isel(time=-1) already wrapped via np.take; the element axis refused it, leaving one
        # method with two index conventions.
        picked = faces.isel(element=-1)
        assert np.asarray(picked["v"].data).tolist() == [20.0]

    def test_an_out_of_range_element_is_still_refused(self, faces):
        with pytest.raises(IndexError, match="out of range"):
            faces.isel(element=5)

    def test_sel_without_a_value_says_so(self):
        temporal = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([[1.0, 2.0], [3.0, 4.0]])},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError, match="needs a time value"):
            temporal.sel()


class TestTidyFrameDtypes:
    def test_a_label_variable_is_not_forced_to_float(self):
        labelled = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"c": np.array(["a", "b"], dtype=object)},
        )
        assert labelled.to_dataframe()["value"].tolist() == ["a", "b"]

    def test_the_empty_frame_matches_the_populated_schema(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([10.0, 20.0])},
        )
        empty = dataset.to_dataframe(variables=[])
        populated = dataset.to_dataframe()
        assert list(empty.columns) == list(populated.columns)
        assert [str(d) for d in empty.dtypes] == [str(d) for d in populated.dtypes]


class TestNanProducingFacadesDeclareIt:
    @pytest.fixture
    def with_nodata(self) -> UgridDataset:
        """A face variable declaring a sentinel no-data value."""
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([10.0, 20.0])},
        )
        dataset["v"].nodata = -9999.0
        return dataset

    def test_to_location_declares_nan(self, with_nodata):
        # An element with no contributor averages to NaN, so carrying the old sentinel through
        # left `to_file` writing NaN under the wrong _FillValue.
        assert np.isnan(with_nodata.to_location("node")["v"].nodata)

    def test_burn_declares_its_fill(self, with_nodata):
        burned = with_nodata.burn(shapely.box(0.0, 0.0, 1.0, 1.0), 1.0)
        assert np.isnan(burned["burned"].nodata)

    def test_burn_does_not_declare_an_explicit_numeric_fill(self, with_nodata):
        """An explicit numeric fill is a value the caller chose, not a sentinel.

        Test scenario:
            This asserted the opposite when it was written, which made `burn(geom, 1.0,
            fill=0.0)` -- the 0/1 mask idiom -- declare `0.0` as missing, so `dropna` and every
            `skipna` reduction discarded the unburned faces. Only a NaN fill means "untouched",
            and the test above keeps that half pinned.
        """
        burned = with_nodata.burn(shapely.box(0.0, 0.0, 1.0, 1.0), 1.0, fill=0.0)

        assert burned["burned"].nodata is None, (
            f"a 0/1 mask has no missing values, got nodata={burned['burned'].nodata!r}"
        )


class TestInterpTimeRefusesDates:
    def test_a_date_coordinate_gets_a_written_error(self):
        dataset = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"v": np.array([[1.0, 2.0], [3.0, 4.0]])},
            data_locations={"v": "face"},
        )
        dataset["v"].attributes["time_values"] = [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
        ]
        targets = [pd.Timestamp("2020-01-01 12:00")]

        with pytest.raises(ValueError, match="needs numeric time coordinates"):
            dataset.interp_time(targets)
