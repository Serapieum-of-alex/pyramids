"""The `UgridDataset` branches the member-by-member suites leave uncovered.

Three families of branch, all reachable from the public API and all previously untested:

- **the unreadable variable.** A variable can exist with neither an eager array nor a lazy loader
  (a file-read dataset whose array GDAL cannot produce). Every member that touches values has to
  report that by name instead of handing `None` to numpy, and the metadata-only members
  (`n_layers`, `to_dataframe`) have to tolerate it rather than raise.
- **the static companion.** Each along-time member carries a non-temporal variable through
  untouched. With a single temporal variable in the dataset that skip never runs, so these pin it
  with a static variable alongside the temporal one.
- **the refusals and the second argument spelling** — the location, CRS, count and method guards of
  `sample` / `weighted` / `zonal_stats` / `sel` / `mesh_names` / `summary`, `to_dataset` and `plot`
  taking a temporal variable's first step, `to_file` stamping the default `Conventions`, `to_crs`
  recomputing face and edge centres, and the ragged-edge fallback in `to_geodataframe`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from shapely.geometry import box

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid import dataset as ugrid_dataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
NON_UGRID_NC = Path("tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc")


def _connectivity(data: np.ndarray, cf_role: str) -> Connectivity:
    """A 0-indexed connectivity with -1 fill."""
    return Connectivity(
        data=np.asarray(data, dtype=np.intp),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


def _unreadable(source: UgridDataset, shape: tuple[int, ...]) -> UgridDataset:
    """`source`'s mesh carrying one temporal variable with no readable array."""
    return UgridDataset(
        mesh=source.mesh,
        data_variables={
            "v": MeshVariable(
                name="v",
                location="face",
                mesh_name=source.mesh_name,
                shape=shape,
                dimensions=("time", "mesh2d_nFaces"),
            )
        },
        global_attributes={},
        crs_wkt=source.crs_wkt,
    )


@pytest.fixture
def mesh() -> UgridDataset:
    """Two triangles carrying one static face variable."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"depth": np.array([2.0, 4.0])},
    )


@pytest.fixture
def temporal() -> UgridDataset:
    """Two triangles carrying a two-step temporal face variable with a time coordinate."""
    out = UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"d": np.array([[1.0, 2.0], [3.0, 4.0]])},
        data_locations={"d": "face"},
    )
    out["d"].attributes = {**out["d"].attributes, "time_values": [10.0, 20.0]}
    return out


@pytest.fixture
def mixed() -> UgridDataset:
    """A temporal variable and a static one on the same mesh."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"d": np.array([[1.0, 2.0], [3.0, 4.0]]), "s": np.array([7.0, 8.0])},
        data_locations={"d": "face", "s": "face"},
    )


@pytest.fixture
def unreadable(mesh) -> UgridDataset:
    """A two-step temporal variable whose array cannot be produced."""
    return _unreadable(mesh, (2, 2))


class TestRenameVariable:
    def test_renaming_an_absent_variable_is_refused(self, mesh):
        """The rename names the missing variable and lists what is available."""
        with pytest.raises(KeyError, match="not found"):
            mesh.rename_variable("nope", "other")

    def test_the_untouched_variables_are_carried_over(self, mesh):
        """A rename rebuilds the mapping, so every other variable survives it unchanged."""
        two = mesh.with_variable("extra", np.array([5.0, 6.0]), location="face")
        renamed = two.rename_variable("depth", "bed_level")
        assert sorted(renamed.data_variable_names) == ["bed_level", "extra"]
        assert renamed["extra"].data.tolist() == [5.0, 6.0]


class TestUnreadableVariable:
    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda d: d.reduce("mean"), "no loaded data to reduce"),
            (lambda d: d.cumsum(), "no loaded data for cumsum"),
            (lambda d: d.isel(time=0), "no loaded data to select"),
            (lambda d: d.head(1), "no loaded data to select"),
            (lambda d: d.diff(), "no loaded data to diff"),
            (lambda d: d.argmin(), "has no loaded data"),
            (lambda d: d.dropna(), "no loaded data to dropna"),
            (lambda d: d.summary(), "has no loaded data"),
            (lambda d: d.to_dataset("v", cell_size=0.5), "has no data loaded"),
            (lambda d: d.sample("v", 0.5, 0.5), "has no loaded data"),
            (lambda d: d.plot("v"), "no loaded data to plot"),
            (lambda d: d.concat(d), "no loaded data to concat"),
        ],
        ids=[
            "reduce",
            "cumsum",
            "isel",
            "head",
            "diff",
            "argmin",
            "dropna",
            "summary",
            "to_dataset",
            "sample",
            "plot",
            "concat",
        ],
    )
    def test_every_value_member_refuses_it_by_name(self, unreadable, call, message):
        """A variable with no readable array is reported, not handed to numpy as `None`."""
        with pytest.raises(ValueError, match=message):
            call(unreadable)

    def test_the_layer_count_tolerates_it(self, unreadable):
        """`n_layers` is metadata-only, so an unreadable variable simply has no layer axis."""
        assert unreadable.n_layers is None

    def test_the_tidy_frame_skips_it_and_keeps_the_columns(self, unreadable):
        """With nothing readable to tabulate the frame is empty but still correctly shaped."""
        frame = unreadable.to_dataframe()
        assert len(frame) == 0
        assert list(frame.columns) == [
            "variable",
            "element",
            "x",
            "y",
            "layer",
            "time",
            "value",
        ]

    def test_squeeze_refuses_a_single_step_variable(self, mesh):
        """A one-step variable is squeezed by reading it, so an unreadable one is refused."""
        unreadable = _unreadable(mesh, (1, 2))
        with pytest.raises(ValueError, match="no loaded data to squeeze"):
            unreadable.squeeze()


class TestStaticVariablesSurviveTheAlongTimeMembers:
    def test_sel_time_keeps_it(self, mixed):
        """`sel_time` picks a step of the temporal variable and leaves the static one alone."""
        picked = mixed.sel_time(1)
        assert picked["d"].data.tolist() == [3.0, 4.0]
        assert picked["s"].data.tolist() == [7.0, 8.0]

    def test_sel_time_range_keeps_it(self, mixed):
        """`sel_time_range` trims the time axis of the temporal variable only."""
        ranged = mixed.sel_time_range(0, 1)
        assert ranged["d"].data.shape == (1, 2)
        assert ranged["s"].data.tolist() == [7.0, 8.0]

    def test_isel_on_a_single_step_keeps_it(self, mixed):
        """An integer `isel` collapses the time axis of the temporal variable only."""
        picked = mixed.isel(time=0)
        assert picked["d"].data.tolist() == [1.0, 2.0]
        assert picked["s"].data.tolist() == [7.0, 8.0]

    def test_head_keeps_it(self, mixed):
        """The positional step selectors carry a static variable through untouched."""
        first = mixed.head(1)
        assert first["d"].data.shape == (1, 2)
        assert first["s"].data.tolist() == [7.0, 8.0]

    def test_diff_keeps_it(self, mixed):
        """Differencing shortens the temporal variable and leaves the static one alone."""
        differenced = mixed.diff()
        assert differenced["d"].data.tolist() == [[2.0, 2.0]]
        assert differenced["s"].data.tolist() == [7.0, 8.0]

    def test_argmin_keeps_it(self, mixed):
        """The extremum search answers per element for the temporal variable only."""
        located = mixed.argmin()
        assert located["d"].data.tolist() == [0, 0]
        assert located["s"].data.tolist() == [7.0, 8.0]

    def test_reduce_keeps_it(self, mixed):
        """A time reduction collapses the temporal variable and copies the static one."""
        reduced = mixed.reduce("mean")
        assert reduced["d"].data.tolist() == [2.0, 3.0]
        assert reduced["s"].data.tolist() == [7.0, 8.0]
        assert reduced["s"].has_time is False

    def test_cumsum_keeps_it(self, mixed):
        """The shape-preserving transforms skip a variable with no time axis."""
        running = mixed.cumsum()
        assert running["d"].data.tolist() == [[1.0, 2.0], [4.0, 6.0]]
        assert running["s"].data.tolist() == [7.0, 8.0]


class TestTimeReductionWrappers:
    def test_mean_collapses_the_time_axis(self, temporal):
        """`mean()` is `reduce("mean")`, so the variable becomes static."""
        reduced = temporal.mean()
        assert reduced["d"].data.tolist() == [2.0, 3.0]
        assert reduced["d"].has_time is False

    def test_var_collapses_the_time_axis(self, temporal):
        """`var()` is `reduce("var")` with the population convention (ddof 0)."""
        assert temporal.var()["d"].data.tolist() == [1.0, 1.0]


class TestFirstStepIsUsedForAGrid:
    def test_a_temporal_variable_is_rasterized_from_its_first_step(self, temporal):
        """`to_dataset` takes step 0 along the variable's own time axis, not the whole cube."""
        raster = temporal.to_dataset("d", cell_size=0.5, method="nearest")
        values = raster.read_array()
        assert float(np.nanmax(values)) == 2.0


class TestFileWriting:
    def test_a_dataset_without_conventions_is_stamped_on_write(self, mesh, tmp_path):
        """`to_file` supplies the CF/UGRID `Conventions` string when the dataset carries none."""
        bare = mesh.delete_global_attribute("Conventions")
        assert "Conventions" not in bare.global_attributes
        path = tmp_path / "bare.nc"
        bare.to_file(path)
        assert UgridDataset.read_file(path).global_attributes["Conventions"] == (
            "CF-1.8 UGRID-1.0"
        )


class TestGeoDataFrameLocations:
    def test_a_variable_on_another_facet_is_not_tabulated(self, mesh):
        """Asked for node geometries, a face variable is left out rather than mis-aligned."""
        gdf = mesh.to_geodataframe("depth", location="node")
        assert len(gdf) == 4
        assert "depth" not in gdf.columns

    def test_a_filled_edge_table_falls_back_to_a_per_edge_build(self, mesh):
        """A padded edge connectivity cannot be read as an (n_edge, 2) block, so each edge is built."""
        padded = Mesh2d(
            node_x=np.array([0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0]),
            face_node_connectivity=_connectivity([[0, 1, 2]], "face_node_connectivity"),
            edge_node_connectivity=_connectivity(
                [[0, 1, -1], [1, 2, -1]], "edge_node_connectivity"
            ),
        )
        ragged = UgridDataset(
            mesh=padded,
            data_variables={},
            global_attributes={},
            crs_wkt=mesh.crs_wkt,
        )
        geometries = ragged.to_geodataframe(location="edge").geometry
        assert len(geometries) == 2
        assert list(geometries.iloc[0].coords) == [(0.0, 0.0), (1.0, 0.0)]


class TestToCrsRecomputesTheCentres:
    def test_face_and_edge_centres_are_reprojected_too(self, mesh):
        """A mesh carrying face / edge centres has them transformed alongside the nodes."""
        centred = Mesh2d(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=_connectivity(
                [[0, 1, 2, 3]], "face_node_connectivity"
            ),
            edge_node_connectivity=_connectivity(
                [[0, 1], [1, 2]], "edge_node_connectivity"
            ),
            face_x=np.array([0.5]),
            face_y=np.array([0.5]),
            edge_x=np.array([0.5, 1.0]),
            edge_y=np.array([0.0, 0.5]),
        )
        source = UgridDataset(
            mesh=centred,
            data_variables={},
            global_attributes={},
            crs_wkt=mesh.crs_wkt,
        )
        moved = source.to_crs(3857)
        assert moved.mesh.has_face_coords is True
        assert moved.mesh.face_x[0] == pytest.approx(55659.745, abs=0.01)
        assert moved.mesh.edge_x[1] == pytest.approx(111319.491, abs=0.01)


class TestMeshNames:
    def test_a_missing_file_is_reported(self):
        """A path that is not a virtual-filesystem name is checked on disk first."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            UgridDataset.mesh_names("no_such_mesh.nc")

    def test_a_file_without_a_topology_is_reported(self):
        """A plain structured NetCDF declares no UGRID topology, which is an error here."""
        with pytest.raises(ValueError, match="No UGRID mesh topology"):
            UgridDataset.mesh_names(NON_UGRID_NC)

    def test_a_virtual_filesystem_path_is_handed_straight_to_gdal(self, mesh, tmp_path):
        """A `/vsimem/` name has no on-disk existence check — GDAL resolves it itself."""
        path = tmp_path / "m.nc"
        mesh.to_file(path)
        name = f"/vsimem/pyramids_mesh_names_{uuid4().hex}.nc"
        gdal.FileFromMemBuffer(name, path.read_bytes())
        try:
            assert UgridDataset.mesh_names(name) == ["mesh2d"]
        finally:
            gdal.Unlink(name)


class TestLayerCoordinates:
    def test_layer_values_skips_an_unlayered_variable(self, mesh):
        """The layer coordinate is read from the first *layered* variable, not the first one."""
        layered = mesh.with_variable("p", np.arange(6.0).reshape(3, 2), location="face")
        layered["p"].dimensions = ("mesh2d_nLayers", "mesh2d_nFaces")
        layered["p"].attributes = {
            **layered["p"].attributes,
            "layer_values": [1.0, 2.0, 3.0],
        }
        assert layered.layer_values == [1.0, 2.0, 3.0]

    def test_sel_layer_without_a_layer_dimension_is_refused(self, mesh):
        """Selecting a layer on a flat dataset says so rather than selecting nothing."""
        with pytest.raises(ValueError, match="sel_layer needs at least one variable"):
            mesh.sel_layer(0)


class TestToDataFrameCoordinates:
    def test_an_empty_selection_returns_the_column_skeleton(self, mesh):
        """Asking for no variables answers an empty frame with the documented columns."""
        frame = mesh.to_dataframe(variables=[])
        assert len(frame) == 0
        assert list(frame.columns) == [
            "variable",
            "element",
            "x",
            "y",
            "layer",
            "time",
            "value",
        ]

    def test_an_edge_variable_without_edge_coordinates_gets_nan_coordinates(self, mesh):
        """A mesh with no edge centres cannot address edge rows, so their coordinates are NaN."""
        edgy = mesh.with_variable("e", np.array([1.0, 2.0, 3.0]), location="edge")
        frame = edgy.to_dataframe(variables=["e"])
        assert len(frame) == 3
        assert frame["x"].isna().all()
        assert frame["value"].tolist() == [1.0, 2.0, 3.0]


class TestSummaryGuards:
    def test_a_non_numeric_variable_is_skipped_with_a_warning(self, mesh):
        """Summarising everything steps over a text variable instead of failing on it."""
        texts = mesh.with_variable("label", np.array(["a", "b"]), location="face")
        with pytest.warns(UserWarning, match="skipping non-numeric variable"):
            table = texts.summary()
        assert table.index.tolist() == ["depth"]

    def test_naming_a_non_numeric_variable_raises(self, mesh):
        """Asked for that variable explicitly, the refusal is an error rather than a warning."""
        texts = mesh.with_variable("label", np.array(["a", "b"]), location="face")
        with pytest.raises(TypeError, match="non-numeric"):
            texts.summary(variables=["label"])


class TestSampleGuards:
    def test_an_unknown_method_on_a_node_variable_is_refused(self, mesh):
        """A node variable supports `nearest` only, and any other method is named back."""
        nodal = mesh.with_variable("z", np.array([1.0, 2.0, 3.0, 4.0]), location="node")
        with pytest.raises(ValueError, match="must be 'contains' or 'nearest'"):
            nodal.sample("z", 0.1, 0.1, method="bogus")

    def test_an_edge_variable_is_unsupported(self, mesh):
        """Sampling resolves points to faces or nodes; an edge variable has no lookup."""
        edgy = mesh.with_variable("e", np.array([1.0, 2.0, 3.0]), location="edge")
        with pytest.raises(ValueError, match="supports 'face' and 'node'"):
            edgy.sample("e", 0.1, 0.1, method="nearest")


class TestWeightedGuards:
    def test_a_node_variable_is_refused(self, mesh):
        """Area weights are per face, so a node variable cannot be area-weighted."""
        nodal = mesh.with_variable("z", np.array([1.0, 2.0, 3.0, 4.0]), location="node")
        with pytest.raises(ValueError, match="need a face-located variable"):
            nodal.weighted("z")

    def test_a_face_count_mismatch_is_refused(self, mesh):
        """A face variable whose length is not the face count cannot be paired with the areas."""
        wrong = mesh.with_variable("x", np.array([1.0, 2.0, 3.0]), location="face")
        with pytest.raises(ValueError, match="face value"):
            wrong.weighted("x")


class TestZonalStatsGuards:
    def test_a_node_variable_is_refused(self, mesh):
        """Zones collect face centroids, so the variable has to be face-located."""
        nodal = mesh.with_variable("z", np.array([1.0, 2.0, 3.0, 4.0]), location="node")
        zones = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="needs a face-located variable"):
            nodal.zonal_stats(zones, variable_name="z")

    def test_zones_without_a_crs_skip_the_crs_check(self, mesh):
        """With no CRS declared on the zones there is nothing to compare, so the zones are used."""
        zones = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0.0, 0.0, 1.0, 1.0)])
        assert zones.crs is None
        table = mesh.zonal_stats(zones, variable_name="depth", stats=("mean", "count"))
        assert table["mean"].tolist() == [3.0]
        assert table["count"].tolist() == [2.0]


class TestTimeCoordinateGuards:
    def test_sel_without_a_time_dimension_is_refused(self, mesh):
        """Selecting by time coordinate needs a time axis; a static dataset has none."""
        with pytest.raises(ValueError, match="sel needs a time dimension"):
            mesh.sel(0.0)

    def test_nearest_needs_numeric_coordinates(self, temporal):
        """Snapping is a distance, so a date-like coordinate cannot be snapped against."""
        dated = temporal
        dated["d"].attributes = {
            **dated["d"].attributes,
            "time_values": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
        }
        wanted = pd.Timestamp("2020-01-01 06:00")
        with pytest.raises(ValueError, match="needs numeric time coordinates"):
            dated.sel(wanted, method="nearest")


class TestJoins:
    @staticmethod
    def _pair(values, stamps=None) -> UgridDataset:
        """One temporal and one static variable, optionally with a time coordinate."""
        out = UgridDataset.from_arrays(
            node_x=NODE_X,
            node_y=NODE_Y,
            face_node_connectivity=TRIANGLES,
            data={"d": np.asarray(values), "s": np.array([7.0, 8.0])},
            data_locations={"d": "face", "s": "face"},
        )
        if stamps is not None:
            out["d"].attributes = {**out["d"].attributes, "time_values": stamps}
        return out

    def test_concat_takes_a_static_variable_from_the_receiver(self):
        """A variable that is static in every part is carried over from `self` unchanged."""
        first = self._pair([[1.0, 2.0], [3.0, 4.0]], stamps=[0.0, 1.0])
        second = self._pair([[5.0, 6.0]], stamps=[2.0])
        joined = first.concat(second)
        assert joined["d"].data.shape == (3, 2)
        assert joined["s"].data.tolist() == [7.0, 8.0]
        assert joined["d"].attributes["time_values"] == [0.0, 1.0, 2.0]

    def test_concat_drops_a_stale_time_coordinate(self):
        """One part without a time coordinate leaves the joined axis with none at all."""
        first = self._pair([[1.0, 2.0], [3.0, 4.0]], stamps=[0.0, 1.0])
        second = self._pair([[5.0, 6.0]])
        joined = first.concat(second)
        assert joined["d"].data.shape == (3, 2)
        assert "time_values" not in joined["d"].attributes


class TestReduceHowGuard:
    def test_quantile_can_be_disallowed(self):
        """`allow_quantile=False` drops `quantile` from the accepted reductions."""
        with pytest.raises(ValueError, match="how must be one of"):
            ugrid_dataset._check_reduce_how("quantile", 0.5, allow_quantile=False)

    def test_quantile_is_accepted_by_default(self):
        """The same call is valid while `quantile` is permitted."""
        assert ugrid_dataset._check_reduce_how("quantile", 0.5) is None


class TestPlotDispatch:
    def test_a_temporal_variable_plots_its_first_step_under_an_explicit_title(
        self, temporal
    ):
        """The render helper receives step 0 of the variable and the caller's own title."""
        with patch(
            "pyramids.netcdf.ugrid.dataset._mesh_render", return_value="sentinel"
        ) as render:
            assert temporal.plot("d", title="custom") == "sentinel"
        kwargs = render.call_args.kwargs
        assert kwargs["title"] == "custom"
        assert np.asarray(kwargs["data"]).tolist() == [1.0, 2.0]


@pytest.mark.interop
@pytest.mark.lazy
class TestZarrRoundTrip:
    def test_from_zarr_reads_back_what_to_zarr_wrote(self, mesh, tmp_path):
        """The zarr pair opens the store once and rebuilds both the mesh and its variables."""
        pytest.importorskip("xarray")
        pytest.importorskip("zarr")
        store = str(tmp_path / "mesh.zarr")
        mesh.to_zarr(store)
        back = UgridDataset.from_zarr(store)
        assert (back.n_node, back.n_face) == (4, 2)
        assert back["depth"].data.tolist() == [2.0, 4.0]
