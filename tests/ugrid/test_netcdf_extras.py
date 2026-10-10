"""N1-N4 — to_dataframe, copy/from_bytes, info, and global-attribute editing."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def mesh() -> UgridDataset:
    """Two triangles with one static face variable."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"depth": np.array([2.0, 4.0])},
    )


@pytest.fixture
def temporal() -> UgridDataset:
    """Two triangles with a two-step temporal face variable."""
    out = UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        data={"d": np.array([[1.0, 2.0], [3.0, 4.0]])},
        data_locations={"d": "face"},
    )
    out["d"].attributes = {**out["d"].attributes, "time_values": [10.0, 20.0]}
    return out


class TestToDataFrame:
    def test_static_variable_is_one_row_per_face(self, mesh):
        frame = mesh.to_dataframe()
        assert list(frame.columns) == [
            "variable",
            "element",
            "x",
            "y",
            "time",
            "value",
        ]
        assert len(frame) == 2
        assert frame["value"].tolist() == [2.0, 4.0]
        assert frame["element"].tolist() == [0, 1]

    def test_centroids_are_carried(self, mesh):
        frame = mesh.to_dataframe()
        assert frame["x"].notna().all()
        assert frame["y"].notna().all()

    def test_temporal_variable_is_one_row_per_element_and_step(self, temporal):
        frame = temporal.to_dataframe()
        assert len(frame) == 4
        assert sorted(set(frame["time"].tolist())) == [10.0, 20.0]

    def test_selecting_variables(self, mesh):
        two = mesh.with_variable("other", np.array([9.0, 9.0]), location="face")
        frame = two.to_dataframe(variables=["other"])
        assert set(frame["variable"]) == {"other"}

    def test_unknown_variable_raises(self, mesh):
        with pytest.raises(KeyError):
            mesh.to_dataframe(variables=["nope"])

    def test_dropna_removes_gaps(self, mesh):
        gappy = mesh.with_variable("g", np.array([1.0, np.nan]), location="face")
        assert len(gappy.to_dataframe(variables=["g"], dropna=True)) == 1


class TestCopy:
    def test_copy_is_independent(self, mesh):
        clone = mesh.copy()
        clone["depth"].data[0] = 99.0
        assert mesh["depth"].data[0] == 2.0

    def test_copy_keeps_values_and_attributes(self, mesh):
        stamped = mesh.set_global_attribute("title", "demo")
        clone = stamped.copy()
        assert clone["depth"].data.tolist() == [2.0, 4.0]
        assert clone.global_attributes["title"] == "demo"


class TestFromBytes:
    def test_round_trips_through_memory(self, mesh, tmp_path: Path):
        path = tmp_path / "m.nc"
        mesh.to_file(path)
        back = UgridDataset.from_bytes(path.read_bytes())
        assert (back.n_face, back.n_node) == (mesh.n_face, mesh.n_node)
        assert back["depth"].data.tolist() == [2.0, 4.0]

    def test_result_survives_the_buffer(self, mesh, tmp_path: Path):
        path = tmp_path / "m.nc"
        mesh.to_file(path)
        back = UgridDataset.from_bytes(path.read_bytes())
        # The vsimem file is unlinked inside from_bytes; the values must still be readable.
        assert float(np.asarray(back["depth"].data).sum()) == 6.0


class TestInfo:
    def test_writes_a_summary(self, mesh):
        buffer = io.StringIO()
        mesh.info(buf=buffer)
        text = buffer.getvalue()
        assert "UgridDataset:" in text
        assert "faces: 2" in text
        assert "depth" in text

    def test_reports_temporal_variables(self, temporal):
        buffer = io.StringIO()
        temporal.info(buf=buffer)
        assert "temporal" in buffer.getvalue()

    def test_reports_global_attributes(self, mesh):
        buffer = io.StringIO()
        mesh.set_global_attribute("title", "demo").info(buf=buffer)
        assert "title: demo" in buffer.getvalue()


class TestGlobalAttributes:
    def test_set_returns_a_new_dataset(self, mesh):
        stamped = mesh.set_global_attribute("title", "demo")
        assert stamped.global_attributes["title"] == "demo"
        assert "title" not in mesh.global_attributes

    def test_delete_removes_it(self, mesh):
        stamped = mesh.set_global_attribute("title", "demo")
        bare = stamped.delete_global_attribute("title")
        assert "title" not in bare.global_attributes
        assert stamped.global_attributes["title"] == "demo"

    def test_deleting_an_absent_attribute_is_not_an_error(self, mesh):
        assert mesh.delete_global_attribute("nope").global_attributes == (
            mesh.global_attributes
        )

    def test_numeric_vector_attribute_survives_a_file_round_trip(
        self, mesh, tmp_path: Path
    ):
        path = tmp_path / "m.nc"
        mesh.set_global_attribute("bounds", [1.0, 2.0, 3.0]).to_file(path)
        back = UgridDataset.read_file(path)
        assert list(np.atleast_1d(back.global_attributes["bounds"])) == [1.0, 2.0, 3.0]
