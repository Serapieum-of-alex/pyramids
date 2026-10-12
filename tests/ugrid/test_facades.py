"""Integration of the capability modules through their UgridDataset facades.

The kernels are unit-tested in `test_graph.py` / `test_fill.py` / `test_vector_ops.py` /
`test_regrid.py` / `test_construct.py` / `test_interop_xarray.py`; this file pins the wiring —
that each facade resolves the right variable, hands the kernel the mesh, and wraps the answer
back into the expected type.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from pyramids.netcdf.ugrid import UgridDataset


@pytest.fixture
def strip() -> UgridDataset:
    """Two quads side by side carrying one face variable."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
        data={"d": np.array([1.0, 2.0])},
    )


@pytest.fixture
def split() -> UgridDataset:
    """Two disjoint triangles, so the graph has two components."""
    return UgridDataset.from_arrays(
        node_x=np.array([0.0, 1.0, 1.0, 5.0, 6.0, 6.0]),
        node_y=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        face_node_connectivity=np.array([[0, 1, 2], [3, 4, 5]]),
        data={"d": np.array([1.0, 2.0])},
    )


class TestGraphFacades:
    def test_connected_components(self, split):
        assert sorted(set(split.connected_components().tolist())) == [0, 1]

    def test_exterior_faces_and_edges(self, strip):
        assert strip.exterior_faces().tolist() == [0, 1]
        assert len(strip.exterior_edges()) > 0

    def test_bounding_polygon_covers_the_mesh(self, strip):
        assert strip.bounding_polygon().area == pytest.approx(2.0)

    def test_sparse_adjacency_shape(self, strip):
        assert strip.sparse_adjacency("face").shape == (2, 2)

    def test_to_location_moves_the_variable(self, strip):
        out = strip.to_location("node")
        assert out["d"].location == "node"
        assert out["d"].data.shape == (6,)

    def test_to_location_rejects_unknown(self, strip):
        with pytest.raises(ValueError, match="location must be one of"):
            strip.to_location("bogus")

    def test_binary_dilation_grows_a_mask(self, strip):
        masked = strip.with_variable("m", np.array([True, False]), location="face")
        out = masked.binary_dilation("m")
        assert out["m"].data.tolist() == [True, True]

    def test_binary_erosion_shrinks_a_mask(self, strip):
        masked = strip.with_variable("m", np.array([True, True]), location="face")
        out = masked.binary_erosion("m")
        assert out["m"].data.tolist() == [False, False]


class TestFillFacades:
    def test_fill_na_spatial_fills_from_the_neighbour(self, strip):
        gappy = strip.with_variable("g", np.array([5.0, np.nan]), location="face")
        assert gappy.fill_na_spatial("g")["g"].data.tolist() == [5.0, 5.0]

    def test_laplace_interpolate_fills_a_hole(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
            node_y=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]),
            data={"d": np.array([0.0, np.nan, 2.0])},
        )
        filled = mesh.laplace_interpolate("d")["d"].data
        assert filled[1] == pytest.approx(1.0, abs=1e-6)


class TestVectorFacades:
    def test_polygonize_returns_a_geodataframe(self, strip):
        out = strip.polygonize("d")
        assert isinstance(out, gpd.GeoDataFrame)
        assert len(out) == 2

    def test_sample_line_crosses_both_faces(self, strip):
        distances, values, faces = strip.sample_line(
            LineString([(0.1, 0.5), (1.9, 0.5)]), "d"
        )
        assert values.tolist() == [1.0, 2.0]
        assert faces.tolist() == [0, 1]
        assert distances[0] < distances[1]

    def test_burn_adds_a_variable(self, strip):
        out = strip.burn(box(-1.0, -1.0, 1.05, 2.0), 9.0, name="b")
        assert out["b"].data[0] == 9.0
        assert "b" in out.data_variable_names


class TestRegridFacades:
    def test_regrid_onto_another_mesh(self, strip):
        target = UgridDataset.from_arrays(
            node_x=np.array([0.0, 2.0, 2.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]]),
            data={"t": np.array([0.0])},
        )
        out = strip.regrid(target, method="overlap", variable_name="d")
        # One target cell covering both sources of equal area -> their mean.
        assert out["d"].data.tolist() == pytest.approx([1.5])

    def test_interp_time_relabels_the_axis(self):
        mesh = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
            data={"d": np.array([[0.0, 0.0], [10.0, 10.0]])},
            data_locations={"d": "face"},
        )
        mesh["d"].attributes = {**mesh["d"].attributes, "time_values": [0.0, 10.0]}
        out = mesh.interp_time([5.0])
        assert out["d"].data.shape == (1, 2)
        assert out["d"].data[0].tolist() == pytest.approx([5.0, 5.0])
        assert out["d"].attributes["time_values"] == [5.0]


class TestConstructors:
    def test_from_geodataframe(self):
        gdf = gpd.GeoDataFrame(
            {"v": [1.0, 2.0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )
        mesh = UgridDataset.from_geodataframe(gdf)
        assert (mesh.n_face, mesh.epsg) == (2, 4326)
        assert mesh["v"].data.tolist() == [1.0, 2.0]

    def test_from_structured(self):
        from pyramids.dataset import Dataset, GeoReference

        raster = Dataset.from_array(
            np.arange(6.0).reshape(2, 3),
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        )
        mesh = UgridDataset.from_structured(raster)
        assert mesh.n_face == 6
        assert "band_1" in mesh.data_variable_names


class TestVariableResolution:
    def test_single_variable_needs_no_name(self, strip):
        assert strip.fill_na_spatial()["d"].data.tolist() == [1.0, 2.0]

    def test_several_variables_require_a_name(self, strip):
        two = strip.with_variable("e", np.array([3.0, 4.0]), location="face")
        with pytest.raises(ValueError, match="name the one to use"):
            two.fill_na_spatial()

    def test_unknown_name_raises_keyerror(self, strip):
        with pytest.raises(KeyError):
            strip.fill_na_spatial("nope")
