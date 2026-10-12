"""The optional-encoding and fall-back branches of the UGRID xarray bridge.

``test_interop_xarray.py`` pins the round trip and the connectivity ``start_index`` /
``_FillValue`` trap, and ``test_interop_xarray_edges.py`` pins edge connectivity, the
dataset-level CRS and the dimension-name fall-backs. What remains unexercised is the part of
:mod:`pyramids.netcdf.ugrid.interop_xarray` that only a mesh built the long way reaches, which
is what this file pins:

- the face-centre coordinates: named in the topology attributes, exported as their own variables,
  and restored by :func:`from_xarray` -- without them a reader has to recompute every centroid;
- a variable's ``standard_name``, and a metadata-only variable with no array (skipped, not
  exported as an empty one);
- the ``Mesh2d``-level CRS fall-back a bare-mesh caller relies on, including a CRS object that
  refuses to serialise (best-effort: the export drops the CRS rather than failing);
- the non-string and unparseable ``crs_wkt`` branches of both CRS readers, which must answer
  ``None`` rather than raising out of a read;
- the two skip paths of :func:`chunked_variables`, where a variable has nothing to chunk.

``xarray`` is pulled at module scope via ``pytest.importorskip`` and the module carries the
``interop`` marker, as :mod:`tests.guards.test_marker_hygiene` requires.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS as PyProjCRS

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.interop_xarray import (
    chunked_variables,
    from_xarray,
    read_crs_wkt,
    to_xarray,
)
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)

#: Face centres of the two triangles above, supplied rather than derived.
FACE_X = np.array([0.6666666666666666, 0.3333333333333333])
FACE_Y = np.array([0.3333333333333333, 0.6666666666666666])


class _UnserialisableCRS:
    """A CRS-like object whose WKT export fails, standing in for a broken/partial CRS."""

    def to_wkt(self) -> str:
        """Always fail, as a CRS carrying an unsupported projection would."""
        raise RuntimeError("this CRS cannot be expressed as WKT")


def _no_array() -> None:
    """A loader that yields no array, so the variable has a source but no data."""
    return None


def _mesh(**kwargs) -> Mesh2d:
    """The two-triangle unit square, with any extra ``Mesh2d`` keyword passed through."""
    return Mesh2d(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=Connectivity(
            data=TRIANGLES,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
        **kwargs,
    )


def _depth(**kwargs) -> MeshVariable:
    """A static face variable named ``depth``, with any extra field overridden."""
    fields = {
        "name": "depth",
        "location": "face",
        "mesh_name": "mesh2d",
        "shape": (2,),
        "dimensions": ("mesh2d_nFaces",),
        "_data": np.array([2.0, 4.0]),
    }
    fields.update(kwargs)
    return MeshVariable(**fields)


@pytest.fixture
def mesh_with_face_coords() -> Mesh2d:
    """The two-triangle mesh carrying explicit face-centre coordinates."""
    return _mesh(face_x=FACE_X, face_y=FACE_Y)


class TestFaceCoordinates:
    """Face-centre coordinates: named, exported, and read back."""

    def test_the_topology_names_the_face_coordinates(self, mesh_with_face_coords):
        """The topology variable advertises the two face-coordinate variables."""
        exported = to_xarray(mesh_with_face_coords, {})
        assert exported["mesh2d"].attrs["face_coordinates"] == (
            "mesh2d_face_x mesh2d_face_y"
        )

    def test_the_face_coordinate_variables_carry_the_centres(
        self, mesh_with_face_coords
    ):
        """The exported face-coordinate variables hold the centres, on the face dimension."""
        exported = to_xarray(mesh_with_face_coords, {})
        np.testing.assert_allclose(exported["mesh2d_face_x"].values, FACE_X)
        np.testing.assert_allclose(exported["mesh2d_face_y"].values, FACE_Y)
        assert exported["mesh2d_face_x"].dims == ("mesh2d_nFaces",)

    def test_the_face_coordinates_survive_the_round_trip(self, mesh_with_face_coords):
        """`from_xarray` restores the face centres instead of leaving them to be recomputed."""
        back, _ = from_xarray(to_xarray(mesh_with_face_coords, {}))
        assert back.has_face_coords is True
        np.testing.assert_allclose(back.face_x, FACE_X)
        np.testing.assert_allclose(back.face_y, FACE_Y)

    def test_a_mesh_without_face_coords_omits_them(self):
        """A mesh with no face centres neither names nor exports them."""
        exported = to_xarray(_mesh(), {})
        assert "face_coordinates" not in exported["mesh2d"].attrs
        assert "mesh2d_face_x" not in exported.variables


class TestVariableAttributes:
    """Per-variable attribute export and the empty-variable skip."""

    def test_a_standard_name_is_stamped_on_the_exported_variable(self):
        """A variable's CF `standard_name` is carried onto the exported DataArray."""
        exported = to_xarray(
            _mesh(), {"depth": _depth(standard_name="sea_floor_depth_below_geoid")}
        )
        assert exported["depth"].attrs["standard_name"] == "sea_floor_depth_below_geoid"

    def test_a_variable_with_no_array_is_left_out(self):
        """A metadata-only variable has nothing to write, so it is skipped entirely."""
        exported = to_xarray(
            _mesh(), {"depth": _depth(), "ghost": _depth(name="ghost", _data=None)}
        )
        assert "depth" in exported.data_vars
        assert "ghost" not in exported.variables

    def test_skipping_an_empty_variable_leaves_the_rest_readable(self):
        """The skip does not disturb the variables around it."""
        _, read_back = from_xarray(
            to_xarray(
                _mesh(), {"ghost": _depth(name="ghost", _data=None), "depth": _depth()}
            )
        )
        assert set(read_back) == {"depth"}
        assert np.asarray(read_back["depth"].data).tolist() == [2.0, 4.0]


class TestMeshLevelCrs:
    """The ``Mesh2d``-level CRS fall-back, used when the caller passes no WKT."""

    def test_a_mesh_crs_is_exported_when_the_caller_passes_none(self):
        """A bare-mesh caller gets the mesh's own CRS written to the grid-mapping variable."""
        exported = to_xarray(_mesh(crs=PyProjCRS.from_epsg(4326)), {}, crs_wkt=None)
        assert "crs" in exported.variables
        assert PyProjCRS.from_wkt(read_crs_wkt(exported)).to_epsg() == 4326

    def test_a_mesh_crs_round_trips_onto_the_rebuilt_mesh(self):
        """The reconstructed mesh carries the same CRS the exported one did."""
        back, _ = from_xarray(to_xarray(_mesh(crs=PyProjCRS.from_epsg(3857)), {}))
        assert back.crs is not None
        assert back.crs.to_epsg() == 3857

    def test_a_crs_without_a_wkt_exporter_is_stringified(self):
        """A CRS-like object with no `to_wkt` is exported through `str`."""
        exported = to_xarray(_mesh(crs="EPSG:28992"), {}, crs_wkt=None)
        assert read_crs_wkt(exported) == "EPSG:28992"

    def test_a_crs_that_cannot_serialise_is_dropped(self):
        """A failing WKT export is best-effort: the CRS variable is omitted, not an error."""
        exported = to_xarray(_mesh(crs=_UnserialisableCRS()), {}, crs_wkt=None)
        assert "crs" not in exported.variables
        assert read_crs_wkt(exported) is None

    def test_an_explicit_caller_wkt_beats_the_mesh_crs(self):
        """The caller's WKT wins over the mesh's own, since the dataset owns the CRS."""
        wanted = PyProjCRS.from_epsg(28992).to_wkt()
        exported = to_xarray(_mesh(crs=PyProjCRS.from_epsg(4326)), {}, crs_wkt=wanted)
        assert PyProjCRS.from_wkt(read_crs_wkt(exported)).to_epsg() == 28992


class TestCrsReadersTolerateBadInput:
    """Both CRS readers answer ``None`` rather than raising out of a read."""

    def _with_crs_attrs(self, attrs: dict) -> object:
        """The exported dataset with its ``crs`` variable replaced by one carrying ``attrs``."""
        exported = to_xarray(_mesh(crs=PyProjCRS.from_epsg(4326)), {})
        return exported.assign(
            crs=xr.DataArray(
                np.array([0], dtype=np.int32),
                dims=exported["crs"].dims,
                attrs=attrs,
            )
        )

    def test_a_dataset_without_a_crs_variable_reads_no_wkt(self):
        """`read_crs_wkt` answers None when the dataset declares no CRS at all."""
        assert read_crs_wkt(to_xarray(_mesh(), {}, crs_wkt=None)) is None

    def test_a_non_string_crs_attribute_reads_no_wkt(self):
        """A numeric `crs_wkt` is not a WKT, so `read_crs_wkt` answers None."""
        assert read_crs_wkt(self._with_crs_attrs({"crs_wkt": 123})) is None

    def test_a_non_string_crs_attribute_leaves_the_mesh_crs_unset(self):
        """The same numeric attribute leaves the rebuilt mesh without a CRS."""
        back, _ = from_xarray(self._with_crs_attrs({"crs_wkt": 123}))
        assert back.crs is None
        assert back.n_face == 2

    def test_an_unparseable_wkt_leaves_the_mesh_crs_unset(self):
        """A string that pyproj cannot parse is dropped, not raised out of `from_xarray`."""
        back, _ = from_xarray(self._with_crs_attrs({"crs_wkt": "not a wkt at all"}))
        assert back.crs is None
        np.testing.assert_array_equal(back.node_x, NODE_X)

    def test_an_unparseable_wkt_is_still_reported_verbatim(self):
        """`read_crs_wkt` reports any string as-is; only the parse is best-effort."""
        assert (
            read_crs_wkt(self._with_crs_attrs({"crs_wkt": "not a wkt at all"}))
            == "not a wkt at all"
        )


@pytest.mark.lazy
class TestChunkedVariablesSkips:
    """The two ``chunked_variables`` skip paths, where a variable has nothing to chunk."""

    def test_a_metadata_only_variable_is_left_out(self):
        """A variable with neither an array nor a loader cannot be chunked, so it is skipped."""
        pytest.importorskip("dask.array")
        chunked = chunked_variables(
            {"depth": _depth(), "ghost": _depth(name="ghost", _data=None)}, chunks=1
        )
        assert set(chunked) == {"depth"}

    def test_a_loader_yielding_no_array_is_left_out(self):
        """A loader that answers None also leaves the variable out of the mapping."""
        pytest.importorskip("dask.array")
        chunked = chunked_variables(
            {"empty": _depth(name="empty", _data=None, _loader=_no_array)}, chunks=1
        )
        assert chunked == {}
