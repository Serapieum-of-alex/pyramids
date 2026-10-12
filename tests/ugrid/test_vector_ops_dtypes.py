"""Value dtypes through `burn` and `polygonize`.

`burn` allocated its output from `fill` alone, so an integer `fill` with a fractional `value`
silently truncated the value. `polygonize` coerced every value array to float to find the NaNs,
which rejected the classified / label field its own docstring recommends it for.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from pyramids.netcdf.ugrid import UgridDataset, vector_ops
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


@pytest.fixture
def mesh():
    """A unit square split into two triangles."""
    return UgridDataset.from_arrays(
        node_x=NODE_X,
        node_y=NODE_Y,
        face_node_connectivity=TRIANGLES,
        data={"v": np.array([1.0, 2.0])},
    ).mesh


@pytest.fixture
def covering() -> shapely.Geometry:
    """A box covering the whole mesh."""
    return shapely.box(0.0, 0.0, 1.0, 1.0)


class TestBurnDtype:
    def test_a_fractional_value_survives_an_integer_fill(self, mesh, covering):
        burned = vector_ops.burn(mesh, covering, 1.5, fill=0)
        assert burned.tolist() == [1.5, 1.5]

    def test_the_output_is_promoted_to_float(self, mesh, covering):
        burned = vector_ops.burn(mesh, covering, 1.5, fill=0)
        assert burned.dtype == np.float64

    def test_an_integer_value_and_fill_stay_integer(self, mesh, covering):
        burned = vector_ops.burn(mesh, covering, 3, fill=0)
        assert burned.dtype.kind == "i"
        assert burned.tolist() == [3, 3]

    def test_the_default_nan_fill_is_unchanged(self, mesh, covering):
        burned = vector_ops.burn(mesh, covering, 1.5)
        assert burned.tolist() == [1.5, 1.5]

    def test_an_untouched_face_keeps_the_fill(self, mesh):
        # Strictly inside the lower-right triangle, clear of the shared diagonal.
        inside_one = shapely.box(0.6, 0.1, 0.9, 0.3)
        burned = vector_ops.burn(mesh, inside_one, 1.5, fill=0)
        assert burned.tolist() == [1.5, 0.0]


class TestPolygonizeLabels:
    def test_a_string_label_field_is_accepted(self, mesh):
        result = vector_ops.polygonize(mesh, np.array(["a", "b"], dtype=object))
        assert len(result) == 2

    def test_equal_string_labels_dissolve_together(self, mesh):
        result = vector_ops.polygonize(mesh, np.array(["a", "a"], dtype=object))
        assert len(result) == 1

    def test_the_label_is_kept_as_the_value(self, mesh):
        result = vector_ops.polygonize(mesh, np.array(["land", "sea"], dtype=object))
        assert sorted(result["value"].tolist()) == ["land", "sea"]

    def test_a_none_label_is_dropped(self, mesh):
        result = vector_ops.polygonize(mesh, np.array(["land", None], dtype=object))
        assert result["value"].tolist() == ["land"]

    def test_numeric_fields_still_drop_nan(self, mesh):
        result = vector_ops.polygonize(mesh, np.array([1.0, np.nan]))
        assert result["value"].tolist() == [1.0]

    def test_numeric_rounding_still_applies(self, mesh):
        result = vector_ops.polygonize(mesh, np.array([1.001, 1.002]), decimals=1)
        assert len(result) == 1


class TestInvalidFace:
    def test_sample_line_names_the_invalid_face(self):
        # A self-intersecting quad is not a valid polygon; GEOS answers with a bare "side
        # location conflict" that names no face, so the mesh-level error has to.
        invalid = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.0, 1.0]),
            node_y=np.array([0.0, 1.0, 1.0, 0.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2, 3]]),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        line = LineString([(-1.0, 0.5), (2.0, 0.5)])
        with pytest.raises(ValueError, match="face 0 is not a valid polygon"):
            vector_ops.sample_line(invalid, np.array([10.0]), line)
