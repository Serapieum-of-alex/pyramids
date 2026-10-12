"""Tests for the standalone mesh regridding / sampling module (``ugrid.regrid``).

Covers the four B-epic capabilities implemented purely in numpy/scipy/shapely:

- :func:`sample` (B2) — point-value sampling, incl. exact barycentric recovery of a plane.
- :func:`rasterize_like` (B1) — snap mesh->raster interpolation onto an existing grid.
- :func:`interp_time` (B4) — temporal interpolation onto new time labels.
- :class:`Regridder` (B3) — overlap / conservative / centroid / barycentric mesh->mesh regridding.

All fixtures are small meshes built with :meth:`UgridDataset.from_arrays`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.regrid import Regridder, interp_time, rasterize_like, sample

pytestmark = pytest.mark.core


def _two_triangle_dataset(node_values=None, face_values=None, epsg=4326):
    """Unit square split into two triangles, optionally carrying node and/or face data.

    Triangle 0 = (0,0),(1,0),(1,1); triangle 1 = (0,0),(1,1),(0,1). The diagonal is ``y = x``.
    """
    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)
    data = {}
    locations = {}
    if node_values is not None:
        data["z"] = np.asarray(node_values, dtype=np.float64)
        locations["z"] = "node"
    if face_values is not None:
        data["f"] = np.asarray(face_values, dtype=np.float64)
        locations["f"] = "face"
    return UgridDataset.from_arrays(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=faces,
        data=data or None,
        data_locations=locations or None,
        epsg=epsg,
    )


def _grid_2x2(face_values=None, epsg=4326):
    """2x2 grid of unit quad faces over ``[0, 2] x [0, 2]`` (4 faces, each area 1)."""
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    faces = np.array(
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.intp
    )
    data = (
        {"v": np.asarray(face_values, dtype=np.float64)}
        if face_values is not None
        else None
    )
    locations = {"v": "face"} if face_values is not None else None
    return UgridDataset.from_arrays(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=faces,
        data=data,
        data_locations=locations,
        epsg=epsg,
    )


def _single_quad_mesh(x0, y0, x1, y1):
    """A one-face mesh: the axis-aligned quad with corners ``(x0, y0)`` and ``(x1, y1)``."""
    node_x = np.array([x0, x1, x1, x0])
    node_y = np.array([y0, y0, y1, y1])
    faces = np.array([[0, 1, 2, 3]], dtype=np.intp)
    return UgridDataset.from_arrays(
        node_x=node_x, node_y=node_y, face_node_connectivity=faces
    ).mesh


class TestSample:
    """Point-value sampling (B2)."""

    def test_barycentric_recovers_plane(self):
        """Barycentric-linear sampling of triangular node data reproduces a plane to 1e-10."""
        node_x = np.array([0.0, 1.0, 1.0, 0.0])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        a, b, c = 2.0, 3.0, 1.0
        z = a * node_x + b * node_y + c
        ds = _two_triangle_dataset(node_values=z)

        xq = np.array([0.5, 0.8, 0.25, 0.3])
        yq = np.array([0.25, 0.3, 0.6, 0.7])
        got = sample(ds.mesh, ds["z"].data, xq, yq, method="linear", location="node")

        expected = a * xq + b * yq + c
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_nearest_face_contains(self):
        """`contains` on face data returns the containing face value; outside -> NaN."""
        ds = _two_triangle_dataset(face_values=[10.0, 20.0])
        xq = np.array([0.5, 0.25, 5.0])
        yq = np.array([0.25, 0.6, 5.0])
        got = sample(ds.mesh, ds["f"].data, xq, yq, method="contains", location="face")
        assert got[0] == 10.0
        assert got[1] == 20.0
        assert np.isnan(got[2])

    def test_nearest_node(self):
        """`nearest` on node data returns the value of the closest node."""
        ds = _two_triangle_dataset(node_values=[1.0, 3.0, 6.0, 4.0])
        got = sample(
            ds.mesh,
            ds["z"].data,
            np.array([0.05]),
            np.array([0.05]),
            method="nearest",
            location="node",
        )
        assert got[0] == 1.0

    def test_outside_point_is_nodata(self):
        """A point outside every face gets the nodata fill, never a silent zero."""
        ds = _two_triangle_dataset(face_values=[10.0, 20.0])
        got = sample(
            ds.mesh,
            ds["f"].data,
            np.array([-9.0]),
            np.array([-9.0]),
            method="contains",
            location="face",
        )
        assert np.isnan(got[0])

    def test_temporal_keeps_time_axis(self):
        """A 2-D (n_time, n_face) variable keeps its leading time axis in the result."""
        ds = _two_triangle_dataset(face_values=[0.0, 0.0])
        values = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        xq = np.array([0.5, 0.25])
        yq = np.array([0.25, 0.6])
        got = sample(ds.mesh, values, xq, yq, method="contains", location="face")
        assert got.shape == (3, 2)
        np.testing.assert_array_equal(got[:, 0], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(got[:, 1], [20.0, 21.0, 22.0])

    def test_input_not_mutated(self):
        """Sampling never writes back into the caller's data array."""
        ds = _two_triangle_dataset(node_values=[1.0, 3.0, 6.0, 4.0])
        values = ds["z"].data
        before = values.copy()
        sample(
            ds.mesh,
            values,
            np.array([0.5]),
            np.array([0.25]),
            method="linear",
            location="node",
        )
        np.testing.assert_array_equal(values, before)


class TestRasterizeLike:
    """Snap mesh->raster interpolation onto an existing grid (B1)."""

    def test_co_registered_with_reference(self):
        """Output matches the reference grid's geotransform, shape, CRS and values."""
        ds = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        reference = ds.to_dataset("v", cell_size=0.5, method="nearest")
        out = rasterize_like(
            ds.mesh, ds["v"].data, reference, location="face", method="nearest"
        )

        assert tuple(out.geotransform) == tuple(reference.geotransform)
        assert out.shape == reference.shape
        assert out.epsg == reference.epsg
        np.testing.assert_array_equal(out.read_array(), reference.read_array())

    def test_nodata_honored(self):
        """With a tiny max_distance every cell falls outside and is filled with nodata."""
        ds = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        reference = ds.to_dataset("v", cell_size=0.5, method="nearest")
        out = rasterize_like(
            ds.mesh,
            ds["v"].data,
            reference,
            location="face",
            method="nearest",
            nodata=-9999.0,
            max_distance=0.01,
        )
        np.testing.assert_array_equal(
            out.read_array(), np.full(reference.read_array().shape, -9999.0)
        )

    def test_idw_method_runs(self):
        """The optional IDW method produces a co-registered raster with finite interior values."""
        ds = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        reference = ds.to_dataset("v", cell_size=0.5, method="nearest")
        out = rasterize_like(
            ds.mesh, ds["v"].data, reference, location="face", method="idw"
        )
        assert tuple(out.geotransform) == tuple(reference.geotransform)
        arr = out.read_array()
        finite = arr[arr != -9999.0]
        assert finite.size > 0
        assert np.all(np.isfinite(finite))

    def test_temporal_uses_first_step(self):
        """A temporal variable rasterizes its first time step (the to_dataset convention)."""
        ds = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        reference = ds.to_dataset("v", cell_size=0.5, method="nearest")
        temporal = np.stack([ds["v"].data, ds["v"].data + 100.0])
        out = rasterize_like(
            ds.mesh, temporal, reference, location="face", method="nearest"
        )
        np.testing.assert_array_equal(out.read_array(), reference.read_array())

    def test_same_epsg_reproject_is_noop(self):
        """Passing source_epsg equal to the reference EPSG changes nothing (reproject no-op)."""
        ds = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        reference = ds.to_dataset("v", cell_size=0.5, method="nearest")
        out = rasterize_like(
            ds.mesh,
            ds["v"].data,
            reference,
            location="face",
            method="nearest",
            source_epsg=reference.epsg,
        )
        np.testing.assert_array_equal(out.read_array(), reference.read_array())


class TestInterpTime:
    """Temporal interpolation onto new time coordinates (B4)."""

    def test_linear_midpoint_exact(self):
        """A field linear in time is reproduced exactly at interior labels."""
        times = np.array([0.0, 1.0, 2.0])
        base = np.array([10.0, 20.0])
        slope = np.array([1.0, -2.0])
        values = np.stack([base + k * slope for k in times])  # (3, 2)
        got = interp_time(times, values, np.array([0.5, 1.5]), method="linear")
        expected = np.stack([base + k * slope for k in (0.5, 1.5)])
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_nearest_picks_closer_step(self):
        """`nearest` snaps each label to the closer source step."""
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 20.0])
        got = interp_time(times, values, np.array([0.4, 0.6]), method="nearest")
        np.testing.assert_array_equal(got, [0.0, 10.0])

    def test_bounds_nan(self):
        """Out-of-range labels become NaN under bounds='nan'."""
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([5.0, 6.0, 7.0])
        got = interp_time(
            times, values, np.array([-1.0, 3.0]), method="linear", bounds="nan"
        )
        assert np.all(np.isnan(got))

    def test_bounds_clamp(self):
        """Out-of-range labels clamp to the first/last step under bounds='clamp'."""
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([5.0, 6.0, 7.0])
        got = interp_time(
            times, values, np.array([-1.0, 3.0]), method="linear", bounds="clamp"
        )
        np.testing.assert_array_equal(got, [5.0, 7.0])

    def test_trailing_time_axis(self):
        """`axis` selects the time axis for a variable storing time last."""
        times = np.array([0.0, 2.0])
        values = np.array([[0.0, 10.0], [1.0, 11.0]])  # (n_elem, n_time)
        got = interp_time(times, values, np.array([1.0]), method="linear", axis=1)
        np.testing.assert_allclose(got[:, 0], [5.0, 6.0])

    def test_duplicate_times_raise(self):
        """Repeated time labels are rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            interp_time(np.array([0.0, 0.0, 1.0]), np.zeros((3, 2)), np.array([0.5]))


class TestRegridder:
    """Mesh->mesh regridders (B3)."""

    def test_overlap_mean_of_constant_is_constant(self):
        """Overlap weights sum to 1 per target, so an overlap-mean of a constant is that constant."""
        source = _grid_2x2(face_values=[5.0, 5.0, 5.0, 5.0])
        target = _single_quad_mesh(0.0, 0.0, 2.0, 2.0)
        regridder = Regridder(source.mesh, target, method="overlap")
        row_sums = np.asarray(regridder.weights.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
        got = regridder.regrid(source["v"].data)
        np.testing.assert_allclose(got, [5.0], atol=1e-12)

    def test_conservative_conserves_integral(self):
        """Conservative regridding conserves sum(value * area) on a refinement pair to 1e-8."""
        source = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        target = _single_quad_mesh(0.0, 0.0, 2.0, 2.0)
        regridder = Regridder(source.mesh, target, method="conservative")
        got = regridder.regrid(source["v"].data)

        source_integral = float(np.sum(source["v"].data * source.mesh.face_areas))
        target_integral = float(np.sum(got * target.face_areas))
        assert abs(source_integral - target_integral) < 1e-8
        np.testing.assert_allclose(got, [2.5], atol=1e-10)

    def test_overlap_outside_target_is_nan(self):
        """A target face with no source overlap stays NaN."""
        source = _grid_2x2(face_values=[5.0, 5.0, 5.0, 5.0])
        node_x = np.array([0.0, 2.0, 2.0, 0.0, 10.0, 11.0, 11.0, 10.0])
        node_y = np.array([0.0, 0.0, 2.0, 2.0, 10.0, 10.0, 11.0, 11.0])
        faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.intp)
        target = UgridDataset.from_arrays(
            node_x=node_x, node_y=node_y, face_node_connectivity=faces
        ).mesh
        got = Regridder(source.mesh, target, method="overlap").regrid(source["v"].data)
        assert got[0] == pytest.approx(5.0)
        assert np.isnan(got[1])

    def test_centroid_picks_nearest(self):
        """CentroidLocator copies the value of the source face nearest the target centroid."""
        source = _grid_2x2(face_values=[10.0, 20.0, 30.0, 40.0])
        target = _single_quad_mesh(
            0.25, 0.25, 0.75, 0.75
        )  # centroid (0.5, 0.5) -> source face 0
        got = Regridder(source.mesh, target, method="centroid").regrid(source["v"].data)
        np.testing.assert_array_equal(got, [10.0])

    def test_barycentric_recovers_linear_field(self):
        """Barycentric regridding of a field linear in centroid coordinates is exact."""
        scx = np.array([0.5, 1.5, 0.5, 1.5])
        scy = np.array([0.5, 0.5, 1.5, 1.5])
        values = 2.0 * scx + 3.0 * scy
        source = _grid_2x2(face_values=values)
        target = _single_quad_mesh(
            0.75, 0.75, 1.25, 1.25
        )  # centroid (1.0, 1.0), inside the hull
        got = Regridder(source.mesh, target, method="barycentric").regrid(
            source["v"].data
        )
        np.testing.assert_allclose(got, [2.0 * 1.0 + 3.0 * 1.0], atol=1e-10)

    def test_time_broadcasting(self):
        """A (n_time, n_source) stack is regridded slab-by-slab, preserving the time axis."""
        source = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        target = _single_quad_mesh(0.0, 0.0, 2.0, 2.0)
        regridder = Regridder(source.mesh, target, method="overlap")
        stack = np.stack(
            [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])]
        )
        got = regridder.regrid(stack)
        assert got.shape == (2, 1)
        np.testing.assert_allclose(got[:, 0], [2.5, 6.5], atol=1e-12)

    def test_weights_reuse_round_trip(self):
        """from_weights rebuilds an equivalent regridder from a stored weight matrix."""
        source = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        target = _single_quad_mesh(0.0, 0.0, 2.0, 2.0)
        regridder = Regridder(source.mesh, target, method="conservative")
        reused = Regridder.from_weights(regridder.weights, target)
        np.testing.assert_allclose(
            reused.regrid(source["v"].data),
            regridder.regrid(source["v"].data),
            atol=1e-12,
        )

    def test_unknown_method_raises(self):
        """An unknown method is rejected at construction."""
        source = _grid_2x2(face_values=[1.0, 2.0, 3.0, 4.0])
        target = _single_quad_mesh(0.0, 0.0, 2.0, 2.0)
        with pytest.raises(ValueError, match="method must be one of"):
            Regridder(source.mesh, target, method="bogus")
