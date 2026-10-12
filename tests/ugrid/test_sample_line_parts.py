"""`sample_line` must answer one sample per crossing, at that crossing's own midpoint.

The implementation projected the *whole* intersection's centroid, which breaks three ways: a line
re-entering a face collapses to one sample placed at a point that is not even on the line, a
non-convex face's sample lands inside its own notch, and a line touching a shared corner produces
two spurious zero-length samples. One sample per 1-D part, each at that part's own midpoint, is
what the docstring promises.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.vector_ops import sample_line

VALUES = np.array([10.0, 20.0, 30.0, 40.0])


@pytest.fixture
def strip() -> Mesh2d:
    """Four unit quads in a row along x, spanning [0, 4] x [0, 1]."""
    node_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0] * 2)
    node_y = np.array([0.0] * 5 + [1.0] * 5)
    faces = np.array([[i, i + 1, i + 6, i + 5] for i in range(4)])
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


class TestOneSamplePerCrossing:
    def test_a_re_entry_is_sampled_twice(self, strip):
        zigzag = LineString([(0.25, 0.25), (1.5, 0.25), (1.5, 0.75), (0.25, 0.75)])
        _, _, faces = sample_line(strip, VALUES, zigzag)
        assert faces.tolist() == [0, 1, 0]

    def test_each_crossing_sits_at_its_own_midpoint(self, strip):
        zigzag = LineString([(0.25, 0.25), (1.5, 0.25), (1.5, 0.75), (0.25, 0.75)])
        distances, _, _ = sample_line(strip, VALUES, zigzag)
        assert distances.tolist() == pytest.approx([0.375, 1.5, 2.625])

    def test_the_values_follow_the_faces(self, strip):
        zigzag = LineString([(0.25, 0.25), (1.5, 0.25), (1.5, 0.75), (0.25, 0.75)])
        _, values, _ = sample_line(strip, VALUES, zigzag)
        assert values.tolist() == [10.0, 20.0, 10.0]


class TestCornerTouchIsNotASample:
    def test_a_corner_touch_yields_no_spurious_sample(self, strip):
        # The endpoint is the node shared by faces 0 and 1; a zero-length touch is not a crossing.
        _, _, faces = sample_line(strip, VALUES, LineString([(0.0, 0.0), (1.0, 1.0)]))
        assert faces.tolist() == [0]

    def test_distances_are_unique_for_that_line(self, strip):
        distances, _, _ = sample_line(
            strip, VALUES, LineString([(0.0, 0.0), (1.0, 1.0)])
        )
        assert len(set(distances.tolist())) == len(distances)


class TestUnchangedCases:
    def test_a_straight_crossing_samples_each_face_once(self, strip):
        distances, values, faces = sample_line(
            strip, VALUES, LineString([(0.5, 0.5), (3.5, 0.5)])
        )
        assert faces.tolist() == [0, 1, 2, 3]
        assert values.tolist() == [10.0, 20.0, 30.0, 40.0]
        assert distances.tolist() == pytest.approx([0.25, 1.0, 2.0, 2.75])

    def test_a_line_off_the_mesh_answers_empty(self, strip):
        distances, values, faces = sample_line(
            strip, VALUES, LineString([(10.0, 10.0), (11.0, 11.0)])
        )
        assert distances.shape == (0,)
        assert values.shape == (0,)
        assert faces.shape == (0,)


class TestFaceIndicesAreNotUnique:
    """The documented consequence of sampling per crossing rather than per face."""

    def test_a_re_entering_line_repeats_a_face(self):
        """A face the line leaves and re-enters appears once per crossing.

        Test scenario:
            The docstring said "each crossed face contributes one sample", which the per-part
            fix made false without updating it. On a 2x2 block of unit quads with a line that
            returns to face 0 three times, face 0 is sampled four times -- so a caller treating
            `face_indices` as unique, or joining on it, gets four rows where the prose promised
            one.
        """
        node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        mesh = UgridDataset.from_arrays(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
            ),
            data={"v": np.array([1.0, 2.0, 3.0, 4.0])},
            data_locations={"v": "face"},
        )
        line = LineString(
            [(0.25, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.25), (0.75, 0.25)]
        )

        distances, values, face_indices = mesh.sample_line(line)
        faces = np.asarray(face_indices).tolist()

        assert len(faces) > len(set(faces)), (
            f"the line re-enters a face, so the indices repeat; got {faces}"
        )
        assert faces.count(0) == 4, (
            f"face 0 is crossed four times, got {faces.count(0)}"
        )
        assert len(distances) == len(faces) == np.asarray(values).shape[-1], (
            "one distance and one value per crossing"
        )
        assert np.all(np.diff(np.asarray(distances)) >= 0), (
            f"distances must stay ascending, got {np.asarray(distances).tolist()}"
        )
