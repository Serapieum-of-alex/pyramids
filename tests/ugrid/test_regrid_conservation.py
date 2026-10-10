"""The two regridder normalisations, pinned on the case that tells them apart.

`overlap` and `conservative` agree whenever the target is fully covered by the source, which is
what most fixtures look like — so a wrong normalisation hides there. These tests use a **partially
covered** target, where the two must diverge:

- `overlap` divides by the covered area, so it answers the *mean of what it covers* and does
  **not** preserve the integral;
- `conservative` divides by the *target* area, so the integral `sum(value * area)` survives.

Dividing by the *source* area (one reading of the task brief) would preserve neither, which is why
this is pinned rather than left to a fixture that cannot see the difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Regridder, UgridDataset


def _quad(node_x: np.ndarray, node_y: np.ndarray, value: float) -> UgridDataset:
    """A single quad face carrying `value`."""
    return UgridDataset.from_arrays(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=np.array([[0, 1, 2, 3]]),
        data={"v": np.array([value])},
    )


@pytest.fixture
def half_covered() -> tuple[UgridDataset, UgridDataset]:
    """A unit source quad covering exactly half of a 2x1 target quad."""
    source = _quad(np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]), 4.0)
    target = _quad(np.array([0.0, 2.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]), 0.0)
    return source, target


def _integral(values: np.ndarray, mesh) -> float:
    """The area integral `sum(value * area)` of a field on a mesh."""
    return float((np.asarray(values) * np.asarray(mesh.face_areas)).sum())


class TestConservativeConservesTheIntegral:
    def test_partially_covered_target_preserves_the_integral(self, half_covered):
        source, target = half_covered
        values = np.array([4.0])
        out = Regridder(source.mesh, target.mesh, method="conservative").regrid(values)
        assert _integral(out, target.mesh) == pytest.approx(
            _integral(values, source.mesh)
        )

    def test_the_weight_is_overlap_over_target_area(self, half_covered):
        source, target = half_covered
        weights = Regridder(
            source.mesh, target.mesh, method="conservative"
        ).weights.toarray()
        # overlap 1.0 / target area 2.0
        assert weights.shape == (1, 1)
        assert weights[0].tolist() == pytest.approx([0.5])

    def test_refinement_preserves_the_integral(self):
        source = _quad(
            np.array([0.0, 2.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]), 3.0
        )
        target = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
            data={"t": np.array([0.0, 0.0])},
        )
        values = np.array([3.0])
        out = Regridder(source.mesh, target.mesh, method="conservative").regrid(values)
        assert _integral(out, target.mesh) == pytest.approx(
            _integral(values, source.mesh)
        )


class TestOverlapAveragesInstead:
    def test_partially_covered_target_answers_the_covered_mean(self, half_covered):
        source, target = half_covered
        out = Regridder(source.mesh, target.mesh, method="overlap").regrid(
            np.array([4.0])
        )
        assert out.tolist() == pytest.approx([4.0])

    def test_overlap_does_not_preserve_the_integral(self, half_covered):
        source, target = half_covered
        values = np.array([4.0])
        out = Regridder(source.mesh, target.mesh, method="overlap").regrid(values)
        # Deliberately different from conservative: an average, not an integral.
        assert _integral(out, target.mesh) != pytest.approx(
            _integral(values, source.mesh)
        )

    def test_overlap_rows_sum_to_one(self, half_covered):
        source, target = half_covered
        weights = Regridder(
            source.mesh, target.mesh, method="overlap"
        ).weights.toarray()
        assert weights.sum(axis=1).tolist() == pytest.approx([1.0])

    def test_overlap_reproduces_a_constant_field(self):
        source = UgridDataset.from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
            data={"v": np.array([7.0, 7.0])},
        )
        target = _quad(
            np.array([0.0, 2.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]), 0.0
        )
        out = Regridder(source.mesh, target.mesh, method="overlap").regrid(
            np.array([7.0, 7.0])
        )
        assert out.tolist() == pytest.approx([7.0])
