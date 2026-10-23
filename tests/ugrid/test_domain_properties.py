"""Tests for UgridDataset.domain_area and count_domain_cells."""

from __future__ import annotations

import numpy as np
import pytest

from tests.ugrid.conftest import make_two_triangle_ds as _make_ds

pytestmark = pytest.mark.core


class TestDomainArea:
    def test_positive(self):
        ds = _make_ds()
        assert ds.domain_area() > 0.0

    def test_matches_face_areas_sum(self):
        ds = _make_ds()
        expected = float(ds.mesh.face_areas.sum())
        assert ds.domain_area() == pytest.approx(expected)


class TestCountDomainCells:
    def test_counts_all_faces(self):
        ds = _make_ds()
        assert ds.count_domain_cells() == 2

    def test_counts_active_faces(self):
        ds = _make_ds()
        ds["depth"].active = np.array([True, False])
        assert ds.count_domain_cells("depth") == 1

    def test_no_mask_counts_all(self):
        ds = _make_ds()
        assert ds.count_domain_cells("depth") == 2
