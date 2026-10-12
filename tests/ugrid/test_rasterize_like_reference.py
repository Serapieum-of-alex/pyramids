"""`rasterize_like` against a reference grid that is CRS-less or anisotropic.

Two defaults are derived from the reference raster: whether the mesh points are reprojected,
and how far a nearest/IDW sample may reach. Both were read off one half of the reference --
the EPSG only when the reference had one, and the cell size only along the row direction --
so a reference with no CRS, or with very different spacing along its two axes, took a
silently different path from the one the caller asked for.
"""

import numpy as np
import pytest
from pyproj import CRS

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.regrid import rasterize_like

pytestmark = pytest.mark.core


def _connectivity(data, cf_role):
    """A connectivity over `data` tagged with `cf_role`, zero-based and `-1`-padded."""
    return Connectivity(
        data=np.asarray(data, dtype="int64"),
        fill_value=-1,
        cf_role=cf_role,
        original_start_index=0,
    )


@pytest.fixture
def mesh():
    """A unit square split into two triangles."""
    return Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=_connectivity(
            [[0, 1, 2], [0, 2, 3]], "face_node_connectivity"
        ),
    )


def _reference(geo, epsg, shape=(3, 3)):
    """A raster carrying `geo` and `epsg`, with no data of its own."""
    return Dataset.from_array(
        np.zeros(shape, dtype="float64"),
        no_data_value=np.nan,
        geo_ref=GeoReference(geo=geo, epsg=epsg),
    )


class TestReferenceWithoutACRS:
    """A reference raster that declares no EPSG."""

    def test_an_explicit_source_epsg_is_refused(self, mesh):
        """Asking to reproject into a CRS-less grid is an error, not a no-op.

        Test scenario:
            The reprojection ran only when *both* EPSGs were known, so passing
            `source_epsg=4326` against a reference with `epsg is None` sampled the mesh's raw
            coordinates and stamped the result as co-registered -- a silent georeferencing
            error in the one call whose whole purpose is co-registration.
        """
        reference = _reference((0.0, 0.5, 0.0, 1.0, 0.0, -0.5), None)

        with pytest.raises(ValueError, match="no CRS") as exc:
            rasterize_like(mesh, np.array([1.0, 2.0]), reference, source_epsg=4326)

        assert "source_epsg" in str(exc.value), (
            f"the message should name the argument that cannot be honoured: {exc.value}"
        )

    def test_no_source_epsg_is_still_fine(self, mesh):
        """Without `source_epsg` the two are assumed to share a CRS, as documented.

        Test scenario:
            The guard must only fire when the caller actually asked for a reprojection; a
            CRS-less reference is a perfectly ordinary input otherwise.
        """
        reference = _reference((0.0, 0.5, 0.0, 1.0, 0.0, -0.5), None)

        out = rasterize_like(mesh, np.array([1.0, 2.0]), reference)

        assert out.shape == reference.shape, (
            f"expected the reference shape {reference.shape}, got {out.shape}"
        )


class TestAnisotropicReference:
    """A reference whose two axes have very different cell lengths."""

    def test_the_default_reach_covers_the_longer_axis(self, mesh):
        """A tall-thin cell does not cut the default cutoff below one row step.

        Test scenario:
            The default was `2 * hypot(geo[1], geo[4])` -- the row direction alone. On a grid
            stepping 1 along a row and 10 down a column that is 2.0, which is a fifth of a
            single row step, so every cell below the first row fell outside the cutoff and came
            back as no-data even though the mesh sits right there. Taking the longer of the two
            cell edges keeps the reach at two steps in both directions.
        """
        reference = _reference((0.0, 1.0, 0.0, 20.0, 0.0, -10.0), 4326, shape=(3, 3))

        out = rasterize_like(mesh, np.array([1.0, 2.0]), reference, nodata=-9999.0)
        filled = int(np.count_nonzero(out.read_array() != -9999.0))

        assert filled > 3, (
            f"only {filled} of 9 cells were sampled; the row-direction-only cutoff left "
            f"everything past the first row as no-data"
        )

    def test_an_explicit_max_distance_still_wins(self, mesh):
        """The caller's own cutoff is used unchanged.

        Test scenario:
            The new default must not override an explicit `max_distance`; a tiny one still
            clips everything, which is what pins that the default is a default.
        """
        reference = _reference((0.0, 1.0, 0.0, 20.0, 0.0, -10.0), 4326, shape=(3, 3))

        out = rasterize_like(
            mesh, np.array([1.0, 2.0]), reference, nodata=-9999.0, max_distance=1e-9
        )
        filled = int(np.count_nonzero(out.read_array() != -9999.0))

        assert filled == 0, (
            f"a 1e-9 cutoff should reach nothing, yet filled {filled} cells"
        )


class TestReferenceWithACrsButNoEpsgCode:
    """A reference that is georeferenced but whose CRS carries no EPSG authority code."""

    @pytest.fixture
    def geostationary(self):
        """A reference in a geostationary fixed-grid projection, which has no EPSG code."""
        crs = CRS.from_proj4(
            "+proj=geos +h=35785831 +lon_0=0 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
        )
        assert crs.to_epsg() is None, "the fixture must be a CRS with no EPSG code"
        reference = Dataset.from_array(
            np.zeros((3, 3), dtype="float64"),
            no_data_value=np.nan,
            geo_ref=GeoReference(
                geo=(0.0, 1000.0, 0.0, 3000.0, 0.0, -1000.0), epsg=4326
            ),
        )
        reference.crs = crs.to_wkt()
        assert reference.epsg is None, (
            "the fixture's CRS must have no EPSG code, which is what `epsg` reports as None"
        )
        assert reference.crs, "the fixture must still carry that CRS as WKT"
        return reference

    def test_it_is_not_refused_as_having_no_crs(self, mesh, geostationary):
        """`Dataset.epsg` answers None for two different reasons, and only one is an error.

        Test scenario:
            `Dataset.epsg`'s own docstring says it returns None both for a raster with no CRS
            *and* for one whose CRS has no EPSG authority code, and tells the caller to read
            `.crs` to tell them apart. The guard tested `epsg`, so a properly georeferenced
            non-EPSG reference got a hard failure whose message stated the opposite of the
            truth -- and whose advice, "Georeference the reference raster", could not be
            followed. The facade always forwards `source_epsg=self.epsg`, so this was
            reachable from the public API.
        """
        out = rasterize_like(
            mesh, np.array([1.0, 2.0]), geostationary, source_epsg=4326
        )

        assert out.shape == geostationary.shape, (
            f"expected the reference shape {geostationary.shape}, got {out.shape}"
        )

    def test_the_mesh_is_reprojected_into_it(self, mesh, geostationary):
        """The non-EPSG reference is a reprojection target, not merely tolerated.

        Test scenario:
            The guard existed so a `source_epsg` is never silently ignored. Honouring a
            non-EPSG reference means actually transforming into its CRS -- a mesh in degrees
            lands at geostationary metres, far outside the degree-sized cells, so with the
            default cutoff nothing is sampled. Sampling the raw degrees instead would have
            filled cells, which is the silent error the guard is for.
        """
        out = rasterize_like(
            mesh, np.array([1.0, 2.0]), geostationary, source_epsg=4326, nodata=-9999.0
        )
        filled = int(np.count_nonzero(out.read_array() != -9999.0))

        assert filled == 0, (
            f"the reprojected mesh is nowhere near this grid, so no cell should be sampled; "
            f"{filled} filled means the mesh was used unprojected"
        )

    def test_a_reference_with_no_crs_at_all_is_still_refused(self, mesh):
        """The guard must keep firing for the case it was written for."""
        reference = _reference((0.0, 0.5, 0.0, 1.0, 0.0, -0.5), None)

        with pytest.raises(ValueError, match="no CRS"):
            rasterize_like(mesh, np.array([1.0, 2.0]), reference, source_epsg=4326)


class TestAnisotropicDefaultReachIsBounded:
    """The documented bound on the default `max_distance`."""

    def test_the_reach_is_two_steps_of_the_longer_axis(self, mesh):
        """The default is `2 * max(cell edges)`, which the docstring must state exactly.

        Test scenario:
            A single isotropic radius cannot be "two steps along either axis" on an
            anisotropic grid: covering two steps of the long axis necessarily reaches many
            steps along the short one. The old default (the column step alone) starved -- 0 of
            9 cells on a 1-by-10 reference -- and `max(...)` is what fixes that, so the
            trade-off is real and is pinned here in both directions rather than left to be
            rediscovered.
        """
        starved = _reference((0.0, 1.0, 0.0, 20.0, 0.0, -10.0), 4326, shape=(3, 3))
        filled = int(
            np.count_nonzero(
                rasterize_like(
                    mesh, np.array([1.0, 2.0]), starved, nodata=-9999.0
                ).read_array()
                != -9999.0
            )
        )

        assert filled == 9, (
            f"the tall-cell reference must be covered, got {filled} of 9 -- this is the "
            f"starvation the default was changed to fix"
        )

    def test_an_explicit_cutoff_bounds_the_over_reach(self, mesh):
        """A caller on a strongly anisotropic grid can and must set `max_distance`.

        Test scenario:
            With the default, a mesh spanning one unit populates cells very far along the
            short axis. That is the documented cost of one isotropic radius, and the remedy is
            an explicit cutoff -- so the remedy is tested.
        """
        wide = _reference((0.0, 1.0, 0.0, 1000.0, 0.0, -1000.0), 4326, shape=(1, 40))

        default_filled = int(
            np.count_nonzero(
                rasterize_like(
                    mesh, np.array([1.0, 2.0]), wide, nodata=-9999.0
                ).read_array()
                != -9999.0
            )
        )
        bounded_filled = int(
            np.count_nonzero(
                rasterize_like(
                    mesh, np.array([1.0, 2.0]), wide, nodata=-9999.0, max_distance=2.0
                ).read_array()
                != -9999.0
            )
        )

        assert default_filled > bounded_filled, (
            f"the default reaches further than an explicit 2.0 cutoff "
            f"({default_filled} vs {bounded_filled} cells)"
        )
        assert bounded_filled <= 3, (
            f"a 2.0 cutoff around a 1-unit mesh should fill only the nearby cells, got "
            f"{bounded_filled}"
        )
