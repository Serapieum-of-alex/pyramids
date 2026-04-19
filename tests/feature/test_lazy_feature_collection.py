"""DASK-40b: LazyFeatureCollection class."""

from __future__ import annotations

import pickle

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base.protocols import SpatialObject, is_lazy
from pyramids.feature import FeatureCollection


try:
    import dask_geopandas  # noqa: F401
    HAS_DASK_GP = True
except ImportError:  # pragma: no cover
    HAS_DASK_GP = False


requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


@pytest.fixture
def small_gdf():
    return gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["a"] * 5 + ["b"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def lfc(small_gdf):
    dg = pytest.importorskip("dask_geopandas")
    from pyramids.feature import LazyFeatureCollection

    ddf = dg.from_geopandas(small_gdf, npartitions=2)
    return LazyFeatureCollection.from_dask_gdf(ddf)


@requires_dask_geopandas
class TestLazyFeatureCollection:
    def test_is_dask_gdf_subclass(self, lfc):
        import dask_geopandas as dg
        from pyramids.feature import LazyFeatureCollection

        assert isinstance(lfc, dg.GeoDataFrame)
        assert isinstance(lfc, LazyFeatureCollection)

    def test_is_not_feature_collection(self, lfc):
        assert not isinstance(lfc, FeatureCollection)

    def test_satisfies_spatial_object_protocol(self, lfc):
        assert isinstance(lfc, SpatialObject)

    def test_is_lazy_protocol_helper(self, lfc):
        assert is_lazy(lfc) is True

    def test_compute_returns_eager_feature_collection(self, lfc, small_gdf):
        eager = lfc.compute()
        assert isinstance(eager, FeatureCollection)
        assert len(eager) == 10
        assert eager.crs == small_gdf.crs

    def test_persist_returns_lazy_fc(self, lfc):
        from pyramids.feature import LazyFeatureCollection

        persisted = lfc.persist()
        assert isinstance(persisted, LazyFeatureCollection)

    def test_to_file_raises_not_implemented(self, lfc, tmp_path):
        with pytest.raises(NotImplementedError, match=r"compute\(\)\.to_file"):
            lfc.to_file(tmp_path / "out.geojson")

    def test_plot_raises_not_implemented(self, lfc):
        with pytest.raises(NotImplementedError, match=r"compute\(\)\.plot"):
            lfc.plot()

    def test_epsg_property(self, lfc):
        assert lfc.epsg == 4326

    def test_top_left_corner_property(self, lfc):
        corner = lfc.top_left_corner
        assert isinstance(corner, list)
        assert len(corner) == 2
        assert corner == [0.0, 9.0]

    def test_repr_is_pyramids_branded(self, lfc):
        text = repr(lfc)
        assert text.startswith("LazyFeatureCollection(")
        assert "npartitions=2" in text

    def test_pickle_roundtrip(self, lfc, small_gdf):
        from pyramids.feature import LazyFeatureCollection

        restored = pickle.loads(pickle.dumps(lfc))
        assert isinstance(restored, LazyFeatureCollection)
        assert len(restored.compute()) == len(small_gdf)

    def test_read_file_classmethod_delegates_to_eager_reader(self, small_gdf, tmp_path):
        from pyramids.feature import LazyFeatureCollection

        p = tmp_path / "pts.geojson"
        small_gdf.to_file(p, driver="GeoJSON")
        lfc_direct = LazyFeatureCollection.read_file(str(p))
        assert isinstance(lfc_direct, LazyFeatureCollection)
        assert len(lfc_direct.compute()) == 10

    def test_from_dask_gdf_public_name_is_canonical(self, small_gdf):
        """ARC-V1: public ``from_dask_gdf`` is the supported constructor."""
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        ddf = dg.from_geopandas(small_gdf, npartitions=2)
        lfc_public = LazyFeatureCollection.from_dask_gdf(ddf)
        lfc_alias = LazyFeatureCollection._from_dask_gdf(ddf)
        assert isinstance(lfc_public, LazyFeatureCollection)
        assert isinstance(lfc_alias, LazyFeatureCollection)
        assert len(lfc_public.compute()) == len(lfc_alias.compute()) == 10

    def test_from_dask_gdf_preserves_state(self, small_gdf):
        """ARC-V2: class-swap drops no instance state.

        Test scenario:
            ``from_dask_gdf`` uses ``result.__class__ = cls``. The
            invariant that enables this safely is "LazyFeatureCollection
            adds no extra instance state beyond dask_geopandas.GeoDataFrame."
            If a future commit adds ``__slots__`` or an ``__init__``,
            this test catches it by comparing non-dunder ``vars()`` keys
            before and after the swap.
        """
        import dask_geopandas as dg

        from pyramids.feature import LazyFeatureCollection

        ddf = dg.from_geopandas(small_gdf, npartitions=2)
        pre = {k for k in vars(ddf).keys() if not k.startswith("__")}
        lfc = LazyFeatureCollection.from_dask_gdf(ddf)
        post = {k for k in vars(lfc).keys() if not k.startswith("__")}
        assert pre == post, (
            f"class-swap leaked state: added={post - pre}, dropped={pre - post}"
        )

    def test_no_extra_slots(self):
        """ARC-V2: pin that LazyFeatureCollection declares no ``__slots__``.

        Test scenario:
            Slotted attributes on the subclass would be silently
            dropped by the class-swap constructor (the source
            dask-geopandas frame wouldn't have them set). Assert at
            the class level that no ``__slots__`` is declared.
        """
        from pyramids.feature import LazyFeatureCollection

        assert "__slots__" not in LazyFeatureCollection.__dict__, (
            "LazyFeatureCollection must not declare __slots__ while the "
            "class-swap constructor (from_dask_gdf) is in use — slotted "
            "attributes would be silently dropped by the swap."
        )
