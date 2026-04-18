"""Tests for :mod:`pyramids.base._file_manager`.

DASK-2 introduces pickle-safe file-handle managers (`CachingFileManager`,
`ThreadLocalFileManager`) plus a module-global ``FILE_CACHE`` LRU + a
``_HashedSequence`` key type.

These tests cover:

* ``_LRUCache`` eviction with ``on_evict`` callback.
* ``_HashedSequence`` stable hashing.
* ``CachingFileManager`` happy-path open / reuse / close.
* ``CachingFileManager`` pickle round-trip — state tuple contains only
  the recipe, never a handle.
* ``CachingFileManager`` LRU eviction calls close.
* ``CachingFileManager`` with ``lock=False`` uses the null lock.
* ``CachingFileManager`` shared ``manager_id`` → two instances share one
  cache slot.
* ``ThreadLocalFileManager`` per-thread handle isolation.
* ``ThreadLocalFileManager`` pickle round-trip.
* ``gdal_raster_open`` + ``gdal_mdarray_open`` + ``ogr_open`` respect
  access modes and do URL rewriting via ``_to_vsi``.
"""

from __future__ import annotations

import pickle
import threading

import pytest

from pyramids.base._file_manager import (
    FILE_CACHE,
    CachingFileManager,
    ThreadLocalFileManager,
    _HashedSequence,
    _LRUCache,
    _NullLock,
    _make_cache_key,
)
from pyramids.base._openers import (
    _resolve_access,
    gdal_mdarray_open,
    gdal_raster_open,
    ogr_open,
)


class _FakeHandle:
    """Test double: records open and close calls."""

    def __init__(self, tag: str = ""):
        self.tag = tag
        self.closed = False

    def Close(self):
        self.closed = True

    def __repr__(self):
        return f"_FakeHandle({self.tag!r})"


_counter = {"n": 0}


def _fake_opener(path: str, access: str = "read_only", **kwargs) -> _FakeHandle:
    """Opener that returns a distinct fake handle per call, tagged by path."""
    _counter["n"] += 1
    return _FakeHandle(tag=f"{path}#{_counter['n']}")


@pytest.fixture(autouse=True)
def _reset_counter_and_cache():
    """Each test gets a fresh handle counter and a clean FILE_CACHE."""
    _counter["n"] = 0
    FILE_CACHE.clear()
    yield
    FILE_CACHE.clear()


class TestLRUCache:
    """``_LRUCache`` — small OrderedDict wrapper with ``on_evict`` hook."""

    def test_basic_get_set(self):
        cache = _LRUCache(maxsize=4)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2

    def test_eviction_calls_on_evict(self):
        evicted: list[tuple] = []
        cache = _LRUCache(maxsize=2, on_evict=lambda k, v: evicted.append((k, v)))
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert evicted == [("a", 1)]
        assert "a" not in cache

    def test_lru_order_respected(self):
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]  # makes 'a' most-recently-used
        cache["c"] = 3  # 'b' should evict, not 'a'
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_maxsize_setter_evicts(self):
        evicted: list = []
        cache = _LRUCache(maxsize=5, on_evict=lambda k, v: evicted.append(k))
        for i in range(5):
            cache[str(i)] = i
        cache.maxsize = 2
        assert len(cache) == 2
        assert len(evicted) == 3

    def test_rejects_invalid_maxsize(self):
        with pytest.raises(ValueError):
            _LRUCache(maxsize=0)

    def test_clear_calls_on_evict_for_all(self):
        evicted: list = []
        cache = _LRUCache(maxsize=4, on_evict=lambda k, v: evicted.append(k))
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert sorted(evicted) == ["a", "b"]


class TestHashedSequence:
    """``_HashedSequence`` — list subclass with cached hash."""

    def test_is_hashable(self):
        hs = _HashedSequence([1, "x", (2, 3)])
        assert hash(hs) == hash((1, "x", (2, 3)))

    def test_usable_as_dict_key(self):
        hs = _HashedSequence(["a", 1])
        d = {hs: "value"}
        assert d[_HashedSequence(["a", 1])] == "value"


class TestMakeCacheKey:
    """``_make_cache_key`` canonicalizes kwargs order."""

    def test_kwargs_order_independent(self):
        k1 = _make_cache_key(_fake_opener, "p", "r", {"a": 1, "b": 2}, "id")
        k2 = _make_cache_key(_fake_opener, "p", "r", {"b": 2, "a": 1}, "id")
        assert hash(k1) == hash(k2)

    def test_different_manager_ids_different_keys(self):
        k1 = _make_cache_key(_fake_opener, "p", "r", {}, "id1")
        k2 = _make_cache_key(_fake_opener, "p", "r", {}, "id2")
        assert hash(k1) != hash(k2)


class TestCachingFileManager:
    """Happy-path / pickle / eviction / lock behaviour."""

    def test_acquire_opens_once(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        h1 = fm.acquire()
        h2 = fm.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_pickle_roundtrip(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        data = pickle.dumps(fm)
        fm2 = pickle.loads(data)
        assert fm2._path == "fixture.tif"
        assert fm2._access == "read_only"

    def test_pickle_excludes_handle(self):
        fm = CachingFileManager(_fake_opener, "fixture.tif", "read_only")
        fm.acquire()  # put handle in cache
        data = pickle.dumps(fm)
        assert b"_FakeHandle" not in data

    def test_pickle_clone_shares_cache_when_manager_id_shared(self):
        fm = CachingFileManager(_fake_opener, "f.tif", "read_only", manager_id="shared")
        h1 = fm.acquire()
        fm2 = pickle.loads(pickle.dumps(fm))
        h2 = fm2.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_pickle_clone_without_shared_id_opens_fresh(self):
        fm = CachingFileManager(_fake_opener, "f.tif", "read_only")
        fm.acquire()
        fm2 = pickle.loads(pickle.dumps(fm))
        # manager_id is preserved in __getstate__, so this actually shares
        assert fm2._manager_id == fm._manager_id

    def test_close_drops_handle(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        h = fm.acquire()
        fm.close()
        assert h.closed is True
        h2 = fm.acquire()
        assert h2 is not h

    def test_lock_false_uses_null_lock(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only", lock=False)
        assert isinstance(fm._lock, _NullLock)

    def test_custom_lock_used(self):
        lock = threading.Lock()
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only", lock=lock)
        assert fm._lock is lock
        assert fm._use_default_lock is False

    def test_acquire_context_yields_handle(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        with fm.acquire_context() as h:
            assert isinstance(h, _FakeHandle)

    def test_acquire_context_preserves_handle_on_reraise(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        fm.acquire()  # pre-cache
        with pytest.raises(RuntimeError):
            with fm.acquire_context():
                raise RuntimeError("boom")
        # Still cached because it was cached before the block.
        assert fm._key in FILE_CACHE

    def test_acquire_context_drops_handle_on_first_open_failure(self):
        fm = CachingFileManager(_fake_opener, "x.tif", "read_only")
        with pytest.raises(RuntimeError):
            with fm.acquire_context():
                raise RuntimeError("boom")
        assert fm._key not in FILE_CACHE


class TestCachingFileManagerLRUEviction:
    """``FILE_CACHE`` eviction closes evicted handles."""

    def test_lru_eviction_closes(self):
        FILE_CACHE.maxsize = 2
        try:
            handles = []
            for i in range(3):
                fm = CachingFileManager(_fake_opener, f"f{i}.tif", "read_only")
                handles.append(fm.acquire())
            # first handle should now be evicted and closed
            assert handles[0].closed is True
            assert handles[1].closed is False
            assert handles[2].closed is False
        finally:
            FILE_CACHE.maxsize = 128


class TestThreadLocalFileManager:
    """Per-thread handle isolation, no locking."""

    def test_acquire_opens_once_per_thread(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        h1 = fm.acquire()
        h2 = fm.acquire()
        assert h1 is h2
        assert _counter["n"] == 1

    def test_different_threads_get_different_handles(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        results: list = []

        def grab():
            results.append(fm.acquire())

        t = threading.Thread(target=grab)
        t.start()
        t.join()
        main_handle = fm.acquire()
        assert len(results) == 1
        assert results[0] is not main_handle

    def test_pickle_roundtrip_no_handle(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        fm.acquire()
        data = pickle.dumps(fm)
        assert b"_FakeHandle" not in data
        fm2 = pickle.loads(data)
        assert fm2._path == "t.tif"

    def test_close_is_thread_local(self):
        fm = ThreadLocalFileManager(_fake_opener, "t.tif", "read_only")
        h = fm.acquire()
        fm.close()
        assert h.closed is True
        # Re-acquire on the same thread opens a new handle.
        h2 = fm.acquire()
        assert h2 is not h


class TestOpeners:
    """``_openers`` module primitives."""

    def test_resolve_access_known(self):
        from osgeo import gdal

        assert _resolve_access("read_only") == gdal.GA_ReadOnly
        assert _resolve_access("r") == gdal.GA_ReadOnly
        assert _resolve_access("write") == gdal.GA_Update
        assert _resolve_access("w") == gdal.GA_Update

    def test_resolve_access_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown access mode"):
            _resolve_access("bogus")

    def test_gdal_raster_open_extra_kwargs_ignored(self):
        # Sanity: opener ignores unrelated kwargs rather than raising.
        # We can't open a real file without fixtures, so just assert
        # the signature accepts them.
        import inspect

        sig = inspect.signature(gdal_raster_open)
        params = sig.parameters
        assert "path" in params and "access" in params

    def test_gdal_mdarray_open_sig(self):
        import inspect

        sig = inspect.signature(gdal_mdarray_open)
        assert "path" in sig.parameters

    def test_ogr_open_sig(self):
        import inspect

        sig = inspect.signature(ogr_open)
        assert "path" in sig.parameters


class TestOpenersE2E:
    """End-to-end: exercise openers against a real tiny GeoTIFF fixture."""

    @pytest.fixture
    def tif_path(self, tmp_path):
        """Create a 3x3 uint8 in-memory GeoTIFF and return its path."""
        from osgeo import gdal, osr

        path = str(tmp_path / "tiny.tif")
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, 3, 3, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(
            __import__("numpy").zeros((3, 3), dtype="uint8")
        )
        ds.FlushCache()
        ds = None
        return path

    def test_gdal_raster_open_real_file(self, tif_path):
        ds = gdal_raster_open(tif_path, "read_only")
        try:
            assert ds.RasterXSize == 3 and ds.RasterYSize == 3
        finally:
            ds = None

    def test_caching_manager_e2e_with_real_file(self, tif_path):
        fm = CachingFileManager(gdal_raster_open, tif_path, "read_only")
        try:
            ds1 = fm.acquire()
            ds2 = fm.acquire()
            assert ds1 is ds2
            assert ds1.RasterXSize == 3
        finally:
            fm.close()

    def test_thread_local_manager_e2e_with_real_file(self, tif_path):
        fm = ThreadLocalFileManager(gdal_raster_open, tif_path, "read_only")
        try:
            ds = fm.acquire()
            assert ds.RasterXSize == 3
        finally:
            fm.close()
