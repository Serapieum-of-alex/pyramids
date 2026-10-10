"""The grouped STAC option objects are a public API: their defaults are the call-site contract.

Every dataclass in :mod:`pyramids.dataset.stac_options` defaults every field, so each one is
optional at the call site and `None` means "all defaults for that group". That makes the default
values, the frozen-ness, and the two `__post_init__` validators the whole observable surface of the
module, and each of them is pinned here.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from pyramids.dataset import stac_options
from pyramids.dataset.grid import Grid
from pyramids.dataset.stac_options import (
    Alignment,
    AssetRead,
    BandMetadata,
    CubeProvenance,
    Footprint,
    GroupFusion,
    ReadTolerance,
    StacExtensions,
)

pytestmark = pytest.mark.core

OPTION_CLASSES = (
    Alignment,
    AssetRead,
    BandMetadata,
    CubeProvenance,
    Footprint,
    GroupFusion,
    ReadTolerance,
    StacExtensions,
)


def _noop_fuser(dst: np.ndarray, src: np.ndarray) -> None:
    """Stand in for an odc-style in-place fuser: a plain module-level function."""
    dst[...] = np.fmax(dst, src)


class _CallableFuser:
    """Stand in for a stateful fuser: an instance that is callable through `__call__`."""

    def __call__(self, dst: np.ndarray, src: np.ndarray) -> None:
        """Mutate the accumulator in place, as a real fuser would."""
        dst[...] = src


class TestBandMetadata:
    """Tests for the `BandMetadata` option group."""

    def test_defaults(self):
        """All five band-measuring opt-ins default to the cheap answer.

        Test scenario:
            `BandMetadata()` must construct with no arguments, leave every measuring block off, ask
            for ten histogram buckets, and let GDAL approximate statistics.
        """
        options = BandMetadata()
        assert options.with_stats is False, (
            f"with_stats should default to False, got {options.with_stats!r}"
        )
        assert options.with_histogram is False, (
            f"with_histogram should default to False, got {options.with_histogram!r}"
        )
        assert options.histogram_bins == 10, (
            f"histogram_bins should default to 10, got {options.histogram_bins!r}"
        )
        assert options.with_eo is False, (
            f"with_eo should default to False, got {options.with_eo!r}"
        )
        assert options.stats_approx_ok is True, (
            f"stats_approx_ok should default to True, got {options.stats_approx_ok!r}"
        )

    def test_every_field_is_settable(self):
        """Each field round-trips the value the call site passed.

        Test scenario:
            Every default is inverted at once, so a field silently wired to the wrong attribute name
            cannot hide behind an identical default.
        """
        options = BandMetadata(
            with_stats=True,
            with_histogram=True,
            histogram_bins=256,
            with_eo=True,
            stats_approx_ok=False,
        )
        assert dataclasses.asdict(options) == {
            "with_stats": True,
            "with_histogram": True,
            "histogram_bins": 256,
            "with_eo": True,
            "stats_approx_ok": False,
        }, f"the constructor did not store every field verbatim: {options!r}"

    def test_equality_and_hash(self):
        """Two equal groups compare equal and hash alike; a differing one does not.

        Test scenario:
            All fields are scalars, so the generated `__eq__`/`__hash__` make the group usable as a
            dict key and safe to compare in assertions.
        """
        left = BandMetadata(with_stats=True)
        right = BandMetadata(with_stats=True)
        assert left is not right, (
            "the two groups must be separate objects to prove anything"
        )
        assert left == right, (
            "two BandMetadata with identical fields should compare equal"
        )
        assert hash(left) == hash(right), "equal BandMetadata should hash alike"
        assert left != BandMetadata(), (
            "BandMetadata(with_stats=True) should not equal the all-defaults group"
        )


class TestFootprint:
    """Tests for the `Footprint` option group and its mode validator."""

    def test_defaults(self):
        """The default footprint is the bbox path at six decimals, with no sampling budget.

        Test scenario:
            `Footprint()` must construct, pick `"bbox"`, footprint band 0, and leave `max_samples`,
            `simplify_tolerance` and `densify` unset.
        """
        options = Footprint()
        assert options.mode == "bbox", (
            f"mode should default to 'bbox', got {options.mode!r}"
        )
        assert options.band == 0, f"band should default to 0, got {options.band!r}"
        assert options.max_samples is None, (
            f"max_samples should default to None, got {options.max_samples!r}"
        )
        assert options.simplify_tolerance is None, (
            f"simplify_tolerance should default to None, got {options.simplify_tolerance!r}"
        )
        assert options.densify is None, (
            f"densify should default to None, got {options.densify!r}"
        )
        assert options.precision == 6, (
            f"precision should default to 6, got {options.precision!r}"
        )

    def test_every_field_is_settable(self):
        """Each field round-trips the value the call site passed.

        Test scenario:
            A fully specified data footprint stores all six values verbatim.
        """
        options = Footprint(
            mode="data",
            band=2,
            max_samples=10_000,
            simplify_tolerance=0.001,
            densify=250.0,
            precision=3,
        )
        assert dataclasses.asdict(options) == {
            "mode": "data",
            "band": 2,
            "max_samples": 10_000,
            "simplify_tolerance": 0.001,
            "densify": 250.0,
            "precision": 3,
        }, f"the constructor did not store every field verbatim: {options!r}"

    @pytest.mark.parametrize("mode", ["bbox", "data"])
    def test_post_init_accepts_the_two_supported_modes(self, mode):
        """Both documented modes construct and keep the mode they were given.

        Args:
            mode: One of the two footprints `__post_init__` allows.

        Test scenario:
            The validator must let `"bbox"` and `"data"` through untouched — it only rejects.
        """
        options = Footprint(mode=mode)
        assert options.mode == mode, (
            f"mode {mode!r} should survive validation, got {options.mode!r}"
        )

    @pytest.mark.parametrize(
        "mode", ["grid", "BBOX", "Data", "", "bboxx", 0, None, 1.0]
    )
    def test_post_init_rejects_any_other_mode(self, mode):
        """An unsupported mode raises `ValueError` quoting the bad value with `repr`.

        Args:
            mode: A value that is neither `"bbox"` nor `"data"` — including near-misses of
                different case and non-string types, since the check is a membership test.

        Test scenario:
            The message is part of the contract: it names both valid modes and echoes the offending
            value through `!r`, so `"BBOX"` reads back quoted and `0` reads back bare.
        """
        expected = f"footprint must be 'bbox' or 'data', got {mode!r}."
        with pytest.raises(ValueError, match=re.escape(expected)) as exc_info:
            Footprint(mode=mode)
        assert str(exc_info.value) == expected, (
            f"expected exactly {expected!r}, got {str(exc_info.value)!r}"
        )

    def test_equality_and_hash(self):
        """Equal footprints compare equal and hash alike.

        Test scenario:
            Every field is a scalar or `None`, so the group is hashable and comparable.
        """
        left = Footprint(mode="data", band=1)
        right = Footprint(mode="data", band=1)
        assert left is not right, (
            "the two footprints must be separate objects to prove anything"
        )
        assert left == right, "two Footprint with identical fields should compare equal"
        assert hash(left) == hash(right), "equal Footprint should hash alike"
        assert Footprint(mode="data") != Footprint(mode="bbox"), (
            "a data footprint should not equal a bbox footprint"
        )


class TestStacExtensions:
    """Tests for the `StacExtensions` option group."""

    def test_defaults(self):
        """Both extension blocks are stamped by default.

        Test scenario:
            `StacExtensions()` must construct and opt *in* to `proj` and `raster:bands` — this group
            is the one whose defaults are `True`.
        """
        options = StacExtensions()
        assert options.with_proj is True, (
            f"with_proj should default to True, got {options.with_proj!r}"
        )
        assert options.with_raster is True, (
            f"with_raster should default to True, got {options.with_raster!r}"
        )

    def test_every_field_is_settable(self):
        """Both flags can be switched off independently.

        Test scenario:
            Turning each flag off in turn proves the two are distinct switches, not one alias.
        """
        assert dataclasses.asdict(StacExtensions(with_proj=False)) == {
            "with_proj": False,
            "with_raster": True,
        }, "with_proj=False should leave with_raster at its default"
        assert dataclasses.asdict(StacExtensions(with_raster=False)) == {
            "with_proj": True,
            "with_raster": False,
        }, "with_raster=False should leave with_proj at its default"

    def test_equality_and_hash(self):
        """Equal extension switchboards compare equal and hash alike."""
        left = StacExtensions(with_proj=False)
        right = StacExtensions(with_proj=False)
        assert left is not right, "the two switchboards must be separate objects"
        assert left == right, (
            "two StacExtensions with identical flags should compare equal"
        )
        assert hash(StacExtensions()) == hash(
            StacExtensions(with_proj=True, with_raster=True)
        ), "the all-defaults group should hash like the explicitly-defaulted one"
        assert StacExtensions() != StacExtensions(with_raster=False), (
            "the all-defaults group should not equal one with raster off"
        )


class TestAlignment:
    """Tests for the `Alignment` option group."""

    def test_defaults(self):
        """No target grid, alignment allowed, nearest-neighbour resampling.

        Test scenario:
            `Alignment()` must construct and keep each timestep's native grid (`grid=None`) while
            still permitting a mixed-resolution item to be resampled (`align=True`).
        """
        options = Alignment()
        assert options.grid is None, (
            f"grid should default to None, got {options.grid!r}"
        )
        assert options.align is True, (
            f"align should default to True, got {options.align!r}"
        )
        assert options.resampling is None, (
            f"resampling should default to None, got {options.resampling!r}"
        )

    def test_every_field_is_settable(self):
        """A target grid, a disabled alignment and a method name all round-trip.

        Test scenario:
            The `grid` field is typed `Grid | None` and the dataclass adds no validation, so an
            empty `Grid()` — the documented "keep the native grid" spelling — is stored as given.
        """
        grid = Grid()
        options = Alignment(grid=grid, align=False, resampling="bilinear")
        assert options.grid is grid, (
            f"grid should be stored by reference, got {options.grid!r}"
        )
        assert options.align is False, f"align should be False, got {options.align!r}"
        assert options.resampling == "bilinear", (
            f"resampling should be 'bilinear', got {options.resampling!r}"
        )

    def test_a_per_asset_resampling_mapping_is_stored_verbatim(self):
        """A `{asset key: method}` mapping is accepted without copying or validation.

        Test scenario:
            `resampling` is documented as either a method name or a per-band mapping; the dataclass
            stores whichever it was handed, leaving the "requires align=True" rule to `from_stac`.
        """
        mapping = {"red": "bilinear", "scl": "nearest"}
        options = Alignment(resampling=mapping)
        assert options.resampling is mapping, (
            f"the mapping should be stored by reference, got {options.resampling!r}"
        )

    def test_a_mapping_resampling_compares_but_does_not_hash(self):
        """With a `dict` field the group stays comparable but becomes unhashable.

        Test scenario:
            `frozen=True` generates a `__hash__` over the fields, so hashability follows the field
            values: two equal mappings compare equal, while hashing either raises `TypeError` from
            the `dict`. Code that keys a cache by an `Alignment` must therefore not pass a mapping.
        """
        left = Alignment(resampling={"red": "cubic"})
        right = Alignment(resampling={"red": "cubic"})
        assert left == right, (
            f"two Alignment with equal mappings should compare equal, got {left!r} != {right!r}"
        )
        with pytest.raises(TypeError, match="unhashable type"):
            hash(left)

    def test_equality_and_hash_with_scalar_fields(self):
        """With hashable fields only, the group hashes like any frozen dataclass."""
        left = Alignment(align=False, resampling="average")
        right = Alignment(align=False, resampling="average")
        assert left is not right, "the two alignments must be separate objects"
        assert left == right, (
            "two Alignment with identical scalar fields should compare equal"
        )
        assert hash(left) == hash(right), (
            "equal Alignment with scalar fields should hash alike"
        )
        assert Alignment(resampling="average") != Alignment(), (
            "an Alignment naming a method should not equal the all-defaults group"
        )


class TestReadTolerance:
    """Tests for the `ReadTolerance` option group."""

    def test_defaults(self):
        """Neither failure is forgiven by default.

        Test scenario:
            `ReadTolerance()` must construct and leave both tolerances off, so a missing asset key
            and an unopenable asset each still fail the build.
        """
        options = ReadTolerance()
        assert options.skip_missing is False, (
            f"skip_missing should default to False, got {options.skip_missing!r}"
        )
        assert options.errors_as_nodata is False, (
            f"errors_as_nodata should default to False, got {options.errors_as_nodata!r}"
        )

    def test_every_field_is_settable(self):
        """The two tolerances are independent switches.

        Test scenario:
            The metadata gap and the IO failure are documented as different events, so enabling one
            must not enable the other.
        """
        assert dataclasses.asdict(ReadTolerance(skip_missing=True)) == {
            "skip_missing": True,
            "errors_as_nodata": False,
        }, "skip_missing=True should leave errors_as_nodata off"
        assert dataclasses.asdict(ReadTolerance(errors_as_nodata=True)) == {
            "skip_missing": False,
            "errors_as_nodata": True,
        }, "errors_as_nodata=True should leave skip_missing off"

    def test_equality_and_hash(self):
        """Equal tolerances compare equal and hash alike."""
        left = ReadTolerance(skip_missing=True)
        right = ReadTolerance(skip_missing=True)
        assert left is not right, "the two tolerances must be separate objects"
        assert left == right, (
            "two ReadTolerance with identical flags should compare equal"
        )
        assert hash(left) == hash(right), "equal ReadTolerance should hash alike"
        assert ReadTolerance(skip_missing=True) != ReadTolerance(
            errors_as_nodata=True
        ), "the two tolerances are different fields and must not compare equal"


class TestGroupFusion:
    """Tests for the `GroupFusion` option group and its `fuse_func` validator."""

    def test_defaults(self):
        """The default fusion is the built-in first-valid-pixel merge.

        Test scenario:
            `GroupFusion()` must construct, select `"first"`, and install no custom fuser — which is
            also the `fuse_func is None` short-circuit of `__post_init__`.
        """
        options = GroupFusion()
        assert options.method == "first", (
            f"method should default to 'first', got {options.method!r}"
        )
        assert options.fuse_func is None, (
            f"fuse_func should default to None, got {options.fuse_func!r}"
        )

    @pytest.mark.parametrize(
        "method", ["first", "last", "min", "max", "sum", "count", "mean"]
    )
    def test_every_documented_method_is_stored_verbatim(self, method):
        """Each documented overlap rule round-trips untouched.

        Args:
            method: One of the merge rules the docstring lists.

        Test scenario:
            `__post_init__` validates `fuse_func` only, so the method name is passed through to
            `merge_rasters` without the dataclass vetting it.
        """
        options = GroupFusion(method=method)
        assert options.method == method, (
            f"method {method!r} should be stored verbatim, got {options.method!r}"
        )

    def test_an_unknown_method_is_not_rejected_here(self):
        """A bogus method name is left for `from_stac` to reject, not the dataclass.

        Test scenario:
            The validator deliberately covers `fuse_func` alone; pinning that keeps a later reader
            from assuming the option object already screened the method.
        """
        options = GroupFusion(method="median")
        assert options.method == "median", (
            f"an unvalidated method should be stored as given, got {options.method!r}"
        )

    def test_post_init_accepts_a_plain_function(self):
        """A module-level `(dst, src) -> None` function is accepted and stored by reference."""
        options = GroupFusion(fuse_func=_noop_fuser)
        assert options.fuse_func is _noop_fuser, (
            f"the fuser should be stored by reference, got {options.fuse_func!r}"
        )

    def test_post_init_accepts_a_lambda(self):
        """A lambda is callable, so the validator lets it through.

        Test scenario:
            The check is `callable(...)`, not an inspection of the signature, so any callable passes
            regardless of how it was spelled.
        """
        fuser = lambda dst, src: None  # noqa: E731
        options = GroupFusion(fuse_func=fuser)
        assert options.fuse_func is fuser, (
            f"the lambda should be stored by reference, got {options.fuse_func!r}"
        )

    def test_post_init_accepts_a_callable_object(self):
        """An instance implementing `__call__` is accepted, not only a function.

        Test scenario:
            A stateful fuser is naturally written as a callable class; `callable()` is true for its
            instances, so `__post_init__` must not demand a `FunctionType`.
        """
        fuser = _CallableFuser()
        options = GroupFusion(fuse_func=fuser)
        assert options.fuse_func is fuser, (
            f"the callable object should be stored by reference, got {options.fuse_func!r}"
        )

    def test_post_init_accepts_a_builtin_callable(self):
        """A C-level callable such as a NumPy ufunc passes the `callable()` check."""
        options = GroupFusion(fuse_func=np.fmax)
        assert options.fuse_func is np.fmax, (
            f"a ufunc should be stored by reference, got {options.fuse_func!r}"
        )

    @pytest.mark.parametrize(
        "fuse_func",
        [3, 0, "first", "", 1.5, True, [1, 2], {"a": 1}, (), np.float64(1.0)],
        ids=[
            "int",
            "zero",
            "str",
            "empty-str",
            "float",
            "bool",
            "list",
            "dict",
            "tuple",
            "np-scalar",
        ],
    )
    def test_post_init_rejects_a_non_callable(self, fuse_func):
        """A non-callable `fuse_func` raises `TypeError` naming its type.

        Args:
            fuse_func: A value that is neither `None` nor callable — including falsy ones, since the
                guard tests `is not None` rather than truthiness.

        Test scenario:
            The message is part of the contract: it restates the expected `(dst, src) -> None` shape
            and ends with `type(fuse_func).__name__`, so an `int` reads back as `int`.
        """
        expected = f"fuse_func must be a callable (dst, src) -> None, got {type(fuse_func).__name__}."
        with pytest.raises(TypeError, match=re.escape(expected)) as exc_info:
            GroupFusion(fuse_func=fuse_func)
        assert str(exc_info.value) == expected, (
            f"expected exactly {expected!r}, got {str(exc_info.value)!r}"
        )

    def test_equality_and_hash(self):
        """Equal fusions compare equal and hash alike when the fuser is a function.

        Test scenario:
            Functions are hashable by identity, so a group carrying one stays usable as a dict key.
        """
        left = GroupFusion(fuse_func=_noop_fuser)
        right = GroupFusion(fuse_func=_noop_fuser)
        assert left is not right, "the two fusions must be separate objects"
        assert left == right, (
            "two GroupFusion holding the same fuser should compare equal"
        )
        assert hash(left) == hash(right), "equal GroupFusion should hash alike"
        assert GroupFusion(method="max") != GroupFusion(), (
            "a non-default method should not equal the all-defaults group"
        )


class TestAssetRead:
    """Tests for the `AssetRead` option group."""

    def test_defaults(self):
        """Stored counts and no metadata overrides.

        Test scenario:
            `AssetRead()` must construct, leave rescaling off (so single-asset builds stay lazy over
            `/vsicurl`) and supply no `stac_cfg`.
        """
        options = AssetRead()
        assert options.rescale is False, (
            f"rescale should default to False, got {options.rescale!r}"
        )
        assert options.cfg is None, f"cfg should default to None, got {options.cfg!r}"

    def test_every_field_is_settable(self):
        """Both fields round-trip, and `cfg` is stored by reference without validation.

        Test scenario:
            `cfg` is typed `Any`, so an arbitrary `stac_cfg`-style mapping reaches `from_stac`
            untouched rather than being normalised here.
        """
        cfg = {"*": {"assets": {"data": {"nodata": -9999}}}}
        options = AssetRead(rescale=True, cfg=cfg)
        assert options.rescale is True, (
            f"rescale should be True, got {options.rescale!r}"
        )
        assert options.cfg is cfg, (
            f"cfg should be stored by reference, got {options.cfg!r}"
        )

    def test_a_mapping_cfg_compares_but_does_not_hash(self):
        """A `dict` `cfg` keeps equality but costs hashability.

        Test scenario:
            As with `Alignment.resampling`, the generated `__hash__` hashes the fields, so a mapping
            `cfg` makes the group unhashable while leaving `__eq__` working.
        """
        left = AssetRead(cfg={"*": {"warnings": "ignore"}})
        right = AssetRead(cfg={"*": {"warnings": "ignore"}})
        assert left == right, (
            f"two AssetRead with equal cfg mappings should compare equal, got {left!r} != {right!r}"
        )
        with pytest.raises(TypeError, match="unhashable type"):
            hash(left)

    def test_equality_and_hash_with_scalar_fields(self):
        """With `cfg=None` the group hashes like any frozen dataclass."""
        left = AssetRead(rescale=True)
        right = AssetRead(rescale=True)
        assert left is not right, "the two readers must be separate objects"
        assert left == right, (
            "two AssetRead with identical scalar fields should compare equal"
        )
        assert hash(left) == hash(right), (
            "equal AssetRead with scalar fields should hash alike"
        )
        assert AssetRead(rescale=True) != AssetRead(), (
            "AssetRead(rescale=True) should not equal the all-defaults group"
        )


class TestCubeProvenance:
    """Tests for the `CubeProvenance` option group."""

    def test_defaults(self):
        """No properties are carried over and bands keep their asset keys.

        Test scenario:
            `CubeProvenance()` must construct and attach nothing: `properties=False` and
            `eo_band_names=False`.
        """
        options = CubeProvenance()
        assert options.properties is False, (
            f"properties should default to False, got {options.properties!r}"
        )
        assert options.eo_band_names is False, (
            f"eo_band_names should default to False, got {options.eo_band_names!r}"
        )

    @pytest.mark.parametrize(
        "properties",
        [True, False, "datetime", ["datetime", "eo:cloud_cover"], ("datetime",)],
        ids=["all", "none", "single-key", "list-of-keys", "tuple-of-keys"],
    )
    def test_properties_accepts_each_documented_spelling(self, properties):
        """Each of the documented `properties` spellings is stored verbatim.

        Args:
            properties: `True`/`False`, a single key, or a sequence of keys.

        Test scenario:
            The field is typed `bool | str | Sequence[str]` and the dataclass normalises nothing, so
            whichever spelling the call site used reaches `from_stac` unchanged.
        """
        options = CubeProvenance(properties=properties)
        assert options.properties == properties, (
            f"properties {properties!r} should be stored verbatim, got {options.properties!r}"
        )

    def test_eo_band_names_is_settable(self):
        """`eo_band_names=True` round-trips and leaves `properties` alone."""
        options = CubeProvenance(eo_band_names=True)
        assert dataclasses.asdict(options) == {
            "properties": False,
            "eo_band_names": True,
        }, f"eo_band_names=True should leave properties at its default, got {options!r}"

    def test_a_list_of_properties_compares_but_does_not_hash(self):
        """A `list` of property keys keeps equality but costs hashability.

        Test scenario:
            A `tuple` of keys hashes fine while the equivalent `list` does not, so the hashable
            spelling is the one to use when a provenance group is cached or deduplicated.
        """
        left = CubeProvenance(properties=["datetime"])
        right = CubeProvenance(properties=["datetime"])
        assert left == right, (
            f"two CubeProvenance with equal key lists should compare equal, got {left!r}"
        )
        with pytest.raises(TypeError, match="unhashable type"):
            hash(left)
        hashable = CubeProvenance(properties=("datetime",))
        twin = CubeProvenance(properties=("datetime",))
        assert hashable is not twin, (
            "the two tuple-keyed groups must be separate objects"
        )
        assert hash(hashable) == hash(twin), (
            "a tuple of keys is hashable, so the group should hash alike"
        )

    def test_equality_and_hash(self):
        """Equal provenance groups compare equal and hash alike for scalar fields."""
        left = CubeProvenance(properties=True)
        right = CubeProvenance(properties=True)
        assert left is not right, "the two provenance groups must be separate objects"
        assert left == right, (
            "two CubeProvenance with identical fields should compare equal"
        )
        assert hash(left) == hash(right), "equal CubeProvenance should hash alike"
        assert CubeProvenance(properties=True) != CubeProvenance(eo_band_names=True), (
            "the two fields are distinct and must not compare equal"
        )


class TestModuleSurface:
    """Tests for the module-level contract every option group shares."""

    @pytest.mark.parametrize(
        "cls", OPTION_CLASSES, ids=[cls.__name__ for cls in OPTION_CLASSES]
    )
    def test_every_group_is_a_frozen_dataclass(self, cls):
        """Each exported name is a dataclass declared `frozen=True`.

        Args:
            cls: One of the eight option groups.

        Test scenario:
            Frozen-ness is what lets a group be shared between calls without a caller mutating
            another's options, so it is checked on the declaration as well as on an assignment.
        """
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} should be a dataclass"
        assert cls.__dataclass_params__.frozen is True, (
            f"{cls.__name__} should be declared frozen=True"
        )

    @pytest.mark.parametrize(
        "cls", OPTION_CLASSES, ids=[cls.__name__ for cls in OPTION_CLASSES]
    )
    def test_every_field_is_defaulted(self, cls):
        """Each group constructs with no arguments, so every field has a default.

        Args:
            cls: One of the eight option groups.

        Test scenario:
            The module's promise is that each group is optional at the call site and `None` means
            "all defaults"; a field added without a default would break every existing call.
        """
        missing = [
            field.name
            for field in dataclasses.fields(cls)
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ]
        assert missing == [], f"{cls.__name__} fields without a default: {missing}"
        assert cls() is not None, f"{cls.__name__}() should construct with no arguments"

    @pytest.mark.parametrize(
        "cls", OPTION_CLASSES, ids=[cls.__name__ for cls in OPTION_CLASSES]
    )
    def test_assignment_raises_frozen_instance_error(self, cls):
        """Assigning to any field of a constructed group raises `FrozenInstanceError`.

        Args:
            cls: One of the eight option groups.

        Test scenario:
            The first field of each group is reassigned on a default instance; a frozen dataclass
            must refuse it rather than silently mutating shared options.
        """
        instance = cls()
        field_name = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, "mutated")
        assert getattr(instance, field_name) == getattr(cls(), field_name), (
            f"{cls.__name__}.{field_name} should be unchanged after the refused assignment"
        )

    @pytest.mark.parametrize(
        "cls", OPTION_CLASSES, ids=[cls.__name__ for cls in OPTION_CLASSES]
    )
    def test_deletion_raises_frozen_instance_error(self, cls):
        """Deleting a field of a constructed group raises `FrozenInstanceError`.

        Args:
            cls: One of the eight option groups.

        Test scenario:
            `frozen=True` blocks `__delattr__` as well as `__setattr__`; both are part of the
            immutability the groups advertise.
        """
        instance = cls()
        field_name = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(instance, field_name)

    def test_dunder_all_lists_exactly_the_eight_groups(self):
        """`__all__` matches the public names the module actually defines, sorted.

        Test scenario:
            `__all__` is what `pyramids.stac` re-exports from, so a group added here but left out of
            `__all__` would be unreachable from the documented import path.
        """
        defined = sorted(
            name
            for name, value in vars(stac_options).items()
            if not name.startswith("_")
            and dataclasses.is_dataclass(value)
            and isinstance(value, type)
        )
        assert stac_options.__all__ == defined, (
            f"__all__ is {stac_options.__all__}, but the module defines {defined}"
        )
        assert stac_options.__all__ == sorted(stac_options.__all__), (
            "__all__ should be sorted"
        )
        assert [cls.__name__ for cls in OPTION_CLASSES] == defined, (
            "this test module does not exercise every group the source defines"
        )

    def test_the_grid_annotation_stays_type_checking_only(self):
        """`Grid` is not bound at run time, so the module remains an import leaf.

        Test scenario:
            `pyramids.dataset._stac` imports this module at load time, which only works while it
            imports nothing from `pyramids` itself; the `Grid` annotation lives behind
            `TYPE_CHECKING` and `from __future__ import annotations` keeps `Alignment.grid`'s
            annotation a string, so the name must be absent from the module namespace.
        """
        assert not hasattr(stac_options, "Grid"), (
            "Grid is bound at run time, so the TYPE_CHECKING-only import was made unconditional"
        )
        annotation = Alignment.__annotations__["grid"]
        assert annotation == "Grid | None", (
            f"the grid annotation should stay a string, got {annotation!r}"
        )
