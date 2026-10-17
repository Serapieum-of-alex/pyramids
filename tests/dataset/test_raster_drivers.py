"""The raster-source driver registry and ``Dataset.read_file`` dispatch, with no plugin installed.

Every test builds its own fake driver *class* (claims are class-level, so discovery never
constructs one). Two fixtures make that possible: ``registered`` injects classes straight into the
registry cache, and ``fake_entry_points`` monkeypatches ``importlib.metadata.entry_points`` with
in-memory ``EntryPoint`` objects so real discovery runs without a distribution on disk.

The negative cases matter most: a plugin that raises on import, declares a future API version,
is missing its ``extra`` dependency, or collides on a name must be skipped with a warning, because
the alternative is a broken third-party package stopping a raster from opening through GDAL.
"""

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint

import numpy as np
import pytest

from pyramids.base._errors import DriverNotExistError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, _drivers
from pyramids.dataset._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    RASTER_DRIVER_API_VERSION,
    clear_driver_cache,
    describe_drivers,
    drivers_for_scheme,
    drivers_for_suffix,
    raster_drivers,
    select_driver,
)
from pyramids.dataset.driver_conformance import (
    assert_raster_driver_conforms,
    check_raster_driver,
)

pytestmark = pytest.mark.core

GEO = GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326)


def _dataset() -> Dataset:
    """Build the smallest Dataset a fake driver can return.

    Returns:
        Dataset: a 2x2 in-memory raster.
    """
    return Dataset.from_array(np.ones((2, 2), dtype="float32"), geo_ref=GEO)


class GoodDriver:
    """A minimal conforming raster driver class."""

    name = "fci"
    long_name = "EUMETSAT FCI"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes = (".nat",)
    schemes = ()
    capabilities = frozenset({"read"})
    extra = None

    @classmethod
    def can_open(cls, source):
        """Claim a ``.nat`` granule."""
        return isinstance(source, str) and source.endswith(".nat")

    def read(self, source, *, bbox=None, bands=None, crs=None, options=None, auth=None):
        """Return a stub dataset."""
        return _dataset()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the discovery cache around every test so one never leaks into the next."""
    clear_driver_cache()
    yield
    clear_driver_cache()


@pytest.fixture
def registered(monkeypatch):
    """Inject driver classes directly into the registry cache.

    Returns:
        A callable taking driver classes and publishing them as the resolved registry.
    """

    def _register(*drivers):
        monkeypatch.setattr(
            _drivers, "_CACHE", {driver.name: driver for driver in drivers}
        )
        return drivers[0] if len(drivers) == 1 else drivers

    return _register


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Monkeypatch ``entry_points`` so real discovery runs over in-memory entry points.

    Returns:
        A callable taking ``(name, class)`` pairs, each published as a module attribute the
        EntryPoint's ``value`` resolves to, so ``EntryPoint.load`` follows its normal path.
    """
    published: list[str] = []

    def _publish(*pairs):
        entries = []
        for index, (name, factory) in enumerate(pairs):
            attribute = f"_fake_raster_driver_{index}"
            published.append(attribute)
            setattr(_drivers, attribute, factory)
            entries.append(
                EntryPoint(
                    name=name,
                    value=f"pyramids.dataset._drivers:{attribute}",
                    group=ENTRY_POINT_GROUP,
                )
            )

        def _entry_points(*, group=None):
            return tuple(entries) if group == ENTRY_POINT_GROUP else ()

        monkeypatch.setattr(_drivers, "entry_points", _entry_points)

    yield _publish
    for attribute in published:
        monkeypatch.delattr(_drivers, attribute, raising=False)


class TestValidation:
    """Discovery rejects a driver class that does not meet the contract."""

    @pytest.mark.parametrize(
        "mutate, fragment",
        [
            ({"name": ""}, "non-blank string name"),
            ({"name": "gdal"}, "reserved name"),
            ({"long_name": 5}, "non-string long_name"),
            ({"api_version": 99}, "speaks version"),
            ({"capabilities": "read"}, "expected a set of tokens"),
            (
                {"capabilities": frozenset({"read", "bogus"})},
                "unknown capability token",
            ),
            ({"capabilities": frozenset()}, "required 'read' capability"),
            ({"suffixes": 5}, "declares suffixes 5"),
            ({"schemes": 5}, "declares schemes 5"),
            ({"suffixes": (), "schemes": ()}, "neither suffixes nor schemes"),
            ({"suffixes": ("nat",)}, "malformed suffix"),
            ({"schemes": ("ee://",)}, "malformed scheme"),
        ],
    )
    def test_a_malformed_claim_is_skipped_with_a_warning(
        self, fake_entry_points, mutate, fragment
    ):
        """Each contract violation skips the driver and warns naming the problem."""
        attrs = {**GoodDriver.__dict__}
        bad = type("Bad", (), {**{k: v for k, v in attrs.items()}, **mutate})
        fake_entry_points(("bad", bad))
        with pytest.warns(RuntimeWarning, match=fragment):
            assert raster_drivers() == {}

    def test_missing_attribute_is_skipped(self, fake_entry_points):
        """A class missing a required claim is skipped with a warning."""

        class Nameless:
            api_version = RASTER_DRIVER_API_VERSION

        fake_entry_points(("nameless", Nameless))
        with pytest.warns(RuntimeWarning, match="missing attribute"):
            assert raster_drivers() == {}

    def test_read_collection_capability_requires_the_method(self, fake_entry_points):
        """Declaring read_collection without implementing it is rejected."""

        class NoMethod(GoodDriver):
            capabilities = frozenset({"read", "read_collection"})

        fake_entry_points(("nomethod", NoMethod))
        with pytest.warns(RuntimeWarning, match="does not implement"):
            assert raster_drivers() == {}

    def test_every_capability_token_is_accepted(self, fake_entry_points):
        """A driver may declare every known token when it implements what they promise."""

        class Everything(GoodDriver):
            name = "everything"
            capabilities = frozenset(CAPABILITIES)

            def read_collection(self, source, *, time=None, **kw):
                return None

        fake_entry_points(("everything", Everything))
        assert sorted(raster_drivers()) == ["everything"]

    def test_driver_failing_to_load_is_skipped_with_a_warning(self, monkeypatch):
        """An entry point whose target cannot be resolved costs itself, not the registry."""
        monkeypatch.setattr(_drivers, "_good_target", GoodDriver, raising=False)
        entries = (
            EntryPoint(
                name="broken",
                value="pyramids.dataset._drivers:_missing_target_xyz",
                group=ENTRY_POINT_GROUP,
            ),
            EntryPoint(
                name="fci",
                value="pyramids.dataset._drivers:_good_target",
                group=ENTRY_POINT_GROUP,
            ),
        )
        monkeypatch.setattr(
            _drivers,
            "entry_points",
            lambda *, group=None: entries if group == ENTRY_POINT_GROUP else (),
        )
        monkeypatch.setattr(_drivers, "_CACHE", None)
        with pytest.warns(RuntimeWarning, match="could not be loaded"):
            assert sorted(raster_drivers()) == ["fci"]
        monkeypatch.delattr(_drivers, "_good_target", raising=False)

    def test_duplicate_name_keeps_the_first(self, fake_entry_points):
        """A name registered twice keeps the first and warns about the second."""

        class Other(GoodDriver):
            long_name = "Other"

        fake_entry_points(("a", GoodDriver), ("b", Other))
        with pytest.warns(RuntimeWarning, match="registered twice"):
            drivers = raster_drivers()
        assert list(drivers) == ["fci"]
        assert drivers["fci"].long_name == "EUMETSAT FCI"


class TestExtraGating:
    """A driver is skipped when its declared optional dependency is absent."""

    def test_missing_extra_is_skipped_silently(self, fake_entry_points):
        """An absent extra skips the driver without importing its backend or warning."""

        class NeedsBackend(GoodDriver):
            extra = "no_such_backend_xyz"

        fake_entry_points(("needs", NeedsBackend))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert raster_drivers() == {}

    def test_present_extra_is_loaded(self, fake_entry_points):
        """An importable extra (here the stdlib ``os``) does not gate the driver out."""

        class HasBackend(GoodDriver):
            extra = "os"

        fake_entry_points(("has", HasBackend))
        assert sorted(raster_drivers()) == ["fci"]

    def test_unresolvable_extra_is_treated_as_missing(self, fake_entry_points):
        """An extra naming a missing parent package (find_spec raises) gates the driver out."""

        class NeedsChild(GoodDriver):
            extra = "no_such_parent_xyz.child"

        fake_entry_points(("needs", NeedsChild))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert raster_drivers() == {}


class TestHelpers:
    """The standalone resolver and message helpers."""

    def test_resolve_driver_returns_the_class(self, registered):
        """resolve_driver returns the registered class for a known name."""
        registered(GoodDriver)
        assert _drivers.resolve_driver("fci") is GoodDriver

    def test_resolve_driver_unknown_raises(self, registered):
        """resolve_driver raises for an unregistered name."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            _drivers.resolve_driver("nope")

    def test_unsupported_source_message_names_installed_drivers(self, registered):
        """The message names the source and the installed drivers."""
        registered(GoodDriver)
        message = _drivers.unsupported_source_message("ee://x")
        assert "ee://x" in message
        assert "fci" in message

    def test_unsupported_source_message_on_a_bare_install(self):
        """With nothing installed the message says so."""
        assert "no raster drivers installed" in _drivers.unsupported_source_message("x")

    def test_select_driver_with_an_explicit_name_delegates_to_resolve(self, registered):
        """select_driver with a name returns that driver, bypassing the can_open scan."""
        registered(GoodDriver)
        assert select_driver("anything", "fci") is GoodDriver


class TestDiscovery:
    """Discovery order, the copy contract, and the lookup helpers."""

    def test_discovery_preserves_entry_point_order(self, fake_entry_points):
        """The registry keeps drivers in the order discovery found them."""

        class Second(GoodDriver):
            name = "second"
            suffixes = (".two",)

        fake_entry_points(("fci", GoodDriver), ("second", Second))
        assert list(raster_drivers()) == ["fci", "second"]

    def test_raster_drivers_returns_a_copy(self, registered):
        """Mutating the returned mapping does not reach the registry."""
        registered(GoodDriver)
        drivers = raster_drivers()
        drivers.clear()
        assert sorted(raster_drivers()) == ["fci"]

    def test_a_concurrent_publish_keeps_the_first_result(self, monkeypatch):
        """If another thread publishes between discovery and the lock, the first result wins."""
        first = {"fci": GoodDriver}

        def _discover_then_race():
            # Simulate a second thread finishing discovery and publishing first.
            _drivers._CACHE = first
            return {"other": GoodDriver}

        monkeypatch.setattr(_drivers, "_CACHE", None)
        monkeypatch.setattr(_drivers, "_discover", _discover_then_race)
        result = raster_drivers()
        assert list(result) == ["fci"], "the already-published mapping must win"
        assert result is not first, "callers get a copy, not the live cache"

    def test_drivers_for_suffix_is_case_and_dot_insensitive(self, registered):
        """A suffix resolves with or without its dot and in any case."""
        registered(GoodDriver)
        assert [d.name for d in drivers_for_suffix(".nat")] == ["fci"]
        assert [d.name for d in drivers_for_suffix("NAT")] == ["fci"]
        assert drivers_for_suffix(".tif") == []

    def test_drivers_for_scheme_strips_separator(self, registered):
        """A scheme resolves with or without a trailing ``://``."""

        class EE(GoodDriver):
            name = "earthengine"
            suffixes = ()
            schemes = ("ee",)

        registered(EE)
        assert [d.name for d in drivers_for_scheme("ee")] == ["earthengine"]
        assert [d.name for d in drivers_for_scheme("EE://")] == ["earthengine"]
        assert drivers_for_scheme("s3") == []

    def test_describe_drivers_is_serialisable(self, registered):
        """describe_drivers returns a plain-dict summary per driver."""
        registered(GoodDriver)
        assert describe_drivers() == [
            {
                "name": "fci",
                "long_name": "EUMETSAT FCI",
                "suffixes": [".nat"],
                "schemes": [],
                "capabilities": ["read"],
            }
        ]


class TestCacheColdness:
    """A bare install does no discovery on a normal GDAL read."""

    def test_bare_registry_is_empty(self):
        """With no plugin installed the registry is empty."""
        assert raster_drivers() == {}

    def test_a_normal_gdal_read_never_consults_the_registry(self, tmp_path):
        """A readable raster opens through GDAL and leaves the discovery cache cold."""
        path = tmp_path / "x.tif"
        _dataset().to_file(str(path))
        clear_driver_cache()
        opened = Dataset.read_file(str(path))
        assert opened.band_count == 1
        assert _drivers._CACHE is None


class TestDispatch:
    """``Dataset.read_file`` routing: driver= override, GDAL, then plugin fallback."""

    def test_explicit_driver_bypasses_gdal(self, registered):
        """``driver=`` routes straight to the named plugin without touching GDAL."""
        registered(GoodDriver)
        opened = Dataset.read_file("whatever.nat", driver="fci")
        assert opened.rows == 2

    def test_unknown_driver_raises(self, registered):
        """An unregistered ``driver=`` name is a clear error."""
        registered(GoodDriver)
        with pytest.raises(DriverNotExistError, match="nope"):
            Dataset.read_file("x.nat", driver="nope")

    def test_gdal_failure_falls_back_to_a_claiming_driver(self, registered):
        """A source GDAL cannot open is handed to the first driver whose can_open claims it."""

        class EE(GoodDriver):
            name = "earthengine"
            suffixes = ()
            schemes = ("ee",)

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("ee://")

        registered(EE)
        opened = Dataset.read_file("ee://MODIS/006/MOD13Q1")
        assert opened.rows == 2

    def test_gdal_failure_with_no_claimant_reraises(self, registered):
        """When GDAL fails and no driver claims the source, GDAL's error is re-raised."""
        registered(GoodDriver)  # claims only .nat, not this path
        with pytest.raises((FileNotFoundError, Exception)):
            Dataset.read_file("definitely-not-a-raster.bogus")

    def test_the_scan_does_not_instantiate_a_non_claiming_driver(self, registered):
        """can_open is a classmethod; only the chosen driver is ever constructed."""
        built = {"nonclaim": 0, "claim": 0}

        class NonClaim(GoodDriver):
            name = "nonclaim"
            schemes = ("zzz",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                return False

            def __init__(self):
                built["nonclaim"] += 1

        class Claim(GoodDriver):
            name = "claim"
            schemes = ("ee",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                return str(source).startswith("ee://")

            def __init__(self):
                built["claim"] += 1

        registered(NonClaim, Claim)
        Dataset.read_file("ee://asset")
        assert built == {"nonclaim": 0, "claim": 1}

    def test_a_can_open_that_raises_is_skipped_with_a_warning(self, registered):
        """A driver whose can_open raises is skipped so one bad driver cannot block the scan."""

        class Explodes(GoodDriver):
            name = "explodes"
            schemes = ("ee",)
            suffixes = ()

            @classmethod
            def can_open(cls, source):
                raise RuntimeError("bad can_open")

        registered(Explodes)
        with pytest.warns(RuntimeWarning, match="raised from can_open"):
            with pytest.raises((FileNotFoundError, Exception)):
                Dataset.read_file("ee://asset")

    def test_select_driver_returns_none_on_a_bare_install(self):
        """With nothing registered, the scan claims nothing and the caller falls back to GDAL."""
        assert select_driver("anything.nat", None) is None


class TestConformanceKit:
    """The reusable conformance kit a downstream plugin runs against its driver."""

    def test_a_conforming_driver_has_no_problems(self):
        """A well-formed class passes both static and behavioural checks."""
        assert check_raster_driver(GoodDriver) == []
        assert_raster_driver_conforms(GoodDriver)

    def test_a_static_violation_is_reported_first(self):
        """A contract violation surfaces as the first listed problem."""

        class Bad(GoodDriver):
            capabilities = frozenset()

        assert "required 'read'" in check_raster_driver(Bad)[0]
        with pytest.raises(AssertionError, match="does not conform"):
            assert_raster_driver_conforms(Bad)

    def test_a_can_open_that_raises_is_reported(self):
        """can_open must never raise; a kit probe catches one that does."""

        class Raises(GoodDriver):
            @classmethod
            def can_open(cls, source):
                raise ValueError("no")

        problems = check_raster_driver(Raises)
        assert any("must never raise" in problem for problem in problems)

    def test_a_non_bool_can_open_is_reported(self):
        """can_open must return a bool; a kit probe catches a truthy non-bool."""

        class Truthy(GoodDriver):
            @classmethod
            def can_open(cls, source):
                return "yes"

        problems = check_raster_driver(Truthy)
        assert any("must return a bool" in problem for problem in problems)
