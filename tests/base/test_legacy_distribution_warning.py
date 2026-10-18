"""Tests for the ``pyramids-gis`` legacy-distribution deprecation notice.

After the rename ``pyramids-gis`` -> ``pyramids-core``, ``pyramids/__init__.py``
emits a ``FutureWarning`` when the deprecated ``pyramids-gis`` alias distribution
is detected in the environment (via ``importlib.metadata``). These tests drive
``pyramids._warn_if_legacy_distribution`` directly, monkeypatching the metadata
lookup so the result does not depend on how the test environment was installed.
"""

from importlib.metadata import PackageNotFoundError

import pytest

import pyramids

pytestmark = pytest.mark.core


def test_future_warning_when_legacy_pyramids_gis_installed(monkeypatch):
    """A ``FutureWarning`` fires when ``pyramids-gis`` metadata is present."""

    def fake_version(name):
        if name == "pyramids-gis":
            return "0.69.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(pyramids, "_get_version", fake_version)
    with pytest.warns(FutureWarning, match=r"pyramids-gis.*deprecated"):
        pyramids._warn_if_legacy_distribution()


def test_no_warning_when_only_pyramids_core_installed(monkeypatch, recwarn):
    """No ``FutureWarning`` fires when the legacy alias is absent."""

    def fake_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(pyramids, "_get_version", fake_version)
    pyramids._warn_if_legacy_distribution()
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_warning_names_pyramids_core_as_the_replacement(monkeypatch):
    """The message points users at ``pyramids-core`` and promises discontinuation."""

    def fake_version(name):
        if name == "pyramids-gis":
            return "0.69.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(pyramids, "_get_version", fake_version)
    with pytest.warns(FutureWarning) as record:
        pyramids._warn_if_legacy_distribution()
    message = str(record[0].message)
    assert "pyramids-core" in message
    assert "discontinued" in message
