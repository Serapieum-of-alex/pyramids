"""Guard tests for ``ci/build-legacy-shim.py`` — the pyramids-gis alias generator.

The shim is what keeps ``pip install pyramids-gis`` working after the rename, so
its generator must keep naming the distribution ``pyramids-gis`` while pinning the
*matching* ``pyramids-core`` version and shipping no importable code. These tests
exercise the generator's pure logic (version read + template render) without
invoking ``python -m build``, so they stay fast and offline.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.core

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    """Import ``ci/build-legacy-shim.py`` by path (``ci`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_legacy_shim", _REPO_ROOT / "ci" / "build-legacy-shim.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reads_core_version_from_pyproject():
    """`_read_core_version` returns the version declared in the core pyproject."""
    version = _load_generator()._read_core_version()
    assert version[0].isdigit() and version.count(".") >= 2


def test_shim_template_pins_the_matching_pyramids_core_version():
    """The rendered shim names pyramids-gis and pins the exact pyramids-core version."""
    gen = _load_generator()
    rendered = gen._SHIM_PYPROJECT.format(version="1.2.3", readme=gen._SHIM_README)
    assert 'name = "pyramids-gis"' in rendered
    assert 'version = "1.2.3"' in rendered
    assert 'dependencies = ["pyramids-core==1.2.3"]' in rendered


def test_shim_declares_no_importable_package():
    """The shim ships no modules/packages, so only pyramids-core owns `pyramids`."""
    gen = _load_generator()
    rendered = gen._SHIM_PYPROJECT.format(version="9.9.9", readme=gen._SHIM_README)
    assert "py-modules = []" in rendered
    assert "[tool.setuptools.packages" not in rendered
