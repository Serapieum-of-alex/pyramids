"""Guard every pinned Python ABI against non-deterministic free-threaded drift.

conda-forge ships two builds of each recent CPython at the same version: ``*_cpNNN`` (standard, GIL)
and ``*_cpNNNt`` (free-threaded / no-GIL, PEP 703). A bare ``python = "3.NN.*"`` lets the conda solver
pick either, and it did so inconsistently per platform — the ``wheel-build`` env picked ``*_cp314t``
on ``osx-64`` / ``win-64`` while the ``py314`` env used ``*_cp314`` everywhere. pyramids has
thread-sensitive code (per-thread GDAL handle management, ``base/_locks.py``), so every version leg
must run one intended ABI (standard GIL). Free-threaded is a separate opt-in leg (#683).

Two complementary guards, because the ABI cannot be constrained the same way everywhere:

- :func:`test_feature_pins_standard_gil_abi` — every pixi *feature* that pins a specific GIL ABI build
  (currently ``py314``, ``py315`` and ``wheel-build``) must pin the standard-GIL ``*_cpNNN`` build for
  its version, never the free-threaded ``*_cpNNNt``. The feature list is derived from the manifest, so
  a newly added version-pinning feature (``py316`` …) is checked automatically.
- :func:`test_lockfile_has_no_free_threaded_python` — the committed ``pixi.lock`` must carry no
  ``cpNNNt`` build of any version on any platform. This is the **universal** backstop: the envs that
  share the ``default`` / ``docs`` solve-groups float Python to the newest with no ABI pin of their own
  and cannot be ABI-constrained at solve time without freezing their version (a shared pin would also
  break the ``py311`` / ``py312`` / ``py313`` envs, which use a bare ``python = "3.NN.*"``). This
  whole-lock scan is therefore what guarantees none of those envs drifts to the free-threaded
  interpreter on the next re-solve, and it also backstops the per-feature guard against manifest forms
  it does not parse (see below).
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_LOCK = REPO_ROOT / "pixi.lock"


def _load_pixi() -> dict:
    """Return the ``[tool.pixi]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["pixi"]


def _python_spec(feature: dict):
    """Return a feature's top-level ``python`` dependency spec (bare string, table, or ``None``).

    Only ``[...dependencies]`` is inspected, not per-platform
    ``target.<platform>.dependencies`` tables. An ABI pin placed under a ``target`` table would be
    missed by the per-feature check, but :func:`test_lockfile_has_no_free_threaded_python` scans the
    whole lock and is the universal backstop for any form this helper does not parse.
    """
    return feature.get("dependencies", {}).get("python")


def _spec_tokens(python) -> list[str]:
    """Tokens of a bare matchspec string, split on whitespace or ``=`` (``"3.15.* *_cp315"`` or
    ``"3.15.*=*_cp315"``)."""
    return [t for t in re.split(r"[ =]+", str(python).strip()) if t] if python else []


def _spec_version(python) -> str:
    """Version half of a pixi python spec — table ``{version=}`` or a bare matchspec string."""
    if isinstance(python, dict):
        return str(python.get("version", ""))
    tokens = _spec_tokens(python)
    return tokens[0] if tokens else ""


def _spec_build(python) -> str:
    """Build half of a pixi python spec (``""`` if the spec carries no build string)."""
    if isinstance(python, dict):
        return str(python.get("build", ""))
    tokens = _spec_tokens(python)
    return tokens[1] if len(tokens) > 1 else ""


def _gil_pinned_features() -> dict[str, str]:
    """Map each feature whose ``python`` spec pins a concrete CPython ABI build to its ``3.NN`` version.

    A feature that pins ``build = "*_cp3NN"`` (``py314``, ``py315``, ``wheel-build``) is asserting one
    interpreter ABI. ``py311``-``py313`` use a bare ``python = "3.NN.*"`` with no build and are covered
    only by the whole-lock free-threaded scan.
    """
    out = {}
    for name, feat in _load_pixi().get("feature", {}).items():
        if re.search(r"cp3\d\d", _spec_build(_python_spec(feat))):
            out[name] = _spec_version(_python_spec(feat))
    return out


def test_known_gil_pinned_features_are_detected():
    """py314, py315 and wheel-build must still be detected as ABI-pinning features."""
    found = set(_gil_pinned_features())
    assert {"py314", "py315", "wheel-build"} <= found, (
        "expected 'py314', 'py315' and 'wheel-build' among the features that pin a CPython ABI build; "
        f"found {sorted(found)}. If a pin was dropped, that version's ABI is no longer constrained."
    )


@pytest.mark.parametrize("feature", sorted(_gil_pinned_features()))
def test_feature_pins_standard_gil_abi(feature: str):
    """Each ABI-pinning feature must pin the standard-GIL ``*_cpNNN`` build for its own version."""
    python = _python_spec(_load_pixi()["feature"][feature])
    version = _spec_version(python)
    minor = re.match(r"3\.(\d+)", version)
    tag = f"cp3{minor.group(1)}" if minor else ""
    # Reject the free-threaded ABI anywhere in the spec (so a cpNNNt pin cannot hide in the version
    # half), then require the standard-GIL cpNNN build that matches the feature's version.
    assert not re.search(r"cp3\d\dt", str(python)), (
        f"feature {feature!r} python spec {python!r} allows a free-threaded 'cpNNNt' ABI."
    )
    assert tag, (
        f"could not derive a cpNNN ABI tag from feature {feature!r} version {version!r}."
    )
    assert tag in _spec_build(python), (
        f"feature {feature!r} python spec {python!r} does not pin the standard-GIL {tag!r} build; a "
        f"bare '{version}' lets the solver pick either ABI."
    )


def test_lockfile_has_no_free_threaded_python():
    """The committed ``pixi.lock`` must carry no free-threaded ``cpNNNt`` build, in any env/platform."""
    lock = PIXI_LOCK.read_text(encoding="utf-8")
    # Any whitespace-delimited token containing a ``cp3NNt`` ABI tag is a free-threaded build — catches
    # both the artifact filenames (``python-3.15.0-..._cp315t.conda``) and the matchspec lines
    # (``python_abi 3.15.* *_cp315t``). A token scan keyed on the regex keeps this linear and
    # version-agnostic (cp313t, cp314t, cp315t, …).
    offenders = sorted(
        {token for token in lock.split() if re.search(r"cp3\d\dt", token)}
    )
    assert not offenders, (
        f"pixi.lock contains free-threaded (cpNNNt) Python builds: {offenders}. Re-pin the ABI in "
        "pyproject.toml and regenerate the lock with `pixi lock`."
    )
