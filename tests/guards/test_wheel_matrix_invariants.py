"""Guard the CPython-version wheel-matrix invariants across ``pyproject.toml`` and the bundle workflow.

The set of CPython versions pyramids ships wheels for is written down in several places that must
agree but are tied together only by prose: the cibuildwheel ``build`` list, the PyPI trove
classifiers, the ``build-winarm64-wheels`` ``CIBW_BUILD`` list, the free-threaded ``skip`` list, and
the release composition gate's ``pyvers`` floor. PR #1322 (adding cp315) showed that a careful author
updating most of them still misses some, and the gate's floor being slack means a release that
silently loses a whole CPython tag would publish as if complete. These guards fail the moment the
next version bump leaves one of those places behind.
"""

import fnmatch
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "bundle-pypi-wheels.yml"


def _cibuildwheel() -> dict:
    """Return the ``[tool.cibuildwheel]`` table from ``pyproject.toml``."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["cibuildwheel"]


def _classifiers() -> set[str]:
    """Return the ``[project].classifiers`` list as a set."""
    with PYPROJECT.open("rb") as fh:
        return set(tomllib.load(fh)["project"]["classifiers"])


def _built_versions() -> list[tuple[int, int]]:
    """Return ``(major, minor)`` for every standard (non-free-threaded) CPython the build selects.

    Parses ``cp311-*`` / ``cp315-*`` style entries in ``[tool.cibuildwheel].build``; a free-threaded
    ``cp314t-*`` entry (should one ever be added) carries a ``t`` and is deliberately not matched.
    """
    versions = []
    for pattern in _cibuildwheel()["build"]:
        match = re.fullmatch(r"cp(\d)(\d+)-\*", pattern)
        if match:
            versions.append((int(match.group(1)), int(match.group(2))))
    return sorted(versions)


def _workflow_text() -> str:
    """Return the bundle workflow as text (it is not valid to parse as a single YAML scalar query)."""
    return WORKFLOW.read_text(encoding="utf-8")


def test_known_versions_are_detected():
    """The build list must still parse to the expected 3.11-3.15 set (guards the parser itself)."""
    assert _built_versions() == [(3, 11), (3, 12), (3, 13), (3, 14), (3, 15)], (
        f"parsed build versions {_built_versions()} do not match the expected cp311-cp315 set; if "
        "the build list changed, update this guard and the siblings that depend on it."
    )


def test_every_built_version_has_a_trove_classifier():
    """Each CPython the wheel build targets must have a matching ``Programming Language`` classifier."""
    classifiers = _classifiers()
    missing = [
        f"{major}.{minor}"
        for major, minor in _built_versions()
        if f"Programming Language :: Python :: {major}.{minor}" not in classifiers
    ]
    assert not missing, (
        f"[tool.cibuildwheel].build ships wheels for Python {missing} but [project].classifiers has "
        "no matching 'Programming Language :: Python :: X.Y' entry, so PyPI would advertise the "
        "wrong supported-version ceiling."
    )


def test_release_gate_floor_matches_the_build_list():
    """The composition gate's ``pyvers`` floor must equal the number of CPython tags built."""
    match = re.search(r'\[ "\$\{pyvers\}" -lt (\d+) \]', _workflow_text())
    assert match, "could not find the `pyvers` floor check in the release composition gate"
    floor = int(match.group(1))
    built = len(_built_versions())
    assert floor == built, (
        f"the release composition gate floors `pyvers` at {floor} but the build list ships {built} "
        "CPython versions; a release run that loses a whole version would still pass the gate. Raise "
        "the floor to match [tool.cibuildwheel].build."
    )


def test_free_threaded_builds_are_skipped_for_every_built_version():
    """For each CPython version built, its free-threaded ``cpXYt`` variant must be excluded by ``skip``."""
    skip = _cibuildwheel()["skip"]
    unskipped = [
        f"cp{major}{minor}t"
        for major, minor in _built_versions()
        if not any(fnmatch.fnmatch(f"cp{major}{minor}t-manylinux_x86_64", pat) for pat in skip)
    ]
    assert not unskipped, (
        f"these free-threaded builds are not excluded by [tool.cibuildwheel].skip: {unskipped}. "
        "pyramids has thread-sensitive code (see tests/guards/test_python_abi_pin.py); exclude them "
        "with explicit cpXYt-* entries or a cp31*t-* glob."
    )


def test_win_arm64_build_list_is_the_matrix_minus_cp311():
    """``build-winarm64-wheels`` must build every CPython the matrix does except cp311.

    This encodes the ``winarm == pyvers - 1`` invariant the composition gate asserts: win_arm64
    omits cp311 (numpy/scipy publish no cp311 ARM64 wheels) and nothing else.
    """
    match = re.search(r'CIBW_BUILD:\s*"([^"]*win_arm64[^"]*)"', _workflow_text())
    assert match, "could not find the win_arm64 CIBW_BUILD line"
    winarm = sorted((int(a), int(b)) for a, b in re.findall(r"cp(\d)(\d+)-win_arm64", match.group(1)))
    expected = [version for version in _built_versions() if version != (3, 11)]
    assert winarm == expected, (
        f"win_arm64 builds {winarm} but the matrix minus cp311 is {expected}; the gate's "
        "`pyvers - 1` assertion and the win_arm64 CIBW_BUILD list must move together."
    )
