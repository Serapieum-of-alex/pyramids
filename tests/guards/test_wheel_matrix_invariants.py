"""Guard the CPython-version wheel-matrix invariants across ``pyproject.toml`` and the bundle workflow.

The set of CPython versions pyramids ships wheels for is written down in many places that must agree but
are tied together only by prose. PR #1322 (adding cp315) showed that a careful author updating most of
them still misses some, so these guards fail the moment the next version bump leaves one behind.

Every place the version set is duplicated is guarded here:

- ``[tool.cibuildwheel].build`` — the source of truth (:func:`_built_versions`).
- ``[project].classifiers`` — one ``Programming Language :: Python :: X.Y`` per built version.
- ``[tool.cibuildwheel].skip`` and the ``build-musl-wheels`` job's ``CIBW_SKIP`` — both must exclude the
  free-threaded ``cpXYt`` variant of every built version.
- the release composition gate's ``pyvers`` floor — must equal the number of built versions (kept exact so
  a release that loses a whole tag fails instead of publishing short).
- ``build-winarm64-wheels``'s ``CIBW_BUILD`` and every ``python-version:`` test/verify matrix
  (``test-wheels``, ``test-wheels-musl``, ``verify-winarm64``, ``verify-macos-x86_64``) — each must list
  every built version, minus cp311 on win_arm64 (numpy/scipy publish no cp311 ARM64 wheels).
- the two explicit ``build-musl-wheels`` ``build:`` lists — one musllinux wheel per built version per arch.

The test/verify matrices are the most important to guard: a missing *build* list fails the release
composition gate (the per-platform count would not match ``pyvers``), but a missing *test* entry has **no
backstop** — the wheel builds, the gate passes, and it publishes without ever being installed in CI.
"""

import fnmatch
import re
import tomllib
from pathlib import Path

import yaml

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


def _version_strings(*, drop_cp311: bool = False) -> list[str]:
    """Return built versions as ``"3.11"`` style strings, optionally without 3.11 (the win_arm64 set)."""
    return [
        f"{major}.{minor}"
        for major, minor in _built_versions()
        if not (drop_cp311 and (major, minor) == (3, 11))
    ]


def _workflow_text() -> str:
    """Return the bundle workflow as raw text (for the few regex-parsed checks)."""
    return WORKFLOW.read_text(encoding="utf-8")


def _workflow() -> dict:
    """Return the parsed bundle workflow (``on:`` becomes the YAML bool key ``True``; harmless here)."""
    return yaml.safe_load(_workflow_text())


def _job_matrix_python_versions() -> dict[str, list[str]]:
    """Map each job that declares a ``strategy.matrix.python-version`` to that list."""
    out = {}
    for name, job in _workflow()["jobs"].items():
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        if isinstance(matrix, dict) and "python-version" in matrix:
            out[name] = [str(v) for v in matrix["python-version"]]
    return out


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
        version
        for version in _version_strings()
        if f"Programming Language :: Python :: {version}" not in classifiers
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


def test_free_threaded_builds_are_skipped_everywhere():
    """Both ``skip`` lists (pyproject and the musl job's ``CIBW_SKIP``) must exclude every ``cpXYt``."""
    pyproject_skip = _cibuildwheel()["skip"]
    musl_skip = next(
        step["env"]["CIBW_SKIP"]
        for step in _workflow()["jobs"]["build-musl-wheels"]["steps"]
        if (step.get("env") or {}).get("CIBW_SKIP")
    ).split()
    for label, skip in (
        ("pyproject [tool.cibuildwheel].skip", pyproject_skip),
        ("build-musl-wheels CIBW_SKIP", musl_skip),
    ):
        unskipped = [
            f"cp{major}{minor}t"
            for major, minor in _built_versions()
            if not any(fnmatch.fnmatch(f"cp{major}{minor}t-manylinux_x86_64", pat) for pat in skip)
        ]
        assert not unskipped, (
            f"{label} does not exclude these free-threaded builds: {unskipped}. pyramids has "
            "thread-sensitive code (see test_python_abi_pin.py); exclude them with a cp31*t-* glob."
        )


def test_win_arm64_build_list_is_the_matrix_minus_cp311():
    """``build-winarm64-wheels`` ``CIBW_BUILD`` must build every CPython the matrix does except cp311."""
    match = re.search(r'CIBW_BUILD:\s*"([^"]*win_arm64[^"]*)"', _workflow_text())
    assert match, "could not find the win_arm64 CIBW_BUILD line"
    winarm = sorted((int(a), int(b)) for a, b in re.findall(r"cp(\d)(\d+)-win_arm64", match.group(1)))
    expected = [version for version in _built_versions() if version != (3, 11)]
    assert winarm == expected, (
        f"win_arm64 builds {winarm} but the matrix minus cp311 is {expected}; the gate's "
        "`pyvers - 1` assertion and the win_arm64 CIBW_BUILD list must move together."
    )


def test_test_and_verify_matrices_cover_every_built_version():
    """Every ``python-version:`` test/verify matrix must list each built version (minus cp311 on winarm)."""
    matrices = _job_matrix_python_versions()
    assert matrices, "found no `python-version` matrices in the workflow — parser or workflow changed"
    for job, versions in matrices.items():
        expected = _version_strings(drop_cp311="winarm" in job)
        assert sorted(versions) == sorted(expected), (
            f"job {job!r} tests Python {sorted(versions)} but the wheel build ships "
            f"{sorted(expected)}; a built wheel with no matching test/verify cell publishes untested "
            "(the release composition gate does not backstop a missing test entry)."
        )


def test_musl_build_lists_cover_every_built_version():
    """Each ``build-musl-wheels`` ``build:`` list must name one musllinux wheel per built version."""
    includes = _workflow()["jobs"]["build-musl-wheels"]["strategy"]["matrix"]["include"]
    assert includes, "build-musl-wheels has no matrix include entries"
    for entry in includes:
        versions = sorted(
            (int(a), int(b)) for a, b in re.findall(r"cp(\d)(\d+)-musllinux", entry["build"])
        )
        assert versions == _built_versions(), (
            f"build-musl-wheels arch {entry.get('arch')!r} builds musllinux wheels for {versions} but "
            f"the matrix ships {_built_versions()}; a missing musl build fails the composition gate, "
            "but keep it in step with [tool.cibuildwheel].build regardless."
        )
