# Option 1 Implementation Plan: Self-Contained Wheels via pixi + conda-forge

## Quick Start

**Goal:** Ship `pyramids-gis` as self-contained platform wheels on PyPI,
where `pip install pyramids-gis` works with no pre-installed GDAL.

**Approach:** Use pixi to install pre-built GDAL + native deps from
conda-forge inside the manylinux build container. Extract the shared
libraries and data files, build the GDAL SWIG Python bindings against
them for each target Python version, vendor everything into the
pyramids package, and let auditwheel/delocate/delvewheel bundle the
native libraries into the wheel.

**Why this instead of from-source?** See
[`build-strategy-alternatives.md`](./build-strategy-alternatives.md)
for the analysis. TL;DR: 6–10 min iteration (vs 45–60 min), no OOM
crashes, reproducible via `pixi.lock`, one toolchain across dev / CI /
wheel-build.

**Current status:** Plan only. No code changes yet. Previous
from-source approach on branch `build/bundle-gdal-linux` is being
superseded by this plan.

**Effort estimate:**

| Phase | Name | Duration | Blocking? |
|-------|------|----------|-----------|
| 1 | Linux POC | 1–2 days | Yes |
| 2 | Multi-Python (3.11/3.12/3.13) | 0.5 day | Can run with 3 |
| 3 | Test suite validation | 1–2 days | Can run with 4, 5 |
| 4 | macOS (x86_64 + arm64) | 2–3 days | Independent |
| 5 | Windows | 2–3 days | Independent |
| 6 | Stabilization + release | 2–3 days | Last |
| **Total** | | **9–14 days** | |

Phase 1 must finish first. Phases 2–5 can proceed in parallel once
Phase 1 proves the architecture.

---

## Architecture Overview

### Build Pipeline

```
cibuildwheel kicks off (ubuntu runner or local Docker)
   │
   ▼
CIBW_BEFORE_ALL  (runs ONCE per platform/OS — not per Python version)
   │   bash ci/setup-gdal-from-pixi.sh
   │     1. curl pixi installer → PIXI_HOME=/usr/local
   │     2. pixi install -e wheel-build --frozen
   │          ↓
   │          Installs the pinned set from pixi.lock into
   │          ./.pixi/envs/wheel-build/:
   │              lib/libgdal.so.*         bin/gdal-config
   │              lib/libproj.so.*         bin/swig
   │              lib/libgeos.so.*         include/gdal.h
   │              lib/libtiff.so.*         include/proj.h
   │              lib/libhdf5.so.*         share/gdal/*
   │              lib/libnetcdf.so.*       share/proj/*
   │              ... (~25 native libs)
   │     3. rsync libs + headers + share/ + build tools into /usr/local
   │     4. strip --strip-unneeded all .so files
   │
   ▼
CIBW_BEFORE_BUILD  (runs per Python version — cp311, cp312, cp313)
   │   PACKAGE_DATA=1 python ci/install-and-vendor-osgeo.py
   │     1. pip download --no-binary :all: GDAL==${GDAL_VERSION}
   │     2. pip install ./GDAL-X.Y.Z/ against our extracted libgdal
   │          (--include-dirs=/usr/local/include
   │           --library-dirs=/usr/local/lib)
   │          Produces osgeo/ in the target Python's site-packages,
   │          with C extensions compiled for that Python's ABI.
   │     3. Copy osgeo/ → src/pyramids/_vendor/osgeo/
   │     4. Copy /usr/local/share/gdal → src/pyramids/_data/gdal_data/
   │     5. Copy /usr/local/share/proj → src/pyramids/_data/proj_data/
   │
   ▼
CIBW_BUILD
   │   pip wheel . -w /tmp/wheel-intermediate/
   │     setuptools packages pyramids + _vendor/ + _data/ as usual.
   │
   ▼
CIBW_REPAIR_WHEEL_COMMAND
   │   Linux:   LD_LIBRARY_PATH=/usr/local/lib auditwheel repair
   │   macOS:   DYLD_LIBRARY_PATH=/usr/local/lib delocate-wheel
   │   Windows: delvewheel repair --add-path C:/.../lib
   │
   │   Walks the _vendor/osgeo/*.so dep graph, copies libgdal.so and
   │   transitive deps into pyramids.libs/, patches RPATH/@loader_path/
   │   DLL imports so the extensions find their bundled libraries.
   │
   ▼
CIBW_TEST_COMMAND
       python -c "from osgeo import gdal; print(gdal.__version__)"
       python -c "import pyramids; print(pyramids.__version__)"
```

### Wheel Layout (Linux, after auditwheel repair)

```
pyramids_gis-X.Y.Z-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
├── pyramids/
│   ├── __init__.py                     # runtime bootstrap
│   ├── dataset.py, multidataset.py, ...
│   ├── _vendor/
│   │   ├── __init__.py
│   │   └── osgeo/
│   │       ├── __init__.py
│   │       ├── gdal.py, ogr.py, osr.py, ...
│   │       └── _gdal.cpython-312-*.so  # links to libgdal.so.36
│   └── _data/
│       ├── gdal_data/gcs.csv, gt_datum.csv, ...
│       └── proj_data/proj.db, *.tif
└── pyramids_gis.libs/                  # auditwheel-bundled deps
    ├── libgdal-<hash>.so.36
    ├── libproj-<hash>.so.25
    ├── libgeos-<hash>.so.3.14.1
    ├── libtiff-<hash>.so.6
    ├── libhdf5-<hash>.so.310
    ├── libnetcdf-<hash>.so.22
    ├── libcurl-<hash>.so.4
    ├── libssl-<hash>.so.3, libcrypto-<hash>.so.3
    └── ... (~20–25 deps)
```

Expected compressed wheel size: **80–120 MB**.

---

## Phase 1: Linux Proof of Concept

**Goal:** Produce a working `cp312-manylinux_2_17_x86_64` wheel that
installs and imports cleanly in `python:3.12-slim` with no GDAL
pre-installed. **This phase is the gate** — if the architecture
doesn't work on Linux, it won't work on macOS or Windows either.

**Acceptance criterion for Phase 1:**

```bash
docker run --rm -v $(pwd)/wheelhouse:/w python:3.12-slim bash -c '
    pip install /w/pyramids_gis-*.whl &&
    python -c "
import pyramids
from osgeo import gdal, ogr, osr
print(f\"pyramids {pyramids.__version__}\")
print(f\"GDAL {gdal.__version__}\")
from osgeo import osr
sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)
assert sr.GetAttrValue(\"AUTHORITY\", 1) == \"4326\", \"EPSG lookup failed\"
print(\"EPSG lookup OK\")
"
'
```

### Task 1.1 — Add `wheel-build` pixi environment to `pyproject.toml`

**Why:** Keeping wheel-build dependencies in pixi (not a separate conda
command) means:
- Pinned versions live in `pixi.lock` → reproducible builds
- Same mechanism dev uses, so GDAL version is consistent
- We can test locally via `pixi install -e wheel-build` before touching
  cibuildwheel

**Actions:**

1. Open `pyproject.toml`.
2. Add a new feature block after the existing feature blocks:
   ```toml
   [tool.pixi.feature.wheel-build.dependencies]
   gdal = ">=3.12,<3.13"
   libgdal-netcdf = ">=3.12,<3.13"
   libgdal-hdf4 = ">=3.12,<3.13"
   swig = ">=4.3,<5"
   ```
3. Add a new environment entry in the `[tool.pixi.environments]` block:
   ```toml
   wheel-build = { features = ["wheel-build"], solve-group = "wheel-build" }
   ```
4. Commit to a new working branch: `git checkout -b build/option-1-pixi`.

**Acceptance criteria:**
- `pixi info` shows `wheel-build` as an available environment.
- `pixi install -e wheel-build` succeeds locally, creates
  `.pixi/envs/wheel-build/`.
- `pixi.lock` is updated with pinned versions for wheel-build.

**Failure modes:**
- If conda-forge can't resolve the pin, relax to `gdal = "*"` and pin
  the minor version via `pixi.lock` instead.
- If `solve-group` conflicts with existing groups, remove `solve-group`
  (optional; used for speed).

**Time:** 15–30 min.

---

### Task 1.2 — Verify pixi wheel-build env locally

**Why:** Before touching Docker/cibuildwheel, we need to know the exact
layout of what pixi installs. The paths we reference in
`setup-gdal-from-pixi.sh` depend on this.

**Actions:**

1. `pixi install -e wheel-build` on your local machine (Windows OK —
   the layout is similar across OSes).
2. Inspect:
   ```bash
   ENV=.pixi/envs/wheel-build
   ls -la $ENV/lib/libgdal* $ENV/lib/libproj* $ENV/lib/libgeos* \
          $ENV/lib/libtiff* $ENV/lib/libhdf* $ENV/lib/libnetcdf*
   ls $ENV/include/ | grep -E '^(gdal|proj|geos|tiff|hdf5|netcdf)'
   ls $ENV/share/gdal/ | head -10
   ls $ENV/share/proj/ | head -10
   ls $ENV/bin/gdal-config $ENV/bin/swig
   ```
3. Note the exact filenames — on Linux, libraries typically appear as:
   - `libgdal.so.36` (main binary)
   - `libgdal.so` (symlink)
   - `libgdal.so.3.12.1` (versioned symlink)
4. Record any surprises (e.g., `lib64/` instead of `lib/`, or a
   `share/licenses/` layer we should skip).

**Acceptance criteria:**
- All core libs present (gdal, proj, geos, tiff, hdf5, hdf4, netcdf, curl).
- `$ENV/bin/gdal-config --version` prints the expected version.
- `$ENV/bin/swig -version` prints the SWIG version.
- GDAL_DATA (share/gdal) and PROJ_DATA (share/proj) directories exist
  and are populated.

**Failure modes:**
- Missing `libgdal-hdf4`: conda-forge may require opting in. Verify the
  feature dependency spec.
- Linux vs Windows: the same pixi config works, but on Windows we'd see
  `bin/gdal.dll` instead of `lib/libgdal.so`. This is expected — Phase
  5 handles Windows separately.

**Time:** 15–30 min.

---

### Task 1.3 — Design `ci/setup-gdal-from-pixi.sh`

**Why:** This script runs inside the manylinux2014 container during
`CIBW_BEFORE_ALL`. It must:
- Install pixi (not present in the base image)
- Run `pixi install -e wheel-build --frozen` to get deterministic deps
- Extract the installed binaries into a location where the pyramids
  build can find them (we'll use `${BUILD_PREFIX:=/usr/local}`)
- Work even if the manylinux container has no previous state

The `--frozen` flag is critical: it forces pixi to use `pixi.lock`
exactly, failing if the lock is stale. This is what makes the build
reproducible across local and CI runs.

**Actions:**

Write `ci/setup-gdal-from-pixi.sh`:

```bash
#!/bin/bash
#
# Install GDAL + native dependencies via pixi (conda-forge), then extract
# the shared libraries, headers, and data files into ${BUILD_PREFIX} so
# downstream steps (vendor-osgeo.py, setuptools, auditwheel) can find
# them without knowing about pixi's internal layout.
#
# Runs once per cibuildwheel platform invocation (not per Python version).
#
set -euo pipefail

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"

# --- 1. Install pixi (~5 seconds) ---
if ! command -v pixi >/dev/null 2>&1; then
    echo "=== Installing pixi ==="
    export PIXI_HOME="${BUILD_PREFIX}"
    export PIXI_NO_PATH_UPDATE=1
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${BUILD_PREFIX}/bin:${PATH}"
fi
pixi --version

# --- 2. Install wheel-build env (~90 seconds first time; cached after) ---
echo "=== Resolving wheel-build environment ==="
pixi install -e wheel-build --frozen

PIXI_ENV="$(pwd)/.pixi/envs/wheel-build"
if [ ! -d "${PIXI_ENV}" ]; then
    echo "ERROR: ${PIXI_ENV} does not exist after pixi install" >&2
    exit 1
fi

# --- 3. Extract into ${BUILD_PREFIX} ---
echo "=== Extracting native artifacts from ${PIXI_ENV} ==="
mkdir -p "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/include" \
         "${BUILD_PREFIX}/share" "${BUILD_PREFIX}/bin"

# Shared libraries — preserve symlinks (-a flag)
cp -a "${PIXI_ENV}/lib/"*.so* "${BUILD_PREFIX}/lib/" 2>/dev/null || true
# Some conda packages install to lib64 on Linux
if [ -d "${PIXI_ENV}/lib64" ]; then
    cp -a "${PIXI_ENV}/lib64/"*.so* "${BUILD_PREFIX}/lib64/" 2>/dev/null || true
fi

# Headers — rsync-style copy of everything under include/
cp -a "${PIXI_ENV}/include/." "${BUILD_PREFIX}/include/"

# GDAL_DATA + PROJ_DATA
cp -a "${PIXI_ENV}/share/gdal" "${BUILD_PREFIX}/share/"
cp -a "${PIXI_ENV}/share/proj" "${BUILD_PREFIX}/share/"

# Build tools needed downstream
for tool in gdal-config swig ogrinfo gdalinfo; do
    if [ -f "${PIXI_ENV}/bin/${tool}" ]; then
        cp "${PIXI_ENV}/bin/${tool}" "${BUILD_PREFIX}/bin/"
    fi
done

# Ensure pkg-config paths are discoverable
if [ -d "${PIXI_ENV}/lib/pkgconfig" ]; then
    mkdir -p "${BUILD_PREFIX}/lib/pkgconfig"
    cp "${PIXI_ENV}/lib/pkgconfig/"*.pc "${BUILD_PREFIX}/lib/pkgconfig/" 2>/dev/null || true
fi

# --- 4. Strip debug symbols to reduce wheel size ---
echo "=== Stripping shared libraries ==="
find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f \
    -exec strip --strip-unneeded {} + 2>/dev/null || true

# --- 5. Diagnostic output ---
echo "=== Setup complete ==="
echo "GDAL version: $("${BUILD_PREFIX}/bin/gdal-config" --version)"
echo "PROJ version: $(pkg-config --modversion proj 2>/dev/null || echo 'unknown')"
echo "Total .so files bundled: $(find "${BUILD_PREFIX}/lib" -name '*.so*' | wc -l)"
du -sh "${BUILD_PREFIX}/lib"
```

**Acceptance criteria:**
- Script is syntactically valid bash (`bash -n ci/setup-gdal-from-pixi.sh`).
- Line endings are LF (not CRLF — see `.gitattributes` rule from the
  previous attempt).
- `chmod +x ci/setup-gdal-from-pixi.sh` applied.

**Failure modes:**
- pixi installer URL changes → pin to a specific pixi release
  (e.g., `https://github.com/prefix-dev/pixi/releases/download/v0.38.0/pixi-x86_64-unknown-linux-musl.tar.gz`).
- manylinux2014 lacks curl (`curl -V` should succeed — it's in the
  base image).
- pixi's default install path conflicts with the mounted project dir
  — that's why we use `PIXI_NO_PATH_UPDATE=1`.

**Time:** 1–2 hours.

---

### Task 1.4 — Smoke-test the setup script in a manylinux container

**Why:** Running this inside cibuildwheel takes minutes per iteration
(container startup, project copy, etc.). Running it in a bare
manylinux container directly — before wiring cibuildwheel — lets us
iterate on the script in seconds.

**Actions:**

```bash
# From a bash shell (WSL2 / Git Bash):
cd C:/gdrive/algorithms/gis/pyramids

docker run --rm -it -v "$(pwd):/project" -w /project \
    quay.io/pypa/manylinux2014_x86_64 \
    bash ci/setup-gdal-from-pixi.sh
```

Then, on success, drop into the container interactively and verify:

```bash
docker run --rm -it -v "$(pwd):/project" -w /project \
    quay.io/pypa/manylinux2014_x86_64 \
    bash -c "bash ci/setup-gdal-from-pixi.sh && \
             /usr/local/bin/gdal-config --version && \
             ls /usr/local/lib/libgdal.so* && \
             ls /usr/local/share/gdal | head && \
             ls /usr/local/share/proj | head"
```

**Acceptance criteria:**
- Script exits 0.
- `gdal-config --version` returns expected version.
- `/usr/local/lib/libgdal.so.36` exists (or equivalent major).
- `/usr/local/share/gdal/gcs.csv` exists.
- `/usr/local/share/proj/proj.db` exists.

**Failure modes:**
- `pixi install` fails: likely the manylinux container can't reach
  conda-forge. Retry or check network.
- Symlink errors (`ln: File exists`): use `cp -a --remove-destination`.
- `strip: not recognized format`: benign — the `|| true` at the end
  swallows it.

**Time:** 1–2 hours (iteration on the script).

---

### Task 1.5 — Update `ci/vendor-osgeo.py` to install GDAL bindings

**Why:** The existing `ci/vendor-osgeo.py` assumes `osgeo` is already
importable. With conda-forge binaries, we've extracted **libgdal** but
NOT the **Python SWIG bindings** — those aren't in the conda `gdal`
package (they're in `python-gdal` or provided via pip).

We need to explicitly install the GDAL Python package (matching our
libgdal version) against the extracted libraries, for the current
Python interpreter.

**Actions:**

Create a new script `ci/install-and-vendor-osgeo.py`:

```python
"""Install GDAL Python bindings against the pixi-extracted libgdal, then
vendor the resulting osgeo package into src/pyramids/_vendor/.

Runs in CIBW_BEFORE_BUILD (once per target Python version).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def install_gdal_python_bindings() -> None:
    """pip install GDAL==$GDAL_VERSION linked against /usr/local/."""
    gdal_version = os.environ["GDAL_VERSION"]
    build_prefix = os.environ.get("BUILD_PREFIX", "/usr/local")
    cmd = [
        sys.executable, "-m", "pip", "install",
        f"GDAL=={gdal_version}",
        "--no-build-isolation",
        "--no-cache-dir",
        "--global-option=build_ext",
        f"--global-option=--include-dirs={build_prefix}/include",
        f"--global-option=--library-dirs={build_prefix}/lib",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def vendor_osgeo_into_package() -> None:
    """Copy osgeo module and GDAL/PROJ data files into src/pyramids/."""
    repo_root = Path(__file__).resolve().parent.parent
    build_prefix = Path(os.environ.get("BUILD_PREFIX", "/usr/local"))
    src_pyramids = repo_root / "src" / "pyramids"

    # Locate the installed osgeo package
    import osgeo
    osgeo_src = Path(osgeo.__file__).parent

    vendor_dst = src_pyramids / "_vendor" / "osgeo"
    if vendor_dst.exists():
        shutil.rmtree(vendor_dst)
    vendor_dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {osgeo_src} -> {vendor_dst}", flush=True)
    shutil.copytree(osgeo_src, vendor_dst)

    # Make _vendor a package
    (vendor_dst.parent / "__init__.py").touch()

    # Copy GDAL_DATA
    gdal_data_src = build_prefix / "share" / "gdal"
    gdal_data_dst = src_pyramids / "_data" / "gdal_data"
    if gdal_data_dst.exists():
        shutil.rmtree(gdal_data_dst)
    gdal_data_dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {gdal_data_src} -> {gdal_data_dst}", flush=True)
    shutil.copytree(gdal_data_src, gdal_data_dst)

    # Copy PROJ_DATA
    proj_data_src = build_prefix / "share" / "proj"
    proj_data_dst = src_pyramids / "_data" / "proj_data"
    if proj_data_dst.exists():
        shutil.rmtree(proj_data_dst)
    print(f"Copying {proj_data_src} -> {proj_data_dst}", flush=True)
    shutil.copytree(proj_data_src, proj_data_dst)

    print("Vendor step complete.", flush=True)


def main() -> None:
    if os.environ.get("PACKAGE_DATA") != "1":
        print("PACKAGE_DATA != 1; skipping install+vendor.", flush=True)
        return
    install_gdal_python_bindings()
    vendor_osgeo_into_package()


if __name__ == "__main__":
    main()
```

Then delete the old `ci/vendor-osgeo.py` (or keep it as a subroutine —
but the standalone script is simpler).

**Acceptance criteria:**
- `python -c "import ast; ast.parse(open('ci/install-and-vendor-osgeo.py').read())"`
  passes syntax check.
- Running `PACKAGE_DATA=1 python ci/install-and-vendor-osgeo.py`
  inside a manylinux container with `setup-gdal-from-pixi.sh` already
  run produces `src/pyramids/_vendor/osgeo/` with both `.py` files
  and a compiled `_gdal.cpython-*.so`.

**Failure modes:**
- `GDAL==3.12.1` pip package not matching the installed libgdal — the
  SWIG bindings are version-locked to libgdal. Keep `GDAL_VERSION` env
  var matching what conda-forge delivered (read from `gdal-config --version`).
- `setuptools not found`: the target Python (cp3XX in
  `/opt/python/cp3XX-cp3XX/bin/python`) needs `setuptools` and
  `wheel`. Add a `pip install setuptools wheel numpy` step before.

**Time:** 1–2 hours.

---

### Task 1.6 — Wire cibuildwheel config in `pyproject.toml`

**Why:** Replace the existing from-source cibuildwheel config with the
pixi-based version. Uses `CIBW_BEFORE_ALL` for one-time native setup
and `CIBW_BEFORE_BUILD` for per-Python bindings.

**Actions:**

Edit `pyproject.toml`, replace the `[tool.cibuildwheel.linux]` block:

```toml
[tool.cibuildwheel]
skip = ["*pp*", "*musllinux*"]
archs = ["auto64"]
build-verbosity = 3
test-command = [
    "python -c \"import pyramids; print(f'pyramids {pyramids.__version__}')\"",
    "python -c \"from osgeo import gdal; print(f'GDAL {gdal.__version__}')\"",
]

[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux2014"
before-all = [
    "bash ./ci/setup-gdal-from-pixi.sh",
]
before-build = [
    "pip install setuptools wheel numpy",
    "PACKAGE_DATA=1 python ./ci/install-and-vendor-osgeo.py",
]
repair-wheel-command = [
    "LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 auditwheel repair -w {dest_dir} {wheel}",
]

[tool.cibuildwheel.linux.environment]
GDAL_VERSION = "3.12.1"  # must match what conda-forge delivered
BUILD_PREFIX = "/usr/local"
GDAL_DATA = "/usr/local/share/gdal"
PROJ_DATA = "/usr/local/share/proj"
PROJ_LIB = "/usr/local/share/proj"
PACKAGE_DATA = "1"
LD_LIBRARY_PATH = "/usr/local/lib:/usr/local/lib64"
PIXI_NO_PATH_UPDATE = "1"
```

**Acceptance criteria:**
- `cibuildwheel --print-build-identifiers --platform linux` still lists
  the expected build targets (cp311, cp312, cp313 × manylinux_x86_64).
- `cibuildwheel` errors-out cleanly with a message if the config is
  wrong (no silent misbehavior).

**Failure modes:**
- Mismatched `GDAL_VERSION` env var vs what conda-forge installed: the
  `pip install GDAL==X.Y.Z` step will fail. Read the actual version
  from `gdal-config --version` dynamically in the python script, OR
  pin both sides together.

**Time:** 30 min.

---

### Task 1.7 — Verify pyramids runtime bootstrap still works

**Why:** `src/pyramids/__init__.py` already has the bootstrap that adds
`_vendor/` to `sys.path` and sets `GDAL_DATA` / `PROJ_DATA`. We need to
ensure:
- It still runs before any `from osgeo import` statement.
- It correctly detects the presence of `_vendor/osgeo/`.
- It correctly sets `PROJ_LIB` (older PROJ versions required this env
  var instead of `PROJ_DATA`).

**Actions:**

1. Open `src/pyramids/__init__.py`.
2. Verify the bootstrap block is at the top, before
   `from pyramids.base.config import Config`.
3. Verify it sets both `GDAL_DATA` and `PROJ_DATA` AND `PROJ_LIB`
   (belt-and-suspenders for older PROJ versions).
4. Add a small diagnostic print (behind an env var) that we can enable
   during debugging:
   ```python
   if _os.environ.get("PYRAMIDS_DEBUG_BOOTSTRAP"):
       print(f"[pyramids] _vendor/osgeo present: {_vendored_osgeo.is_dir()}")
       print(f"[pyramids] GDAL_DATA: {_os.environ.get('GDAL_DATA')}")
       print(f"[pyramids] PROJ_DATA: {_os.environ.get('PROJ_DATA')}")
   ```
5. Save.

**Acceptance criteria:**
- In a venv where `_vendor/osgeo/` exists, `import pyramids` then
  `from osgeo import gdal` resolves to the vendored copy:
  ```python
  import osgeo
  assert "_vendor" in osgeo.__file__, f"Got {osgeo.__file__}"
  ```
- `gdal.SpatialReference().ImportFromEPSG(4326)` succeeds (proves
  `PROJ_DATA`/`GDAL_DATA` point somewhere readable).

**Failure modes:**
- `_vendor/osgeo/__init__.py` present but extensions fail to load:
  likely missing RPATH patching (auditwheel handles this — only a risk
  if testing without repair).
- Import order: if any earlier module already imported `osgeo`,
  vendored path injection comes too late. Verify no `import osgeo` in
  modules that `__init__.py` imports eagerly.

**Time:** 30 min.

---

### Task 1.8 — Archive the from-source build script

**Why:** `ci/build-gdal-linux.sh` represents ~6 hours of work and is
90% working. Keep it as a fallback and for reference, but mark it as
deprecated so no one accidentally uses it going forward.

**Actions:**

1. Add a deprecation header to `ci/build-gdal-linux.sh`:
   ```bash
   #!/bin/bash
   #
   # =============================================================
   # DEPRECATED AS OF 2026-04-13
   # =============================================================
   # This from-source build approach has been superseded by
   # ci/setup-gdal-from-pixi.sh, which uses conda-forge binaries
   # via pixi. See planning/bundle/build-strategy-alternatives.md
   # for the rationale.
   #
   # Kept as a reference and as a fallback if pixi/conda-forge
   # becomes insufficient (e.g. a required feature isn't packaged).
   # =============================================================
   ```
2. Do NOT delete it yet — Phase 6 will decide final archival.

**Acceptance criteria:** Header present, script still syntactically valid.

**Time:** 10 min.

---

### Task 1.9 — Move `ci/vendor-osgeo.py` to an importable module

**Why:** The old `ci/vendor-osgeo.py` is now a sub-step inside
`ci/install-and-vendor-osgeo.py`. Either:

- **Option A:** Inline the vendor logic into
  `install-and-vendor-osgeo.py` (simpler — one script).
- **Option B:** Keep `vendor-osgeo.py` and have the new script call it
  via `subprocess` or `import`.

Pick **Option A** for simplicity — the logic is ~40 lines.

**Actions:**

1. Ensure the vendoring logic lives inside
   `ci/install-and-vendor-osgeo.py::vendor_osgeo_into_package()`.
2. Delete `ci/vendor-osgeo.py`.

**Acceptance criteria:** Only one "vendor" script exists under `ci/`.

**Time:** 10 min.

---

### Task 1.10 — Run cibuildwheel for cp312 locally

**Why:** First full end-to-end test of the new pipeline. If this
succeeds, Phase 1 is essentially done.

**Actions:**

```bash
cd C:/gdrive/algorithms/gis/pyramids

# Clean slate
rm -rf wheelhouse/ build/ dist/ src/pyramids/_vendor/ src/pyramids/_data/
mkdir -p wheelhouse

# Build cp312 only
CIBW_BUILD="cp312-manylinux_x86_64" \
  pixi run -e dev cibuildwheel --platform linux --output-dir wheelhouse . \
  2>&1 | tee wheelhouse/build.log
```

**Acceptance criteria:**
- Exit code 0.
- `wheelhouse/pyramids_gis-*-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  exists.
- Wheel size between 40–150 MB.

**Failure modes:** Many possible — see Task 1.11 debug playbook.

**Time:** First-run: 15–30 min. Subsequent iteration: 5–10 min.

---

### Task 1.11 — Debug any first-build failures

**Why:** Likely issues on first run, with remedies:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pixi: command not found` | Installer didn't add to PATH | Verify `PATH=${BUILD_PREFIX}/bin:${PATH}` is set in the script |
| `pixi install` times out | Network from inside container | Try again; check if manylinux2014 has DNS issues |
| `ERROR: Frozen install: lockfile out of date` | pixi.lock doesn't cover `wheel-build` | Run `pixi lock` on host; re-commit |
| `gdal-config not found` | pixi installs it to `libexec/` or a sub-dir | `find .pixi -name gdal-config` and adjust cp path |
| `libgdal.so: cannot open shared object` during SWIG binding install | LD_LIBRARY_PATH missing | Already set in `[environment]`; verify with `echo $LD_LIBRARY_PATH` |
| GDAL `pip install` fails with "setuptools not found" | Target Python lacks build deps | Add `pip install setuptools wheel` to `before-build` |
| `osgeo.__file__` points to `/opt/python/...`, not `_vendor/` | Wheel test runs before vendor — OR sys.path bootstrap not wired | Verify `pyramids/__init__.py` bootstrap + `_vendor/` is in wheel |
| auditwheel error: "cannot find libgdal.so" | `/usr/local/lib` not in `LD_LIBRARY_PATH` when repair runs | Repair command already has `LD_LIBRARY_PATH=/usr/local/lib` — verify |
| auditwheel policy violation: `CXXABI_1.3.13 not found` | Newer C++ stdlib than manylinux2014 supports | Upgrade to `manylinux_2_28` — change `manylinux-x86_64-image` |

**Actions:** Iterate on each failure one at a time.

**Time:** 2–6 hours of debug work.

---

### Task 1.12 — Verify wheel layout with `zipfile -l`

**Why:** Before testing the wheel in a clean container, confirm the
files we expect are actually in it.

**Actions:**

```bash
WHEEL=wheelhouse/pyramids_gis-*.whl

python -m zipfile -l $WHEEL | grep -E 'osgeo/(gdal|ogr|osr)\.py'
# Expect: 3 matches (pure-Python SWIG wrappers)

python -m zipfile -l $WHEEL | grep -E 'osgeo/_(gdal|ogr|osr).*\.so'
# Expect: 3+ matches (C extensions for cp312)

python -m zipfile -l $WHEEL | grep -E '_data/gdal_data/gcs\.csv'
# Expect: 1 match

python -m zipfile -l $WHEEL | grep -E '_data/proj_data/proj\.db'
# Expect: 1 match

python -m zipfile -l $WHEEL | grep -E '\.libs/libgdal'
# Expect: libgdal-<hash>.so.36 (auditwheel-bundled)

python -m zipfile -l $WHEEL | wc -l
# Expect: 500-2000 files (depending on data file count)
```

**Acceptance criteria:** All 5 checks pass.

**Failure modes:**
- Missing `osgeo/*.py`: vendor step didn't run or copied to wrong path.
  Check `src/pyramids/_vendor/osgeo/` exists after `CIBW_BEFORE_BUILD`.
- Missing `_data/`: GDAL_DATA/PROJ_DATA not in `package-data` globs
  (verify `pyproject.toml` has `_data/gdal_data/*` and `_data/proj_data/*`).
- Missing `.libs/libgdal`: auditwheel didn't bundle. Rerun repair with
  verbose: `auditwheel -v repair wheel.whl`.

**Time:** 15 min.

---

### Task 1.13 — Smoke-test in clean `python:3.12-slim`

**Why:** Final Phase 1 gate. The wheel must install and import in an
environment that has NO system GDAL.

**Actions:**

```bash
WHEEL=$(ls wheelhouse/pyramids_gis-*cp312*.whl)

docker run --rm -v "$(pwd)/wheelhouse:/wh" python:3.12-slim bash -c '
    pip install /wh/'"$(basename $WHEEL)"' &&
    python << PY
import pyramids
from osgeo import gdal, ogr, osr
print("pyramids:", pyramids.__version__)
print("GDAL:", gdal.__version__)

# EPSG lookup — exercises PROJ_DATA
sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)
auth = sr.GetAttrValue("AUTHORITY", 1)
assert auth == "4326", f"Expected 4326, got {auth}"
print("EPSG 4326 lookup: OK")

# Raster creation — exercises GDAL_DATA
driver = gdal.GetDriverByName("MEM")
ds = driver.Create("", 10, 10, 1, gdal.GDT_Byte)
ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
ds.SetProjection(sr.ExportToWkt())
band = ds.GetRasterBand(1)
band.Fill(42)
arr = band.ReadAsArray()
assert arr.shape == (10, 10) and (arr == 42).all()
print("MEM raster round-trip: OK")

# osgeo resolves from vendored copy
import osgeo
assert "_vendor" in osgeo.__file__, f"Not from vendor: {osgeo.__file__}"
print("osgeo from _vendor: OK")

print("ALL CHECKS PASSED")
PY
'
```

**Acceptance criteria:** Script prints `ALL CHECKS PASSED`.

**Failure modes:**
- `cannot import _gdal`: RPATH not patched by auditwheel. Inspect with
  `readelf -d _vendor/osgeo/_gdal.*.so` → should show `RPATH $ORIGIN/...`.
- `EPSG 4326 lookup failed`: `PROJ_DATA` points to a missing dir.
  Re-check bootstrap in `__init__.py`.
- `osgeo from /usr/local/lib/python3.12/site-packages/osgeo`: vendor
  copy lost the sys.path race. Verify bootstrap runs before any other
  module imports osgeo.

**Time:** 30 min - 2 hours depending on failure.

---

### Phase 1 Deliverable — summary commit

Once all Task 1.x items are Green:

- `pyproject.toml` — wheel-build feature + cibuildwheel config
- `ci/setup-gdal-from-pixi.sh` — new
- `ci/install-and-vendor-osgeo.py` — new
- `ci/vendor-osgeo.py` — deleted (inlined into above)
- `ci/build-gdal-linux.sh` — deprecation header added
- `src/pyramids/__init__.py` — bootstrap verified (maybe minor edits)
- `.pixi/envs/wheel-build/` — not committed (in `.gitignore` via `.pixi`)
- `pixi.lock` — updated

Commit message: `1: Phase 1 — Linux POC with pixi/conda-forge bundling`.

**Phase 1 done criterion:** `wheelhouse/*.whl` imports and passes the
smoke test in `python:3.12-slim` with no pre-installed GDAL.

---

## Phase 2: Multi-Python Coverage

**Goal:** Produce wheels for cp311, cp312, cp313 (and cp314 once
Python 3.14 ships GA).

### Task 2.1 — Enable multi-Python matrix in cibuildwheel

**Why:** Each Python version needs its own SWIG binding build
(different C API). The `CIBW_BEFORE_BUILD` step handles this — it's
already per-Python. We just need to let cibuildwheel iterate.

**Actions:**

Edit `pyproject.toml`:
```toml
[tool.cibuildwheel]
skip = ["*pp*", "*musllinux*", "cp310-*", "cp39-*"]
```

Or explicitly:
```toml
[tool.cibuildwheel]
build = ["cp311-manylinux_x86_64", "cp312-manylinux_x86_64", "cp313-manylinux_x86_64"]
```

**Acceptance criteria:**
- `cibuildwheel --print-build-identifiers` shows 3 targets.
- Local cibuildwheel run produces 3 wheels in `wheelhouse/`.

**Failure modes:**
- cp313 may have newer C API differences (stable enough by 3.13.0,
  usually fine).
- Build time scales linearly (30–60 min for 3 wheels).

**Time:** 30 min config + 45 min first full build.

---

### Task 2.2 — Smoke-test each Python wheel

**Why:** Each Python's SWIG bindings are separate; one could work
while another fails.

**Actions:** Run Task 1.13's smoke test against each wheel, in the
matching `python:3.XX-slim` container.

```bash
for py in 3.11 3.12 3.13; do
    WHEEL=$(ls wheelhouse/pyramids_gis-*cp${py//./}*.whl)
    echo "=== Testing cp${py//./} ==="
    docker run --rm -v "$(pwd)/wheelhouse:/wh" python:${py}-slim bash -c "
        pip install /wh/$(basename $WHEEL) &&
        python -c 'from osgeo import gdal; print(gdal.__version__)'
    "
done
```

**Acceptance criteria:** All 3 succeed.

**Time:** 30 min.

---

### Phase 2 Deliverable — commit

`2: Phase 2 — Multi-Python wheels (cp311, cp312, cp313)`.

---

## Phase 3: Test-Suite Validation

**Goal:** Run the full pyramids test suite against the produced platform
wheel to catch regressions before shipping.

### Task 3.1 — Add a "platform wheel" test job to `.github/workflows/wheel-test.yml`

**Why:** The existing `wheel-test.yml` tests pure-Python wheels atop
conda-installed GDAL. A new parallel job tests the self-contained
platform wheel with NO conda.

**Actions:**

Edit `.github/workflows/wheel-test.yml` or create a new
`wheel-platform-test.yml`:

```yaml
name: Platform Wheel Tests
on:
  workflow_call:
  workflow_dispatch:

jobs:
  build:
    uses: ./.github/workflows/build-wheels.yml@build/option-1-pixi

  test:
    needs: build
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/download-artifact@v4
        with: {name: wheels-linux-x86_64, path: dist}
      - name: Install platform wheel (NO conda)
        run: |
          pip install dist/*-cp${{ matrix.python-version }}*-manylinux*.whl || \
          pip install dist/*.whl
          pip install pytest pytest-env pytest-mock
      - name: Verify import
        run: |
          python -c "from osgeo import gdal; print(gdal.__version__)"
          python -c "import pyramids; print(pyramids.__version__)"
      - name: Run test suite
        run: pytest -vvv -sv -m "not plot" tests/
```

**Acceptance criteria:** Workflow runs all 3 Python versions, tests pass.

**Failure modes:**
- Tests reference `tests/data/` paths relative to repo root — fine if
  we run from repo root.
- Tests assuming conda env (e.g., `GDAL_DATA` at conda path) — these
  need fixing. Should be rare since pyramids' tests are GDAL-agnostic.
- Data file size — large test data may be excluded from the repo; only
  a problem for tests that need them.

**Time:** 2–4 hours (the test fixture fixes are the variable part).

---

### Task 3.2 — Run the test suite locally against the wheel

**Why:** Faster iteration than CI.

**Actions:**

```bash
python -m venv /tmp/pytest-env
source /tmp/pytest-env/bin/activate
pip install wheelhouse/pyramids_gis-*cp312*.whl pytest pytest-env pytest-mock
cd /path/to/pyramids   # for tests/ directory
pytest -vvv -sv -m "not plot" tests/ 2>&1 | tee /tmp/test-output.log
```

**Acceptance criteria:** All non-plot tests pass (or same count as pixi
dev env, minus plot tests).

**Failure modes:** Test-by-test triage. Usually data-path fixtures.

**Time:** 2–6 hours.

---

### Task 3.3 — Fix any tests that break under platform-wheel install

**Why:** Tests may assume `pyramids` is installed editable (in-tree),
not as a wheel. Fixture paths often fail.

**Actions:** Iterate. Typical fixes:
- Change hardcoded `"src/pyramids/base/data/config.yaml"` to use
  `importlib.resources` or `pkgutil.get_data`.
- Ensure `tests/data/*.tif` are in `tests/data/` relative to `tests/`
  dir, not relative to `src/`.

**Time:** Variable (hours to day).

---

### Phase 3 Deliverable — commit

`3: Phase 3 — Test suite passes against platform wheel`.

---

## Phase 4: macOS Support

**Goal:** Produce `macosx_*` wheels for `x86_64` and `arm64`.

### Task 4.1 — Confirm pixi works on macOS runners

**Why:** pixi is cross-platform; conda-forge has `osx-64` and `osx-arm64`
channels. The same `wheel-build` feature should resolve identically.

**Actions:** In the pixi feature, add macos-specific pins only if needed.
Usually not:
```toml
[tool.pixi.feature.wheel-build.target.linux-64.dependencies]
# Linux-specific
[tool.pixi.feature.wheel-build.target.osx-arm64.dependencies]
# macOS arm64-specific
```

For now, the base `[tool.pixi.feature.wheel-build.dependencies]` should
cover macOS too.

**Acceptance criteria:** `pixi install -e wheel-build` succeeds on a
macOS runner.

**Time:** 30 min (verify only).

---

### Task 4.2 — Write `ci/setup-gdal-from-pixi-macos.sh` (or adapt Linux one)

**Why:** Same logic, but on macOS:
- Libraries are `.dylib`, not `.so`
- `install_name_tool` instead of `patchelf` (handled by `delocate`)
- No `strip --strip-unneeded`; use `strip -S` on macOS

**Actions:** Create macOS variant or wrap in a single script with
platform detection.

**Acceptance criteria:** Inside macOS runner, same verification as Task 1.4.

**Time:** 2 hours.

---

### Task 4.3 — Add macOS to `[tool.cibuildwheel]` in `pyproject.toml`

**Why:** cibuildwheel needs per-OS `before-all` and `repair-wheel-command`.

```toml
[tool.cibuildwheel.macos]
before-all = [
    "bash ./ci/setup-gdal-from-pixi-macos.sh",  # or unified script
]
before-build = [
    "pip install setuptools wheel numpy",
    "PACKAGE_DATA=1 python ./ci/install-and-vendor-osgeo.py",
]
repair-wheel-command = [
    "DYLD_LIBRARY_PATH=/usr/local/lib delocate-wheel --require-archs {delocate_archs} -w {dest_dir} {wheel}",
]

[tool.cibuildwheel.macos.environment]
GDAL_VERSION = "3.12.1"
BUILD_PREFIX = "/usr/local"
GDAL_DATA = "/usr/local/share/gdal"
PROJ_DATA = "/usr/local/share/proj"
MACOSX_DEPLOYMENT_TARGET = "11.0"
```

**Acceptance criteria:** Build succeeds on macOS runner.

**Time:** 2–4 hours (delocate quirks).

---

### Task 4.4 — Handle macOS arm64 vs x86_64

**Why:** Different architectures need separate runners (cross-compile
is possible but flaky).

**Actions:** In `.github/workflows/build-wheels.yml`:

```yaml
matrix:
  include:
    - os: macos-13     # x86_64
      arch: x86_64
    - os: macos-14     # arm64 (M-series)
      arch: arm64
```

**Time:** 1 hour.

---

### Task 4.5 — Smoke-test macOS wheels

Similar to Task 1.13, but on the relevant macOS host or Orbstack.

**Time:** 1 hour.

---

### Phase 4 Deliverable — commit

`4: Phase 4 — macOS wheels (x86_64 + arm64)`.

---

## Phase 5: Windows Support

**Goal:** Produce `win_amd64` wheels.

### Task 5.1 — Confirm pixi works on Windows

**Why:** Windows uses `.dll` and has different path conventions. pixi
handles this transparently (conda-forge has `win-64` channel).

**Actions:** Same as Task 4.1 but on Windows runner.

**Time:** 30 min.

---

### Task 5.2 — Write `ci/setup-gdal-from-pixi.ps1` (PowerShell)

**Why:** Windows doesn't natively run bash. The setup logic is similar
but uses `Copy-Item` instead of `cp`, and handles DLLs instead of `.so`.

**Actions:** PowerShell rewrite of the extraction logic. Key paths:
- Libraries: `.pixi\envs\wheel-build\Library\bin\*.dll`
- Headers: `.pixi\envs\wheel-build\Library\include\`
- Data: `.pixi\envs\wheel-build\Library\share\{gdal,proj}\`

Note: on Windows, conda packages use `Library/` prefix due to how
Anaconda installs on Windows.

**Acceptance criteria:** In a Windows runner, `pyramids/_vendor/osgeo/`
populated after running.

**Time:** 3–5 hours.

---

### Task 5.3 — Add Windows to `pyproject.toml` cibuildwheel config

```toml
[tool.cibuildwheel.windows]
before-all = [
    "powershell.exe -File ./ci/setup-gdal-from-pixi.ps1",
]
before-build = [
    "pip install setuptools wheel numpy",
    "python ./ci/install-and-vendor-osgeo.py",
]
repair-wheel-command = [
    "delvewheel repair --add-path C:/pyramids-lib -w {dest_dir} {wheel}",
]
```

**Time:** 2 hours.

---

### Task 5.4 — `os.add_dll_directory` at import

**Why:** Windows DLL loading doesn't respect RPATH. `delvewheel`
bundles DLLs but the wheel's `__init__.py` must add the dir to the
search path.

**Actions:** Already handled in `pyramids/__init__.py` bootstrap for
`sys.platform == "win32"`. Verify `pyramids.libs/` (or `.libs/`) is
the right path after delvewheel.

**Time:** 1 hour.

---

### Task 5.5 — Smoke-test Windows wheel

On a Windows runner or local Windows install.

**Time:** 1–2 hours.

---

### Phase 5 Deliverable — commit

`5: Phase 5 — Windows wheels`.

---

## Phase 6: Stabilization & Release

### Task 6.1 — Measure and optimize wheel sizes

Target: ≤ 80 MB per wheel compressed. If over 120 MB:
- Disable unused GDAL drivers in `[tool.pixi.feature.wheel-build.dependencies]`
  (choose a more targeted conda-forge build)
- Strip `.so` files more aggressively
- Drop rarely-used data files in `_data/proj_data/` (but keep `proj.db`)

**Time:** 1 day.

---

### Task 6.2 — Document release process

Create `planning/bundle/release-process.md` covering:
- Version bump via `cz-bump`
- Trigger `build-wheels.yml` on a release
- Pre-release test on `test.pypi.org`
- Yank procedure for bad wheels

**Time:** 2 hours.

---

### Task 6.3 — Wire `build-wheels.yml` to PyPI trusted publishing

Uncomment the `publish:` job in `build-wheels.yml` once testing is stable.

**Time:** 1 hour.

---

### Task 6.4 — User-facing docs update

- `docs/installation.md` — add "Install via pip with bundled GDAL"
- `README.md` — update install instructions
- `docs/troubleshooting.md` — common failures (glibc, Alpine, Windows DLLs)
- `CHANGELOG.md` — note the distribution change

**Time:** 3 hours.

---

### Task 6.5 — Namespace collision mitigation (optional)

If field testing shows `osgeo` collision with users' pre-installed
GDAL causes issues, pivot to "Strategy B" (rename vendored module to
`pyramids._vendor._pyramids_osgeo`). Documented in
`gdal-bundling-plan.md` §6.5.

**Time:** 1–2 days if needed.

---

### Phase 6 Deliverable — Public release

Once CI is green across all 3 platforms and test.pypi.org install is
verified, do a real release:

```bash
pixi run -e dev cz-bump
git push --follow-tags
gh workflow run build-wheels.yml
# When green, the publish job uploads to PyPI
```

---

## Glossary

- **pixi** — Conda-compatible package manager (https://pixi.sh)
- **wheel-build** — The name of the pixi feature/env used only during
  wheel build (separate from `dev`).
- **conda-forge** — Community-maintained conda channel; the source of
  pre-compiled GDAL/PROJ/GEOS binaries.
- **manylinux2014** — Linux wheel compatibility tag (glibc 2.17+,
  CentOS 7-era); covers Ubuntu 16.04+, RHEL 7+.
- **CIBW_BEFORE_ALL** — Runs once per platform, shared across Python
  versions. Long-running setup belongs here.
- **CIBW_BEFORE_BUILD** — Runs per Python version. Python-specific
  build steps belong here.
- **auditwheel** — Linux tool that bundles shared libs and patches
  RPATH.
- **delocate** — macOS equivalent (patches `install_name`).
- **delvewheel** — Windows equivalent (bundles DLLs).
- **osgeo** — The Python module name for GDAL's SWIG bindings.
- **GDAL_DATA / PROJ_DATA** — Runtime data directories that GDAL and
  PROJ look up at runtime; must be shipped with the wheel.

---

## Issue Tracker

| #    | Task                                                     | Phase | Priority | State  |
|------|----------------------------------------------------------|-------|----------|--------|
| 1.1  | Add `wheel-build` pixi feature + env to `pyproject.toml` | 1     | P0       | Solved |
| 1.2  | Verify pixi wheel-build env resolves locally             | 1     | P0       | Solved |
| 1.3  | Design `ci/setup-gdal-from-pixi.sh`                      | 1     | P0       | Open  |
| 1.4  | Smoke-test setup script in manylinux container           | 1     | P0       | Open  |
| 1.5  | Create `ci/install-and-vendor-osgeo.py`                  | 1     | P0       | Open  |
| 1.6  | Wire cibuildwheel config in `pyproject.toml`             | 1     | P0       | Open  |
| 1.7  | Verify pyramids runtime bootstrap                        | 1     | P1       | Open  |
| 1.8  | Archive `ci/build-gdal-linux.sh` with deprecation header | 1     | P2       | Open  |
| 1.9  | Delete/inline `ci/vendor-osgeo.py`                       | 1     | P2       | Open  |
| 1.10 | Run cibuildwheel for cp312 locally                       | 1     | P0       | Open  |
| 1.11 | Debug first-build failures                               | 1     | P0       | Open  |
| 1.12 | Verify wheel layout via `python -m zipfile -l`           | 1     | P0       | Open  |
| 1.13 | Smoke-test wheel in `python:3.12-slim`                   | 1     | P0       | Open  |
| 2.1  | Enable multi-Python matrix (cp311/cp312/cp313)           | 2     | P1       | Open  |
| 2.2  | Smoke-test each Python wheel                             | 2     | P1       | Open  |
| 3.1  | Add platform-wheel test job to `wheel-test.yml`          | 3     | P0       | Open  |
| 3.2  | Run test suite locally against platform wheel            | 3     | P0       | Open  |
| 3.3  | Fix tests that break under platform-wheel install        | 3     | P1       | Open  |
| 4.1  | Confirm pixi works on macOS runners                      | 4     | P1       | Open  |
| 4.2  | Write `ci/setup-gdal-from-pixi-macos.sh`                 | 4     | P1       | Open  |
| 4.3  | Add macOS to `[tool.cibuildwheel]`                       | 4     | P1       | Open  |
| 4.4  | Handle macOS arm64 vs x86_64 separately                  | 4     | P1       | Open  |
| 4.5  | Smoke-test macOS wheels                                  | 4     | P1       | Open  |
| 5.1  | Confirm pixi works on Windows runners                    | 5     | P1       | Open  |
| 5.2  | Write `ci/setup-gdal-from-pixi.ps1` (PowerShell)         | 5     | P1       | Open  |
| 5.3  | Add Windows to `[tool.cibuildwheel]`                     | 5     | P1       | Open  |
| 5.4  | Verify `os.add_dll_directory` bootstrap works            | 5     | P1       | Open  |
| 5.5  | Smoke-test Windows wheel                                 | 5     | P1       | Open  |
| 6.1  | Measure and optimize wheel sizes                         | 6     | P1       | Open  |
| 6.2  | Document release process in `release-process.md`         | 6     | P1       | Open  |
| 6.3  | Wire `build-wheels.yml` to PyPI trusted publishing       | 6     | P0       | Open  |
| 6.4  | Update user-facing installation docs                     | 6     | P1       | Open  |
| 6.5  | Namespace collision mitigation (if needed)               | 6     | P2       | Open  |
