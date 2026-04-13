#!/bin/bash
#
# Install GDAL + native dependencies via pixi (conda-forge), then extract
# the shared libraries, headers, and data files into ${BUILD_PREFIX} so
# downstream steps (install-and-vendor-osgeo.py, setuptools,
# auditwheel/delocate/delvewheel) can find them without knowing about
# pixi's internal layout.
#
# Runs once per cibuildwheel platform invocation (CIBW_BEFORE_ALL), i.e.
# shared across all Python versions in the matrix.
#
# See planning/bundle/option-1-implementation-plan.md Tasks 1.3–1.4.
set -euo pipefail

BUILD_PREFIX="${BUILD_PREFIX:-/usr/local}"

echo "=== setup-gdal-from-pixi.sh ==="
echo "BUILD_PREFIX=${BUILD_PREFIX}"

# ---------------------------------------------------------------------------
# 1. Install pixi (static binary, ~50 MB, ~5 seconds)
# ---------------------------------------------------------------------------
if ! command -v pixi >/dev/null 2>&1; then
    echo "--- Installing pixi ---"
    export PIXI_HOME="${BUILD_PREFIX}"
    export PIXI_NO_PATH_UPDATE=1
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="${BUILD_PREFIX}/bin:${PATH}"
fi
pixi --version

# ---------------------------------------------------------------------------
# 2. Install the wheel-build environment using the committed pixi.lock.
#
# `--frozen` forces pixi to use exactly what's in pixi.lock (no re-solving)
# which (a) keeps builds reproducible and (b) avoids pixi trying to solve
# other envs like 'docs' for platforms not available in the container.
# ---------------------------------------------------------------------------
echo "--- Resolving wheel-build environment ---"
pixi install -e wheel-build --frozen

PIXI_ENV="$(pwd)/.pixi/envs/wheel-build"
if [ ! -d "${PIXI_ENV}" ]; then
    echo "ERROR: ${PIXI_ENV} does not exist after pixi install" >&2
    exit 1
fi
echo "wheel-build env: ${PIXI_ENV}"

# ---------------------------------------------------------------------------
# 3. Extract native artifacts into ${BUILD_PREFIX}
# ---------------------------------------------------------------------------
echo "--- Extracting native artifacts into ${BUILD_PREFIX} ---"
mkdir -p "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" \
         "${BUILD_PREFIX}/include" "${BUILD_PREFIX}/share" \
         "${BUILD_PREFIX}/bin"

# Shared libraries — preserve symlinks with -a
cp -a "${PIXI_ENV}/lib/"*.so* "${BUILD_PREFIX}/lib/" 2>/dev/null || true
if [ -d "${PIXI_ENV}/lib64" ]; then
    cp -a "${PIXI_ENV}/lib64/"*.so* "${BUILD_PREFIX}/lib64/" 2>/dev/null || true
fi

# Headers
cp -a "${PIXI_ENV}/include/." "${BUILD_PREFIX}/include/"

# GDAL_DATA + PROJ_DATA — required at runtime
cp -a "${PIXI_ENV}/share/gdal" "${BUILD_PREFIX}/share/"
cp -a "${PIXI_ENV}/share/proj" "${BUILD_PREFIX}/share/"

# Build tooling needed downstream
for tool in gdal-config swig ogrinfo gdalinfo; do
    src="${PIXI_ENV}/bin/${tool}"
    if [ -f "${src}" ]; then
        cp "${src}" "${BUILD_PREFIX}/bin/"
    fi
done

# pkg-config files (some build tools consult pkg-config)
if [ -d "${PIXI_ENV}/lib/pkgconfig" ]; then
    mkdir -p "${BUILD_PREFIX}/lib/pkgconfig"
    cp "${PIXI_ENV}/lib/pkgconfig/"*.pc "${BUILD_PREFIX}/lib/pkgconfig/" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. Strip debug symbols to reduce wheel size
# ---------------------------------------------------------------------------
echo "--- Stripping shared libraries ---"
find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f \
    -exec strip --strip-unneeded {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Diagnostic output
# ---------------------------------------------------------------------------
echo "=== setup-gdal-from-pixi.sh complete ==="
echo "GDAL version: $("${BUILD_PREFIX}/bin/gdal-config" --version)"
echo "libgdal: $(ls "${BUILD_PREFIX}/lib/libgdal.so"* 2>/dev/null | head -1)"
echo "libproj: $(ls "${BUILD_PREFIX}/lib/libproj.so"* 2>/dev/null | head -1)"
echo "libgeos: $(ls "${BUILD_PREFIX}/lib/libgeos.so"* 2>/dev/null | head -1)"
echo "Total .so files: $(find "${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/lib64" -name '*.so*' -type f 2>/dev/null | wc -l)"
echo "Total size: $(du -sh "${BUILD_PREFIX}/lib" 2>/dev/null | cut -f1)"
