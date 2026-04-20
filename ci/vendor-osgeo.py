"""Vendor the osgeo package and GDAL/PROJ data files into the pyramids source tree.

Run this BEFORE building the wheel (in CIBW_BEFORE_BUILD) so that setuptools
includes the vendored files in the wheel.

Usage:
    PACKAGE_DATA=1 python ci/vendor-osgeo.py

Environment variables:
    PACKAGE_DATA    - Must be set to "1" to enable vendoring (safety guard).
    GDAL_DATA       - Path to GDAL data directory (default: auto-detect via gdal-config).
    PROJ_DATA       - Path to PROJ data directory (default: /usr/local/share/proj).
    BUILD_PREFIX    - Build prefix where GDAL was installed (default: /usr/local).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if os.environ.get("PACKAGE_DATA") != "1":
        print("PACKAGE_DATA not set to '1', skipping vendoring.")
        return

    src_root = Path(__file__).resolve().parent.parent / "src" / "pyramids"
    vendor_dir = src_root / "_vendor"
    data_dir = src_root / "_data"
    build_prefix = Path(os.environ.get("BUILD_PREFIX", "/usr/local"))

    # --- Vendor osgeo Python package ---
    osgeo_dest = vendor_dir / "osgeo"
    if osgeo_dest.exists():
        shutil.rmtree(osgeo_dest)

    try:
        import osgeo
        osgeo_src = Path(osgeo.__file__).parent
    except ImportError:
        print("ERROR: osgeo module not importable. Build GDAL with -DBUILD_PYTHON_BINDINGS=ON first.")
        sys.exit(1)

    print(f"Vendoring osgeo from {osgeo_src} -> {osgeo_dest}")
    vendor_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(osgeo_src, osgeo_dest)

    # Create __init__.py for the _vendor package
    vendor_init = vendor_dir / "__init__.py"
    if not vendor_init.exists():
        vendor_init.write_text("")

    # --- Vendor GDAL data files ---
    gdal_data_src = os.environ.get("GDAL_DATA")
    if not gdal_data_src:
        try:
            gdal_data_src = subprocess.check_output(
                ["gdal-config", "--datadir"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            gdal_data_src = str(build_prefix / "share" / "gdal")

    gdal_data_dest = data_dir / "gdal_data"
    if gdal_data_dest.exists():
        shutil.rmtree(gdal_data_dest)

    gdal_data_path = Path(gdal_data_src)
    if gdal_data_path.is_dir():
        print(f"Vendoring GDAL data from {gdal_data_path} -> {gdal_data_dest}")
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(gdal_data_path, gdal_data_dest)
    else:
        print(f"WARNING: GDAL data directory not found at {gdal_data_path}")

    # --- Vendor PROJ data files ---
    proj_data_src = os.environ.get("PROJ_DATA")
    if not proj_data_src:
        proj_data_src = str(build_prefix / "share" / "proj")

    proj_data_dest = data_dir / "proj_data"
    if proj_data_dest.exists():
        shutil.rmtree(proj_data_dest)

    proj_data_path = Path(proj_data_src)
    if proj_data_path.is_dir():
        print(f"Vendoring PROJ data from {proj_data_path} -> {proj_data_dest}")
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(proj_data_path, proj_data_dest)
    else:
        print(f"WARNING: PROJ data directory not found at {proj_data_path}")

    print("Vendoring complete.")


if __name__ == "__main__":
    main()
