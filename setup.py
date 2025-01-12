import os
import urllib.request
import tarfile
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    """Custom build_ext to automate GDAL header fetching and configuration."""

    GDAL_VERSION = "3.10.0"
    GDAL_SOURCE_URL = (
        f"https://github.com/OSGeo/gdal/archive/refs/tags/v{GDAL_VERSION}.tar.gz"
    )
    GDAL_TARGET_DIR = "build/gdal_sources"

    def initialize_options(self):
        super().initialize_options()
        self.include_dirs = []
        self.library_dirs = []
        self.libraries = []

    def fetch_gdal_sources(self):
        """Download and extract GDAL source files."""
        os.makedirs(self.GDAL_TARGET_DIR, exist_ok=True)
        tar_file = os.path.join(self.GDAL_TARGET_DIR, "gdal_sources.tar.gz")

        # Download GDAL source tarball
        print(f"Downloading GDAL source files from {self.GDAL_SOURCE_URL}...")
        urllib.request.urlretrieve(self.GDAL_SOURCE_URL, tar_file)

        # Extract source files
        print("Extracting GDAL source files...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=self.GDAL_TARGET_DIR)

    def finalize_options(self):
        super().finalize_options()
        try:
            self.fetch_gdal_sources()

            # Configure paths for GDAL
            gdal_base = os.path.join(self.GDAL_TARGET_DIR, f"gdal-{self.GDAL_VERSION}")
            gdal_include = os.path.join(gdal_base, "gdal")
            gdal_lib = os.path.join(gdal_base, "lib")

            self.include_dirs.append(gdal_include)
            self.library_dirs.append(gdal_lib)
            self.libraries.append("gdal")

        except Exception as e:
            raise RuntimeError(f"Failed to configure GDAL: {e}")


# GDAL extensions
extensions = [
    Extension(
        "osgeo._gdal",
        sources=[],
        libraries=["gdal"],
    ),
    Extension(
        "osgeo._gdalconst",
        sources=[],
        libraries=["gdal"],
    ),
    Extension(
        "osgeo._ogr",
        sources=[],
        libraries=["gdal"],
    ),
    Extension(
        "osgeo._osr",
        sources=[],
        libraries=["gdal"],
    ),
]

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": CustomBuildExt},
)
