# Installation

This page explains how to install pyramids-gis and its dependencies using Pixi/conda or pip. The instructions and versions below are aligned with the project’s pyproject.toml.

Package name: pyramids-gis
Current version: 0.7.3
Supported Python versions: 3.11 – 3.13 (requires Python >=3.11,<4)

## Dependencies

### Core runtime (PyPI)
- numpy >=2.0.0
- pandas >=2.0.0
- geopandas >=1.0.0
- Shapely >=2.0.0
- pyproj >=3.7.0
- PyYAML >=6.0.0
- loguru >=0.7.2
- hpc-utils >=0.1.5

### GIS stack (recommended via conda-forge)
- GDAL >=3.10,<4
- libgdal-netcdf >=3.10,<4
- libgdal-hdf4 >=3.10,<4

### Optional extras
- viz: cleopatra >=0.5.1
- dev: nbval, pre-commit, pytest, coverage, build, twine, etc.
- docs: mkdocs, mkdocs-material, mkdocstrings, mike, etc.

## Recommended: Pixi/Conda environment
This repository includes a Pixi configuration to create fully-solvable environments with the right GDAL build from conda-forge.

Prerequisites: Install Pixi (https://pixi.sh/) or have conda/mamba with the conda-forge channel available.

### Using Pixi
From the project root:

```console
pixi run main          # runs the main test suite to ensure the env is solvable
pixi shell             # enter the Pixi environment
```

Pixi environments provided:
- default: includes dev + viz extras
- docs: documentation toolchain
- py311 / py312 / py313: pinned Python versions

To install the package in editable mode inside the Pixi environment:

```console
pip install -e .
```

### Using conda/mamba directly
Create and activate an environment (example with Python 3.12):

```console
mamba create -n pyramids -c conda-forge python=3.12 gdal libgdal-netcdf libgdal-hdf4
mamba activate pyramids-gis
```

Then install the package from PyPI (release):

```console
pip install pyramids-gis==0.7.3
```

Optionally include extras (examples):

```console
pip install "pyramids-gis[viz]"        # installs cleopatra
pip install "pyramids-gis[dev]"        # developer tools
pip install "pyramids-gis[docs]"       # docs toolchain
```

## Installing with pip only (advanced)
Installing GDAL wheels via pip can be platform-specific. We strongly recommend installing GDAL from conda-forge first, then using pip for pyramids-gis:

```console
conda install -c conda-forge gdal libgdal-netcdf libgdal-hdf4
pip install pyramids-gis
```

If you insist on a pip-only approach, consult the GDAL wheel guidance for your platform and ensure gdal is available at runtime before installing pyramids-gis.

## Install from source
Clone the repository and install:

```console
git clone https://github.com/Serapieum-of-alex/pyramids.git
cd pyramids
python -m pip install .
```

Editable (development) install:

```console
git clone https://github.com/Serapieum-of-alex/pyramids.git
cd pyramids
pip install -e .[dev]
```

Install directly from GitHub (latest main):

```console
pip install "git+https://github.com/Serapieum-of-alex/pyramids.git"
```

Install a specific tagged release from GitHub:

```console
pip install "git+https://github.com/Serapieum-of-alex/pyramids.git@v0.7.3"
```

## Quick check
After installation, open Python and run:

```python
import pyramids
print(pyramids.__version__)
```

You can also run the test suite via Pixi:
```console
pixi run -e dev pytest -m "not plot" -q
```

To run a specific test file:
```console
pixi run -e dev pytest tests/netcdf/test_dimensions.py
```

## Notes
- Supported Python versions are 3.11–3.13.
- Prefer conda-forge for GDAL and related libraries.
- Documentation: https://serapieum-of-alex.github.io/pyramids/latest
- Source repository: https://github.com/Serapieum-of-alex/pyramids
