[![Documentations](https://img.shields.io/badge/Documentations-blue?logo=github&logoColor=white)](https://serapeum-org.github.io/pyramids/main/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyramids-gis.png)](https://img.shields.io/pypi/pyversions/pyramids-gis)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![GitHub last commit](https://img.shields.io/github/last-commit/serapeum-org/pyramids)
![GitHub Repo stars](https://img.shields.io/github/stars/serapeum-org/pyramids?style=social)
[![codecov](https://codecov.io/gh/serapeum-org/pyramids/graph/badge.svg?token=g0DV4dCa8N)](https://codecov.io/gh/serapeum-org/pyramids)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5e3aa4d0acc843d1a91caf33545ecf03)](https://www.codacy.com/gh/serapeum-org/pyramids/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=serapeum-org/pyramids&amp;utm_campaign=Badge_Grade)

![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/serapeum-org/pyramids/latest?include_prereleases&style=plastic)

[![pages-build-deployment](https://github.com/serapeum-org/pyramids/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/serapeum-org/pyramids/actions/workflows/pages/pages-build-deployment)

Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyramids-green.svg)](https://anaconda.org/conda-forge/pyramids) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Downloads](https://pepy.tech/badge/pyramids-gis)](https://pepy.tech/project/pyramids-gis) [![Downloads](https://pepy.tech/badge/pyramids-gis/month)](https://pepy.tech/project/pyramids-gis)  [![Downloads](https://pepy.tech/badge/pyramids-gis/week)](https://pepy.tech/project/pyramids-gis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/pyramids-gis?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![PyPI version](https://badge.fury.io/py/pyramids-gis.svg)](https://badge.fury.io/py/pyramids-gis) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) |

### conda-forge feedstock
[Conda-forge feedstock](https://github.com/conda-forge/pyramids-feedstock)


pyramids - GIS utility package
=====================================================================
**pyramids** is a GIS utility package built on top of GDAL/OGR for working with raster data (GeoTIFF, NetCDF),
vector data (shapefiles, GeoJSON), and multi-temporal datacubes.

```mermaid
graph TD
    subgraph IO["I/O Formats"]
        GeoTIFF["GeoTIFF"]
        NC["NetCDF"]
        UGRID_NC["UGRID NetCDF"]
        SHP["Shapefile / GeoJSON"]
        ZIP["Compressed (zip/gzip/tar)"]
    end

    subgraph Core["Core Components"]
        DS["<b>Dataset</b><br/>Single raster<br/>read · write · crop<br/>reproject · align"]
        NCD["<b>NetCDF</b><br/>extends Dataset<br/>time/variable dimensions<br/>CF conventions"]
        UGDS["<b>UgridDataset</b><br/>Unstructured mesh<br/>UGRID-1.0 conventions"]
        DC["<b>DatasetCollection</b><br/>Temporal raster stack<br/>multi-temporal analysis"]
        FC["<b>FeatureCollection</b><br/>Vector data<br/>GeoDataFrame + OGR"]
    end

    subgraph Ops["Operations"]
        Spatial["Spatial<br/>crop · reproject · align"]
        Math["Array / Band<br/>math · statistics · no-data"]
        Convert["Convert<br/>raster ↔ vector<br/>raster ↔ NetCDF"]
    end

    GeoTIFF --> DS
    NC --> NCD
    UGRID_NC --> UGDS
    SHP --> FC
    ZIP --> DS
    ZIP --> NCD

    DS --> Spatial
    DS --> Math
    DS --> Convert
    NCD --> Spatial
    NCD --> Math
    UGDS --> |"mesh-to-raster"| DS
    UGDS --> |"mesh-to-vector"| FC
    DC --> |"contains aligned<br/>Dataset stack"| DS
    FC --> Convert
    FC --> |"rasterize"| DS
    Convert --> FC
    Convert --> DS
```

Main Features
-------------

- **Dataset** - Read, write, crop, reproject, and align single-band and multi-band rasters (GeoTIFF)
  with full no-data handling and coordinate reference system support.
- **NetCDF** - Extends Dataset for NetCDF files with time/variable dimensions and CF conventions metadata.
  Optional xarray interoperability.
- **UgridDataset** - Read and visualize UGRID-1.0 unstructured meshes (triangles, quads, mixed).
  Supports mesh-to-raster interpolation and mesh-to-vector export.
- **DatasetCollection** - Manage time-series of co-registered rasters as a temporal stack for
  multi-temporal analysis.
- **FeatureCollection** - Work with vector data (shapefiles, GeoJSON) through a unified GeoDataFrame and
  OGR DataSource interface, including rasterization and geometry operations.
- **Spatial operations** - Align rasters to a reference grid, reproject between coordinate systems,
  crop to vector boundaries, and convert between raster, NetCDF, and vector formats.

Installing pyramids
===============

Installing `pyramids` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge pyramids
```

It is possible to list all the versions of `pyramids` available on your platform with:

```
conda search pyramids --channel conda-forge
```

## Install from GitHub

To install the latest development version, you can install the library from GitHub:

```
pip install git+https://github.com/serapeum-org/pyramids
```

## pip

To install the latest release from PyPI:

```
pip install pyramids-gis
```

## Optional extras

```
pip install pyramids-gis[viz]      # cleopatra plotting support
pip install pyramids-gis[xarray]   # xarray/NetCDF4 interoperability
```

Quick start
===========

```python
from pyramids.dataset import Dataset

# Open a raster file
src = Dataset.read_file("path/to/raster.tif")
print(src.epsg)        # coordinate reference system EPSG code
print(src.cell_size)   # pixel resolution
print(src.shape)       # (rows, columns)

# Get the raster data as a NumPy array
arr = src.raster.ReadAsArray()
```

```python
from pyramids.netcdf import NetCDF

# Open a NetCDF file
nc = NetCDF.read_file("path/to/data.nc")
print(nc.variables)
```

```python
from pyramids.feature import FeatureCollection

# Open a vector file
vector = FeatureCollection.read_file("path/to/shapefile.shp")
print(vector.shape)
```

Testing
=======

This project uses [pixi](https://pixi.sh) as the environment and task manager.

```console
# Install dependencies and create dev environment
pixi install -e dev

# Run all tests (excluding plot tests)
pixi run -e dev main

# Run plot tests only
pixi run -e dev plot

# Run a specific test file
pixi run -e dev pytest tests/netcdf/test_dimensions.py -v

# Run a single test by node id
pixi run -e dev pytest tests/netcdf/test_dimensions.py::TestStripBraces::test_with_braces -q
```

Docker
======

A Dockerfile is provided to run pyramids-gis in a controlled environment with the correct GDAL stack
preinstalled via conda-forge. The image uses a multi-stage pixi build for a minimal production container.

Build the image:

```
docker build -t pyramids-gis:latest .
```

Run the container (mount your current folder as /workspace):

```
docker run --rm -it -v ${PWD}:/workspace pyramids-gis:latest bash
```

Inside the container you can verify the package is installed:

```
python -c "import pyramids; print('pyramids', pyramids.__version__)"
```
