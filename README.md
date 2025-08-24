[![Documentations](https://img.shields.io/badge/Documentations-blue?logo=github&logoColor=white)](https://serapieum-of-alex.github.io/pyramids/main/)
[![pages-build-deployment](https://github.com/Serapieum-of-alex/pyramids/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Serapieum-of-alex/pyramids/actions/workflows/pages/pages-build-deployment)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyramids-gis.png)](https://img.shields.io/pypi/pyversions/pyramids-gis)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


![GitHub last commit](https://img.shields.io/github/last-commit/Serapieum-of-alex/pyramids)
![GitHub Repo stars](https://img.shields.io/github/stars/Serapieum-of-alex/pyramids?style=social)
[![codecov](https://codecov.io/gh/Serapieum-of-alex/pyramids/graph/badge.svg?token=g0DV4dCa8N)](https://codecov.io/gh/Serapieum-of-alex/pyramids)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5e3aa4d0acc843d1a91caf33545ecf03)](https://www.codacy.com/gh/Serapieum-of-alex/pyramids/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Serapieum-of-alex/pyramids&amp;utm_campaign=Badge_Grade)

![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/Serapieum-of-alex/pyramids/latest?include_prereleases&style=plastic)

Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyramids-green.svg)](https://anaconda.org/conda-forge/pyramids) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Downloads](https://pepy.tech/badge/pyramids-gis)](https://pepy.tech/project/pyramids-gis) [![Downloads](https://pepy.tech/badge/pyramids-gis/month)](https://pepy.tech/project/pyramids-gis)  [![Downloads](https://pepy.tech/badge/pyramids-gis/week)](https://pepy.tech/project/pyramids-gis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/pyramids-gis?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![PyPI version](https://badge.fury.io/py/pyramids-gis.svg)](https://badge.fury.io/py/pyramids-gis) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

conda-forge feedstock
=====================
[Conda-forge feedstock](https://github.com/conda-forge/pyramids-feedstock)


pyramids - GIS utility package
=====================================================================
**pyramids** is a GIS utility package using gdal, ....

pyramids

![1](/docs/_images/package-work-flow/overall.png)

Main Features
-------------

- GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
  needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
  convert different forms of distributed data (rasters, NetCDF, shapefiles)

Installing pyramids
===============

Installing `pyramids` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge pyramids=0.7.3
```

It is possible to list all the versions of `pyramids` available on your platform with:

```
conda search pyramids --channel conda-forge
```

## Install from GitHub

to install the last development to time, you can install the library from GitHub

```
pip install git+https://github.com/Serapieum-of-alex/pyramids
```

## pip

to install the last release, you can easily use pip

```
pip install pyramids-gis==0.7.3
```

Quick start
===========

```
  >>> import pyramids
```

Docker
======

A Dockerfile is provided to run pyramids-gis in a controlled environment with the correct GDAL stack preinstalled via conda-forge.

Build the image:

```
# from the repository root
docker build -t pyramids-gis:latest .
```

Run the container (mount your current folder as /workspace):

```
# Windows PowerShell
docker run --rm -it -v ${PWD}:/workspace pyramids-gis:latest bash
```

Inside the container you can verify the package is installed:

```
python -c "import pyramids; import sys; print('pyramids', getattr(pyramids, '__version__', 'dev'), 'Python', sys.version.split()[0])"
```

![Dataset diagram](./docs/_images/pyramids-dataset.svg)