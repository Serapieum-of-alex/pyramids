[![Python Versions](https://img.shields.io/pypi/pyversions/pyramids-gis.png)](https://img.shields.io/pypi/pyversions/pyramids-gis)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MAfarrag/Hapi.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/Hapi/context:python)
[![Documentation Status](https://readthedocs.org/projects/pyramids-gis/badge/?version=latest)](https://pyramids-gis.readthedocs.io/en/latest/?badge=latest)


![GitHub last commit](https://img.shields.io/github/last-commit/MAfarrag/pyramids)
![GitHub forks](https://img.shields.io/github/forks/MAfarrag/pyramids?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/MAfarrag/pyramids?style=social)
[![Coverage Status](https://coveralls.io/repos/github/MAfarrag/pyramids/badge.svg?branch=main)](https://coveralls.io/github/MAfarrag/pyramids?branch=main)


![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/mafarrag/pyramids/0.2.1?include_prereleases&style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/mafarrag/pyramids)

Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Version                                                                                                                                                                                                                                                                                                                                                 | Platforms |
|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyramids-green.svg)](https://anaconda.org/conda-forge/pyramids) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Downloads](https://pepy.tech/badge/pyramids-gis)](https://pepy.tech/project/pyramids-gis) [![Downloads](https://pepy.tech/badge/pyramids-gis/month)](https://pepy.tech/project/pyramids-gis)  [![Downloads](https://pepy.tech/badge/pyramids-gis/week)](https://pepy.tech/project/pyramids-gis)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/pyramids-gis?color=blue&style=flat-square) ![GitHub all releases](https://img.shields.io/github/downloads/MAfarrag/pyramids/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/MAfarrag/pyramids/0.2.1/total) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![PyPI version](https://badge.fury.io/py/pyramids-gis.svg)](https://badge.fury.io/py/pyramids-gis)  | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyramids.svg)](https://anaconda.org/conda-forge/pyramids) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

pyramids - GIS utility package
=====================================================================
**pyramids** is a GIS utility package using gdal, rasterio, ....

pyramids

Main Features
-------------
  - GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
    needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
    convert different forms of distributed data (rasters, NetCDF, shapefiles)


Future work
-------------
  - Developing a DEM processing module for generating the river network at different DEM spatial resolutions.



Installing pyramids
===============

Installing `pyramids` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge pyramids=0.2.7
```

It is possible to list all of the versions of `pyramids` available on your platform with:

```
conda search pyramids --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/MAfarrag/pyramids
```

## pip
to install the last release you can easly use pip
```
pip install pyramids-gis==0.2.8
```

Quick start
===========

```
  >>> import pyramids
```

[other code samples](https://pyramids-gis.readthedocs.io/en/latest/?badge=latest)
