[![Python Versions](https://img.shields.io/pypi/pyversions/pyramids-gis.png)](https://img.shields.io/pypi/pyversions/pyramids-gis)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MAfarrag/Hapi.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/Hapi/context:python)



![GitHub last commit](https://img.shields.io/github/last-commit/MAfarrag/pyramids)
![GitHub forks](https://img.shields.io/github/forks/MAfarrag/pyramids?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/MAfarrag/pyramids?style=social)



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
conda install -c conda-forge pyramids
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
pip install pyramids-gis==0.1.0
```

Quick start
===========

```
  >>> import pyramids
```

[other code samples](https://pyramids-gis.readthedocs.io/en/latest/?badge=latest)
