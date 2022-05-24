[![Python Versions](https://img.shields.io/pypi/pyversions/HAPI-Nile.png)](https://img.shields.io/pypi/pyversions/HAPI-Nile)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MAfarrag/Hapi.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MAfarrag/Hapi/context:python)



![GitHub last commit](https://img.shields.io/github/last-commit/MAfarrag/Hapi)
![GitHub forks](https://img.shields.io/github/forks/MAfarrag/pyramids?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/MAfarrag/pyramids?style=social)


Current release info
====================

| Name | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Version | Platforms |
| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-hapi-green.svg)](https://anaconda.org/conda-forge/hapi) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![Downloads](https://pepy.tech/badge/hapi-nile)](https://pepy.tech/project/hapi-nile) [![Downloads](https://pepy.tech/badge/hapi-nile/month)](https://pepy.tech/project/hapi-nile)  [![Downloads](https://pepy.tech/badge/hapi-nile/week)](https://pepy.tech/project/hapi-nile)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/hapi-nile?color=blue&style=flat-square) ![GitHub all releases](https://img.shields.io/github/downloads/MAfarrag/Hapi/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/MAfarrag/hapi/1.2.3/total) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![PyPI version](https://badge.fury.io/py/HAPI-Nile.svg)](https://badge.fury.io/py/HAPI-Nile) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/hapi/badges/version.svg)](https://anaconda.org/conda-forge/hapi) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |


Hapi - Hydrological library for Python
=====================================================================
**Hapi** is an open-source Python Framework for building raster-based conceptual distributed hydrological models using HBV96 lumped
model & Muskingum routing method at a catchment scale (Farrag & Corzo, 2021), Hapi gives a high degree of flexibility to all components of the model
(spatial discretization - cell size, temporal resolution, parameterization approaches and calibration (Farrag et al., 2021)).


![1](/docs/img/Picture1.png)  ![2](/docs/img/Picture2.png)

Hapi

Main Features
-------------
  - Modified version of HBV96 hydrological model (Bergström, 1992) with 15 parameters in case of considering
   snow processes, and 10 parameters without snow, in addition to 2 parameters of Muskingum routing method
  - Remote sensing module to download the meteorological inputs required for the hydrologic model simulation (ECMWF)
  - GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
    needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
    convert different forms of distributed data (rasters, NetCDF, shapefiles)
  - Sensitivity analysis module based on the concept of one-at-a-time OAT and analysis of the interaction among
    model parameters using the Sobol concept ((Rusli et al., 2015)) and a visualization
  - Statistical module containing interpolation methods for generating distributed data from gauge data, some
    distribution for frequency analysis and Maximum likelihood method for distribution parameter estimation.
  - Visualization module for animating the results of the distributed model, and the meteorological inputs
  - Optimization module, for calibrating the model based on the Harmony search method

The recent version of Hapi (Hapi 1.0.1) integrates the global hydrological parameters obtained by Beck et al., (2016),
to reduce model complexity and uncertainty of parameters.

Future work
-------------
  - Developing a regionalization method for connection model parameters with some catchment characteristics for better model calibration.
  - Developing and integrate river routing method (kinematic and diffusive wave approximation)
  - Apply the model for large scale (regional/continental) cases
  - Developing a DEM processing module for generating the river network at different DEM spatial resolutions.

For using Hapi please cite Farrag et al. (2021) and Farrag & Corzo (2021)

IHE-Delft sessions
------------------
- In April 14-15 we had a two days session for Masters and PhD student in IHE-Delft to explain the different modules and the distributed hydrological model in Hapi [Day 1](https://youtu.be/HbmUdN9ehSo) ,  [Day 2](https://youtu.be/m7kHdOFQFIY)

References
-------------
Farrag, M. & Corzo, G. (2021) MAfarrag/Hapi: Hapi. doi:10.5281/ZENODO.4662170

Farrag, M., Perez, G. C. & Solomatine, D. (2021) Spatio-Temporal Hydrological Model Structure and Parametrization Analysis. J. Mar. Sci. Eng. 9(5), 467. doi:10.3390/jmse9050467 [Link](https://www.researchgate.net/publication/351143581_Spatio-Temporal_Hydrological_Model_Structure_and_Parametrization_Analysis)

Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles, T. R. M. & Jaap Schellekens,  and L. A. B. (2016) Global-scale regionalization of hydrologic model parameters-Supporting materials 3599–3622. doi:10.1002/2015WR018247.Received

Bergström, S. (1992) The HBV model - its structure and applications. Smhi Rh 4(4), 35.

Rusli, S. R., Yudianto, D. & Liu, J. tao. (2015) Effects of temporal variability on HBV model calibration. Water Sci. Eng. 8(4), 291–300. Elsevier Ltd. doi:10.1016/j.wse.2015.12.002


Installing hapi
===============

Installing `hapi` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge hapi
```

It is possible to list all of the versions of `hapi` available on your platform with:

```
conda search hapi --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/MAfarrag/HAPI
```

## pip
to install the last release you can easly use pip
```
pip install HAPI-Nile==1.2.3
```

Quick start
===========

```
  >>> import Hapi
```

[other code samples](https://hapi-hm.readthedocs.io/en/latest/?badge=latest)
