
Current release info
====================

.. image:: https://readthedocs.org/projects/pyramids-gis/badge/?version=latest
    :target: https://pyramids-gis.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/pyramids-gis
  :target: https://pypi.org/project/pyramids-gis


.. image:: https://img.shields.io/conda/v/conda-forge/pyramids?label=conda-forge
  :target: https://pypi.org/project/pyramids-gis


.. image:: https://img.shields.io/pypi/pyversions/pyramids-gis
  :target: https://pypi.org/project/pyramids-gis


.. image:: https://img.shields.io/github/forks/mafarrag/hapi?style=social   :alt: GitHub forks


.. image:: https://anaconda.org/conda-forge/pyramids/badges/downloads.svg
  :target: https://anaconda.org/conda-forge/pyramids-gis


.. image:: https://img.shields.io/conda/vn/conda-forge/pyramids   :alt: Conda (channel only)
  :target: https://pypi.org/project/pyramids-gis


.. image:: https://img.shields.io/gitter/room/mafarrag/pyramids
  :alt: Gitter


.. image:: https://static.pepy.tech/personalized-badge/pyramids-gis?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pypi.org/project/pyramids-gis


.. image:: https://img.shields.io/pypi/v/pyramids-gis
  :alt: PyPI


.. image:: https://anaconda.org/conda-forge/pyramids/badges/platforms.svg
  :target: https://anaconda.org/conda-forge/pyramids


.. image:: https://static.pepy.tech/personalized-badge/pyramids-gis?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/pyramids-gis



.. image:: https://static.pepy.tech/personalized-badge/pyramids-gis?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/pyramids-gis


.. image:: https://static.pepy.tech/personalized-badge/pyramids-gis?period=week&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pepy.tech/project/pyramids-gis


.. image:: https://coveralls.io/repos/github/MAfarrag/pyramids/badge.svg?branch=main
  :target: https://coveralls.io/github/MAfarrag/pyramids?branch=main

.. image:: https://img.shields.io/github/commits-since/mafarrag/pyramids/0.3.1?include_prereleases&style=plastic
    :alt: GitHub commits since latest release (by SemVer including pre-releases)

.. image:: https://img.shields.io/github/last-commit/mafarrag/pyramids
    :alt: GitHub last commit

pyramids - GIS utility package
==============================


**pyramids** is a Python package providing utility function to manipulate various types and formats of geo-spatial data

Main Features
-------------
- GDAL/ogr API that tackle various processes using gdal python functions, and gdal CLI.
- rasterio api that uses rasterio function in manipulating
- GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
  needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
  convert different forms of distributed data (rasters, NetCDF, shapefiles)

.. digraph:: Linking

    pyramids -> raster;
    pyramids -> vector;
    pyramids -> netcdf;
    pyramids -> convert;
    pyramids -> array;
    pyramids -> catchment;
    pyramids -> indexing;
    pyramids -> utils;
    dpi=200;

.. toctree::
   :hidden:
   :maxdepth: 1

    Installation <installation.rst>
    Raster <raster.rst>
    Vector <vector.rst>
    NetCDF <netcdf.rst>
    Convert <convert.rst>
    Indeixing <indexing.rst>
    Catchment <catchment.rst>
