
Current release info
====================

.. image:: https://img.shields.io/pypi/v/pyramids-gis
  :target: https://pypi.org/project/pyramids-gis/0.1.0/


.. image:: https://img.shields.io/conda/v/conda-forge/pyramids?label=conda-forge
  :target: https://pypi.org/project/pyramids-gis/0.1.0/


.. image:: https://img.shields.io/pypi/pyversions/pyramids-gis
  :target: https://pypi.org/project/pyramids-gis/0.1.0/


.. image:: https://img.shields.io/github/forks/mafarrag/hapi?style=social   :alt: GitHub forks


.. image:: https://anaconda.org/conda-forge/pyramids-gis/badges/downloads.svg
  :target: https://anaconda.org/conda-forge/pyramids


.. image:: https://img.shields.io/conda/vn/conda-forge/pyramids   :alt: Conda (channel only)
  :target: https://pypi.org/project/pyramids-gis/0.1.0/


.. image:: https://img.shields.io/gitter/room/mafarrag/pyramids
  :alt: Gitter


.. image:: https://static.pepy.tech/personalized-badge/pyramids-gis?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
  :target: https://pypi.org/project/pyramids-gis/0.1.0/


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




pyramids - GIS utility package
=====================================================================


**pyramids** is a Python package providing utility function to manipulate various types and formats of geo-spatial data

Main Features
-------------
- GDAL API that tackle various processes using gdal python functions, and gdal CLI.
- rasterio api that uses rasterio function in manipulating
- GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing
  needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and
  convert different forms of distributed data (rasters, NetCDF, shapefiles)

.. digraph:: Linking

    pyramids -> raster;
    pyramids -> vector;
    pyramids -> catchment;
    pyramids -> netcdf;
    pyramids -> convert;
    dpi=200;

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <00Installation.rst>
   Tutorial <03tutorial.rst>
   GIS <05GIS.rst>
