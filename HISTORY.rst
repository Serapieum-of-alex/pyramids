=======
History
=======

0.1.0 (2022-05-24)
------------------

* First release on PyPI.

0.2.9 (2022-12-19)
------------------

* Use environment.yaml and requirements.txt instead of pyproject.toml and replace poetry env byconda env

0.2.10 (2022-12-25)
------------------

* lock numpy version to 1.23.5 as conda-forge can not install 1.24.0

0.2.11 (2022-12-27)
------------------

* fix bug in pypi package names in requirements.txt file

0.2.11 (2023-01-14)
------------------

* add utils module for functions dealing with compressing files, and others utility functions

0.3.0 (2023-01-23)
------------------

* add array module to deal with any array operations.
* add openDataset, getEPSG create SpatialReference, and setNoDataValue utility function, getCellCoords, ...
* add rasterToPolygon, PolygonToRaster, rasterToGeoDataFrame, conversion between ogr DataSource and GeoDataFrame.

0.3.1 (2023-01-25)
------------------
* add pyarrow to use parquet data type for saving dataframes and geodataframes
* add H3 indexing package, and add new module indexing with functions to convert geometries to indeces back and forth.
* fix bug in calculating pivot point of the netcdf file
* rasterToDataFrame function will create geometries of the cells only based on the add_geometry parameter.

0.3.2 (2023-01-29)
------------------
* refactor code
* add documentation
* fix creating memory driver with compression in _createDataset
