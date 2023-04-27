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

0.3.3 (2023-02-06)
------------------
* fix bug in reading the ogr drivers catalog for the vector class
* fix bug in creating rasterLike in the asciiToRaster method

0.4.0 (2023-04-11)
------------------
* Restructure the whole package to two main objects Dataset and FeatureCollection
* Add class for multiple Dataset "DataCube".
* Link both Dataset and FeatureCollection to convert between raster and vector data types.
* Remove rasterio and netcdf from dependencies and depend only on gdal.
* Test read rasters/netcdf from virtual file systems (aws, compressed)
* Add dunder methods for all classes.
* add plotting functionality and cleopatra (plotting package) as an optional package.
* remove loops and replace it with ufunc from numpy.

0.4.1 (2023-04-23)
------------------
* adjust all spatial operation functions to work with multi-band rasters.
* use gdal exceptions to capture runtime error of not finding the the file.
* add cluster method to dataset class.
* time_stamp attribute returns None if there is no time_stamp.
* restructure the no_data_value related functions.
* plot function can plot rgb imagr for multi-band rasters.
* to_file detect the driver type from the extension in the path.

0.4.2 (2023-04-27)
------------------
* fix bug in plotting dataset without specifying the band
* fix bug in passing ot not passing band index in case of multi band rasters
* change the bounds in to_dataset method to total_bounds tp get the bbox of the whole geometries in the gdf
* add convert_longitude method to convert longitude to range between -180 and 180
