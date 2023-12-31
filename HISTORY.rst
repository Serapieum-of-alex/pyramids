=======
History
=======

0.1.0 (2022-05-24)
------------------

* First release on PyPI.

0.2.9 (2022-12-19)
------------------

* Use environment.yaml and requirements.txt instead of pyproject.toml and replace poetry env by conda env

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
* add H3 indexing package, and add new module indexing with functions to convert geometries to indices back and forth.
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
* plot function can plot rgb image for multi-band rasters.
* to_file detect the driver type from the extension in the path.

0.4.2 (2023-04-27)
------------------
* fix bug in plotting dataset without specifying the band
* fix bug in passing ot not passing band index in case of multi band rasters
* change the bounds in to_dataset method to total_bounds tp get the bbox of the whole geometries in the gdf
* add convert_longitude method to convert longitude to range between -180 and 180

0.5.0 (2023-10-01)
------------------
Dataset
"""""""
* The dtype attribute is not initialized in the __init__, but initialized when using the dtype property.
* Create band_names setter method.
* add gdal_dtype, numpy_dtype, and dtype attribute to change between the dtype (data type general name), and the corresponding data type in numpy and gdal.
* Create color_table attribute, getter & setter property to set a symbology (assign color to different values in each band).
* The read_array method returns array with the same type as the dtype of the first band in the raster.
* add a setter method to the band_names property.
* The methods (create_driver_from_scratch, get_band_names) is converted to private method.
* The no_data_value check the dtype before setting any value.
* The convert_longitude used the gdal.Wrap method instead of making the calculation step by step.
* The to_polygon is converted to _band_to_polygon private method used in the clusters method.
* The to_geodataframe is converted to to_feature_collection. change the backend of the function to use the crop function is a vector mask is given.
* the to_crs takes an extra parameter "inplace".
* The locate_points is converted to map_to_array_coordinates.
* Create array_to_map_coordinates to translate the array indices into real map coordinates.

DataCube
""""""""
* rename the read_separate_files to read_multiple_files, and enable it to use regex strings to filter files in a given directory.
* rename read_dataset to open_datacube.
* rename the data attribute to values

FeatureCollection
"""""""""""""""""
* Add a pivot_point attribute to return the top left corner/first coordinates of the polygon.
* Add a layers_count property to return the number of layers in the file.
* Add a layer_names property to return the layers names.
* Add a column property to return column names.
* Add the file_name property to store the file name.
* Add the dtypes property to retrieve the data types of the columns in the file.
* Rename bounds to total_bounds.
* The _gdf_to_ds can convert the GeoDataFrame to a ogr.DataSource and to a gdal.Dataset.
* The create_point method returns a shapely point object or a GeoDataFrame if an epsg number is given.

0.5.1 (2023-11-27)
------------------
Dataset
"""""""
* revert the convert_longitude method to not use the gdal_wrap method as it is not working with the new version of gdal (newer tan 3.7.1).
* bump up versions.

0.5.2 (2023-12-27)
------------------
Dataset
"""""""
* add _iloc method to get the gdal band object by index.
* add stats method to calculate the statistics of the raster bands.

0.5.3 (2023-12-28)
------------------
Dataset
"""""""
* Introduce a new parameter touch to the crop method in the Dataset to enable considering the cells that most of the
cell lies inside the mask, not only the cells that lie entirely inside the mask.
* Introduce a new parameter inplace to the crop method in the Dataset to enable replacing the dataset object with the
new cropped dataset.
* Adjust the stats method to take a mask to calculate the stats inside this mask.

0.5.4 (2023-12-31)
------------------
Dataset
"""""""
* fix the un-updated array dimension bug in the crop method when the mask is a vector mask and the touch parameter is
True.
