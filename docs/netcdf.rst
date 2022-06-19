******
netCDF
******

getProjectionData
-----------------

- GetProjectionData returns the projection details of a given gdal.Dataset

Parameters
==========

    src: [gdal.Dataset]
        raster read by gdal

Returns
=======

    epsg: [integer]
         integer reference number that defines the projection (https://epsg.io/)
    geo: [tuple]
        geotransform data of the upper left corner of the raster
        (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation, pixelsize).
