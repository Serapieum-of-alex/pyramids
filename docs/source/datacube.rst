########
DataCube
########

- `DataCube` class contains  is made to operate in multiple single files .

- DataCube represent a stack of rasters which have the same dimensions, contains data that have same dimensions (rows
    & columns).

.. image:: /_images/datacube/logo.png
   :width: 700pt
   :align: center

The datacube object has some attributes and methods to help working with multiple rasters files, or to repeat thesame
operation on multiple rasters.

- To import the raster module

.. code:: py

    from pyramids.dataset import Datacube

- The detailed module attributes and methods are summarized in the following figure.

.. image:: /_images/datacube/detailed.png
   :width: 700pt
   :align: center

**********
Attributes
**********

The `DataCube` object will have the following attributes

#. base: Dataset object
#. columns: number of columns in the dataset.
#. rows: number of rows in the dataset.
#. time_length: number of files/considering the each file represent a timestamp.
#. shape: (time_length, rows, columns).
#. files: file that have been read.

.. image:: /_images/datacube/attributes.png
   :width: 150pt
   :align: center


*******
Methods
*******


===================
read_multiple_files
===================

- `read_multiple_files` parse files in a directory and construct the array with the dimension of the first reads
    rasters from a folder and creates a 3d array with the same 2d dimensions of the first raster in the folder and length
    as the number of files.

inside the folder.
    - All rasters should have the same dimensions
    - If you want to read the rasters with a certain order, then all raster file names should have a date that follows
        the same format (YYYY.MM .DD / YYYY-MM-DD or YYYY_MM_DD) (i.e. "MSWEP_1979.01.01.tif").

.. note::

    `read_multiple_files` only parse the files names' in the given directory, to open each raster and read a specific,
    band from each raster and add it to the `DataCube` you have to do one step further using the `open_datacube`_ method.

Parameters
----------
path:[str/list]
    path of the folder that contains all the rasters, ora list contains the paths of the rasters to read.
with_order: [bool]
    True if the rasters names' follows a certain order, then the rasters names should have a date that follows
    the same format (YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD).
    >>> "MSWEP_1979.01.01.tif"
    >>> "MSWEP_1979.01.02.tif"
    >>> ...
    >>> "MSWEP_1979.01.20.tif"
regex_string: [str]
    a regex string that we can use to locate the date in the file names.Default is r"\d{4}.\d{2}.\d{2}".
    >>> fname = "MSWEP_YYYY.MM.DD.tif"
    >>> regex_string = r"\d{4}.\d{2}.\d{2}"
    - or
    >>> fname = "MSWEP_YYYY_M_D.tif"
    >>> regex_string = r"\d{4}_\d{1}_\d{1}"
    - if there is a number at the beginning of the name
    >>> fname = "1_MSWEP_YYYY_M_D.tif"
    >>> regex_string = r"\d+"
date: [bool]
    True if the number in the file name is a date. Default is True.
file_name_data_fmt : [str]
    if the files names' have a date and you want to read them ordered .Default is None
    >>> "MSWEP_YYYY.MM.DD.tif"
    >>> file_name_data_fmt = "%Y.%m.%d"
start: [str]
    start date if you want to read the input raster for a specific period only and not all rasters,
    if not given all rasters in the given path will be read.
end: [str]
    end date if you want to read the input temperature for a specific period only,
    if not given all rasters in the given path will be read.
fmt: [str]
    format of the given date in the start/end parameter.
extension: [str]
    the extension of the files you want to read from the given path. Default is ".tif".

Cases
-----

with_order = False
^^^^^^^^^^^^^^^^^^
- if you want to make some mathematical operation on all the raster, then the order of the rasters does not matter.

.. code:: py

    rasters_folder_path = "examples/data/geotiff/raster-folder"
    datacube = Datacube.read_multiple_files(rasters_folder_path)
    print(datacube)
    >>>     Files: 6
    >>>     Cell size: 5000.0
    >>>     EPSG: 4647
    >>>     Dimension: 125 * 93
    >>>     Mask: 2147483648.0

with_order = True
^^^^^^^^^^^^^^^^^
- If the order in which each raster represent is important (each raster is represents a time stamp)
- To read the rasters with a certain order, each raster has to have a date in its file name, and using the format of
    this name the method is going to read the file in right order.

- the raster directory contents are files with a date in each file name

.. code:: py

    >>> MSWEP_1979.01.01.tif
    >>> MSWEP_1979.01.02.tif
    >>> MSWEP_1979.01.03.tif
    >>> MSWEP_1979.01.04.tif
    >>> MSWEP_1979.01.05.tif
    >>> MSWEP_1979.01.06.tif


.. code:: py

    rasters_folder_path = "examples/data/geotiff/raster-folder"
    datacube = Datacube.read_multiple_files(
        rasters_folder_path, regex_string=r"\d{4}.\d{2}.\d{2}", date=True, file_name_data_fmt="%Y.%m.%d",
    )
    print(datacube)
    >>>     Files: 6
    >>>     Cell size: 5000.0
    >>>     EPSG: 4647
    >>>     Dimension: 125 * 93
    >>>     Mask: 2147483648.0

- the raster directory contents are files with a number in each file name

.. code:: py

    >>> 0_MSWEP.tif
    >>> 1_MSWEP.tif
    >>> 2_MSWEP.tif
    >>> 3_MSWEP.tif
    >>> 4_MSWEP.tif

.. code:: py

    rasters_folder_path = "tests/data/geotiff/rhine"
    datacube = Datacube.read_multiple_files(
        rasters_folder_path, with_order=True, regex_string=r"\d+", date=False,
    )
    print(datacube)
    >>>     Files: 3
    >>>     Cell size: 5000.0
    >>>     EPSG: 4647
    >>>     Dimension: 125 * 93
    >>>     Mask: 2147483648.0

=============
open_datacube
=============

- After using the `read_multiple_files` method to parse the files in the directory, you can read the values of a
    specific band from each raster using the `open_datacube` method.


.. code:: py

    rasters_folder_path = "examples/data/geotiff/raster-folder"
    datacube = Datacube.read_multiple_files(rasters_folder_path, file_name_data_fmt="%Y.%m.%d", separator=".")
    dataset.open_datacube()
    print(dataset.values.shape)
    >>>     (6, 125, 93)


===========
create_cube
===========
- Create a `DataCube` object.

===========
Propteties
===========
- update the data in the `DataCube` object


.. automodule:: pyramids.dataset
   :members: DataCube
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
