"""Array I/O and file serialization mixin for Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, ogr
from osgeo_utils import gdal2xyz
from pandas import DataFrame

from pyramids import _io
from pyramids.base._errors import (
    FailedToSaveError,
    OutOfBoundsError,
    ReadOnlyError,
)
from pyramids.base._utils import (
    DTYPE_CONVERSION_DF,
    gdal_to_numpy_dtype,
    numpy_to_gdal_dtype,
)
from pyramids.dataset.abstract_dataset import CATALOG, OVERVIEW_LEVELS, RESAMPLING_METHODS
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


class IO:

    def read_array(
        self: Dataset, band: int | None = None, window: GeoDataFrame | list[int] | None = None
    ) -> np.ndarray:
        """Read the values stored in a given band.

        Data Chuncks/blocks
            When a raster dataset is stored on disk, it might not be stored as one continuous chunk of data. Instead,
            it can be divided into smaller rectangular blocks or tiles. These blocks can be individually accessed,
            which is particularly useful for large datasets:

                - Efficiency: Reading or writing small blocks requires less memory than dealing with the entire dataset
                      at once. This is especially beneficial when only a small portion of the data needs to be processed.
                - Performance: For certain file formats and operations, working with optimal block sizes can significantly
                      improve performance. For example, if the block size matches the reading or processing window,
                      Pyramids can minimize disk access and data transfer.

        Args:
            band (int, optional):
                The band you want to get its data. If None, data of all bands will be read. Default is None.
            window (List[int] | GeoDataFrame, optional):
                Specify a block of data to read from the dataset. The window can be specified in two ways:

                - List:
                    Window specified as a list of 4 integers [offset_x, offset_y, window_columns, window_rows].

                    - offset_x/column index: x offset of the block.
                    - offset_y/row index: y offset of the block.
                    - window_columns: number of columns in the block.
                    - window_rows: number of rows in the block.

                - GeoDataFrame:
                    GeoDataFrame with a geometry column filled with polygon geometries; the function will get the
                    total_bounds of the GeoDataFrame and use it as a window to read the raster.

        Returns:
            np.ndarray:
                array with all the values in the raster.

        Examples:
            - Create `Dataset` consisting of 4 bands, 5 rows, and 5 columns at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```

            - Read all the values stored in a given band:

              ```python
              >>> arr = dataset.read_array(band=0) # doctest: +SKIP
              array([[0.50482225, 0.45678043, 0.53294294, 0.28862223, 0.66753579],
                     [0.38471912, 0.14617829, 0.05045189, 0.00761358, 0.25501918],
                     [0.32689036, 0.37358843, 0.32233918, 0.75450564, 0.45197608],
                     [0.22944676, 0.2780928 , 0.71605189, 0.71859309, 0.61896933],
                     [0.47740168, 0.76490779, 0.07679277, 0.16142599, 0.73630836]])

              ```

            - Read a 2x2 block from the first band. The block starts at the 2nd column (index 1) and 2nd row (index 1)
                (the first index is the column index):

              ```python
              >>> arr = dataset.read_array(band=0, window=[1, 1, 2, 2])
              >>> print(arr) # doctest: +SKIP
              array([[0.14617829, 0.05045189],
                     [0.37358843, 0.32233918]])

              ```

            - If you check the values of the 2x2 block, you will find them the same as the values in the entire array
                of band 0, starting at the 2nd row and 2nd column.

            - Read a block using a GeoDataFrame polygon that covers the same area as the window above:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import Polygon
              >>> poly = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])], crs=4326)
              >>> arr = dataset.read_array(band=0, window=poly)
              >>> print(arr) # doctest: +SKIP
              array([[0.14617829, 0.05045189],
                     [0.37358843, 0.32233918]])

              ```

        See Also:
            - Dataset.get_tile: Read the dataset in chunks.
            - Dataset.get_block_arrangement: Get block arrangement to read the dataset in chunks.
        """
        if band is None and self.band_count > 1:
            rows = self.rows if window is None else window[3]
            columns = self.columns if window is None else window[2]
            arr = np.ones(
                (
                    self.band_count,
                    rows,
                    columns,
                ),
                dtype=self.numpy_dtype[0],
            )

            for i in range(self.band_count):
                if window is None:
                    # this line could be replaced with the following line
                    # arr[i, :, :] = self._iloc(i).ReadAsArray()
                    arr[i, :, :] = self._raster.GetRasterBand(i + 1).ReadAsArray()
                else:
                    arr[i, :, :] = self._read_block(i, window)
        else:
            # given band number or the raster has only one band
            if band is None:
                band = 0
            else:
                if band > self.band_count - 1:
                    raise ValueError(
                        f"band index should be between 0 and {self.band_count - 1}"
                    )
            if window is None:
                arr = self._iloc(band).ReadAsArray()
            else:
                arr = self._read_block(band, window)

        return arr

    def _read_block(
        self: Dataset, band: int, window: list[int] | GeoDataFrame | None = None
    ) -> np.ndarray:
        """Read block of data from the dataset.

        Args:
            band (int):
                Band index.
            window (List[int] | GeoDataFrame):
                - List[int]: Window to specify a block of data to read from the dataset.
                    The window should be a list of 4 integers [offset_x, offset_y, window_columns, window_rows].
                    - offset_x: x offset of the block.
                    - offset_y: y offset of the block.
                    - window_columns: number of columns in the block.
                    - window_rows: number of rows in the block.
                - GeoDataFrame:
                    A GeoDataFrame with a polygon geometry. The function will get the total_bounds of the
                    GeoDataFrame and use it as a window to read the raster.

        Returns:
            np.ndarray:
                Array with the values of the block. The shape of the array is (window[2], window[3]), and the
                location of the block in the raster is (window[0], window[1]).
        """
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
        if not isinstance(window, (list, tuple)):
            raise ValueError(f"window must be a list of 4 integers, got {type(window)}")
        try:
            block = self._iloc(band).ReadAsArray(
                window[0], window[1], window[2], window[3]
            )
        except Exception as e:
            if e.args[0].__contains__("Access window out of range in RasterIO()"):
                raise OutOfBoundsError(
                    f"The window you entered ({window})is out of the raster bounds: {self.rows, self.columns}"
                )
            else:
                raise e
        return np.asarray(block)

    def _convert_polygon_to_window(
        self: Dataset, poly: GeoDataFrame | FeatureCollection
    ) -> list[Any]:
        poly = FeatureCollection(poly)
        bounds = poly.total_bounds
        df = pd.DataFrame(columns=["id", "x", "y"])
        df.loc["top_left", ["x", "y"]] = bounds[0], bounds[3]
        df.loc["bottom_right", ["x", "y"]] = bounds[2], bounds[1]
        arr_indeces = self.map_to_array_coordinates(df)
        xoff = arr_indeces[0, 1]
        yoff = arr_indeces[0, 0]
        x_size = arr_indeces[1, 0] - arr_indeces[0, 0]
        y_size = arr_indeces[1, 1] - arr_indeces[0, 1]
        return [xoff, yoff, x_size, y_size]

    def write_array(self: Dataset, array: np.ndarray, top_left_corner: list[int]):
        """Write an array to the dataset at the given xoff, yoff position.

        Args:
            array (np.ndarray):
                The array to write
            top_left_corner (list[int]):
                indices [row, column]/[y_offset, x_offset] of the cell to write the array to.

        Raises:
            Exception: If the array is not written successfully.

        Hint:
            - The `Dataset` has to be opened in a write mode `read_only=False`.

        Returns:
        None

        Examples:
            - First, create a dataset on disk:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> path = 'write_array.tif'
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326, path=path
              ... )
              >>> dataset = None

              ```

            - In a later session you can read the dataset in a `write` mode and update it:

              ```python
              >>> dataset = Dataset.read_file(path, read_only=False)
              >>> arr = np.array([[1, 2], [3, 4]])
              >>> dataset.write_array(arr, top_left_corner=[1, 1])
              >>> dataset.read_array()    # doctest: +SKIP
              array([[0.77359738, 0.64789596, 0.37912658, 0.03673771, 0.69571106],
                     [0.60804387, 1.        , 2.        , 0.501909  , 0.99597122],
                     [0.83879291, 3.        , 4.        , 0.33058081, 0.59824467],
                     [0.774213  , 0.94338147, 0.16443719, 0.28041457, 0.61914179],
                     [0.97201104, 0.81364799, 0.35157525, 0.65554998, 0.8589739 ]])

              ```
        """
        yoff, xoff = top_left_corner
        try:
            self._raster.WriteArray(array, xoff=xoff, yoff=yoff)
            self._raster.FlushCache()
        except Exception as e:
            raise e

    def get_block_arrangement(
        self: Dataset,
        band: int = 0,
        x_block_size: int | None = None,
        y_block_size: int | None = None,
    ) -> DataFrame:
        """Get Block Arrangement.

        Args:
            band (int, optional):
                band index, by default 0
            x_block_size (int, optional):
                x block size/number of columns, by default None
            y_block_size (int, optional):
                y block size/number of rows, by default None

        Returns:
            DataFrame:
                with the following columns: [x_offset, y_offset, window_xsize, window_ysize]

        Examples:
            - Example of getting block arrangement:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(13, 14)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> df = dataset.get_block_arrangement(x_block_size=5, y_block_size=5)
              >>> print(df)
                 x_offset  y_offset  window_xsize  window_ysize
              0         0         0             5             5
              1         5         0             5             5
              2        10         0             4             5
              3         0         5             5             5
              4         5         5             5             5
              5        10         5             4             5
              6         0        10             5             3
              7         5        10             5             3
              8        10        10             4             3

              ```
        """
        block_sizes = self.block_size[band]
        x_block_size = block_sizes[0] if x_block_size is None else x_block_size
        y_block_size = block_sizes[1] if y_block_size is None else y_block_size

        df = pd.DataFrame(
            [
                {
                    "x_offset": x,
                    "y_offset": y,
                    "window_xsize": min(x_block_size, self.columns - x),
                    "window_ysize": min(y_block_size, self.rows - y),
                }
                for y in range(0, self.rows, y_block_size)
                for x in range(0, self.columns, x_block_size)
            ],
            columns=["x_offset", "y_offset", "window_xsize", "window_ysize"],
        )
        return df

    def to_file(
        self: Dataset,
        path: str | Path,
        band: int = 0,
        tile_length: int | None = None,
        creation_options: list[str] | None = None,
    ) -> None:
        """Save dataset to tiff file.

            `to_file` saves a raster to disk, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Args:
            path (str):
                A path including the name of the dataset.
            band (int):
                Band index, needed only in case of ascii drivers. Default is 0.
            tile_length (int, optional):
                Length of the tiles in the driver. Default is 256.
            creation_options: List[str], Default is None
                List of strings that will be passed to the GDAL driver during the creation of the dataset.
                i.e., ['PREDICTOR=2']

        Examples:
            - Create a Dataset with 4 bands, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.file_name)
              <BLANKLINE>

              ```

            - Now save the dataset as a geotiff file:

              ```python
              >>> dataset.to_file("my-dataset.tif")
              >>> print(dataset.file_name)
              my-dataset.tif

              ```
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"path input should be string or Path type, given: {type(path)}"
            )

        path = Path(path)
        extension = path.suffix[1:]
        driver = CATALOG.get_driver_name_by_extension(extension)
        driver_name = CATALOG.get_gdal_name(driver)

        if driver == "ascii":
            arr = self.read_array(band=band)
            no_data_value = self.no_data_value[band]
            xmin, ymin, _, _ = self.bbox
            _io.to_ascii(arr, self.cell_size, xmin, ymin, no_data_value, path)
        else:
            # saving rasters with color table fails with a runtime error
            options = ["COMPRESS=DEFLATE"]
            if tile_length is not None:
                options += [
                    "TILED=YES",
                    f"TILE_LENGTH={tile_length}",
                ]
                if self._block_size is not None and self._block_size != []:
                    options += [
                        "BLOCKXSIZE={}".format(self._block_size[0][0]),
                        "BLOCKYSIZE={}".format(self._block_size[0][1]),
                    ]
            if creation_options is not None:
                options += creation_options

            try:
                self.raster.FlushCache()
                dst = gdal.GetDriverByName(driver_name).CreateCopy(
                    str(path), self.raster, 0, options=options
                )
                self._update_inplace(dst, "write")
                # flush the data to the dataset on disk.
                dst.FlushCache()
            except RuntimeError:
                if not path.exists():
                    raise FailedToSaveError(
                        f"Failed to save the {driver_name} raster to the path: {path}"
                    )

    def _window(self: Dataset, size: int = 256):
        """Dataset square window size/offsets.

        Args:
            size (int):
                Size of the window in pixels. One value required which is used for both the x and y size. e.g.,
                256 means a 256x256 window. Default is 256.

        Yields:
            tuple[int, int, int, int]:
                (x-offset/column-index, y-offset/row-index, x-size, y-size).

        Examples:
            - Generate 2x2 windows over a 3x5 dataset:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> tile_dimensions = list(dataset._window(2))
              >>> print(tile_dimensions)
              [(0, 0, 2, 2), (2, 0, 2, 2), (4, 0, 1, 2), (0, 2, 2, 1), (2, 2, 2, 1), (4, 2, 1, 1)]

              ```
        """
        cols = self.columns
        rows = self.rows
        for yoff in range(0, rows, size):
            ysize = size if size + yoff <= rows else rows - yoff
            for xoff in range(0, cols, size):
                xsize = size if size + xoff <= cols else cols - xoff
                yield xoff, yoff, xsize, ysize

    def get_tile(self: Dataset, size=256) -> Generator[np.ndarray]:
        """Get tile.

        Args:
            size (int):
                Size of the window in pixels. One value is required which is used for both the x and y size. e.g., 256
                means a 256x256 window. Default is 256.

        Yields:
            np.ndarray:
                Dataset array with a shape `[band, y, x]`.

        Examples:
            - First, we will create a dataset with 3 rows and 5 columns.

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 3 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              >>> print(dataset.read_array())   # doctest: +SKIP
              [[0.55332314 0.48364841 0.67794589 0.6901816  0.70516817]
               [0.82518332 0.75657103 0.45693945 0.44331782 0.74677865]
               [0.22231314 0.96283065 0.15201337 0.03522544 0.44616888]]

              ```
            - The `get_tile` method splits the domain into tiles of the specified `size` using the `_window` function.

              ```python
              >>> tile_dimensions = list(dataset._window(2))
              >>> print(tile_dimensions)
              [(0, 0, 2, 2), (2, 0, 2, 2), (4, 0, 1, 2), (0, 2, 2, 1), (2, 2, 2, 1), (4, 2, 1, 1)]

              ```
              ![get_tile](./../../_images/dataset/get_tile.png)

            - So the first two chunks are 2*2, 2*1 chunk, then two 1*2 chunks, and the last chunk is 1*1.
            - The `get_tile` method returns a generator object that can be used to iterate over the smaller chunks of
                the data.

              ```python
              >>> tiles_generator = dataset.get_tile(size=2)
              >>> print(tiles_generator)  # doctest: +SKIP
              <generator object Dataset.get_tile at 0x00000145AA39E680>
              >>> print(list(tiles_generator))  # doctest: +SKIP
              [
                  array([[0.55332314, 0.48364841],
                         [0.82518332, 0.75657103]]),
                  array([[0.67794589, 0.6901816 ],
                         [0.45693945, 0.44331782]]),
                  array([[0.70516817], [0.74677865]]),
                  array([[0.22231314, 0.96283065]]),
                  array([[0.15201337, 0.03522544]]),
                  array([[0.44616888]])
              ]

              ```
        """
        for xoff, yoff, xsize, ysize in self._window(size=size):
            # read the array at certain indices
            yield self.raster.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )

    def to_xyz(
        self: Dataset, bands: list[int] | None = None, path: str | Path | None = None
    ) -> DataFrame | None:
        """Convert to XYZ.

        Args:
            path (str, optional):
                path to the file where the data will be saved. If None, the data will be returned as a DataFrame.
                default is None.
            bands (List[int], optional):
                indices of the bands. If None, all bands will be used. default is None

        Returns:
            DataFrame/File:
                DataFrame with columns: lon, lat, band_1, band_2,... . If a path is provided the data will be saved to
                disk as a .xyz file

        Examples:
            - First we will create a dataset from a float32 array with values between 1 and 10, and then we will
                assign a scale of 0.1 to the dataset.
                ```python
                >>> import numpy as np
                >>> arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326)
                >>> print(dataset)
                <BLANKLINE>
                            Top Left Corner: (0.0, 0.0)
                            Cell size: 0.05
                            Dimension: 2 * 2
                            EPSG: 4326
                            Number of Bands: 2
                            Band names: ['Band_1', 'Band_2']
                            Band colors: {0: 'undefined', 1: 'undefined'}
                            Band units: ['', '']
                            Scale: [1.0, 1.0]
                            Offset: [0, 0]
                            Mask: -9999.0
                            Data type: int64
                            File: ...
                <BLANKLINE>
                >>> df = dataset.to_xyz()
                >>> print(df)
                     lon    lat  Band_1  Band_2
                0  0.025 -0.025       1       5
                1  0.075 -0.025       2       6
                2  0.025 -0.075       3       7
                3  0.075 -0.075       4       8
                ```
        """
        if bands is None:
            bands = range(1, self.band_count + 1)
        elif isinstance(bands, int):
            bands = [bands + 1]
        elif isinstance(bands, list):
            bands = [band + 1 for band in bands]
        else:
            raise ValueError("bands must be an integer or a list of integers.")

        band_nums = bands
        arr = gdal2xyz.gdal2xyz(
            self.raster,
            str(path) if path is not None else None,
            skip_nodata=True,
            return_np_arrays=True,
            band_nums=band_nums,
        )
        if path is None:
            band_names = []
            if bands is not None:
                for band in bands:
                    band_names.append(self.band_names[band - 1])
            else:
                band_names = self.band_names

            df = pd.DataFrame(columns=["lon", "lat"] + band_names)
            df["lon"] = arr[0]
            df["lat"] = arr[1]
            df[band_names] = arr[2].transpose()
            result = df
        else:
            result = None
        return result
    @property
    def overview_count(self: Dataset) -> list[int]:
        """Number of the overviews for each band."""
        overview_number = []
        for i in range(self.band_count):
            overview_number.append(self._iloc(i).GetOverviewCount())
        return overview_number
    def create_overviews(
        self: Dataset,
        resampling_method: str = "nearest",
        overview_levels: list | None = None,
    ) -> None:
        """Create overviews for the dataset.
        Args:
            resampling_method (str):
                The resampling method used to create the overviews. Possible values are
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                "AVERAGE_MAGPHASE", "RMS", "BILINEAR". Defaults to "nearest".
            overview_levels (list, optional):
                The overview levels. Restricted to typical power-of-two reduction factors. Defaults to [2, 4, 8, 16,
                32].
        Returns:
            None:
                Creates internal or external overviews depending on the dataset access mode. See Notes.
        Notes:
            - External (.ovr file): If the dataset is read with `read_only=True` then the overviews file will be created
              as an external .ovr file in the same directory of the dataset.
            - Internal: If the dataset is read with `read_only=False` then the overviews will be created internally in
              the dataset, and the dataset needs to be saved/flushed to persist the changes to disk.
            - You can check the count per band via the `overview_count` property.
        Examples:
            - Create a Dataset with 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              ```
            - Now, create overviews using the default parameters:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]
              ```
            - For each band, there are 4 overview levels you can use to plot the bands:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=0) # doctest: +SKIP
              ```
              ![overviews-level-0](./../../_images/dataset/overviews-level-0.png)
            - However, the dataset originally is 10*10, but the first overview level (2) displays half of the cells by
              aggregating all the cells using the nearest neighbor. The second level displays only 3 cells in each:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=1)   # doctest: +SKIP
              ```
              ![overviews-level-1](./../../_images/dataset/overviews-level-1.png)
            - For the third overview level:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=2)       # doctest: +SKIP
              ```
              ![overviews-level-2](./../../_images/dataset/overviews-level-2.png)
        See Also:
            - Dataset.recreate_overviews: Recreate the dataset overviews if they exist
            - Dataset.get_overview: Get an overview of a band
            - Dataset.overview_count: Number of overviews
            - Dataset.read_overview_array: Read overview values
            - Dataset.plot: Plot a band
        """
        if overview_levels is None:
            overview_levels = OVERVIEW_LEVELS
        else:
            if not isinstance(overview_levels, list):
                raise TypeError("overview_levels should be a list")
            # if self.raster.HasArbitraryOverviews():
            if not all(elem in OVERVIEW_LEVELS for elem in overview_levels):
                raise ValueError(
                    "overview_levels are restricted to the typical power-of-two reduction factors "
                    "(like 2, 4, 8, 16, etc.)"
                )
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Define the overview levels (the reduction factor).
        # e.g., 2 means the overview will be half the resolution of the original dataset.
        # Build overviews using nearest neighbor resampling
        # NEAREST is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        self.raster.BuildOverviews(resampling_method, overview_levels)
    def recreate_overviews(self: Dataset, resampling_method: str = "nearest"):
        """Recreate overviews for the dataset.
        Args:
            resampling_method (str): Resampling method used to recreate overviews. Possible values are
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                "AVERAGE_MAGPHASE", "RMS", "BILINEAR". Defaults to "nearest".
        Raises:
            ValueError:
                If resampling_method is not one of the allowed values above.
            ReadOnlyError:
                If overviews are internal and the dataset is opened read-only. Read with read_only=False.
        See Also:
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
            - Dataset.get_overview: Get an overview of a band.
            - Dataset.overview_count: Number of overviews.
            - Dataset.read_overview_array: Read overview values.
            - Dataset.plot: Plot a band.
        """
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Build overviews using nearest neighbor resampling
        # nearest is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        try:
            for i in range(self.band_count):
                band = self._iloc(i)
                for j in range(self.overview_count[i]):
                    ovr = self.get_overview(i, j)
                    # TODO: if this method takes a long time, we can use the gdal.RegenerateOverviews() method
                    #  which is faster but it does not give the option to choose the resampling method. and the
                    #  overviews has to be given to the function as a list.
                    #  overviews = [band.GetOverview(i) for i in range(band.GetOverviewCount())]
                    #  band.RegenerateOverviews(overviews) or gdal.RegenerateOverviews(overviews)
                    gdal.RegenerateOverview(band, ovr, resampling_method)
        except RuntimeError:
            raise ReadOnlyError(
                "The Dataset is opened with a read only. Please read the dataset using read_only=False"
            )
    def get_overview(self: Dataset, band: int = 0, overview_index: int = 0) -> gdal.Band:
        """Get an overview of a band.
        Args:
            band (int):
                The band index. Defaults to 0.
            overview_index (int):
                Index of the overview. Defaults to 0.
        Returns:
            gdal.Band:
                GDAL band object.
        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 10, size=(4, 10, 10))
              >>> print(arr[0, :, :]) # doctest: +SKIP
              array([[6, 3, 3, 7, 4, 8, 4, 3, 8, 7],
                     [6, 7, 3, 7, 8, 6, 3, 4, 3, 8],
                     [5, 8, 9, 6, 7, 7, 5, 4, 6, 4],
                     [2, 9, 9, 5, 8, 4, 9, 6, 8, 7],
                     [5, 8, 3, 9, 1, 5, 7, 9, 5, 9],
                     [8, 3, 7, 2, 2, 5, 2, 8, 7, 7],
                     [1, 1, 4, 2, 2, 2, 6, 5, 9, 2],
                     [6, 3, 2, 9, 8, 8, 1, 9, 7, 7],
                     [4, 1, 3, 1, 6, 7, 5, 4, 8, 7],
                     [9, 7, 2, 1, 4, 6, 1, 2, 3, 3]], dtype=int32)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              ```
            - Now, create overviews using the default parameters and inspect them:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]
              >>> ovr = dataset.get_overview(band=0, overview_index=0)
              >>> print(ovr)  # doctest: +SKIP
              <osgeo.gdal.Band; proxy of <Swig Object of type 'GDALRasterBandShadow *' at 0x0000017E2B5AF1B0> >
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 3, 4, 4, 8],
                     [5, 9, 7, 5, 6],
                     [5, 3, 1, 7, 5],
                     [1, 4, 2, 6, 9],
                     [4, 3, 6, 5, 8]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=1)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 7, 3],
                     [2, 5, 6],
                     [6, 9, 9]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=2)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 8],
                     [8, 5]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=3)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6]], dtype=int32)
              ```
        See Also:
            - Dataset.create_overviews: Create the dataset overviews if they exist.
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
            - Dataset.overview_count: Number of overviews.
            - Dataset.read_overview_array: Read overview values.
            - Dataset.plot: Plot a band.
        """
        band_obj = self._iloc(band)
        n_views = band_obj.GetOverviewCount()
        if n_views == 0:
            raise ValueError(
                "The band has no overviews, please use the `create_overviews` method to build the overviews"
            )
        if overview_index >= n_views:
            raise ValueError(f"overview_level should be less than {n_views}")
        # TODO:find away to create a Dataset object from the overview band and to return the Dataset object instead
        #  of the gdal band.
        return band_obj.GetOverview(overview_index)
    def read_overview_array(
        self: Dataset, band: int | None = None, overview_index: int = 0
    ) -> np.ndarray:
        """Read overview values.
            - Read the values stored in a given band or overview.
        Args:
            band (int | None):
                The band to read. If None and multiple bands exist, reads all bands at the given overview.
            overview_index (int):
                Index of the overview. Defaults to 0.
        Returns:
            np.ndarray:
                Array with the values in the raster.
        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 10, size=(4, 10, 10))
              >>> print(arr[0, :, :])     # doctest: +SKIP
              array([[6, 3, 3, 7, 4, 8, 4, 3, 8, 7],
                     [6, 7, 3, 7, 8, 6, 3, 4, 3, 8],
                     [5, 8, 9, 6, 7, 7, 5, 4, 6, 4],
                     [2, 9, 9, 5, 8, 4, 9, 6, 8, 7],
                     [5, 8, 3, 9, 1, 5, 7, 9, 5, 9],
                     [8, 3, 7, 2, 2, 5, 2, 8, 7, 7],
                     [1, 1, 4, 2, 2, 2, 6, 5, 9, 2],
                     [6, 3, 2, 9, 8, 8, 1, 9, 7, 7],
                     [4, 1, 3, 1, 6, 7, 5, 4, 8, 7],
                     [9, 7, 2, 1, 4, 6, 1, 2, 3, 3]], dtype=int32)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              ```
            - Create overviews using the default parameters and read overview arrays:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]
              >>> arr = dataset.read_overview_array(band=0, overview_index=0)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 3, 4, 4, 8],
                     [5, 9, 7, 5, 6],
                     [5, 3, 1, 7, 5],
                     [1, 4, 2, 6, 9],
                     [4, 3, 6, 5, 8]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=1)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 7, 3],
                     [2, 5, 6],
                     [6, 9, 9]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=2)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 8],
                     [8, 5]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=3)
              >>> print(arr)  # doctest: +SKIP
              array([[6]], dtype=int32)
              ```
        See Also:
            - Dataset.create_overviews: Create the dataset overviews.
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
            - Dataset.get_overview: Get an overview of a band.
            - Dataset.overview_count: Number of overviews.
            - Dataset.plot: Plot a band.
        """
        if band is None and self.band_count > 1:
            if any(elem == 0 for elem in self.overview_count):
                raise ValueError(
                    "Some bands do not have overviews, please create overviews first"
                )
            # read the array from the first overview to get the size of the array.
            ovr_arr = np.asarray(self.get_overview(0, 0).ReadAsArray())
            arr: np.ndarray = np.ones(
                (
                    self.band_count,
                    ovr_arr.shape[0],
                    ovr_arr.shape[1],
                ),
                dtype=self.numpy_dtype[0],
            )
            for i in range(self.band_count):
                arr[i, :, :] = self.get_overview(i, overview_index).ReadAsArray()
        else:
            if band is None:
                band = 0
            else:
                if band > self.band_count - 1:
                    raise ValueError(
                        f"band index should be between 0 and {self.band_count - 1}"
                    )
                if self.overview_count[band] == 0:
                    raise ValueError(
                        f"band {band} has no overviews, please create overviews first"
                    )
            arr = np.asarray(self.get_overview(band, overview_index).ReadAsArray())
        return arr
