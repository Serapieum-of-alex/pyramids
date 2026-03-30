"""Overviews mixin for Dataset raster overview operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset.abstract_dataset import (
    OVERVIEW_LEVELS,
    RESAMPLING_METHODS,
)

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


class OverviewsMixin:
    """Mixin providing overview-related methods for Dataset."""

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
              ![overviews-level-0](./../_images/dataset/overviews-level-0.png)

            - However, the dataset originally is 10*10, but the first overview level (2) displays half of the cells by
              aggregating all the cells using the nearest neighbor. The second level displays only 3 cells in each:

              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=1)   # doctest: +SKIP

              ```
              ![overviews-level-1](./../_images/dataset/overviews-level-1.png)

            - For the third overview level:

              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=2)       # doctest: +SKIP

              ```
              ![overviews-level-2](./../_images/dataset/overviews-level-2.png)

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
