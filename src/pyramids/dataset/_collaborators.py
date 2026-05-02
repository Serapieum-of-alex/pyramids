"""Collaborator objects for Dataset operations (L-2 Stage 1 stubs).

The L-2 composition refactor (see
``planning/architecture-review/L-2-dataset-mixin-refactor.md``)
replaces the seven mixins inherited by ``Dataset`` with seven
collaborator instances accessible as ``ds.io``, ``ds.spatial``,
``ds.bands``, ``ds.analysis``, ``ds.cell``, ``ds.vectorize``,
``ds.cog``. During Stage 1 the collaborators are forwarder stubs
— each method delegates back to ``self._ds.<method>(...)``, which
resolves to the existing mixin via the unchanged MRO. Stage 2
PRs migrate method bodies into the collaborators one at a time
and remove the corresponding mixin from ``Dataset``'s base list.

Three design notes:

1.  **Back-reference**. Every collaborator holds ``self._ds``,
    a reference to the parent Dataset. Operations that need
    state (``self._ds.crs``, ``self._ds._raster``) reach through
    that handle.
2.  **Pickle**. Collaborators are NOT pickled as state.
    ``Dataset.__reduce__`` short-circuits the entire pickle graph
    (verified in the Stage 0 audit, §3) and reconstructs via
    ``cls.read_file(...)``, which calls ``Dataset.__init__``,
    which creates fresh collaborators on the new instance. The
    defensive ``_Collaborator.__reduce__`` returning a
    ``_Placeholder`` is only needed if a caller pickles a
    collaborator *directly* — e.g. ``pickle.dumps(ds.io)``.
3.  **Naming**. The collaborator class names (``IO``, ``Spatial``,
    ``Bands``, ``Analysis``, ``Cell``, ``Vectorize``, ``COG``)
    intentionally collide with the existing mixin classes for
    five of the seven. ``dataset.py`` resolves the collision by
    importing the mixin classes under ``_<X>Mixin`` aliases.

Method-level docstrings on the forwarders are intentionally
one-liners that reference the canonical implementation on
``Dataset``. Duplicating the full Args/Returns/Examples blocks on
both the mixin method and the forwarder would create two sources
of truth that drift apart; readers who need the contract should
follow the cross-reference.
"""

from __future__ import annotations

import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import geopandas as gpd
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, locate_values
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset.cog import (
    ValidationReport,
    merge_options,
    translate_to_cog,
    validate,
    validate_blocksize,
)
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


_AVERAGING_RESAMPLERS: frozenset[str] = frozenset(
    {"average", "bilinear", "cubic", "cubicspline", "lanczos"}
)
"""Overview resampling methods that smooth pixel values.

Incorrect for categorical rasters (land cover, basin IDs, classification
masks). Using any of these on a categorical dataset emits a
``UserWarning`` from :meth:`COG.to_cog`.
"""


_INTEGER_DTYPES: frozenset[int] = frozenset(
    {
        gdal.GDT_Byte,
        gdal.GDT_UInt16,
        gdal.GDT_Int16,
        gdal.GDT_UInt32,
        gdal.GDT_Int32,
        gdal.GDT_UInt64,
        gdal.GDT_Int64,
        gdal.GDT_Int8,
    }
)


class _Placeholder:
    """Stand-in returned by ``_Collaborator.__reduce__``.

    Exists only as the unpickle target for a directly-pickled
    collaborator. ``Dataset.__init__`` creates fresh collaborators
    on Dataset unpickle, overwriting any placeholder that would
    otherwise be attached. If user code ever observes a
    ``_Placeholder`` instance, the unpickle sequence has been
    interrupted — open a bug.
    """


def _recreate_placeholder() -> _Placeholder:
    return _Placeholder()


class _Collaborator:
    """Base class for every Dataset collaborator.

    Holds a **weak** back-reference to the parent ``Dataset``. The
    weakref is essential: a strong ``_ds`` reference creates a cycle
    (``ds -> ds.spatial -> ds``) that the cycle collector eventually
    breaks but that delays GDAL handle release long enough to fail
    Windows file-unlink in tests (and to leak file descriptors in
    long-running processes). xarray uses the same pattern for
    accessors. ``weakref.proxy`` is transparent — ``self._ds.crs``
    works as if ``_ds`` were a real reference — so collaborator
    method bodies don't need to know the back-reference is weak.

    Also overrides ``__reduce__`` so direct collaborator pickling
    (``pickle.dumps(ds.io)``) produces a placeholder rather than a
    circular pickle through ``_ds``.
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        # ``weakref.proxy`` so the back-reference does not create a
        # strong cycle with the parent Dataset. See class docstring.
        self._ds = weakref.proxy(ds)

    def __reduce__(self) -> tuple[Any, tuple]:
        return (_recreate_placeholder, ())


class IO(_Collaborator):
    """IO operations on a Dataset (read_array, to_file, overviews, …).

    Stage 1 stub: every method forwards to the equivalent on the
    underlying Dataset (which still inherits the IO mixin). Stage 2
    PR2.6 migrates method bodies onto this class and deletes the
    IO mixin from ``Dataset``'s base list.
    """

    def read_array(self, *args: Any, **kwargs: Any) -> Any:
        """Read raster cell values into a NumPy array (forwarder to ``Dataset.read_array``)."""
        return self._ds.read_array(*args, **kwargs)

    def write_array(self, *args: Any, **kwargs: Any) -> Any:
        """Write a NumPy array into the raster (forwarder to ``Dataset.write_array``)."""
        return self._ds.write_array(*args, **kwargs)

    def to_file(self, *args: Any, **kwargs: Any) -> Any:
        """Save the dataset to disk (forwarder to ``Dataset.to_file``)."""
        return self._ds.to_file(*args, **kwargs)

    def to_raster(self, *args: Any, **kwargs: Any) -> Any:
        """Write the dataset to a raster file (forwarder to ``Dataset.to_raster``)."""
        return self._ds.to_raster(*args, **kwargs)

    def get_block_arrangement(self, *args: Any, **kwargs: Any) -> Any:
        """Return the block layout used to tile reads (forwarder to ``Dataset.get_block_arrangement``)."""
        return self._ds.get_block_arrangement(*args, **kwargs)

    def get_tile(self, *args: Any, **kwargs: Any) -> Any:
        """Return one tile of cells by index (forwarder to ``Dataset.get_tile``)."""
        return self._ds.get_tile(*args, **kwargs)

    def map_blocks(self, *args: Any, **kwargs: Any) -> Any:
        """Apply a callable across all blocks of the raster (forwarder to ``Dataset.map_blocks``)."""
        return self._ds.map_blocks(*args, **kwargs)

    def to_xyz(self, *args: Any, **kwargs: Any) -> Any:
        """Export the raster as an XYZ point list (forwarder to ``Dataset.to_xyz``)."""
        return self._ds.to_xyz(*args, **kwargs)

    @property
    def overview_count(self) -> list[int]:
        """Per-band overview level counts (forwarder to ``Dataset.overview_count``)."""
        return self._ds.overview_count

    def create_overviews(self, *args: Any, **kwargs: Any) -> Any:
        """Build overview pyramids for the dataset (forwarder to ``Dataset.create_overviews``)."""
        return self._ds.create_overviews(*args, **kwargs)

    def recreate_overviews(self, *args: Any, **kwargs: Any) -> Any:
        """Discard and rebuild overview pyramids (forwarder to ``Dataset.recreate_overviews``)."""
        return self._ds.recreate_overviews(*args, **kwargs)

    def get_overview(self, *args: Any, **kwargs: Any) -> Any:
        """Return one overview level as a sub-Dataset (forwarder to ``Dataset.get_overview``)."""
        return self._ds.get_overview(*args, **kwargs)

    def read_overview_array(self, *args: Any, **kwargs: Any) -> Any:
        """Read one overview level into a NumPy array (forwarder to ``Dataset.read_overview_array``)."""
        return self._ds.read_overview_array(*args, **kwargs)


class Spatial(_Collaborator):
    """Spatial operations on a Dataset (crop, to_crs, align, …).

    Stage 1 stub.
    """

    def crop(self, *args: Any, **kwargs: Any) -> Any:
        """Crop the dataset to a polygon, raster, or bounding box (forwarder to ``Dataset.crop``)."""
        return self._ds.crop(*args, **kwargs)

    def to_crs(self, *args: Any, **kwargs: Any) -> Any:
        """Reproject the dataset into a new CRS (forwarder to ``Dataset.to_crs``)."""
        return self._ds.to_crs(*args, **kwargs)

    def set_crs(self, *args: Any, **kwargs: Any) -> Any:
        """Set the dataset CRS in place without reprojecting (forwarder to ``Dataset.set_crs``)."""
        return self._ds.set_crs(*args, **kwargs)

    def convert_longitude(self, *args: Any, **kwargs: Any) -> Any:
        """Shift the longitude axis between 0..360 and -180..180 (forwarder to ``Dataset.convert_longitude``)."""
        return self._ds.convert_longitude(*args, **kwargs)

    def resample(self, *args: Any, **kwargs: Any) -> Any:
        """Resample the dataset to a new cell size (forwarder to ``Dataset.resample``)."""
        return self._ds.resample(*args, **kwargs)

    def align(self, *args: Any, **kwargs: Any) -> Any:
        """Align this dataset to another's grid and CRS (forwarder to ``Dataset.align``)."""
        return self._ds.align(*args, **kwargs)

    def fill_gaps(self, *args: Any, **kwargs: Any) -> Any:
        """Fill no-data gaps using neighbouring cells (forwarder to ``Dataset.fill_gaps``)."""
        return self._ds.fill_gaps(*args, **kwargs)


class Bands(_Collaborator):
    """Band-metadata operations on a Dataset (attribute table, color, …).

    Stage 1 stub.
    """

    def get_attribute_table(self, *args: Any, **kwargs: Any) -> Any:
        """Return the GDAL raster attribute table (forwarder to ``Dataset.get_attribute_table``)."""
        return self._ds.get_attribute_table(*args, **kwargs)

    def set_attribute_table(self, *args: Any, **kwargs: Any) -> Any:
        """Set the GDAL raster attribute table (forwarder to ``Dataset.set_attribute_table``)."""
        return self._ds.set_attribute_table(*args, **kwargs)

    def add_band(self, *args: Any, **kwargs: Any) -> Any:
        """Append a new band to the dataset (forwarder to ``Dataset.add_band``)."""
        return self._ds.add_band(*args, **kwargs)

    @property
    def band_color(self) -> dict[int, str]:
        """Per-band color interpretation (forwarder to ``Dataset.band_color``)."""
        return self._ds.band_color

    @band_color.setter
    def band_color(self, values: dict[int, str]) -> None:
        self._ds.band_color = values

    @property
    def color_table(self) -> Any:
        """Categorical color table for paletted bands (forwarder to ``Dataset.color_table``)."""
        return self._ds.color_table

    @color_table.setter
    def color_table(self, df: Any) -> None:
        self._ds.color_table = df

    def get_band_by_color(self, *args: Any, **kwargs: Any) -> Any:
        """Look up a band index by its color interpretation (forwarder to ``Dataset.get_band_by_color``)."""
        return self._ds.get_band_by_color(*args, **kwargs)

    def change_no_data_value(self, *args: Any, **kwargs: Any) -> Any:
        """Replace the no-data sentinel for one or more bands (forwarder to ``Dataset.change_no_data_value``)."""
        return self._ds.change_no_data_value(*args, **kwargs)


class Analysis(_Collaborator):
    """Analysis / statistics / plot operations on a Dataset.

    Stage 1 stub.
    """

    def stats(self, *args: Any, **kwargs: Any) -> Any:
        """Compute per-band statistics (min/max/mean/std) (forwarder to ``Dataset.stats``)."""
        return self._ds.stats(*args, **kwargs)

    def count_domain_cells(self, *args: Any, **kwargs: Any) -> Any:
        """Count cells with valid (non-no-data) values (forwarder to ``Dataset.count_domain_cells``)."""
        return self._ds.count_domain_cells(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Apply a callable element-wise across cells (forwarder to ``Dataset.apply``)."""
        return self._ds.apply(*args, **kwargs)

    def fill(self, *args: Any, **kwargs: Any) -> Any:
        """Fill no-data cells with a constant or neighbour value (forwarder to ``Dataset.fill``)."""
        return self._ds.fill(*args, **kwargs)

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        """Extract a sub-region of the dataset (forwarder to ``Dataset.extract``)."""
        return self._ds.extract(*args, **kwargs)

    def overlay(self, *args: Any, **kwargs: Any) -> Any:
        """Overlay another dataset on top of this one (forwarder to ``Dataset.overlay``)."""
        return self._ds.overlay(*args, **kwargs)

    def get_mask(self, *args: Any, **kwargs: Any) -> Any:
        """Return the no-data mask as a boolean array (forwarder to ``Dataset.get_mask``)."""
        return self._ds.get_mask(*args, **kwargs)

    def footprint(self, *args: Any, **kwargs: Any) -> Any:
        """Return the data footprint as a polygon (forwarder to ``Dataset.footprint``)."""
        return self._ds.footprint(*args, **kwargs)

    def get_histogram(self, *args: Any, **kwargs: Any) -> Any:
        """Return per-band histograms (forwarder to ``Dataset.get_histogram``)."""
        return self._ds.get_histogram(*args, **kwargs)

    @staticmethod
    def normalize(array: Any) -> Any:
        """Min-max normalize a NumPy array into the [0, 1] range.

        Implemented by delegating to the staticmethod on the Analysis
        mixin so callers who reach this method through the collaborator
        get the same body as callers who reach it through ``Dataset``.

        Args:
            array: NumPy array of numeric values. Must contain at least
                one finite value or division-by-zero will occur when
                ``array.max() == array.min()``.

        Returns:
            NumPy array of the same shape, scaled so that the minimum
            value maps to ``0.0`` and the maximum to ``1.0``.

        Examples:
            - Normalize a one-dimensional array of integers:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._collaborators import Analysis
                >>> Analysis.normalize(np.array([0.0, 5.0, 10.0])).tolist()
                [0.0, 0.5, 1.0]

                ```
            - Normalize a two-dimensional float array and inspect the result:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._collaborators import Analysis
                >>> out = Analysis.normalize(np.array([[2.0, 4.0], [6.0, 8.0]]))
                >>> float(out.min()), float(out.max())
                (0.0, 1.0)
                >>> out.shape
                (2, 2)

                ```
        """
        # Defer to the staticmethod on the Analysis mixin so callers
        # who take the collaborator's normalize get the same body.
        from pyramids.dataset.ops.analysis import Analysis as _AnalysisMixin

        return _AnalysisMixin.normalize(array)

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Render the dataset as a matplotlib figure (forwarder to ``Dataset.plot``)."""
        return self._ds.plot(*args, **kwargs)


class Cell(_Collaborator):
    """Cell-geometry operations on a Dataset.

    Owns the real implementations of ``get_cell_coords``,
    ``get_cell_polygons``, ``get_cell_points``,
    ``map_to_array_coordinates``, and ``array_to_map_coordinates`` after
    L-2 PR 2.1. ``Dataset`` exposes a same-named facade for each method
    that delegates to this collaborator, so ``ds.get_cell_coords(...)``
    and ``ds.cell.get_cell_coords(...)`` are equivalent.
    """

    def get_cell_coords(
        self, location: str = "center", domain_only: bool = False
    ) -> np.ndarray:
        """Get coordinates for the center/corner of cells inside the dataset domain.

        Returns the coordinates of the cell centers inside the domain (only the cells that
        do not have nodata value)

        Args:
            location (str):
                Location of the coordinates. Use `center` for the center of a cell, `corner` for the corner of the
                cell (top-left corner).
            domain_only (bool):
                True to exclude the cells out of the domain. Default is False.

        Returns:
            np.ndarray:
                Array with a list of the coordinates to be interpolated, without the NaN.
            np.ndarray:
                Array with all the centers of cells in the domain of the DEM.

        Examples:
            - Create `Dataset` consists of 1 bands, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> coords = dataset.get_cell_coords()
              >>> print(coords)
              [[ 0.025 -0.025]
               [ 0.075 -0.025]
               [ 0.125 -0.025]
               [ 0.025 -0.075]
               [ 0.075 -0.075]
               [ 0.125 -0.075]
               [ 0.025 -0.125]
               [ 0.075 -0.125]
               [ 0.125 -0.125]]

              ```

            - Get the coordinates of the top left corner of cells inside the domain.

              ```python
              >>> coords = dataset.get_cell_coords(location="corner")
              >>> print(coords)
              [[ 0.    0.  ]
               [ 0.05  0.  ]
               [ 0.1   0.  ]
               [ 0.   -0.05]
               [ 0.05 -0.05]
               [ 0.1  -0.05]
               [ 0.   -0.1 ]
               [ 0.05 -0.1 ]
               [ 0.1  -0.1 ]]

              ```
        """
        location = location.lower()
        if location not in ["center", "corner"]:
            raise ValueError(
                "The location parameter can have one of these values: 'center', 'corner', "
                f"but the value: {location} is given."
            )

        if location == "center":
            add_value = 0.5
        else:
            add_value = 0
        (
            x_init,
            cell_size_x,
            xy_span,
            y_init,
            yy_span,
            cell_size_y,
        ) = self._ds.geotransform

        if cell_size_x != cell_size_y:
            if np.abs(cell_size_x) != np.abs(cell_size_y):
                self._ds.logger.warning(
                    f"The given raster does not have a square cells, the cell size is "
                    f"{cell_size_x}*{cell_size_y} "
                )

        no_val = (
            self._ds.no_data_value[0] if self._ds.no_data_value[0] is not None else np.nan
        )
        arr = self._ds.read_array(band=0)
        if domain_only and no_val not in arr:
            self._ds.logger.warning(
                "The no data value does not exist in the band, so all the cells will be considered, and the "
                "domain_only filter will not be applied."
            )

        mask_values: list[Any] | None = [no_val] if domain_only else None
        indices = get_indices2(arr, mask=mask_values)

        f1 = [i[0] for i in indices]
        f2 = [i[1] for i in indices]
        x = [x_init + cell_size_x * (i + add_value) for i in f2]
        y = [y_init + cell_size_y * (i + add_value) for i in f1]
        coords = np.array(list(zip(x, y)))

        return coords

    def get_cell_polygons(self, domain_only: bool = False) -> GeoDataFrame:
        """Get a polygon shapely geometry for the raster cells.

        Args:
            domain_only (bool):
                True to get the polygons of the cells inside the domain.

        Returns:
            GeoDataFrame:
                With two columns, geometry, and id.

        Examples:
            - Create `Dataset` consists of 1 band, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_polygons()
              >>> print(gdf)
                                                     geometry  id
              0  POLYGON ((0 0, 0.05 0, 0.05 -0.05, 0 -0.05, 0 0))   0
              1  POLYGON ((0.05 0, 0.1 0, 0.1 -0.05, 0.05 -0.05...   1
              2  POLYGON ((0.1 0, 0.15 0, 0.15 -0.05, 0.1 -0.05...   2
              3  POLYGON ((0 -0.05, 0.05 -0.05, 0.05 -0.1, 0 -0...   3
              4  POLYGON ((0.05 -0.05, 0.1 -0.05, 0.1 -0.1, 0.0...   4
              5  POLYGON ((0.1 -0.05, 0.15 -0.05, 0.15 -0.1, 0....   5
              6  POLYGON ((0 -0.1, 0.05 -0.1, 0.05 -0.15, 0 -0....   6
              7  POLYGON ((0.05 -0.1, 0.1 -0.1, 0.1 -0.15, 0.05...   7
              8  POLYGON ((0.1 -0.1, 0.15 -0.1, 0.15 -0.15, 0.1...   8
              >>> fig, ax = dataset.plot()
              >>> gdf.plot(ax=ax, facecolor='none', edgecolor="gray", linewidth=2)
              <Axes: >

              ```

        ![get_cell_polygons](./../../_images/dataset/get_cell_polygons.png)
        """
        coords = self.get_cell_coords(location="corner", domain_only=domain_only)
        cell_size = self._ds.geotransform[1]
        epsg = self._ds._get_epsg()
        x = np.zeros((coords.shape[0], 4))
        y = np.zeros((coords.shape[0], 4))
        x[:, 0] = coords[:, 0]
        y[:, 0] = coords[:, 1]
        x[:, 1] = x[:, 0] + cell_size
        y[:, 1] = y[:, 0]
        x[:, 2] = x[:, 0] + cell_size
        y[:, 2] = y[:, 0] - cell_size
        x[:, 3] = x[:, 0]
        y[:, 3] = y[:, 0] - cell_size

        coords_tuples = [list(zip(x[:, i], y[:, i])) for i in range(4)]
        polys_coords = [
            (
                coords_tuples[0][i],
                coords_tuples[1][i],
                coords_tuples[2][i],
                coords_tuples[3][i],
            )
            for i in range(len(x))
        ]
        polygons = list(map(FeatureCollection.create_polygon, polys_coords))
        gdf = gpd.GeoDataFrame(geometry=polygons)
        gdf.set_crs(epsg=epsg, inplace=True)
        gdf["id"] = gdf.index
        return gdf

    def get_cell_points(
        self, location: str = "center", domain_only: bool = False
    ) -> GeoDataFrame:
        """Get a point shapely geometry for the raster cells center point.

        Args:
            location (str):
                Location of the point, ["corner", "center"]. Default is "center".
            domain_only (bool):
                True to get the points of the cells inside the domain only.

        Returns:
            GeoDataFrame:
                With two columns, geometry, and id.

        Examples:
            - Create `Dataset` consists of 1 band, 3 rows, 3 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1,3, size=(3, 3))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```

            - Get the coordinates of the center of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_points()
              >>> print(gdf)
                             geometry  id
              0  POINT (0.025 -0.025)   0
              1  POINT (0.075 -0.025)   1
              2  POINT (0.125 -0.025)   2
              3  POINT (0.025 -0.075)   3
              4  POINT (0.075 -0.075)   4
              5  POINT (0.125 -0.075)   5
              6  POINT (0.025 -0.125)   6
              7  POINT (0.075 -0.125)   7
              8  POINT (0.125 -0.125)   8
              >>> fig, ax = dataset.plot()
              >>> gdf.plot(ax=ax, facecolor='black', linewidth=2)
              <Axes: >

              ```

            ![get_cell_points](./../../_images/dataset/get_cell_points.png)

            - Get the coordinates of the top left corner of cells inside the domain.

              ```python
              >>> gdf = dataset.get_cell_points(location="corner")
              >>> print(gdf)
                          geometry  id
              0         POINT (0 0)   0
              1      POINT (0.05 0)   1
              2       POINT (0.1 0)   2
              3     POINT (0 -0.05)   3
              4  POINT (0.05 -0.05)   4
              5   POINT (0.1 -0.05)   5
              6      POINT (0 -0.1)   6
              7   POINT (0.05 -0.1)   7
              8    POINT (0.1 -0.1)   8
              >>> fig, ax = dataset.plot()
              >>> gdf.plot(ax=ax, facecolor='black', linewidth=4)
              <Axes: >

              ```

            ![get_cell_points-corner](./../../_images/dataset/get_cell_points-corner.png)
        """
        coords = self.get_cell_coords(location=location, domain_only=domain_only)
        epsg = self._ds._get_epsg()

        coords_tuples = list(zip(coords[:, 0], coords[:, 1]))
        points = FeatureCollection.create_points(coords_tuples)
        gdf = gpd.GeoDataFrame(geometry=points)
        gdf.set_crs(epsg=epsg, inplace=True)
        gdf["id"] = gdf.index
        return gdf

    def map_to_array_coordinates(
        self,
        points: GeoDataFrame | FeatureCollection | DataFrame,
    ) -> np.ndarray:
        """Convert coordinates of points to array indices.

        - map_to_array_coordinates locates a point with real coordinates (x, y) or (lon, lat) on the array by finding
            the cell indices (row, column) of the nearest cell in the raster.
        - The point coordinate system of the raster has to be projected to be able to calculate the distance.

        Args:
            points (GeoDataFrame | pandas.DataFrame | FeatureCollection):
                - GeoDataFrame: GeoDataFrame with POINT geometry.
                - DataFrame: DataFrame with x, y columns.

        Returns:
            np.ndarray:
                Array with shape (N, 2) containing the row and column indices in the array.

        Examples:
            - Create `Dataset` consisting of 2 bands, 10 rows, 10 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> arr = np.random.randint(1, 3, size=(2, 10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```
            - DataFrame with x, y columns:

              - We can give the function a DataFrame with x, y columns to array the coordinates of the points that are located within the dataset domain.

              ```python
              >>> points = pd.DataFrame({"x": [0.025, 0.175, 0.375], "y": [0.025, 0.225, 0.125]})
              >>> indices = dataset.map_to_array_coordinates(points)
              >>> print(indices)
              [[0 0]
               [0 3]
               [0 7]]

              ```
            - GeoDataFrame with POINT geometry:

              - We can give the function a GeoDataFrame with POINT geometry to array the coordinates of the points that locate within the dataset domain.

              ```python
              >>> from shapely.geometry import Point
              >>> from geopandas import GeoDataFrame
              >>> points = GeoDataFrame({"geometry": [Point(0.025, 0.025), Point(0.175, 0.225), Point(0.375, 0.125)]})
              >>> indices = dataset.map_to_array_coordinates(points)
              >>> print(indices)
              [[0 0]
               [0 3]
               [0 7]]

              ```
        """
        if isinstance(points, FeatureCollection):
            verts = points.with_coordinates()
            points = verts.loc[:, ["x", "y"]].values
        elif isinstance(points, GeoDataFrame):
            verts = FeatureCollection(points).with_coordinates()
            points = verts.loc[:, ["x", "y"]].values
        elif isinstance(points, DataFrame):
            if all(elem not in points.columns for elem in ["x", "y"]):
                raise ValueError(
                    "If the input is a DataFrame, it should have two columns x, and y"
                )
            points = points.loc[:, ["x", "y"]].values
        else:
            raise TypeError(
                "please check points input it should be GeoDataFrame/DataFrame/FeatureCollection - given"
                f" {type(points)}"
            )

        indices = locate_values(points, self._ds.x, self._ds.y)
        indices = indices[:, [1, 0]]
        return np.asarray(indices)

    def array_to_map_coordinates(
        self,
        rows_index: list[Number] | np.ndarray,
        column_index: list[Number] | np.ndarray,
        center: bool = False,
    ) -> tuple[list[Number], list[Number]]:
        """Convert array indices to map coordinates.

        array_to_map_coordinates converts the array indices (rows, cols) to real coordinates (x, y) or (lon, lat).

        Args:
            rows_index (List[Number] | np.ndarray):
                The row indices of the cells in the raster array.
            column_index (List[Number] | np.ndarray):
                The column indices of the cells in the raster array.
            center (bool):
                If True, the coordinates will be the center of the cell. Default is False.

        Returns:
            Tuple[List[Number], List[Number]]:
                A tuple of two lists: the x coordinates and the y coordinates of the cells.

        Examples:
            - Create `Dataset` consisting of 1 band, 10 rows, 10 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> import pandas as pd
              >>> arr = np.random.randint(1, 3, size=(10, 10))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

              ```

            - Now call the function with two lists of row and column indices:

              ```python
              >>> rows_index = [1, 3, 5]
              >>> column_index = [2, 4, 6]
              >>> coords = dataset.array_to_map_coordinates(rows_index, column_index)
              >>> print(coords) # doctest: +SKIP
              ([0.1, 0.2, 0.3], [-0.05, -0.15, -0.25])

              ```
        """
        top_left_x, top_left_y = self._ds.top_left_corner
        cell_size = self._ds.cell_size
        if center:
            top_left_x += cell_size / 2
            top_left_y -= cell_size / 2

        x_coord_fn = lambda x: top_left_x + x * cell_size
        y_coord_fn = lambda y: top_left_y - y * cell_size

        x_coords = list(map(x_coord_fn, column_index))
        y_coords = list(map(y_coord_fn, rows_index))

        return x_coords, y_coords


class Vectorize(_Collaborator):
    """Vectorisation / clustering operations on a Dataset.

    Stage 1 stub.
    """

    def to_feature_collection(self, *args: Any, **kwargs: Any) -> Any:
        """Vectorise the raster into a FeatureCollection (forwarder to ``Dataset.to_feature_collection``)."""
        return self._ds.to_feature_collection(*args, **kwargs)

    def translate(self, *args: Any, **kwargs: Any) -> Any:
        """Translate cell values into vector attributes (forwarder to ``Dataset.translate``)."""
        return self._ds.translate(*args, **kwargs)

    def cluster(self, *args: Any, **kwargs: Any) -> Any:
        """Cluster cells of equal value into connected groups (forwarder to ``Dataset.cluster``)."""
        return self._ds.cluster(*args, **kwargs)

    def cluster2(self, *args: Any, **kwargs: Any) -> Any:
        """Alternative connected-component clustering algorithm (forwarder to ``Dataset.cluster2``)."""
        return self._ds.cluster2(*args, **kwargs)


class COG(_Collaborator):
    """Cloud Optimized GeoTIFF read/write/validate operations for ``Dataset``.

    Owns the real implementations of ``to_cog``, ``is_cog`` (property),
    and ``validate_cog`` after L-2 PR 2.2. ``Dataset`` exposes a
    same-named facade for each so ``ds.to_cog(...)`` and
    ``ds.cog.to_cog(...)`` are equivalent. The categorical-raster
    resampling guardrail (``_warn_if_categorical_with_averaging``)
    lives here too.
    """

    def to_cog(
        self,
        path: str | Path,
        *,
        compress: str = "DEFLATE",
        level: int | None = None,
        quality: int | None = None,
        blocksize: int = 512,
        predictor: str | int | None = None,
        bigtiff: str = "IF_SAFER",
        num_threads: int | str = "ALL_CPUS",
        overview_resampling: str = "nearest",
        overview_count: int | None = None,
        overview_compress: str | None = None,
        tiling_scheme: str | None = None,
        zoom_level: int | None = None,
        zoom_level_strategy: str = "auto",
        aligned_levels: int | None = None,
        resampling: str = "nearest",
        add_mask: bool = False,
        sparse_ok: bool = False,
        target_srs: int | str | None = None,
        statistics: bool = True,
        extra: Mapping[str, Any] | list[str] | None = None,
    ) -> Path:
        """Save the dataset as a Cloud Optimized GeoTIFF.

        Args:
            path: Destination path. Parent directory must exist.
            compress: Compression method. ``DEFLATE``, ``LZW``, and
                ``NONE`` are guaranteed by every GDAL build. ``JPEG``
                is almost always available. ``ZSTD``, ``WEBP``,
                ``LERC``, ``LERC_DEFLATE``, and ``LERC_ZSTD`` require
                the GDAL build to have been compiled with the
                corresponding library (libzstd / libwebp / LERC); on
                a GDAL build lacking them, the COG driver will raise
                at write time. To probe what your GDAL supports:

                ```python
                from osgeo import gdal
                meta = gdal.GetDriverByName("GTiff").GetMetadataItem(
                    "DMD_CREATIONOPTIONLIST"
                )
                print("ZSTD" in (meta or ""))
                ```
            level: Compression level (e.g., 1-12 for DEFLATE, 1-22 ZSTD).
            quality: Lossy-compression quality 1-100 (JPEG/WEBP).
            blocksize: Internal tile size; power of 2 in [64, 4096].
            predictor: ``"YES"``/``"STANDARD"``/``"FLOATING_POINT"`` or 1/2/3.
            bigtiff: ``"IF_SAFER"`` (default), ``"YES"``, ``"NO"``,
                ``"IF_NEEDED"``.
            num_threads: Worker threads; ``"ALL_CPUS"`` or an int.
            overview_resampling: ``nearest``, ``average``, ``bilinear``,
                ``cubic``, ``cubicspline``, ``lanczos``, ``mode``,
                ``rms``, ``gauss``.
            overview_count: Number of overview levels (default: auto).
            overview_compress: Compression for overview IFDs.
            tiling_scheme: e.g., ``"GoogleMapsCompatible"`` for a
                web-optimized COG (EPSG:3857).
            zoom_level, zoom_level_strategy, aligned_levels: Advanced
                tiling-scheme knobs.
            resampling: Warp resampling when ``tiling_scheme`` or
                ``target_srs`` reprojects.
            add_mask: Add an alpha band for transparency.
            sparse_ok: Allow sparse (unfilled) tiles.
            target_srs: Reproject before write. Int for EPSG or a WKT
                / PROJ string.
            statistics: Compute and embed band statistics.
            extra: Additional GDAL creation options as a mapping or
                legacy ``['KEY=VALUE', ...]`` list. Overrides
                conflicting kwargs.

        Returns:
            Path: The resolved destination path.

        Raises:
            ValueError: Invalid blocksize or unknown option key.
            FileNotFoundError: Parent directory does not exist.
            FailedToSaveError: GDAL CreateCopy failed.
            DriverNotExistError: GDAL build lacks the COG driver.

        Warnings:
            UserWarning: When the source looks categorical (integer
                dtype or has a color table) and ``overview_resampling``
                is an averaging method.

        Note:
            Setting ``tiling_scheme`` (e.g., ``GoogleMapsCompatible``)
            implies a specific SRS — ``target_srs`` is ignored in that
            case. A ``UserWarning`` is emitted if both are provided.

        Examples:
            - Write a compressed COG from an in-memory Dataset:
                ```python
                >>> import numpy as np  # doctest: +SKIP
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> arr = np.random.rand(256, 256).astype("float32")  # doctest: +SKIP
                >>> ds = Dataset.create_from_array(  # doctest: +SKIP
                ...     arr, top_left_corner=(0, 0), cell_size=0.001, epsg=4326,
                ... )
                >>> out = ds.to_cog("out.tif", compress="ZSTD")  # doctest: +SKIP
                >>> out.name  # doctest: +SKIP
                'out.tif'

                ```
            - Produce a web-optimized COG for a tile server:
                ```python
                >>> web = ds.to_cog("web.tif", tiling_scheme="GoogleMapsCompatible")  # doctest: +SKIP
                >>> reopened = Dataset.read_file(web)  # doctest: +SKIP
                >>> reopened.epsg  # doctest: +SKIP
                3857

                ```
            - Forward additional GDAL options through `extra`:
                ```python
                >>> _ = ds.to_cog(  # doctest: +SKIP
                ...     "precise.tif",
                ...     compress="LERC",
                ...     extra={"MAX_Z_ERROR": 0.001},
                ... )

                ```
        """
        validate_blocksize(blocksize)
        self._warn_if_categorical_with_averaging(overview_resampling)
        if tiling_scheme is not None and target_srs is not None:
            warnings.warn(
                "Both tiling_scheme and target_srs provided; "
                "tiling_scheme wins and target_srs is ignored.",
                UserWarning,
                stacklevel=2,
            )
            target_srs = None

        num_threads_str = (
            num_threads if isinstance(num_threads, str) else str(num_threads)
        )
        defaults: dict[str, Any] = {
            "COMPRESS": compress,
            "LEVEL": level,
            "QUALITY": quality,
            "BLOCKSIZE": blocksize,
            "PREDICTOR": predictor,
            "BIGTIFF": bigtiff,
            "NUM_THREADS": num_threads_str,
            "OVERVIEW_RESAMPLING": overview_resampling,
            "OVERVIEW_COUNT": overview_count,
            "OVERVIEW_COMPRESS": overview_compress,
            "TILING_SCHEME": tiling_scheme,
            "ZOOM_LEVEL": zoom_level,
            "ZOOM_LEVEL_STRATEGY": zoom_level_strategy,
            "ALIGNED_LEVELS": aligned_levels,
            "WARP_RESAMPLING": (resampling if (tiling_scheme or target_srs) else None),
            "ADD_ALPHA": True if add_mask else None,
            "SPARSE_OK": True if sparse_ok else None,
            "STATISTICS": "YES" if statistics else None,
        }
        if target_srs is not None:
            defaults["TARGET_SRS"] = (
                f"EPSG:{target_srs}" if isinstance(target_srs, int) else target_srs
            )

        options = merge_options(defaults, extra)

        dst: gdal.Dataset | None = None
        try:
            dst = translate_to_cog(self._ds._raster, path, options)
            dst.FlushCache()
        finally:
            dst = None

        return Path(path)

    @property
    def is_cog(self) -> bool:
        """``True`` iff the backing file on disk is a valid COG.

        ``False`` for MEM datasets, ``/vsimem/`` paths, and unsaved
        datasets (empty :attr:`file_name`).

        Examples:
            - Check the backing file of a newly-opened COG:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> ds.is_cog  # doctest: +SKIP
                True

                ```
            - Plain GeoTIFFs and MEM datasets return False:
                ```python
                >>> plain = Dataset.read_file("plain.tif")  # doctest: +SKIP
                >>> plain.is_cog  # doctest: +SKIP
                False

                ```
            - Use in a conditional pipeline:
                ```python
                >>> if not ds.is_cog:  # doctest: +SKIP
                ...     ds.to_cog("fixed.tif")

                ```
        """
        result: bool
        fn = self._ds.file_name
        if not fn or fn.startswith("/vsimem/"):
            result = False
        else:
            try:
                result = validate(fn).is_valid
            except FileNotFoundError:
                result = False
        return result

    def validate_cog(self, strict: bool = False) -> ValidationReport:
        """Validate the backing file as a COG.

        Args:
            strict: If ``True``, warnings are treated as errors.

        Returns:
            ValidationReport with errors, warnings, and structural details.

        Raises:
            FileNotFoundError: Dataset has no on-disk backing file
                (MEM-only or ``/vsimem/``).

        Examples:
            - Validate and branch on the result:
                ```python
                >>> from pyramids.dataset import Dataset  # doctest: +SKIP
                >>> ds = Dataset.read_file("scene.tif")  # doctest: +SKIP
                >>> report = ds.validate_cog()  # doctest: +SKIP
                >>> bool(report)  # doctest: +SKIP
                True

                ```
            - Strict mode promotes warnings to errors:
                ```python
                >>> strict = ds.validate_cog(strict=True)  # doctest: +SKIP
                >>> if not strict:  # doctest: +SKIP
                ...     for err in strict.errors: print(err)

                ```
            - Inspect structural details from the report:
                ```python
                >>> report.details.get("blocksize")  # doctest: +SKIP
                [512, 512]

                ```
        """
        fn = self._ds.file_name
        if not fn or fn.startswith("/vsimem/"):
            raise FileNotFoundError(
                "Dataset has no on-disk backing file to validate "
                "(is this a MEM or /vsimem/ dataset?)"
            )
        return validate(fn, strict=strict)

    def _warn_if_categorical_with_averaging(self, overview_resampling: str) -> None:
        """Emit a ``UserWarning`` if an averaging resampler is used on categorical data.

        Args:
            overview_resampling: The resampling method requested by the
                caller. Case-insensitive. Only averaging-family methods
                (``average``, ``bilinear``, ``cubic``, ``cubicspline``,
                ``lanczos``) trigger the check.

        Warns:
            UserWarning: When ``overview_resampling`` is an averaging
                method and the source has a color table OR integer
                dtype — both strong signals of categorical data.

        Note:
            Silent when ``overview_resampling`` is ``nearest`` or
            ``mode`` (both category-safe) or when the source is
            floating-point and has no color table (continuous data).

        Examples:
            - Integer dataset + averaging method emits a warning:
                ```python
                >>> import warnings  # doctest: +SKIP
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     byte_ds.cog._warn_if_categorical_with_averaging("average")
                ...     [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
                ['overview_resampling=\\'average\\' averages pixel values, ...']

                ```
            - Nearest resampling is always silent:
                ```python
                >>> with warnings.catch_warnings(record=True) as caught:  # doctest: +SKIP
                ...     warnings.simplefilter("always")
                ...     byte_ds.cog._warn_if_categorical_with_averaging("nearest")
                ...     len(caught)
                0

                ```
        """
        if overview_resampling.lower() not in _AVERAGING_RESAMPLERS:
            return
        first_band = self._ds._raster.GetRasterBand(1)
        has_color_table = first_band.GetColorTable() is not None
        is_integer = first_band.DataType in _INTEGER_DTYPES
        if has_color_table or is_integer:
            warnings.warn(
                f"overview_resampling={overview_resampling!r} averages pixel "
                "values, which corrupts categorical rasters (land cover, IDs). "
                "Use overview_resampling='nearest' or 'mode' instead.",
                UserWarning,
                stacklevel=3,
            )


__all__ = [
    "IO",
    "Spatial",
    "Bands",
    "Analysis",
    "Cell",
    "Vectorize",
    "COG",
]
