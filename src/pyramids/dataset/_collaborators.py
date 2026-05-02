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

import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


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

    Stage 1 stub.
    """

    def get_cell_coords(self, *args: Any, **kwargs: Any) -> Any:
        """Return per-cell map coordinates (forwarder to ``Dataset.get_cell_coords``)."""
        return self._ds.get_cell_coords(*args, **kwargs)

    def get_cell_polygons(self, *args: Any, **kwargs: Any) -> Any:
        """Return per-cell polygon geometries (forwarder to ``Dataset.get_cell_polygons``)."""
        return self._ds.get_cell_polygons(*args, **kwargs)

    def get_cell_points(self, *args: Any, **kwargs: Any) -> Any:
        """Return per-cell centroid points (forwarder to ``Dataset.get_cell_points``)."""
        return self._ds.get_cell_points(*args, **kwargs)

    def map_to_array_coordinates(self, *args: Any, **kwargs: Any) -> Any:
        """Convert map (x, y) coordinates to array (row, col) indices.

        Forwarder to ``Dataset.map_to_array_coordinates``.
        """
        return self._ds.map_to_array_coordinates(*args, **kwargs)

    def array_to_map_coordinates(self, *args: Any, **kwargs: Any) -> Any:
        """Convert array (row, col) indices to map (x, y) coordinates.

        Forwarder to ``Dataset.array_to_map_coordinates``.
        """
        return self._ds.array_to_map_coordinates(*args, **kwargs)


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
    """Cloud Optimized GeoTIFF operations on a Dataset.

    Stage 1 stub.
    """

    def to_cog(self, *args: Any, **kwargs: Any) -> Any:
        """Save the dataset as a Cloud Optimized GeoTIFF (forwarder to ``Dataset.to_cog``)."""
        return self._ds.to_cog(*args, **kwargs)

    @property
    def is_cog(self) -> bool:
        """Whether the source file is a valid COG (forwarder to ``Dataset.is_cog``)."""
        return self._ds.is_cog

    def validate_cog(self, *args: Any, **kwargs: Any) -> Any:
        """Validate the dataset against the COG specification (forwarder to ``Dataset.validate_cog``)."""
        return self._ds.validate_cog(*args, **kwargs)


__all__ = [
    "IO",
    "Spatial",
    "Bands",
    "Analysis",
    "Cell",
    "Vectorize",
    "COG",
]
