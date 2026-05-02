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
        return self._ds.read_array(*args, **kwargs)

    def write_array(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.write_array(*args, **kwargs)

    def to_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_file(*args, **kwargs)

    def to_raster(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_raster(*args, **kwargs)

    def get_block_arrangement(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_block_arrangement(*args, **kwargs)

    def get_tile(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_tile(*args, **kwargs)

    def map_blocks(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.map_blocks(*args, **kwargs)

    def to_xyz(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_xyz(*args, **kwargs)

    @property
    def overview_count(self) -> list[int]:
        return self._ds.overview_count

    def create_overviews(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.create_overviews(*args, **kwargs)

    def recreate_overviews(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.recreate_overviews(*args, **kwargs)

    def get_overview(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_overview(*args, **kwargs)

    def read_overview_array(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.read_overview_array(*args, **kwargs)


class Spatial(_Collaborator):
    """Spatial operations on a Dataset (crop, to_crs, align, …).

    Stage 1 stub.
    """

    def crop(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.crop(*args, **kwargs)

    def to_crs(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_crs(*args, **kwargs)

    def set_crs(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.set_crs(*args, **kwargs)

    def convert_longitude(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.convert_longitude(*args, **kwargs)

    def resample(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.resample(*args, **kwargs)

    def align(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.align(*args, **kwargs)

    def fill_gaps(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.fill_gaps(*args, **kwargs)


class Bands(_Collaborator):
    """Band-metadata operations on a Dataset (attribute table, color, …).

    Stage 1 stub.
    """

    def get_attribute_table(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_attribute_table(*args, **kwargs)

    def set_attribute_table(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.set_attribute_table(*args, **kwargs)

    def add_band(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.add_band(*args, **kwargs)

    @property
    def band_color(self) -> dict[int, str]:
        return self._ds.band_color

    @band_color.setter
    def band_color(self, values: dict[int, str]) -> None:
        self._ds.band_color = values

    @property
    def color_table(self) -> Any:
        return self._ds.color_table

    @color_table.setter
    def color_table(self, df: Any) -> None:
        self._ds.color_table = df

    def get_band_by_color(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_band_by_color(*args, **kwargs)

    def change_no_data_value(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.change_no_data_value(*args, **kwargs)


class Analysis(_Collaborator):
    """Analysis / statistics / plot operations on a Dataset.

    Stage 1 stub.
    """

    def stats(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.stats(*args, **kwargs)

    def count_domain_cells(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.count_domain_cells(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.apply(*args, **kwargs)

    def fill(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.fill(*args, **kwargs)

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.extract(*args, **kwargs)

    def overlay(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.overlay(*args, **kwargs)

    def get_mask(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_mask(*args, **kwargs)

    def footprint(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.footprint(*args, **kwargs)

    def get_histogram(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_histogram(*args, **kwargs)

    @staticmethod
    def normalize(array: Any) -> Any:
        # Defer to the staticmethod on the Analysis mixin so callers
        # who take the collaborator's normalize get the same body.
        from pyramids.dataset.ops.analysis import Analysis as _AnalysisMixin

        return _AnalysisMixin.normalize(array)

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.plot(*args, **kwargs)


class Cell(_Collaborator):
    """Cell-geometry operations on a Dataset.

    Stage 1 stub.
    """

    def get_cell_coords(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_cell_coords(*args, **kwargs)

    def get_cell_polygons(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_cell_polygons(*args, **kwargs)

    def get_cell_points(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.get_cell_points(*args, **kwargs)

    def map_to_array_coordinates(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.map_to_array_coordinates(*args, **kwargs)

    def array_to_map_coordinates(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.array_to_map_coordinates(*args, **kwargs)


class Vectorize(_Collaborator):
    """Vectorisation / clustering operations on a Dataset.

    Stage 1 stub.
    """

    def to_feature_collection(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_feature_collection(*args, **kwargs)

    def translate(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.translate(*args, **kwargs)

    def cluster(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.cluster(*args, **kwargs)

    def cluster2(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.cluster2(*args, **kwargs)


class COG(_Collaborator):
    """Cloud Optimized GeoTIFF operations on a Dataset.

    Stage 1 stub.
    """

    def to_cog(self, *args: Any, **kwargs: Any) -> Any:
        return self._ds.to_cog(*args, **kwargs)

    @property
    def is_cog(self) -> bool:
        return self._ds.is_cog

    def validate_cog(self, *args: Any, **kwargs: Any) -> Any:
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
