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

import collections
import logging
import warnings
import weakref
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from hpc.indexing import get_indices2, get_pixels, get_pixels2, locate_values
from osgeo import gdal, ogr, osr
from pandas import DataFrame

from pyramids.base._domain import inside_domain, is_no_data
from pyramids.base._errors import AlignmentError, CRSError
from pyramids.base._utils import (
    INTERPOLATION_METHODS,
    gdal_to_ogr_dtype,
    import_cleopatra,
)
from pyramids.base.crs import (
    epsg_from_wkt,
    reproject_coordinates,
    sr_from_epsg,
    sr_from_wkt,
)
from pyramids.dataset.abstract_dataset import AbstractDataset
from pyramids.dataset.cog import (
    ValidationReport,
    merge_options,
    translate_to_cog,
    validate,
    validate_blocksize,
)
from pyramids.feature import FeatureCollection
from pyramids.feature import _ogr as _feature_ogr

if TYPE_CHECKING:
    from cleopatra.array_glyph import ArrayGlyph

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

    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        return str(self._ds.raster.GetProjection())

    def set_crs(self, crs: str | None = None, epsg: int | None = None) -> None:
        """Set the Coordinate Reference System (CRS).

            Set the Coordinate Reference System (CRS) of a

        Args:
            crs (str):
                Optional if epsg is specified. WKT string. i.e.
                    ```
                    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84", 6378137,298.257223563,AUTHORITY["EPSG","7030"],
                    AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
                    0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],
                    AUTHORITY["EPSG","4326"]]'
                    ```
            epsg (int):
                Optional if crs is specified. EPSG code specifying the projection.
        """
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if self._ds.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then "
                "reset the crs"
            )
        else:
            if crs is not None:
                self._ds.raster.SetProjection(crs)
                # ARC-7: fallback to 4326 when crs is an empty string
                # (get_epsg_from_prj raises in that case); epsg_from_wkt
                # absorbs the fallback in one place.
                self._ds._epsg = epsg_from_wkt(crs)
            elif epsg is not None:
                sr = sr_from_epsg(epsg)
                self._ds.raster.SetProjection(sr.ExportToWkt())
                self._ds._epsg = epsg
            else:
                raise ValueError("Either crs or epsg must be provided.")

    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
    ) -> Dataset:
        """Reproject the dataset to any projection.

            (default the WGS84 web mercator projection, without resampling)

        Args:
            to_epsg (int):
                reference number to the new projection (https://epsg.io/). Default 3857 is the reference number of WGS84
                web mercator.
            method (str):
                resampling method. Default is "nearest neighbor". See https://gisgeography.com/raster-resampling/.
                Allowed values: "nearest neighbor", "cubic", "bilinear".
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the raster the same after reprojection.
                Default is False.

        Returns:
            Dataset:
                A new reprojected Dataset.

        Examples:
            - Create a dataset and reproject it:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> print(dataset.crs)
              GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
              >>> print(dataset.epsg)
              4326
              >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)
              >>> print(reprojected_dataset)
              <BLANKLINE>
                          Cell size: 5565.983370404396
                          Dimension: 5 * 5
                          EPSG: 3857
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> print(reprojected_dataset.crs)
              PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]
              >>> print(reprojected_dataset.epsg)
              3857

              ```

        """
        if not isinstance(to_epsg, int):
            raise TypeError(
                "please enter correct integer number for to_epsg more information "
                f"https://epsg.io/, given {type(to_epsg)}"
            )
        if not isinstance(method, str):
            raise TypeError(
                "Please enter a correct method, for more information, see documentation "
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method: {method} does not exist, existing methods are "
                f"{INTERPOLATION_METHODS.keys()}"
            )

        resampling_method: Any = INTERPOLATION_METHODS.get(method)

        if maintain_alignment:
            dst_obj = self._reproject_with_ReprojectImage(to_epsg, resampling_method)
        else:
            dst = gdal.Warp("", self._ds.raster, dstSRS=f"EPSG:{to_epsg}", format="VRT")
            dst_obj = self._ds.__class__(dst)

        return dst_obj

    def _get_epsg(self) -> int:
        """Get the EPSG number.

            This function reads the projection of a GEOGCS file or tiff file.

        Returns:
            int: EPSG number.
        """
        prj = self._get_crs()
        # ARC-7: get_epsg_from_prj raises on empty input; epsg_from_wkt
        # absorbs the historical 4326 fallback for datasets without a
        # projection.
        epsg = epsg_from_wkt(prj)

        return epsg

    def convert_longitude(self) -> Dataset:
        """Convert Longitude.

        - convert the longitude from 0-360 to -180 - 180.
        - currently the function works correctly if the raster covers the whole world, it means that the columns
            in the rasters covers from longitude 0 to 360.

        Returns:
            Dataset:
                A new Dataset with longitude converted to -180/180.
        """
        # dst = gdal.Warp(
        #     "",
        #     self._ds.raster,
        #     dstSRS="+proj=longlat +ellps=WGS84 +datum=WGS84 +lon_0=0 +over",
        #     format="VRT",
        # )
        lon = self._ds.lon
        src = self._ds.raster
        # create a copy
        drv = gdal.GetDriverByName("MEM")
        dst = drv.CreateCopy("", src, 0)
        # convert the 0 to 360 to -180 to 180
        if lon[-1] <= 180:
            raise ValueError("The raster should cover the whole globe")

        first_to_translated = np.where(lon > 180)[0][0]

        ind = list(range(first_to_translated, len(lon)))
        ind_2 = list(range(0, first_to_translated))

        for band in range(self._ds.band_count):
            arr = self._ds.read_array(band=band)
            arr_rearranged = arr[:, ind + ind_2]
            dst.GetRasterBand(band + 1).WriteArray(arr_rearranged)

        # correct the geotransform
        top_left_corner = self._ds.top_left_corner
        gt = list(self._ds.geotransform)
        if lon[-1] > 180:
            new_gt = top_left_corner[0] - 180
            gt[0] = new_gt

        dst.SetGeoTransform(gt)
        return self._ds.__class__(dst)

    def resample(
        self, cell_size: int | float, method: str = "nearest neighbor"
    ) -> Dataset:
        """resample.

        resample method reprojects a raster to any projection (default the WGS84 web mercator projection,
        without resampling). The function returns a GDAL in-memory file object.

        Args:
            cell_size (int):
                New cell size to resample the raster. If None, raster will not be resampled.
            method (str):
                Resampling method: "nearest neighbor", "cubic", or "bilinear". Default is "nearest neighbor".

        Returns:
            Dataset:
                A new resampled Dataset.

        Examples:
            - Create a Dataset with 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 10 * 10
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File: ...
              <BLANKLINE>
              >>> dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
              ![resample-source](./../../_images/dataset/resample-source.png)

            - Resample the raster to a new cell size of 0.1:

              ```python
              >>> new_dataset = dataset.resample(cell_size=0.1)
              >>> print(new_dataset)
              <BLANKLINE>
                          Cell size: 0.1
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 4
                          Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>
              >>> new_dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
              ![resample-new](./../../_images/dataset/resample-new.png)

            - Resampling the dataset from cell_size 0.05 to 0.1 degrees reduced the number of cells to 5 in each
              dimension instead of 10.
        """
        if not isinstance(method, str):
            raise TypeError(
                "Please enter a correct method, for more information, see documentation"
            )
        if method not in INTERPOLATION_METHODS.keys():
            raise ValueError(
                f"The given interpolation method does not exist, existing methods are "
                f"{INTERPOLATION_METHODS.keys()}"
            )

        resampling_method: Any = INTERPOLATION_METHODS.get(method)

        sr_src = sr_from_wkt(self._ds.crs)

        ulx = self._ds.geotransform[0]
        uly = self._ds.geotransform[3]
        # transform the right lower corner point
        lrx = self._ds.geotransform[0] + self._ds.geotransform[1] * self._ds.columns
        lry = self._ds.geotransform[3] + self._ds.geotransform[5] * self._ds.rows

        # new geotransform
        new_geo = (
            self._ds.geotransform[0],
            cell_size,
            self._ds.geotransform[2],
            self._ds.geotransform[3],
            self._ds.geotransform[4],
            -1 * cell_size,
        )
        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / cell_size))
        rows = int(np.round(abs(uly - lry) / cell_size))
        dtype = self._ds.gdal_dtype[0]
        bands = self._ds.band_count

        dst_obj = self._ds.__class__._build_dataset(
            cols,
            rows,
            bands,
            dtype,
            new_geo,
            sr_src.ExportToWkt(),
            self._ds.no_data_value,
        )
        gdal.ReprojectImage(
            self._ds.raster,
            dst_obj.raster,
            sr_src.ExportToWkt(),
            sr_src.ExportToWkt(),
            resampling_method,
        )

        return dst_obj

    def _reproject_with_ReprojectImage(
        self, to_epsg: int, method: str = "nearest neighbor"
    ) -> Dataset:
        src_gt = self._ds.geotransform
        src_x = self._ds.columns
        src_y = self._ds.rows

        src_sr = sr_from_wkt(self._ds.crs)
        src_epsg = self._ds.epsg

        dst_sr = sr_from_epsg(to_epsg)

        # in case the source crs is GCS and longitude is in the west hemisphere, gdal
        # reads longitude from 0 to 360 and a transformation factor wont work with values
        # greater than 180
        if src_epsg != to_epsg:
            if src_epsg == "4326" and src_gt[0] > 180:
                lng_new = src_gt[0] - 360
                # transformation factors
                tx = osr.CoordinateTransformation(src_sr, dst_sr)
                # transform the right upper corner point
                ulx, uly, ulz = tx.TransformPoint(lng_new, src_gt[3])
                # transform the right lower corner point
                lrx, lry, lrz = tx.TransformPoint(
                    lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                )
            else:
                xs = [src_gt[0], src_gt[0] + src_gt[1] * src_x]
                ys = [src_gt[3], src_gt[3] + src_gt[5] * src_y]

                # ARC-14: reproject_coordinates takes (x, y) and returns (x, y).
                [ulx, lrx], [uly, lry] = reproject_coordinates(
                    xs, ys, from_crs=src_epsg, to_crs=to_epsg
                )
                # old transform
                # # transform the right upper corner point
                # (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
                # # transform the right lower corner point
                # (lrx, lry, lrz) = tx.TransformPoint(
                #     src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
                # )

        else:
            ulx = src_gt[0]
            uly = src_gt[3]
            lrx = src_gt[0] + src_gt[1] * src_x
            lry = src_gt[3] + src_gt[5] * src_y

        # get the cell size in the source raster and convert it to the new crs
        # x coordinates or longitudes
        xs = [src_gt[0], src_gt[0] + src_gt[1]]
        # y coordinates or latitudes
        ys = [src_gt[3], src_gt[3]]

        if src_epsg != to_epsg:
            # transform the two-point coordinates to the new crs to calculate the new cell size
            # ARC-14: reproject_coordinates takes (x, y) and returns (x, y).
            new_xs, new_ys = reproject_coordinates(
                xs, ys, from_crs=src_epsg, to_crs=to_epsg, precision=6
            )
        else:
            new_xs = xs
            # new_ys = ys

        # TODO: the function does not always maintain alignment, based on the conversion of the cell_size and the
        # pivot point
        pixel_spacing = np.abs(new_xs[0] - new_xs[1])

        # create a new raster
        cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
        rows = int(np.round(abs(uly - lry) / pixel_spacing))

        dtype = self._ds.gdal_dtype[0]
        new_geo = (
            ulx,
            pixel_spacing,
            src_gt[2],
            uly,
            src_gt[4],
            np.sign(src_gt[-1]) * pixel_spacing,
        )
        dst_obj = self._ds.__class__._build_dataset(
            cols,
            rows,
            self._ds.band_count,
            dtype,
            new_geo,
            dst_sr.ExportToWkt(),
            self._ds.no_data_value,
        )
        gdal.ReprojectImage(
            self._ds.raster,
            dst_obj.raster,
            src_sr.ExportToWkt(),
            dst_sr.ExportToWkt(),
            method,
        )
        return dst_obj

    def fill_gaps(self, mask, src_array: np.ndarray) -> np.ndarray:
        """Fill gaps in src_array using nearest neighbors where mask indicates valid cells.

        Args:
            mask (Dataset | np.ndarray):
                Mask dataset or array used to determine valid cells.
            src_array (np.ndarray):
                Source array whose gaps will be filled.

        Returns:
            np.ndarray: The source array with gaps filled where applicable.
        """
        # align function only equate the no of rows and columns only
        # match no_data_value inserts no_data_value in src raster to all places like mask
        # still places that has no_data_value in the src raster, but it is not no_data_value in the mask
        # and now has to be filled with values
        # compare no of element that is not no_data_value in both rasters to make sure they are matched
        # if both inputs are rasters
        mask_array = mask.read_array()
        row = mask.rows
        col = mask.columns
        mask_noval = mask.no_data_value[0]

        if isinstance(mask, AbstractDataset) and isinstance(self._ds, AbstractDataset):
            src_no_data = is_no_data(src_array, self._ds.no_data_value[0])
            mask_no_data = is_no_data(mask_array, mask_noval)
            elem_src = src_array.size - np.count_nonzero(src_array[src_no_data])
            elem_mask = mask_array.size - np.count_nonzero(mask_array[mask_no_data])

            # Cells that are out-of-domain in src but in-domain in mask
            # need to be interpolated from neighbors.
            if elem_mask > elem_src:
                gap_rows, gap_cols = np.where(src_no_data & ~mask_no_data)
                src_array = Vectorize._nearest_neighbour(
                    src_array,
                    self._ds.no_data_value[0],
                    gap_rows.tolist(),
                    gap_cols.tolist(),
                )
        return src_array

    def _crop_aligned(
        self,
        mask: gdal.Dataset | np.ndarray,
        mask_noval: int | float | None = None,
        fill_gaps: bool = False,
    ) -> Dataset:
        """Clip/crop by matching the nodata layout from mask to the source raster.

        Both rasters must have the same dimensions (rows and columns). Use MatchRasterAlignment prior to this
        method to align both rasters.

        Args:
            mask (Dataset | np.ndarray):
                Mask raster to get the location of the NoDataValue and where it is in the array.
            mask_noval (int | float, optional):
                In case the mask is a numpy array, the mask_noval has to be given.
            fill_gaps (bool):
                Whether to fill gaps after cropping. Default is False.

        Returns:
            Dataset:
                The raster with NoDataValue stored in its cells exactly the same as the source raster.
        """
        if isinstance(mask, AbstractDataset):
            mask_gt = mask.geotransform
            mask_epsg = mask.epsg
            row = mask.rows
            col = mask.columns
            mask_noval = mask.no_data_value[0]
            mask_array = mask.read_array(band=0)
        elif isinstance(mask, np.ndarray):
            if mask_noval is None:
                raise ValueError(
                    "You have to enter the value of the no_val parameter when the mask is a numpy array"
                )
            mask_array = mask.copy()
            row, col = mask.shape
        else:
            raise TypeError(
                "The second parameter 'mask' has to be either gdal.Dataset or numpy array"
                f"given - {type(mask)}"
            )

        band_count = self._ds.band_count
        src_sref = sr_from_wkt(self._ds.crs)
        src_array = self._ds.read_array()

        if not row == self._ds.rows or not col == self._ds.columns:
            raise ValueError(
                "Two rasters have different number of columns or rows, please resample or match both rasters"
            )

        if isinstance(mask, AbstractDataset):
            if (
                not self._ds.top_left_corner == mask.top_left_corner
                or not self._ds.cell_size == mask.cell_size
            ):
                raise ValueError(
                    "the location of the upper left corner of both rasters is not the same or cell size is "
                    "different please match both rasters first "
                )

            if not mask_epsg == self._ds.epsg:
                raise ValueError(
                    "Dataset A & B are using different coordinate systems please reproject one of them to "
                    "the other raster coordinate system"
                )

        mask_no_data = is_no_data(mask_array, mask_noval)
        if band_count > 1:
            # check if the no data value for the src complies with the dtype of the src as sometimes the band is full
            # of values and the no_data_value is not used at all in the band, and when we try to replace any value in
            # the array with the no_data_value it will raise an error.
            no_data_value = self._ds._check_no_data_value(self._ds.no_data_value)
            for band in range(self._ds.band_count):
                src_array[band, mask_no_data] = no_data_value[band]
        else:
            src_array[mask_no_data] = self._ds.no_data_value[0]

        if fill_gaps:
            src_array = self.fill_gaps(mask, src_array)

        dst = self._ds.__class__._create_dataset(
            col, row, band_count, self._ds.gdal_dtype[0], driver="MEM"
        )
        # but with a lot of computations,
        # if the mask is an array and the mask_gt is not defined, use the src_gt as both the mask and the src
        # are aligned, so they have the sam gt
        try:
            # set the geotransform
            dst.SetGeoTransform(mask_gt)
            # set the projection
            dst.SetProjection(mask.crs)
        except UnboundLocalError:
            dst.SetGeoTransform(self._ds.geotransform)
            dst.SetProjection(src_sref.ExportToWkt())

        dst_obj = self._ds.__class__(dst)
        # set the no data value
        dst_obj._set_no_data_value(self._ds.no_data_value)
        if band_count > 1:
            for band in range(band_count):
                dst_obj.raster.GetRasterBand(band + 1).WriteArray(src_array[band, :, :])
        else:
            dst_obj.raster.GetRasterBand(1).WriteArray(src_array)
        return dst_obj

    def _check_alignment(self, mask) -> bool:
        """Check if raster is aligned with a given mask raster."""
        if not isinstance(mask, AbstractDataset):
            raise TypeError("The second parameter should be a Dataset")

        return self._ds.rows == mask.rows and self._ds.columns == mask.columns

    def align(
        self,
        alignment_src: Dataset,
    ) -> Dataset:
        """Align the current dataset (rows and columns) to match a given dataset.

        Copies spatial properties from alignment_src to the current raster:
            - The coordinate system
            - The number of rows and columns
            - Cell size
        Then resamples values from the current dataset using the nearest neighbor interpolation.

        Args:
            alignment_src (Dataset):
                Spatial information source raster to get the spatial information (coordinate system, number of rows and
                columns). The data values of the current dataset are resampled to this alignment.

        Returns:
            Dataset: A new aligned Dataset.

        Examples:
            - The source dataset has a `top_left_corner` at (0, 0) with a 5*5 alignment, and a 0.05 degree cell size.

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              ```

            - The dataset to be aligned has a top_left_corner at (-0.1, 0.1) (i.e., it has two more rows on top of the
              dataset, and two columns on the left of the dataset).

              ```python
              >>> arr = np.random.rand(10, 10)
              >>> top_left_corner = (-0.1, 0.1)
              >>> cell_size = 0.07
              >>> dataset_target = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,
              ... epsg=4326)
              >>> print(dataset_target)
              <BLANKLINE>
                          Cell size: 0.07
                          Dimension: 10 * 10
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              ```

            ![align-source-target](./../../_images/dataset/align-source-target.png)

            - Now call the `align` method and use the dataset as the alignment source.

              ```python
              >>> aligned_dataset = dataset_target.align(dataset)
              >>> print(aligned_dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 5 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              ```

            ![align-result](./../../_images/dataset/align-result.png)
        """
        if isinstance(alignment_src, AbstractDataset):
            src = alignment_src
        else:
            raise TypeError(
                "First parameter should be a Dataset read using Dataset.openRaster or a path to the raster, "
                f"given {type(alignment_src)}"
            )

        # reproject the raster to match the projection of alignment_src
        reprojected_raster_b: Dataset = self._ds
        if self._ds.epsg != src.epsg:
            reprojected_raster_b = self.to_crs(src.epsg)  # type: ignore[assignment]
        dst_obj = self._ds.__class__._build_dataset(
            src.columns,
            src.rows,
            self._ds.band_count,
            src.gdal_dtype[0],
            src.geotransform,
            src.crs,
            self._ds.no_data_value,
        )
        method = gdal.GRA_NearestNeighbour
        # resample the reprojected_RasterB
        gdal.ReprojectImage(
            reprojected_raster_b.raster,
            dst_obj.raster,
            src.crs,
            src.crs,
            method,
        )

        return dst_obj

    def _crop_with_raster(
        self,
        mask: gdal.Dataset | str,
    ) -> Dataset:
        """Crop this raster using another raster as a mask.

        Args:
            mask (Dataset | str):
                The raster you want to use as a mask to crop this raster; it can be a path or a GDAL Dataset.

        Returns:
            Dataset:
                The cropped raster.
        """
        # get information from the mask raster
        if isinstance(mask, (str, Path)):
            mask = self._ds.__class__.read_file(mask)
        elif isinstance(mask, AbstractDataset):
            mask = mask
        else:
            raise TypeError(
                "The second parameter has to be either path to the mask raster or a gdal.Dataset object"
            )
        if not self._check_alignment(mask):
            # first align the mask with the src raster
            mask = mask.align(self._ds)
        # crop the src raster with the aligned mask
        dst_obj = self._crop_aligned(mask)

        dst_obj = Spatial._correct_wrap_cutline_error(dst_obj)
        return dst_obj

    def _crop_with_polygon_warp(
        self, feature: FeatureCollection | GeoDataFrame, touch: bool = True
    ) -> Dataset:
        """Crop raster with polygon.

            - Do not convert the polygon into a raster but rather use it directly to crop the raster using the
            gdal.warp function.

        Args:
            feature (FeatureCollection | GeoDataFrame):
                Vector mask.
            touch (bool):
                Include cells that touch the polygon, not only those entirely inside the polygon mask. Defaults to True.

        Returns:
            Dataset:
                Cropped dataset.
        """
        if isinstance(feature, GeoDataFrame):
            feature = FeatureCollection(feature)
        else:
            if not isinstance(feature, FeatureCollection):
                raise TypeError(
                    f"The function takes only a FeatureCollection or GeoDataFrame, given {type(feature)}"
                )

        # gdal.Warp's cutlineDSName needs a *path*; stage the vector in
        # /vsimem/ through the internal OGR bridge. The path is unlinked
        # automatically when the with-block exits.
        # Use the base Dataset class (not a subclass like NetCDF) for intermediate GDAL warp results
        # because _correct_wrap_cutline_error calls create_from_array which has different behavior in
        # subclasses.
        base_cls = next(
            c
            for c in self._ds.__class__.__mro__
            if AbstractDataset in getattr(c, "__bases__", ())
        )

        # The warp output (VRT) may resolve the cutline lazily, so we must
        # complete every access that could touch the cutline path inside
        # the with-block that keeps that path alive.
        with _feature_ogr.as_vsimem_path(feature) as cutline_path:
            warp_options = gdal.WarpOptions(
                format="VRT",
                cropToCutline=not touch,
                cutlineDSName=cutline_path,
                multithread=True,
            )
            dst = gdal.Warp("", self._ds.raster, options=warp_options)
            dst_obj = base_cls(dst)
            if touch:
                dst_obj = Spatial._correct_wrap_cutline_error(dst_obj)

        return dst_obj

    @staticmethod
    def _correct_wrap_cutline_error(src: Dataset) -> Dataset:
        """Correct wrap cutline error.

        https://github.com/serapeum-org/pyramids/issues/74
        """
        big_array = src.read_array()
        value_to_remove = src.no_data_value[0]
        """Remove rows and columns that are all filled with a certain value from a 2D array."""
        # Find rows and columns to be removed
        if big_array.ndim == 2:
            rows_to_remove = np.all(big_array == value_to_remove, axis=1)
            cols_to_remove = np.all(big_array == value_to_remove, axis=0)
            # Use boolean indexing to remove rows and columns
            small_array = big_array[~rows_to_remove][:, ~cols_to_remove]
        elif big_array.ndim == 3:
            rows_to_remove = np.all(big_array == value_to_remove, axis=(0, 2))
            cols_to_remove = np.all(big_array == value_to_remove, axis=(0, 1))
            # Use boolean indexing to remove rows and columns
            # first remove the rows then the columns
            small_array = big_array[:, ~rows_to_remove, :]
            small_array = small_array[:, :, ~cols_to_remove]
            n_rows = np.count_nonzero(~rows_to_remove)
            n_cols = np.count_nonzero(~cols_to_remove)
            small_array = small_array.reshape((src.band_count, n_rows, n_cols))
        else:
            raise ValueError("Array must be 2D or 3D")

        x_ind = np.where(~rows_to_remove)[0][0]
        y_ind = np.where(~cols_to_remove)[0][0]
        new_x = src.x[y_ind] - src.cell_size / 2
        new_y = src.y[x_ind] + src.cell_size / 2
        new_gt = (new_x, src.cell_size, 0, new_y, 0, -src.cell_size)
        new_src = src.create_from_array(
            small_array, geo=new_gt, epsg=src.epsg, no_data_value=src.no_data_value
        )
        return new_src

    def crop(
        self,
        mask: GeoDataFrame | FeatureCollection,
        touch: bool = True,
    ) -> Dataset:
        """Crop dataset using dataset/feature collection.

            Crop/Clip the Dataset object using a polygon/raster.

        Args:
            mask (GeoDataFrame | Dataset):
                GeoDataFrame with a polygon geometry, or a Dataset object.
            touch (bool):
                Include the cells that touch the polygon, not only those that lie entirely inside the polygon mask.
                Default is True.

        Returns:
            Dataset:
                A new cropped Dataset.

        Hint:
            - If the mask is a dataset with multi-bands, the `crop` method will use the first band as the mask.

        Examples:
            - Crop the raster using a polygon mask.

              - The polygon covers 4 cells in the 3rd and 4th rows and 3rd and 4th column `arr[2:4, 2:4]`, so the result
                dataset will have the same number of bands `4`, 2 rows and 2 columns.
              - First, create the dataset to have 4 bands, 10 rows and 10 columns; the dataset has a cell size of 0.05
                degree, the top left corner of the dataset is (0, 0).

              ```python
              >>> import numpy as np
              >>> import geopandas as gpd
              >>> from shapely.geometry import Polygon
              >>> arr = np.random.rand(4, 10, 10)
              >>> cell_size = 0.05
              >>> top_left_corner = (0, 0)
              >>> dataset = Dataset.create_from_array(
              ...         arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```
            - Second, create the polygon using shapely polygon, and use the xmin, ymin, xmax, ymax = [0.1, -0.2, 0.2 -0.1]
                to cover the 4 cells.

                ```python
                >>> mask = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])], crs=4326)

                ```
            - Pass the `geodataframe` to the crop method using the `mask` parameter.

              ```python
              >>> cropped_dataset = dataset.crop(mask=mask)

              ```
            - Check the cropped dataset:

              ```python
              >>> print(cropped_dataset.shape)
              (4, 2, 2)
              >>> print(cropped_dataset.geotransform)
              (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> print(cropped_dataset.read_array(band=0))# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]
              >>> print(arr[0, 2:4, 2:4])# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]

              ```
            - Crop a raster using another raster mask:

              - Create a mask dataset with the same extent of the polygon we used in the previous example.

              ```python
              >>> geotransform = (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> mask_dataset = Dataset.create_from_array(np.random.rand(2, 2), geo=geotransform, epsg=4326)

              ```
            - Then use the mask dataset to crop the dataset.

              ```python
              >>> cropped_dataset_2 = dataset.crop(mask=mask_dataset)
              >>> print(cropped_dataset_2.shape)
              (4, 2, 2)

              ```
            - Check the cropped dataset:

              ```python
              >>> print(cropped_dataset_2.geotransform)
              (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
              >>> print(cropped_dataset_2.read_array(band=0))# doctest: +SKIP
              [[0.00921161 0.90841171]
               [0.355636   0.18650262]]
              >>> print(arr[0, 2:4, 2:4])# doctest: +SKIP
               [[0.00921161 0.90841171]
               [0.355636   0.18650262]]

              ```

        """
        if isinstance(mask, GeoDataFrame):
            dst = self._crop_with_polygon_warp(mask, touch=touch)
        elif isinstance(mask, AbstractDataset):
            dst = self._crop_with_raster(mask)
        else:
            raise TypeError(
                "The second parameter: mask could be either GeoDataFrame or Dataset object"
            )

        return dst


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
    """Mixin providing analysis, statistics, and data extraction operations for Dataset."""

    def stats(
        self, band: int | None = None, mask: GeoDataFrame | None = None
    ) -> DataFrame:
        """Get statistics of a band [Min, max, mean, std].

        Args:
            band (int, optional):
                Band index. If None, the statistics of all bands will be returned.
            mask (Polygon GeoDataFrame or Dataset, optional):
                GeodataFrame with a geometry of polygon type.

        Returns:
            DataFrame:
                DataFrame wit the stats of each band, the dataframe has the following columns
                [min, max, mean, std], the index of the dataframe is the band names.

                ```text

                                   Min         max        mean       std
                    Band_1  270.369720  270.762299  270.551361  0.154270
                    Band_2  269.611938  269.744751  269.673645  0.043788
                    Band_3  273.641479  274.168823  273.953979  0.198447
                    Band_4  273.991516  274.540344  274.310669  0.205754
                ```

        Notes:
            - The value of the stats will be stored in an xml file by the name of the raster file with the extension of
              .aux.xml.
            - The content of the file will be like the following:

              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Metadata>
                        <MDI key="RepresentationType">ATHEMATIC</MDI>
                        <MDI key="STATISTICS_MAXIMUM">88</MDI>
                        <MDI key="STATISTICS_MEAN">7.9662921348315</MDI>
                        <MDI key="STATISTICS_MINIMUM">0</MDI>
                        <MDI key="STATISTICS_STDDEV">18.294377743948</MDI>
                        <MDI key="STATISTICS_VALID_PERCENT">48.9</MDI>
                      </Metadata>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Get the statistics of all bands in the dataset:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> geotransform = (0, 0.05, 0, 0, 0, -0.05)
              >>> dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
              >>> print(dataset.stats()) # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.006443  0.942943  0.468935  0.266634
              Band_2  0.020377  0.978130  0.477189  0.306864
              Band_3  0.019652  0.992184  0.537215  0.286502
              Band_4  0.011955  0.984313  0.503616  0.295852
              >>> print(dataset.stats(band=1))  # doctest: +SKIP
                           min      max      mean       std
              Band_2  0.020377  0.97813  0.477189  0.306864

              ```

            - Get the statistics of all the bands using a mask polygon.

              - Create the polygon using shapely polygon, and use the xmin, ymin, xmax, ymax = [0.1, -0.2,
                0.2 -0.1] to cover the 4 cells.
              ```python
              >>> from shapely.geometry import Polygon
              >>> import geopandas as gpd
              >>> mask = gpd.GeoDataFrame(geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],crs=4326)
              >>> print(dataset.stats(mask=mask))  # doctest: +SKIP
                           min       max      mean       std
              Band_1  0.193441  0.702108  0.541478  0.202932
              Band_2  0.281281  0.932573  0.665602  0.239410
              Band_3  0.031395  0.982235  0.493086  0.377608
              Band_4  0.079562  0.930965  0.591025  0.341578

              ```

        """
        dst: Dataset | None = None
        if mask is not None:
            dst = self._ds.crop(mask, touch=True)

        if band is None:
            df = pd.DataFrame(
                index=self._ds.band_names,
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            for i in range(self._ds.band_count):
                if mask is not None and dst is not None:
                    df.iloc[i, :] = dst.analysis._get_stats(i)
                else:
                    df.iloc[i, :] = self._get_stats(i)
        else:
            df = pd.DataFrame(
                index=[self._ds.band_names[band]],
                columns=["min", "max", "mean", "std"],
                dtype=np.float32,
            )
            if mask is not None and dst is not None:
                df.iloc[0, :] = dst.analysis._get_stats(band)
            else:
                df.iloc[0, :] = self._get_stats(band)

        return df

    def _get_stats(self, band: int | None = None) -> list[float]:
        """_get_stats."""
        band_index = band if band is not None else 0
        band_i = self._ds._iloc(band_index)
        try:
            vals = band_i.GetStatistics(True, True)
        except RuntimeError:
            # when the GetStatistics gives an error "RuntimeError: Failed to compute statistics, no valid pixels
            # found in sampling."
            vals = [0]

        if sum(vals) == 0:
            warnings.warn(
                f"Band {band} has no statistics, and the statistics are going to be calculate"
            )
            vals = band_i.ComputeStatistics(False)

        return list(vals)

    def count_domain_cells(self, band: int = 0) -> int:
        """Count cells inside the domain.

        Args:
            band (int):
                Band index. Default is 0.

        Returns:
            int:
                Number of cells.
        """
        arr = self._ds.read_array(band=band)
        domain_count = np.size(arr[:, :]) - np.count_nonzero(
            arr[is_no_data(arr, self._ds.no_data_value[band])]
        )
        return int(domain_count)

    def apply(self, func, band: int = 0, inplace: bool = False) -> Dataset:
        """Apply a function to all domain cells.

        - apply method executes a mathematical operation on the raster array.
        - The function is applied to all domain cells at once using vectorized NumPy operations.

        Args:
            func (function):
                Defined function that takes one input (the cell value).
            band (int):
                Band number.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created.
                Default is False.

        Returns:
            Dataset:
                A new Dataset with the function applied. If inplace is True, returns self.

        Examples:
            - Create a dataset from an array filled with values between -1 and 1:

              ```python
              >>> import numpy as np
              >>> arr = np.random.uniform(-1, 1, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[ 0.94997539 -0.80083622 -0.30948769 -0.77439961 -0.83836424]
               [-0.36810158 -0.23979251  0.88051216 -0.46882913  0.64511056]
               [ 0.50585374 -0.46905902  0.67856589  0.2779605   0.05589759]
               [ 0.63382852 -0.49259597  0.18471423 -0.49308984 -0.52840286]
               [-0.34076174 -0.53073014 -0.18485789 -0.40033474 -0.38962938]]

              ```

            - Apply the absolute function to the dataset:

              ```python
              >>> abs_dataset = dataset.apply(np.abs)
              >>> print(abs_dataset.read_array()) # doctest: +SKIP
              [[0.94997539 0.80083622 0.30948769 0.77439961 0.83836424]
               [0.36810158 0.23979251 0.88051216 0.46882913 0.64511056]
               [0.50585374 0.46905902 0.67856589 0.2779605  0.05589759]
               [0.63382852 0.49259597 0.18471423 0.49308984 0.52840286]
               [0.34076174 0.53073014 0.18485789 0.40033474 0.38962938]]

              ```
        """
        if not callable(func):
            raise TypeError("The second argument should be a function")

        no_data_value = self._ds.no_data_value[band]
        src_array = self._ds.read_array(band)
        dtype = self._ds.gdal_dtype[band]

        new_array = np.full(
            (self._ds.rows, self._ds.columns), no_data_value, dtype=src_array.dtype
        )
        domain_mask = inside_domain(src_array, no_data_value)
        domain_values = src_array[domain_mask]
        try:
            new_array[domain_mask] = func(domain_values)
        except (ValueError, TypeError):
            new_array[domain_mask] = np.vectorize(func)(domain_values)

        dst_obj = self._ds.__class__._build_dataset(
            self._ds.columns,
            self._ds.rows,
            1,
            dtype,
            self._ds.geotransform,
            self._ds.crs,
            no_data_value,
        )
        dst_obj.raster.GetRasterBand(1).WriteArray(new_array)

        if inplace:
            self._ds._update_inplace(dst_obj.raster)
            return None
        return dst_obj

    def fill(
        self, value: float | int, inplace: bool = False, path: str | Path | None = None
    ) -> Dataset:
        """Fill the domain cells with a certain value.

            Fill takes a raster and fills it with one value

        Args:
            value (float | int):
                Numeric value to fill.
            inplace (bool):
                If True, the original dataset will be modified. If False, a new dataset will be created. Default is False.
            path (str):
                Path including the extension (.tif).

        Returns:
            Dataset:
                A new Dataset with cells filled. If inplace is True, returns self.

        Examples:
            - Create a Dataset with 1 band, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 5, size=(5, 5))
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.read_array()) # doctest: +SKIP
              [[1 1 3 1 2]
               [2 2 2 1 2]
               [2 2 3 1 3]
               [3 4 3 3 4]
               [4 4 2 1 1]]
              >>> new_dataset = dataset.fill(10)
              >>> print(new_dataset.read_array())
              [[10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]
               [10 10 10 10 10]]

              ```
        """
        no_data_value = self._ds.no_data_value[0]
        src_array = self._ds.raster.ReadAsArray()

        # rtol=1e-6 is intentionally tighter than the package default
        # (1e-3): ``fill`` writes user-supplied values into every domain
        # cell, so a too-loose match would clobber legitimate cells that
        # happen to lie within ~0.1% of the no-data sentinel.
        src_array[inside_domain(src_array, no_data_value, rtol=0.000001)] = value

        dst = self._ds.__class__.dataset_like(self._ds, src_array, path=path)
        if inplace:
            self._ds._update_inplace(dst.raster)
            return None
        return dst

    def extract(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        mask: FeatureCollection | GeoDataFrame | None = None,
    ) -> np.ndarray:
        """Extract.

        - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
        - If the mask parameter is given, the raster will be clipped to the extent of the given mask and the
          values within the mask are extracted.

        Args:
            band (int, optional):
                Band index. Default is None.
            exclude_value (Numeric, optional):
                Values to exclude from extracted values. If the dataset is multi-band, the values in `exclude_value`
                will be filtered out from the first band only.
            mask (FeatureCollection | GeoDataFrame, optional):
                Vector data containing point geometries at which to extract the values. Default is None.

        Returns:
            np.ndarray:
                The extracted values from each band in the dataset will be in one row in the returned array.

        Examples:
            - Extract all values from the dataset:

              - First, create a dataset with 2 bands, 4 rows and 4 columns:

                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(1, 5, size=(2, 4, 4))
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
                >>> print(dataset)
                <BLANKLINE>
                            Cell size: 0.05
                            Dimension: 4 * 4
                            EPSG: 4326
                            Number of Bands: 2
                            Band names: ['Band_1', 'Band_2']
                            Mask: -9999.0
                            Data type: int32
                            File:...
                <BLANKLINE>
                >>> print(dataset.read_array()) # doctest: +SKIP
                [[[1 3 3 4]
                  [1 4 2 4]
                  [2 4 2 1]
                  [1 3 2 3]]
                 [[3 2 1 3]
                  [4 3 2 2]
                  [2 2 3 4]
                  [1 4 1 4]]]

                ```

              - Now, extract the values in the dataset:

                ```python
                >>> values = dataset.extract()
                >>> print(values) # doctest: +SKIP
                [[1 3 3 4 1 4 2 4 2 4 2 1 1 3 2 3]
                 [3 2 1 3 4 3 2 2 2 2 3 4 1 4 1 4]]

                ```

              - Extract all the values except 2:

                ```python
                >>> values = dataset.extract(exclude_value=2)
                >>> print(values) # doctest: +SKIP

                ```

            - Extract values at the location of the given point geometries:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import Point
              ```

              - Create the points using shapely and GeoPandas to cover the 4 cells with xmin, ymin, xmax, ymax = [0.1, -0.2, 0.2, -0.1]:

                ```python
                >>> points = gpd.GeoDataFrame(geometry=[Point(0.1, -0.1), Point(0.1, -0.2), Point(0.2, -0.2), Point(0.2, -0.1)],crs=4326)
                >>> values = dataset.extract(mask=points)
                >>> print(values) # doctest: +SKIP
                [[4 3 3 4]
                 [3 4 4 2]]

                ```
        """
        # Optimize: make the read_array return only the array for inside the mask feature, and not to read the whole
        #  raster
        arr = self._ds.read_array(band=band)
        no_data_value = (
            self._ds.no_data_value[0] if self._ds.no_data_value[0] is not None else np.nan
        )
        if mask is None:
            exclude_list = (
                [no_data_value, exclude_value]
                if exclude_value is not None
                else [no_data_value]
            )
            values = get_pixels2(arr, exclude_list)
        else:
            indices = self._ds.map_to_array_coordinates(mask)
            if arr.ndim > 2:
                values = arr[:, indices[:, 0], indices[:, 1]]
            else:
                values = arr[indices[:, 0], indices[:, 1]]

        return np.asarray(values)

    def overlay(
        self: Dataset,
        classes_map,
        band: int = 0,
        exclude_value: float | int | None = None,
    ) -> dict[list[float], list[float]]:
        """Overlay.

        Overlay method extracts all the values in the dataset for each class in the given class map.

        Args:
            classes_map (Dataset):
                Dataset object for the raster that has classes you want to overlay with the raster.
            band (int):
                If the raster is multi-band, choose the band you want to overlay with the classes map. Default is 0.
            exclude_value (Numeric, optional):
                Values you want to exclude from extracted values. Default is None.

        Returns:
            Dict:
                Dictionary with class values as keys (from the class map), and for each key a list of all the intersected
                values in the base map.

        Examples:
            - Read the dataset:

              ```python
              >>> dataset = Dataset.read_file("examples/data/geotiff/raster-folder/MSWEP_1979.01.01.tif")
              >>> dataset.plot(figsize=(6, 8)) # doctest: +SKIP

              ```

              ![rhine-rainfall](./../../_images/dataset/rhine-rainfall.png)

            - Read the classes dataset:

              ```python
              >>> classes = Dataset.read_file("examples/data/geotiff/rhine-classes.tif")
              >>> classes.plot(figsize=(6, 8), color_scale=4, bounds=[1,2,3,4,5,6]) # doctest: +SKIP

              ```

              ![rhine-classes](./../../_images/dataset/rhine-classes.png)

            - Overlay the dataset with the classes dataset:

              ```python
              >>> classes_dict = dataset.overlay(classes)
              >>> print(classes_dict.keys()) # doctest: +SKIP
              dict_keys([1, 2, 3, 4, 5])

              ```

            - You can use the key `1` to get the values that overlay class 1.
        """
        if not self._ds.spatial._check_alignment(classes_map):
            raise AlignmentError(
                "The class Dataset is not aligned with the current raster, please use the method "
                "'align' to align both rasters."
            )
        arr = self._ds.read_array(band=band)
        no_data_value = (
            self._ds.no_data_value[0] if self._ds.no_data_value[0] is not None else np.nan
        )
        mask = (
            [no_data_value, exclude_value]
            if exclude_value is not None
            else [no_data_value]
        )
        ind = get_indices2(arr, mask)
        classes = classes_map.read_array()
        values: dict[Any, list[Any]] = dict()

        # extract values
        for i, ind_i in enumerate(ind):
            # first check if the sub-basin has a list in the dict if not create a list
            key = classes[ind_i[0], ind_i[1]]
            if key not in list(values.keys()):
                values[key] = list()

            values[key].append(arr[ind_i[0], ind_i[1]])

        return values

    def get_mask(self, band: int = 0) -> np.ndarray:
        """Get the mask array.

        Args:
            band (int):
                Band index. Default is 0.

        Returns:
            np.ndarray:
                Array of the mask. 0 value for cells out of the domain, and 255 for cells in the domain.
        """
        # TODO: there is a CreateMaskBand method in the gdal.Dataset class, it creates a mask band for the dataset
        #   either internally or externally.
        arr = np.asarray(self._ds._iloc(band).GetMaskBand().ReadAsArray())
        return arr

    def footprint(
        self: Dataset,
        band: int = 0,
        exclude_values: list[Any] | None = None,
    ) -> GeoDataFrame | None:
        """Extract the real coverage of the values in a certain band.

        Args:
            band (int):
                Band index. Default is 0.
            exclude_values (List[Any] | None):
                If you want to exclude a certain value in the raster with another value inter the two values as a
                list of tuples a [(value_to_be_exclude_valuesd, new_value)].

                - Example of exclude_values usage:

                  ```python
                  >>> exclude_values = [0]

                  ```

                - This parameter is introduced particularly in the case of rasters that has the no_data_value stored in
                  the `no_data_value` property does not match the value stored in the band, so this option can correct
                  this behavior.

        Returns:
            GeoDataFrame:
                - geodataframe containing the polygon representing the extent of the raster. the extent column should
                  contain a value of 2 only.
                - if the dataset had separate polygons, each polygon will be in a separate row.

        Examples:
            - The following raster dataset has flood depth stored in its values, and the non-flooded cells are filled with
              zero, so to extract the flood extent, we need to exclude the zero flood depth cells.

              ```python
              >>> dataset = Dataset.read_file("examples/data/geotiff/rhine-flood.tif")
              >>> dataset.plot()
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            ![dataset-footprint-rhine-flood](./../../_images/dataset/dataset-footprint-rhine-flood.png)

            - Now, to extract the footprint of the dataset band, we need to specify the `exclude_values` parameter with the
              value of the non-flooded cells.

              ```python
              >>> extent = dataset.footprint(band=0, exclude_values=[0])
              >>> print(extent)
                 Band_1                                           geometry
              0     2.0  POLYGON ((4070974.182 3181069.473, 4070974.182...
              1     2.0  POLYGON ((4077674.182 3181169.473, 4077674.182...
              2     2.0  POLYGON ((4091174.182 3169169.473, 4091174.182...
              3     2.0  POLYGON ((4088574.182 3176269.473, 4088574.182...
              4     2.0  POLYGON ((4082974.182 3167869.473, 4082974.182...
              5     2.0  POLYGON ((4092274.182 3168269.473, 4092274.182...
              6     2.0  POLYGON ((4072474.182 3181169.473, 4072474.182...

              >>> extent.plot()
              <Axes: >

              ```

            ![dataset-footprint-rhine-flood-extent](./../../_images/dataset/dataset-footprint-rhine-flood-extent.png)

        """
        arr = self._ds.read_array(band=band)
        no_data_val = self._ds.no_data_value[band]

        if no_data_val is None:
            if not (np.isnan(arr)).any():
                self._ds.logger.warning(
                    "The nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the no_data_value stored in the raster is"
                    " not correct"
                )
        else:
            if not (np.isclose(arr, no_data_val, rtol=0.00001)).any():
                self._ds.logger.warning(
                    "the nodata value stored in the raster does not exist in the raster "
                    "so either the raster extent is all full of data, or the no_data_value stored in the raster is"
                    " not correct"
                )
        # if you want to exclude_values any value in the raster
        if exclude_values:
            for val in exclude_values:
                try:
                    # in case the val2 is None, and the array is int type, the following line will give error as None
                    # is considered as float
                    arr[np.isclose(arr, val)] = no_data_val
                except TypeError:
                    arr = arr.astype(np.float32)
                    arr[np.isclose(arr, val)] = no_data_val

        # replace all the values with 2
        if no_data_val is None:
            # check if the whole raster is full of no_data_value
            if (np.isnan(arr)).all():
                self._ds.logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isnan(arr)] = 2
        else:
            # check if the whole raster is full of no_data_value
            if (np.isclose(arr, no_data_val, rtol=0.00001)).all():
                self._ds.logger.warning("the raster is full of no_data_value")
                return None

            arr[~np.isclose(arr, no_data_val, rtol=0.00001)] = 2
        new_dataset = self._ds.create_from_array(
            arr, geo=self._ds.geotransform, epsg=self._ds.epsg, no_data_value=self._ds.no_data_value
        )
        # then convert the raster into polygon
        gdf = new_dataset.cluster2(band=band)
        gdf.rename(columns={"Band_1": self._ds.band_names[band]}, inplace=True)

        return gdf

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        """Normalize numpy arrays into scale 0.0-1.0.

        Args:
            array (np.ndarray): Numpy array to normalize.

        Returns:
            np.ndarray: Normalized array.
        """
        array_min = array.min()
        array_max = array.max()
        val = (array - array_min) / (array_max - array_min)
        return np.asarray(val)

    @staticmethod
    def _rescale(array: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
        val = (array - min_value) / (max_value - min_value)
        return val

    def get_histogram(
        self: Dataset,
        band: int = 0,
        bins: int = 6,
        min_value: float | None = None,
        max_value: float | None = None,
        include_out_of_range: bool = False,
        approx_ok: bool = False,
    ) -> tuple[list, list[tuple[Any, Any]]]:
        """Get histogram.

        Args:
            band (int, optional):
                Band index. Default is 1.
            bins (int, optional):
                Number of bins. Default is 6.
            min_value (float, optional):
                Minimum value. Default is None.
            max_value (float, optional):
                Maximum value. Default is None.
            include_out_of_range (bool, optional):
                If True, add out-of-range values into the first and last buckets. Default is False.
            approx_ok (bool, optional):
                If True, compute an approximate histogram by using subsampling or overviews. Default is False.

        Returns:
            tuple[list, list[tuple[Any, Any]]]:
                Histogram values and bin edges.

        Hint:
            - The value of the histogram will be stored in an xml file by the name of the raster file with the extension
                of .aux.xml.

            - The content of the file will be like the following:
              ```xml

                  <PAMDataset>
                    <PAMRasterBand band="1">
                      <Description>Band_1</Description>
                      <Histograms>
                        <HistItem>
                          <HistMin>0</HistMin>
                          <HistMax>88</HistMax>
                          <BucketCount>6</BucketCount>
                          <IncludeOutOfRange>0</IncludeOutOfRange>
                          <Approximate>0</Approximate>
                          <HistCounts>75|6|0|4|2|1</HistCounts>
                        </HistItem>
                      </Histograms>
                    </PAMRasterBand>
                  </PAMDataset>

              ```

        Examples:
            - Create `Dataset` consists of 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0).

              ```python
              >>> import numpy as np
              >>> arr = np.random.randint(1, 12, size=(10, 10))
              >>> print(arr)    # doctest: +SKIP
              [[ 4  1  1  2  6  9  2  5  1  8]
               [ 1 11  5  6  2  5  4  6  6  7]
               [ 5  2 10  4  8 11  4 11 11  1]
               [ 2  3  6  3  1  5 11 10 10  7]
               [ 8  2 11  3  1  3  5  4 10 10]
               [ 1  2  1  6 10  3  6  4  2  8]
               [ 9  5  7  9  7  8  1 11  4  4]
               [ 7  7  2  2  5  3  7  2  9  9]
               [ 2 10  3  2  1 11  5  9  8 11]
               [ 1  5  6 11  3  3  8  1  2  1]]
               >>> top_left_corner = (0, 0)
               >>> cell_size = 0.05
               >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)

               ```

            - Now, let's get the histogram of the first band using the `get_histogram` method with the default
                parameters:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0)
                >>> print(hist)  # doctest: +SKIP
                [28, 17, 10, 15, 13, 7]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 2.67), (2.67, 4.34), (4.34, 6.0), (6.0, 7.67), (7.67, 9.34), (9.34, 11.0)]

                ```
            - we can also exclude values from the histogram by using the `min_value` and `max_value`:
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, min_value=5, max_value=10)
                >>> print(hist)  # doctest: +SKIP
                [10, 8, 7, 7, 6, 0]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 1.835), (1.835, 2.67), (2.67, 3.5), (3.5, 4.34), (4.34, 5.167), (5.167, 6.0)]

                ```
            - For datasets with big dimensions, computing the histogram can take some time; approximating the computation
                of the histogram can save a lot of computation time. When using the parameter `approx_ok` with a `True`
                value the histogram will be calculated from resampling the band or from the overviews if they exist.
                ```python
                >>> hist, ranges = dataset.get_histogram(band=0, approx_ok=True)
                >>> print(hist)  # doctest: +SKIP
                [28, 17, 10, 15, 13, 7]
                >>> print(ranges)   # doctest: +SKIP
                [(1.0, 2.67), (2.67, 4.34), (4.34, 6.0), (6.0, 7.67), (7.67, 9.34), (9.34, 11.0)]

                ```
            - As you see for small datasets, the approximation of the histogram will be the same as without approximation.

        """
        band_obj = self._ds._iloc(band)
        min_val, max_val = band_obj.ComputeRasterMinMax()
        if min_value is None:
            min_value = min_val
        if max_value is None:
            max_value = max_val

        bin_width = (max_value - min_value) / bins
        ranges = [
            (min_val + i * bin_width, min_val + (i + 1) * bin_width)
            for i in range(bins)
        ]

        hist = band_obj.GetHistogram(
            min=min_value,
            max=max_value,
            buckets=bins,
            include_out_of_range=include_out_of_range,
            approx_ok=approx_ok,
        )
        return hist, ranges

    def plot(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        percentile: int | None = None,
        basemap: bool | str | None = None,
        **kwargs: Any,
    ) -> ArrayGlyph:
        """Plot the values/overviews of a given band.
        The plot function uses the `cleopatra` as a backend to plot the raster data, for more information check
        [ArrayGlyph](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/#cleopatra.array_glyph.ArrayGlyph.plot).
        Args:
            band (int, optional):
                The band you want to get its data. Default is 0.
            exclude_value (Any, optional):
                Value to exclude from the plot. Default is None.
            rgb (List[int], optional):
                The indices of the red, green, and blue bands in the `Dataset`. the `rgb` parameter can be a list of
                three values, or a list of four values if the alpha band is also included.
                The `plot` method will check if the rgb bands are defined in the `Dataset`, if all the three bands (
                red, green, blue)) are defined, the method will use them to plot the real image, if not the rgb bands
                will be considered as [2,1,0] as the default order for sentinel tif files.
            surface_reflectance (int, optional):
                Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff (List, optional):
                clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
                and scale them back to between 0 and 1). Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            percentile: int
                The percentile value to be used for scaling.
            basemap (bool or str, optional):
                If True, add an OpenStreetMap basemap underneath the plot. If a string, use it as
                the tile provider name (e.g. "CartoDB.Positron"). Default is None (no basemap).
                Requires the [viz] extra (mercantile, xyzservices, Pillow).
        kwargs:
                | Parameter                   | Type                | Description |
                |-----------------------------|---------------------|-------------|
                | `points`                    | array               | 3 column array with the first column as the value to display for the point, the second as the row index, and the third as the column index in the array. The second and third columns tell the location of the point. |
                | `point_color`               | str                 | Color of the point. |
                | `point_size`                | Any                 | Size of the point. |
                | `pid_color`                 | str                 | Color of the annotation of the point. Default is blue. |
                | `pid_size`                  | Any                 | Size of the point annotation. |
                | `figsize`                   | tuple, optional     | Figure size. Default is `(8, 8)`. |
                | `title`                     | str, optional       | Title of the plot. Default is `'Total Discharge'`. |
                | `title_size`                | int, optional       | Title size. Default is `15`. |
                | `orientation`               | str, optional       | Orientation of the color bar (`horizontal` or `vertical`). Default is `'vertical'`. |
                | `rotation`                  | number, optional    | Rotation of the color bar label. Default is `-90`. |
                | `cbar_length`               | float, optional     | Ratio to control the height of the color bar. Default is `0.75`. |
                | `ticks_spacing`             | int, optional       | Spacing between color bar ticks. Default is `2`. |
                | `cbar_label_size`           | int, optional       | Size of the color bar label. Default is `12`. |
                | `cbar_label`                | str, optional       | Label of the color bar. Default is `'Discharge m\u00b3/s'`. |
                | `color_scale`               | int, optional       | Scale mode for colors. Options: 1 = normal, 2 = power, 3 = SymLogNorm, 4 = PowerNorm, 5 = BoundaryNorm. Default is `1`. |
                | `gamma`                     | float, optional     | Value needed for color scale option 2. Default is `1/2`. |
                | `line_threshold`            | float, optional     | Value needed for color scale option 3. Default is `0.0001`. |
                | `line_scale`                | float, optional     | Value needed for color scale option 3. Default is `0.001`. |
                | `bounds`                    | list, optional      | Discrete bounds for color scale option 4. Default is `None`. |
                | `midpoint`                  | float, optional     | Value needed for color scale option 5. Default is `0`. |
                | `cmap`                      | str, optional       | Color map style. Default is `'coolwarm_r'`. |
                | `display_cell_value`        | bool, optional      | Whether to display cell values as text. |
                | `num_size`                  | int, optional       | Size of numbers plotted on top of each cell. Default is `8`. |
                | `background_color_threshold`| float or int, optional | Threshold for deciding text color over cells: if value > threshold -> black text; else white text. If `None`, max value / 2 is used. Default is `None`. |
        Returns:
            ArrayGlyph:
                ArrayGlyph object. For more details of the ArrayGlyph object check the [ArrayGlyph](https://serapeum-org.github.io/cleopatra/latest/api/array-glyph-class/).
        Examples:
            - Plot a certain band:
              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326)
              >>> dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using power scale.
              ```python
              >>> dataset.plot(band=0, color_scale="power")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using SymLogNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="sym-lognorm")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using PowerNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="boundary-norm", bounds=[0, 0.2, 0.4, 0.6, 0.8, 1])
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
            - plot using BoundaryNorm scale.
              ```python
              >>> dataset.plot(band=0, color_scale="midpoint")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)
              ```
        """
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/serapeum-org/cleopatra"
        )
        from cleopatra.array_glyph import ArrayGlyph

        no_data_value = [np.nan if i is None else i for i in self._ds.no_data_value]
        if overview:
            arr = self._ds.read_overview_array(
                band=band,
                overview_index=overview_index if overview_index is not None else 0,
            )
        else:
            arr = self._ds.read_array(band=band)
        # if the raster has three bands or more.
        if self._ds.band_count >= 3:
            if band is None:
                if rgb is None:
                    rgb_candidate: list[int | None] = [
                        self._ds.get_band_by_color("red"),
                        self._ds.get_band_by_color("green"),
                        self._ds.get_band_by_color("blue"),
                    ]
                    if None in rgb_candidate:
                        rgb = [2, 1, 0]
                    else:
                        rgb = [int(v) for v in rgb_candidate if v is not None]
                # first make the band index the first band in the rgb list (red band)
                band = rgb[0]
        # elif self._ds.band_count == 1:
        #     band = 0
        else:
            if band is None:
                band = 0
        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )
        ax = kwargs.pop("ax", None)
        fig = kwargs.pop("fig", None)
        cleo = ArrayGlyph(
            arr,
            exclude_value=exclude_value,
            extent=self._ds.bbox,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            percentile=percentile,
            ax=ax,
            fig=fig,
            **kwargs,
        )
        cleo.plot(**kwargs)

        if basemap:
            if self._ds.epsg is None:
                raise ValueError("Dataset must have a CRS (epsg) to use basemap.")
            from pyramids.basemap.basemap import add_basemap

            source = basemap if isinstance(basemap, str) else None
            add_basemap(cleo.ax, crs=self._ds.epsg, source=source)

        return cleo

    @staticmethod
    def _process_color_table(color_table: DataFrame) -> DataFrame:
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info"
            " check https://github.com/serapeum-org/cleopatra"
        )
        from cleopatra.colors import Colors

        # if the color_table does not contain the red, green, and blue columns, assume it has one column with
        # the color as hex and then, convert the color to rgb.
        if all(elem in color_table.columns for elem in ["red", "green", "blue"]):
            color_df = color_table.loc[:, ["values", "red", "green", "blue"]]
        elif "color" in color_table.columns:
            color = Colors(color_table["color"].tolist())
            color_rgb = color.to_rgb(normalized=False)
            color_df = DataFrame(columns=["values"])
            color_df["values"] = color_table["values"].to_list()
            color_df.loc[:, ["red", "green", "blue"]] = color_rgb
        else:
            raise ValueError(
                f"color_table must contain either red, green, blue, or color columns. given columns are: "
                f"{color_table.columns}"
            )
        if "alpha" not in color_table.columns:
            color_df.loc[:, "alpha"] = 255
        else:
            color_df.loc[:, "alpha"] = color_table["alpha"]
        return color_df


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
    """Mixin providing vectorization, clustering, and translate methods for Dataset."""

    def _band_to_polygon(self, band: int, col_name: str) -> GeoDataFrame:
        gdal_band = self._ds.raster.GetRasterBand(band + 1)
        srs = sr_from_wkt(self._ds.crs)

        # Build the OGR DataSource directly â€” FeatureCollection.create_ds
        # was deleted because it exposed ogr.DataSource on the public API.
        # Here the DataSource is purely local scratch space for gdal.Polygonize.
        dst_ds = ogr.GetDriverByName("Memory").CreateDataSource("memData")
        if dst_ds is None:
            raise RuntimeError("Failed to create in-memory OGR DataSource")
        dst_layer = dst_ds.CreateLayer(col_name, srs=srs)
        dtype = gdal_to_ogr_dtype(self._ds.raster)
        new_field = ogr.FieldDefn(col_name, dtype)
        dst_layer.CreateField(new_field)
        gdal.Polygonize(gdal_band, gdal_band, dst_layer, 0, [], callback=None)

        return _feature_ogr.datasource_to_gdf(dst_ds)

    def to_feature_collection(
        self,
        mask: GeoDataFrame | None = None,
        add_geometry: str | None = None,
        tile: bool = False,
        tile_size: int = 256,
        touch: bool = True,
    ) -> DataFrame | GeoDataFrame:
        """Convert a dataset to a vector.

        The function does the following:
            - Flatten the array in each band in the raster then mask the values if a mask is given
                otherwise it will flatten all values.
            - Put the values for each band in a column in a dataframe under the name of the raster band,
                but if no meta-data in the raster band exists, an index number will be used [1, 2, 3, ...]
            - The function has an add_geometry parameter with two possible values ["point", "polygon"], which you can
                specify the type of shapely geometry you want to create from each cell,

                - If point is chosen, the created point will be at the center of each cell
                - If a polygon is chosen, a square polygon will be created that covers the entire cell.

        Args:
            mask (GeoDataFrame, optional):
                GeoDataFrame to clip the raster. If given, the raster will be cropped to the mask extent.
            add_geometry (str):
                "Polygon" or "Point" if you want to add a polygon geometry of the cells as column in dataframe.
                Default is None.
            tile (bool):
                True to use tiles in extracting the values from the raster. Default is False.
            tile_size (int):
                Tile size. Default is 1500.
            touch (bool):
                Include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
                Default is True.

        Returns:
            DataFrame | GeoDataFrame:
                The resulting frame will have the band value under the name of the band (if the raster file has
                metadata; if not, the bands will be indexed from 1 to the number of bands).

        Examples:
            - Create a dataset from array with 2 bands and 3*3 array each:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(2, 3, 3)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.read_array(band=0)) # doctest: +SKIP
              [[0.88625832 0.81804328 0.99372706]
               [0.85333054 0.35448201 0.78079262]
               [0.43887136 0.68166208 0.53170966]]
              >>> print(dataset.read_array(band=1)) # doctest: +SKIP
              [[0.07051872 0.67650833 0.17625027]
               [0.41258071 0.38327938 0.18783139]
               [0.83741314 0.70446373 0.64913575]]

              ```

            - Convert the dataset to dataframe by calling the `to_feature_collection` method:

              ```python
              >>> df = dataset.to_feature_collection()
              >>> print(df) # doctest: +SKIP
                   Band_1    Band_2
              0  0.886258  0.070519
              1  0.818043  0.676508
              2  0.993727  0.176250
              3  0.853331  0.412581
              4  0.354482  0.383279
              5  0.780793  0.187831
              6  0.438871  0.837413
              7  0.681662  0.704464
              8  0.531710  0.649136

              ```

            - Convert the dataset into geodataframe with either a polygon or a point geometry that represents each cell.
                To specify the geometry type use the parameter `add_geometry`:

                  ```python
                  >>> gdf = dataset.to_feature_collection(add_geometry="point")
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2                  geometry
                  0  0.886258  0.070519  POINT (0.02500 -0.02500)
                  1  0.818043  0.676508  POINT (0.07500 -0.02500)
                  2  0.993727  0.176250  POINT (0.12500 -0.02500)
                  3  0.853331  0.412581  POINT (0.02500 -0.07500)
                  4  0.354482  0.383279  POINT (0.07500 -0.07500)
                  5  0.780793  0.187831  POINT (0.12500 -0.07500)
                  6  0.438871  0.837413  POINT (0.02500 -0.12500)
                  7  0.681662  0.704464  POINT (0.07500 -0.12500)
                  8  0.531710  0.649136  POINT (0.12500 -0.12500)
                  >>> gdf = dataset.to_feature_collection(add_geometry="polygon")
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2                                           geometry
                  0  0.886258  0.070519  POLYGON ((0.00000 0.00000, 0.05000 0.00000, 0....
                  1  0.818043  0.676508  POLYGON ((0.05000 0.00000, 0.10000 0.00000, 0....
                  2  0.993727  0.176250  POLYGON ((0.10000 0.00000, 0.15000 0.00000, 0....
                  3  0.853331  0.412581  POLYGON ((0.00000 -0.05000, 0.05000 -0.05000, ...
                  4  0.354482  0.383279  POLYGON ((0.05000 -0.05000, 0.10000 -0.05000, ...
                  5  0.780793  0.187831  POLYGON ((0.10000 -0.05000, 0.15000 -0.05000, ...
                  6  0.438871  0.837413  POLYGON ((0.00000 -0.10000, 0.05000 -0.10000, ...
                  7  0.681662  0.704464  POLYGON ((0.05000 -0.10000, 0.10000 -0.10000, ...
                  8  0.531710  0.649136  POLYGON ((0.10000 -0.10000, 0.15000 -0.10000, ...

                  ```

            - Use a mask to crop part of the dataset, and then convert the cropped part to a dataframe/geodataframe:

              - Create a mask that covers only the cell in the middle of the dataset.

                  ```python
                  >>> import geopandas as gpd
                  >>> from shapely.geometry import Polygon
                  >>> poly = gpd.GeoDataFrame(
                  ...             geometry=[Polygon([(0.05, -0.05), (0.05, -0.1), (0.1, -0.1), (0.1, -0.05)])], crs=4326
                  ... )
                  >>> df = dataset.to_feature_collection(mask=poly)
                  >>> print(df) # doctest: +SKIP
                       Band_1    Band_2
                  0  0.354482  0.383279

                  ```

            - If you have a big dataset, and you want to convert it to dataframe in tiles (do not read the whole dataset
                at once but in tiles), you can use the `tile` and the `tile_size` parameters. The values will be the
                same as above; the difference is reading in chunks:

                  ```python
                  >>> gdf = dataset.to_feature_collection(tile=True, tile_size=1)
                  >>> print(gdf) # doctest: +SKIP
                       Band_1    Band_2
                  0  0.886258  0.070519
                  1  0.818043  0.676508
                  2  0.993727  0.176250
                  3  0.853331  0.412581
                  4  0.354482  0.383279
                  5  0.780793  0.187831
                  6  0.438871  0.837413
                  7  0.681662  0.704464
                  8  0.531710  0.649136

                  ```

        """
        band_names = self._ds.band_names

        if mask is not None:
            src_ds = self._ds.crop(mask=mask, touch=touch)
        else:
            src_ds = self._ds

        if tile:
            df = self._extract_values_tiled(band_names, tile_size)
        else:
            df = src_ds.vectorize._extract_values_full(band_names)

        df.drop(columns=["burn_value", "geometry"], errors="ignore", inplace=True)

        if add_geometry:
            df = self._attach_geometry(src_ds, df, add_geometry)

        return df

    def _extract_values_tiled(self, band_names: list, tile_size: int) -> pd.DataFrame:
        """Extract raster band values into a DataFrame using tiles.

        Args:
            band_names (list): Band names for the DataFrame columns.
            tile_size (int): Tile size in pixels.

        Returns:
            pd.DataFrame: Concatenated DataFrame from all tiles.
        """
        no_data_value = self._ds.no_data_value[0]
        df_list = []
        for arr in self._ds.get_tile(tile_size):
            idx = (1, 2) if arr.ndim > 2 else (0, 1)
            mask_arr = np.ones((arr.shape[idx[0]], arr.shape[idx[1]]))
            pixels = get_pixels(arr, mask_arr).transpose()
            df = pd.DataFrame(pixels, columns=band_names)
            if no_data_value is not None:
                df.replace(no_data_value, np.nan, inplace=True)
            df.dropna(axis=0, inplace=True, ignore_index=True)
            if not df.empty:
                df_list.append(df)

        if not df_list:
            return pd.DataFrame(columns=band_names)

        return pd.concat(df_list, ignore_index=True)

    def _extract_values_full(self, band_names: list) -> pd.DataFrame:
        """Extract all raster band values into a DataFrame (no tiling).

        Args:
            band_names (list): Band names for the DataFrame columns.

        Returns:
            pd.DataFrame: DataFrame with one column per band, no-data rows removed.
        """
        arr = self._ds.read_array()

        if self._ds.band_count == 1:
            pixels = arr.flatten()
        else:
            pixels = (
                arr.flatten()
                .reshape(self._ds.band_count, self._ds.columns * self._ds.rows)
                .transpose()
            )
        df = pd.DataFrame(pixels, columns=band_names)
        if self._ds.no_data_value[0] is not None:
            df.replace(self._ds.no_data_value[0], np.nan, inplace=True)
        df.dropna(axis=0, inplace=True, ignore_index=True)
        return df

    @staticmethod
    def _attach_geometry(src, df: pd.DataFrame, geometry_type: str) -> gpd.GeoDataFrame:
        """Attach point or polygon geometry to a DataFrame.

        Args:
            src: The dataset to derive cell geometries from.
            df (pd.DataFrame): DataFrame with band values.
            geometry_type (str): "point" or "polygon".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with geometry column.
        """
        if geometry_type.lower() == "point":
            coords = src.get_cell_points(domain_only=True)
        else:
            coords = src.get_cell_polygons(domain_only=True)

        gdf = gpd.GeoDataFrame(df.loc[:], geometry=coords["geometry"].to_list())
        gdf = gdf.set_crs(coords.crs.to_epsg())
        return gdf

    def translate(self, path: str | Path | None = None, **kwargs) -> Dataset:
        """Translate.

        The translate function can be used to
        - Convert Between Formats: Convert a raster from one format to another (e.g., from GeoTIFF to JPEG).
        - Subset: Extract a subregion of a raster.
        - Resample: Change the resolution of a raster.
        - Reproject: Change the coordinate reference system of a raster.
        - Scale Values: Scale pixel values to a new range.
        - Change Data Type: Convert the data type of the raster.
        - Apply Compression: Apply compression to the output raster.
        - Apply No-Data Values: Define no-data values for the output raster.


        Parameters
        ----------
        path: str, optional, default is None.
            path to save the output, if None, the output will be saved in memory.
        kwargs:
            unscale:
                unscale values with scale and offset metadata.
            scaleParams:
                list of scale parameters, each of the form [src_min,src_max] or [src_min,src_max,dst_min,dst_max]
            outputType:
                output type (gdalconst.GDT_Byte, etc...)
            exponents:
                list of exponentiation parameters
            bandList:
                array of band numbers (index start at 1)
            maskBand:
                mask band to generate or not ("none", "auto", "mask", 1, ...)
            creationOptions:
                list or dict of creation options
            srcWin:
                subwindow in pixels to extract: [left_x, top_y, width, height]
            projWin:
                subwindow in projected coordinates to extract: [ulx, uly, lrx, lry]
            projWinSRS:
                SRS in which projWin is expressed
            outputBounds:
                assigned output bounds: [ulx, uly, lrx, lry]
            outputGeotransform:
                assigned geotransform matrix (array of 6 values) (mutually exclusive with outputBounds)
            metadataOptions:
                list or dict of metadata options
            outputSRS:
                assigned output SRS
            noData:
                nodata value (or "none" to unset it)
            rgbExpand:
                Color palette expansion mode: "gray", "rgb", "rgba"
            xmp:
                whether to copy XMP metadata
            resampleAlg:
                resampling mode
            overviewLevel:
                To specify which overview level of source files must be used
            domainMetadataOptions:
                list or dict of domain-specific metadata options

        Returns
        -------
        Dataset

        Examples
        --------
        Scale & offset:
            - the translate function can be used to get rid of the scale and offset that are used to manipulate the
            dataset, to get the real values of the dataset.

            Scale:
                - First we will create a dataset from a float32 array with values between 1 and 10, and then we will
                    assign a scale of 0.1 to the dataset.

                    >>> import numpy as np
                    >>> arr = np.random.randint(1, 10, size=(5, 5)).astype(np.float32)
                    >>> print(arr) # doctest: +SKIP
                    [[5. 5. 3. 4. 2.]
                     [2. 5. 5. 8. 5.]
                     [7. 5. 6. 1. 2.]
                     [6. 8. 1. 5. 8.]
                     [2. 5. 2. 2. 9.]]
                    >>> top_left_corner = (0, 0)
                    >>> cell_size = 0.05
                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )
                    >>> print(dataset)
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File: ...
                    <BLANKLINE>
                    >>> dataset.scale = [0.1]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset) # doctest: +SKIP
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File:
                    <BLANKLINE>
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[0.5 0.5 0.3 0.4 0.2]
                     [0.2 0.5 0.5 0.8 0.5]
                     [0.7 0.5 0.6 0.1 0.2]
                     [0.6 0.8 0.1 0.5 0.8]
                     [0.2 0.5 0.2 0.2 0.9]]

            offset:
                - You can also unshift the values of the dataset if the dataset has an offset. To remove the offset
                    from all values in the dataset, you can read the values using the `read_array` and then add the
                    offset value to the array. we will create a dataset from the same array we created above (values
                    are between 1, and 10) with an offset of 100.

                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )
                    >>> print(dataset)
                    <BLANKLINE>
                                Top Left Corner: (0.0, 0.0)
                                Cell size: 0.05
                                Dimension: 5 * 5
                                EPSG: 4326
                                Number of Bands: 1
                                Band names: ['Band_1']
                                Band colors: {0: 'undefined'}
                                Band units: ['']
                                Scale: [1.0]
                                Offset: [0]
                                Mask: -9999.0
                                Data type: float32
                                File: ...
                    <BLANKLINE>

                - set the offset to 100.

                    >>> dataset.offset = [100]

                - check if the offset has been set.

                    >>> print(dataset.offset)
                    [100.0]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[105. 105. 103. 104. 102.]
                     [102. 105. 105. 108. 105.]
                     [107. 105. 106. 101. 102.]
                     [106. 108. 101. 105. 108.]
                     [102. 105. 102. 102. 109.]]

                - as you see, all the values have been shifted by 100. now if you check the offset of the dataset

                    >>> print(unscaled_dataset.offset)
                    [0]

            Offset and Scale together:
                - we can unscale and get rid of the offset at the same time.

                    >>> dataset = Dataset.create_from_array(
                    ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                    ... )

                - set the offset to 100, and a scale of 0.1.

                    >>> dataset.offset = [100]
                    >>> dataset.scale = [0.1]

                - check if the offset has been set.

                    >>> print(dataset.offset)
                    [100.0]
                    >>> print(dataset.scale)
                    [0.1]

                - now lets unscale the dataset values.

                    >>> unscaled_dataset = dataset.translate(unscale=True)
                    >>> print(unscaled_dataset.read_array()) # doctest: +SKIP
                    [[100.5 100.5 100.3 100.4 100.2]
                     [100.2 100.5 100.5 100.8 100.5]
                     [100.7 100.5 100.6 100.1 100.2]
                     [100.6 100.8 100.1 100.5 100.8]
                     [100.2 100.5 100.2 100.2 100.9]]

                - Now you can see that the values were multiplied first by the scale; then the offset value was added.
                    `value * scale + offset`

                    >>> print(unscaled_dataset.offset)
                    [0]
                    >>> print(unscaled_dataset.scale)
                    [1.0]

        Scale between two values:
            - you can scale the values of the dataset between two values, for example, you can scale the values
                between two values 0 and 1.

                >>> dataset = Dataset.create_from_array(
                ...     arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326
                ... )
                >>> print(dataset.stats()) # doctest: +SKIP
                        min  max  mean      std
                Band_1  1.0  9.0   4.0  2.19089
                >>> scaled_dataset = dataset.translate(scaleParams=[[1, 9, 0, 255]], outputType=gdal.GDT_Byte)
                >>> print(scaled_dataset.read_array()) # doctest: +SKIP
                [[128 128  64  96  32]
                 [ 32 128 128 223 128]
                 [191 128 159   0  32]
                 [159 223   0 128 223]
                 [ 32 128  32  32 255]]


        """
        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"

        options = gdal.TranslateOptions(format=driver, **kwargs)
        dst = gdal.Translate(str(path), self._ds.raster, options=options)
        result = self._ds.__class__(dst, access="write")
        return result

    @staticmethod
    def _nearest_neighbour(
        array: np.ndarray, no_data_value: float | int, rows: list, cols: list
    ) -> np.ndarray:
        """Fill specified cells with the value of the nearest neighbor.

            - The _nearest_neighbour method fills the cells with the given indices in rows and cols with the value
                of the nearest neighbor.
            - The raster grid is square, so the 4 perpendicular directions are of the same proximity; the function
                gives priority to the right, left, bottom, and then top, and similarly for 45-degree directions:
                right-bottom, left-bottom, left-top, right-top.

        Args:
            array (np.ndarray):
                Array to fill some of its cells with the nearest value.
            no_data_value (float | int):
                Value stored in cells that are out of the domain.
            rows (list[int]):
                Row indices of the cells you want to fill with the nearest neighbor.
            cols (list[int]):
                Column indices of the cells you want to fill with the nearest neighbor.

        Returns:
            np.ndarray:
                Cells of given indices filled with the value of the nearest neighbor.

        Examples:
            - Basic usage:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> req_rows = [1,3]
              >>> req_cols = [2,4]
              >>> no_data_value = dataset.no_data_value[0]
              >>> new_array = Dataset._nearest_neighbour(arr, no_data_value, req_rows, req_cols)

              ```
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "src should be read using gdal (gdal dataset please read it using gdal library) "
            )
        if not isinstance(rows, list):
            raise TypeError("The `rows` input has to be of type list")
        if not isinstance(cols, list):
            raise TypeError("The `cols` input has to be of type list")

        no_rows = np.shape(array)[0]
        no_cols = np.shape(array)[1]

        for i in range(len(rows)):
            # give the cell the value of the cell that is at the right
            if cols[i] + 1 < no_cols:
                if array[rows[i], cols[i] + 1] != no_data_value:
                    array[rows[i], cols[i]] = array[rows[i], cols[i] + 1]

            elif array[rows[i], cols[i] - 1] != no_data_value and cols[i] - 1 > 0:
                # give the cell the value of the cell that is at the left
                array[rows[i], cols[i]] = array[rows[i], cols[i] - 1]

            elif array[rows[i] - 1, cols[i]] != no_data_value and rows[i] - 1 > 0:
                # give the cell the value of the cell that is at the bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i]]

            elif array[rows[i] + 1, cols[i]] != no_data_value and rows[i] + 1 < no_rows:
                # give the cell the value of the cell that is at the Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i]]

            elif (
                array[rows[i] - 1, cols[i] + 1] != no_data_value
                and rows[i] - 1 > 0
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] + 1]

            elif (
                array[rows[i] - 1, cols[i] - 1] != no_data_value
                and rows[i] - 1 > 0
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left bottom
                array[rows[i], cols[i]] = array[rows[i] - 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] - 1] != no_data_value
                and rows[i] + 1 <= no_rows
                and cols[i] - 1 > 0
            ):
                # give the cell the value of the cell that is at the left Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] - 1]

            elif (
                array[rows[i] + 1, cols[i] + 1] != no_data_value
                and rows[i] + 1 <= no_rows
                and cols[i] + 1 <= no_cols
            ):
                # give the cell the value of the cell that is at the right Top
                array[rows[i], cols[i]] = array[rows[i] + 1, cols[i] + 1]
            else:
                logger.warning("the cell is isolated (No surrounding cells exist)")
        return array

    @staticmethod
    def _group_neighbours(
        array, i, j, lower_bound, upper_bound, position, values, count, cluster
    ) -> None:
        """Group neighboring cells with the same values using iterative BFS.

        Uses a queue-based breadth-first search instead of recursion to avoid
        hitting Python's recursion limit on large connected regions.

        Note: The starting cell (i, j) is enqueued but not marked. When a
        discovered neighbor later checks its own neighbors, it will find (i, j)
        still unmarked and add it to position/values. Therefore the starting
        cell appears in the output whenever it has at least one in-bound
        neighbor. The caller (cluster) handles truly isolated cells separately.
        """
        rows, cols = array.shape
        queue = collections.deque()
        queue.append((i, j))

        while queue:
            ci, cj = queue.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if (
                            cluster[ni, nj] == 0
                            and lower_bound <= array[ni, nj] <= upper_bound
                        ):
                            cluster[ni, nj] = count
                            position.append([ni, nj])
                            values.append(array[ni, nj])
                            queue.append((ni, nj))

    def cluster(
        self, lower_bound: Any, upper_bound: Any
    ) -> tuple[np.ndarray, int, list, list]:
        """Group all the connected values between two bounds.

        Args:
            lower_bound (Number):
                Lower bound of the cluster.
            upper_bound (Number):
                Upper bound of the cluster.

        Returns:
            tuple[np.ndarray, int, list, list]:
                - cluster (np.ndarray):
                    Array with integers representing the cluster number per cell.
                - count (int):
                    Number of clusters in the array.
                - position (list[list[int, int]]):
                    List of [row, col] indices for the position of each value.
                - values (list[Number]):
                    Values stored in each cell in the cluster.

        Examples:
            - First, we will create a dataset with 10 rows and 10 columns.

              ```python
              >>> import numpy as np
              >>> np.random.seed(10)
              >>> arr = np.random.randint(1, 5, size=(5, 5))
              >>> print(arr) # doctest: +SKIP
              [[2 3 3 2 3]
               [3 4 1 1 1]
               [1 3 3 2 2]
               [4 1 1 3 2]
               [2 4 2 3 2]]
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )
              >>> dataset.plot(
              ...     color_scale=4, bounds=[1, 1.9, 4.1, 5], display_cell_value=True, num_size=12,
              ...     background_color_threshold=5
              ... )  # doctest: +SKIP

              ```
              ![cluster](./../../_images/dataset/cluster.png)

            - Now let's cluster the values in the dataset that are between 2 and 4.

              ```python
              >>> lower_value = 2
              >>> upper_value = 4
              >>> cluster_array, count, position, values = dataset.cluster(lower_value, upper_value)

              ```
            - The first returned output is a binary array with 1 indicating that the cell value is inside the
                cluster, and 0 is outside.

              ```python
              >>> print(cluster_array)  # doctest: +SKIP
              [[1. 1. 1. 1. 1.]
               [1. 1. 0. 0. 0.]
               [0. 1. 1. 1. 1.]
               [1. 0. 0. 1. 1.]
               [1. 1. 1. 1. 1.]]

              ```
            - The second returned value is the number of connected clusters.

              ```python
              >>> print(count) # doctest: +SKIP
              2

              ```
            - The third returned value is the indices of the cells that belong to the cluster.

              ```python
              >>> print(position) # doctest: +SKIP
              [[1, 0], [2, 1], [2, 2], [3, 3], [4, 3], [4, 4], [3, 4], [2, 4], [2, 3], [4, 2], [4, 1], [3, 0], [4, 0], [1, 1], [0, 2], [0, 3], [0, 4], [0, 1], [0, 0]]

              ```
            - The fourth returned value is a list of the values that are in the cluster (extracted from these cells).

              ```python
              >>> print(values) # doctest: +SKIP
              [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 2, 4, 3, 2, 3, 3, 2]

              ```

        """
        data = self._ds.read_array()
        position: list[list[int]] = []
        values: list[Any] = []
        count = 1
        cluster = np.zeros(shape=(data.shape[0], data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if lower_bound <= data[i, j] <= upper_bound and cluster[i, j] == 0:
                    self._group_neighbours(
                        data,
                        i,
                        j,
                        lower_bound,
                        upper_bound,
                        position,
                        values,
                        count,
                        cluster,
                    )
                    if cluster[i, j] == 0:
                        position.append([i, j])
                        values.append(data[i, j])
                        cluster[i, j] = count
                    count += 1

        return cluster, count, position, values

    def cluster2(
        self,
        band: int | list[int] | None = None,
    ) -> GeoDataFrame:
        """Cluster the connected equal cells into polygons.

        - Creates vector polygons for all connected regions of pixels in the raster sharing a common
            pixel value (group neighboring cells with the same value into one polygon).

        Args:
            band (int | List[int] | None):
                Band index 0, 1, 2, 3, ...

        Returns:
            GeoDataFrame:
                GeodataFrame containing polygon geomtries for all connected regions.

        Examples:
            - First, we will create a 10*10 dataset full of random integer between 1, and 5.

              ```python
              >>> import numpy as np
              >>> np.random.seed(200)
              >>> arr = np.random.randint(1, 5, size=(10, 10))
              >>> print(arr)  # doctest: +SKIP
              [[3 2 1 1 3 4 1 4 2 3]
               [4 2 2 4 3 3 1 2 4 4]
               [4 2 4 2 3 4 2 1 4 3]
               [3 2 1 4 3 3 4 1 1 4]
               [1 2 4 2 2 1 3 2 3 1]
               [1 4 4 4 1 1 4 2 1 1]
               [1 3 2 3 3 4 1 3 1 3]
               [4 1 3 3 3 4 1 4 1 1]
               [2 1 3 3 4 2 2 1 3 4]
               [2 3 2 2 4 2 1 3 2 2]]
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
              ... )

              ```

            - Now, let's cluster the connected equal cells into polygons.
              ```python
              >>> gdf = dataset.cluster2()
              >>> print(gdf)  # doctest: +SKIP
                  Band_1                                           geometry
              0        3  POLYGON ((0 0, 0 -0.05, 0.05 -0.05, 0.05 0, 0 0))
              1        1  POLYGON ((0.1 0, 0.1 -0.05, 0.2 -0.05, 0.2 0, ...
              2        4  POLYGON ((0.25 0, 0.25 -0.05, 0.3 -0.05, 0.3 0...
              3        4  POLYGON ((0.35 0, 0.35 -0.05, 0.4 -0.05, 0.4 0...
              4        2  POLYGON ((0.4 0, 0.4 -0.05, 0.45 -0.05, 0.45 0...
              5        3  POLYGON ((0.45 0, 0.45 -0.05, 0.5 -0.05, 0.5 0...
              6        1  POLYGON ((0.3 0, 0.3 -0.1, 0.35 -0.1, 0.35 0, ...
              7        4  POLYGON ((0.15 -0.05, 0.15 -0.1, 0.2 -0.1, 0.2...
              8        2  POLYGON ((0.35 -0.05, 0.35 -0.1, 0.4 -0.1, 0.4...
              9        4  POLYGON ((0 -0.05, 0 -0.15, 0.05 -0.15, 0.05 -...
              10       4  POLYGON ((0.4 -0.05, 0.4 -0.15, 0.45 -0.15, 0....
              11       4  POLYGON ((0.1 -0.1, 0.1 -0.15, 0.15 -0.15, 0.1...

              ```

        """
        if band is None:
            band = 0

        if isinstance(band, int):
            name = self._ds.band_names[band]
            gdf = self._band_to_polygon(band, name)
        else:
            gdfs = []
            for b in band:
                name = self._ds.band_names[b]
                gdfs.append(self._band_to_polygon(b, name))
            gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        return gdf


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
