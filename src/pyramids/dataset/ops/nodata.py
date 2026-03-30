"""NoData-related mixin for the Dataset class."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal

from pyramids.base._errors import NoDataValueError, ReadOnlyError
from pyramids.base._utils import gdal_to_numpy_dtype
from pyramids.dataset.abstract_dataset import DEFAULT_NO_DATA_VALUE
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


class NoData:

    def _check_no_data_value(self, no_data_value: list):
        """Validate the no_data_value with the dtype of the object.

        Args:
            no_data_value:
                No-data value(s) to validate.

        Returns:
            Any:
                Convert the no_data_value to comply with the dtype.
        """
        # convert the no_data_value based on the dtype of each raster band.
        for i, val in enumerate(self.gdal_dtype):
            try:
                val = no_data_value[i]
                # if not None or np.nan
                if val is not None and not np.isnan(val):
                    # if val < np.iinfo(self.dtype[i]).min or val > np.iinfo(self.dtype[i]).max:
                    # if the no_data_value is out of the range of the data type
                    no_data_value[i] = self.numpy_dtype[i](val)
                else:
                    # None and np.nan
                    if self.dtype[i].startswith("u"):
                        # only Unsigned integer data types.
                        # if None or np.nan it will make a problem with the unsigned integer data type
                        # use the max bound of the data type as a no_data_value
                        no_data_value[i] = np.iinfo(self.dtype[i]).max
                    else:
                        # no_data_type is None/np,nan and all other data types that is not Unsigned integer
                        no_data_value[i] = val

            except OverflowError:
                # no_data_value = -3.4028230607370965e+38, numpy_dtype = np.int64
                warnings.warn(
                    f"The no_data_value:{no_data_value[i]} is out of range, Band data type is {self.numpy_dtype[i]}"
                )
                no_data_value[i] = self.numpy_dtype[i](DEFAULT_NO_DATA_VALUE)
        return no_data_value

    def _set_no_data_value(self, no_data_value: Any | list = DEFAULT_NO_DATA_VALUE):
        """setNoDataValue.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - used only when creating an empty driver.

            now the no_data_value is converted to the dtype of the raster bands and updated in the
            dataset attribute, gdal no_data_value attribute, used to fill the raster band.
            from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.

        Args:
            no_data_value (numeric):
                No data value to fill the masked part of the array.
        """
        if not isinstance(no_data_value, list):
            no_data_value = [no_data_value] * self.band_count

        no_data_value = self._check_no_data_value(no_data_value)

        for band in range(self.band_count):
            try:
                # now the no_data_value is converted to the dtype of the raster bands and updated in the
                # dataset attribute, gdal no_data_value attribute, used to fill the raster band.
                # from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.
                self._set_no_data_value_backend(band, no_data_value[band])
            except Exception as e:
                if str(e).__contains__(
                    "Attempt to write to read only dataset in GDALRasterBand::Fill()."
                ):
                    raise ReadOnlyError(
                        "The Dataset is open with a read only, please read the raster using update access mode"
                    )
                elif str(e).__contains__(
                    "in method 'Band_SetNoDataValue', argument 2 of type 'double'"
                ):
                    self._set_no_data_value_backend(
                        band, np.float64(no_data_value[band])
                    )
                else:
                    self._set_no_data_value_backend(band, DEFAULT_NO_DATA_VALUE)
                    self.logger.warning(
                        "the type of the given no_data_value differs from the dtype of the raster"
                        f"no_data_value now is set to {DEFAULT_NO_DATA_VALUE} in the raster"
                    )

    def _calculate_bbox(self) -> list:
        """Calculate bounding box."""
        xmin, ymax = self.top_left_corner
        ymin = ymax - self.rows * self.cell_size
        xmax = xmin + self.columns * self.cell_size
        return [xmin, ymin, xmax, ymax]

    def _calculate_bounds(self) -> GeoDataFrame:
        """Get the bbox as a geodataframe with a polygon geometry."""
        xmin, ymin, xmax, ymax = self._calculate_bbox()
        coords = [(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)]
        poly = FeatureCollection.create_polygon(coords)
        gdf = gpd.GeoDataFrame(geometry=[poly])
        gdf.set_crs(epsg=self.epsg, inplace=True)
        return gdf

    def _set_no_data_value_backend(self, band_i: int, no_data_value: Any):
        """
            - band_i starts from 0 to the number of bands-1.

        Args:
            band_i:
                Band index, starts from 0.
            no_data_value:
                Numerical value.
        """
        # check if the dtype of the no_data_value complies with the dtype of the raster itself.
        self._change_no_data_value_attr(band_i, no_data_value)
        # initialize the band with the nodata value instead of 0
        # the no_data_value may have changed inside the _change_no_data_value_attr method to float64, so redefine it.
        no_data_value = self.no_data_value[band_i]
        try:
            self.raster.GetRasterBand(band_i + 1).Fill(no_data_value)
        except Exception as e:
            if str(e).__contains__(" argument 2 of type 'double'"):
                self.raster.GetRasterBand(band_i + 1).Fill(np.float64(no_data_value))
            elif str(e).__contains__(
                "Attempt to write to read only dataset in GDALRasterBand::Fill()."
            ) or str(e).__contains__(
                "attempt to write to dataset opened in read-only mode."
            ):
                raise ReadOnlyError(
                    "The Dataset is open with a read only, please read the raster using update access mode"
                )
            else:
                raise ValueError(
                    f"Failed to fill the band {band_i} with value: {no_data_value}, because of {e}"
                )
        # update the no_data_value in the Dataset object
        self.no_data_value[band_i] = no_data_value

    def _change_no_data_value_attr(self, band: int, no_data_value):
        """Change the no_data_value attribute.

            - Change only the no_data_value attribute in the gdal Dataset object.
            - Change the no_data_value in the Dataset object for the given band index.
            - The corresponding value in the array will not be changed.

        Args:
            band (int):
                Band index, starts from 0.
            no_data_value (Any):
                No data value.
        """
        try:
            self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)
        except Exception as e:
            if str(e).__contains__(
                "Attempt to write to read only dataset in GDALRasterBand::Fill()."
            ):
                raise ReadOnlyError(
                    "The Dataset is open with a read only, please read the raster using update "
                    "access mode"
                )
            # TypeError
            elif e.args == (
                "in method 'Band_SetNoDataValue', argument 2 of type 'double'",
            ):
                no_data_value = np.float64(no_data_value)
                self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value)

        self._no_data_value[band] = no_data_value

    def change_no_data_value(self, new_value: Any, old_value: Any | None = None):
        """Change No Data Value.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.

        Args:
            new_value (numeric):
                No data value to set in the raster bands.
            old_value (numeric):
                Old no data value that is already in the raster bands.

        Warning:
            The `change_no_data_value` method creates a new dataset in memory in order to change the `no_data_value` in the raster bands.

        Examples:
            - Create a Dataset (4 bands, 10 rows, 10 columns) at lon/lat (0, 0):

              ```python
              >>> dataset = Dataset.create(
              ...     cell_size=0.05, rows=3, columns=3, bands=1, top_left_corner=(0, 0),dtype="float32",
              ...     epsg=4326, no_data_value=-9
              ... )
              >>> arr = dataset.read_array()
              >>> print(arr)
              [[-9. -9. -9.]
               [-9. -9. -9.]
               [-9. -9. -9.]]
              >>> print(dataset.no_data_value) # doctest: +SKIP
              [-9.0]

              ```

            - The dataset is full of the no_data_value. Now change it using `change_no_data_value`:

              ```python
              >>> new_dataset = dataset.change_no_data_value(-10, -9)
              >>> arr = new_dataset.read_array()
              >>> print(arr)
              [[-10. -10. -10.]
               [-10. -10. -10.]
               [-10. -10. -10.]]
              >>> print(new_dataset.no_data_value) # doctest: +SKIP
              [-10.0]

              ```
        """
        if not isinstance(new_value, list):
            new_value = [new_value] * self.band_count

        if old_value is not None and not isinstance(old_value, list):
            old_value = [old_value] * self.band_count

        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.raster, 0)
        # create a new dataset
        new_dataset = type(self)(dst, "write")
        # the new_value could change inside the _set_no_data_value method before it is used to set the no_data_value
        # attribute in the gdal object/pyramids object and to fill the band.
        new_dataset._set_no_data_value(new_value)
        # now we have to use the no_data_value value in the no_data_value attribute in the Dataset object as it is
        # updated.
        new_value = new_dataset.no_data_value
        for band in range(self.band_count):
            arr = self.read_array(band)
            try:
                if old_value is not None:
                    arr[np.isclose(arr, old_value, rtol=0.001)] = new_value[band]
                else:
                    arr[np.isnan(arr)] = new_value[band]
            except TypeError:
                raise NoDataValueError(
                    f"The dtype of the given no_data_value: {new_value[band]} differs from the dtype of the "
                    f"band: {gdal_to_numpy_dtype(self.gdal_dtype[band])}"
                )
            new_dataset.raster.GetRasterBand(band + 1).WriteArray(arr)
        return new_dataset
