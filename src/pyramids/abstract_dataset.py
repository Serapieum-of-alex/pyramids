"""
Abstract Dataset.

raster contains python functions to handle raster data align them together based on a source raster, perform any
algebraic operation on cell's values. gdal class: https://gdal.org/java/org/gdal/gdal/package-summary.html.
"""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr
from osgeo.osr import SpatialReference

from pyramids._utils import (
    Catalog,
)
from pyramids.featurecollection import FeatureCollection


DEFAULT_NO_DATA_VALUE = -9999
CATALOG = Catalog(raster_driver=True)
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
RESAMPLING_METHODS = [
    "NEAREST",
    "CUBIC",
    "AVERAGE",
    "GAUSS",
    "CUBICSPLINE",
    "LANCZOS",
    "MODE",
    "AVERAGE_MAGPHASE",
    "RMS",
    "BILINEAR",
]


class AbstractDataset(ABC):
    """AbstractDataset."""

    default_no_data_value = DEFAULT_NO_DATA_VALUE

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        """__init__."""
        if not isinstance(src, gdal.Dataset):
            raise TypeError(  # pragma: no cover
                "src should be read using gdal (gdal dataset please read it using gdal"
                f" library) given {type(src)}"
            )
        self._access = access
        self._raster = src
        self._geotransform = src.GetGeoTransform()
        self._cell_size = self.geotransform[1]
        # replace with a loop over the GetMetadata for each separate band
        self._meta_data = src.GetMetadata()
        self._file_name = src.GetDescription()
        # projection data
        # the epsg property returns the value of the _epsg attribute, so if the projection changes in any function, the
        # function should also change the value of the _epsg attribute.
        self._epsg = self._get_epsg()
        # array and dimensions
        self._rows = src.RasterYSize
        self._columns = src.RasterXSize
        self._band_count = src.RasterCount
        self._block_size = [
            src.GetRasterBand(i).GetBlockSize() for i in range(1, self._band_count + 1)
        ]

    @abstractmethod
    def __str__(self):
        """__str__."""
        pass

    @abstractmethod
    def __repr__(self):
        """__repr__."""
        pass

    @property
    @abstractmethod
    def access(self):
        """Access mode (read_only/write)."""
        return self._access

    @property
    @abstractmethod
    def raster(self) -> gdal.Dataset:
        """The ase GDAL Dataset."""
        return self._raster

    @raster.setter
    @abstractmethod
    def raster(self, value: gdal.Dataset):
        """Contains GDAL Dataset."""
        self._raster = value

    @property
    def values(self) -> np.ndarray:
        """Values of all the bands."""
        pass

    @property
    def rows(self) -> int:
        """Number of rows in the raster array."""
        pass

    @property
    def columns(self) -> int:
        """Number of columns in the raster array."""
        pass

    @property
    @abstractmethod
    def shape(self):
        """Shape (bands, rows, columns)."""
        pass

    @property
    @abstractmethod
    def geotransform(self):
        """WKT projection.(x, cell_size, 0, y, 0, -cell_size)."""
        return self._geotransform

    @property
    @abstractmethod
    def top_left_corner(self):
        """Top left corner coordinates."""
        xmin, _, _, ymax, _, _ = self._geotransform
        return xmin, ymax

    @property
    @abstractmethod
    def epsg(self):
        """EPSG number."""
        pass

    @property
    @abstractmethod
    def crs(self) -> str:
        """Coordinate reference system."""
        pass

    @crs.setter
    @abstractmethod
    def crs(self, value: str):
        """Coordinate reference system."""
        pass

    @property
    @abstractmethod
    def cell_size(self) -> int:
        """Cell size."""
        pass

    @property
    @abstractmethod
    def no_data_value(self):
        """No data value that marks the cells out of the domain."""
        pass

    @no_data_value.setter
    @abstractmethod
    def no_data_value(self, value: Union[List, Number]):
        """no_data_value.

        No data value that marks the cells out of the domain

        Notes
        -----
            - the setter does not change the values of the cells to the new no_data_value, it only changes the
            `no_data_value` attribute.
            - use this method to change the `no_data_value` attribute to match the value that is stored in the cells.
            - to change the values of the cells to the new no_data_value, use the `change_no_data_value` method.
        """
        pass

    @property
    @abstractmethod
    def meta_data(self):
        """Meta data."""
        return self._raster.GetMetadata()

    @property
    def block_size(self) -> List[Tuple[int, int]]:
        """Block Size.

        The block size is the size of the block that the raster is divided into, the block size is used to read and
        write the raster data in blocks.

        Examples
        --------
        >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
        >>> size = dataset.block_size
        >>> print(size)
        >>> [(128, 128)]
        """
        return self._block_size

    @block_size.setter
    def block_size(self, value: List[Tuple[int, int]]):
        """Block Size.

        Parameters
        ----------
        value : List[Tuple[int, int]]
            block size for each band in the raster(512, 512).
        """
        if len(value[0]) != 2:
            raise ValueError("block size should be a tuple of 2 integers")

        self._block_size = value

    @property
    @abstractmethod
    def file_name(self):
        """File name."""
        if self._file_name.startswith("NETCDF"):
            name = self._file_name.split(":")[1][1:-1]
        else:
            name = self._file_name
        return name

    @property
    @abstractmethod
    def driver_type(self):
        """Driver Type."""
        drv = self.raster.GetDriver()
        driver_type = drv.GetDescription() if drv is not None else None
        return CATALOG.get_driver_name(driver_type)

    @classmethod
    @abstractmethod
    def read_file(cls, path: str, read_only=True) -> "AbstractDataset":
        """Read file.

        Parameters
        ----------
        path: [str]
            Path of file to open.
        read_only: [bool]
            File mode, set as False, to open in "update" mode.

        Returns
        -------
        Dataset
        """
        pass

    @abstractmethod
    def read_array(self, band: int = None, window: List[int] = None) -> np.ndarray:
        """Read Array.

            - read the values stored in a given band.

        Data Chuncks/blocks
            When a raster dataset is stored on disk, it might not be stored as one continuous chunk of data. Instead,
            it can be divided into smaller rectangular blocks or tiles. These blocks can be individually accessed,
            which is particularly useful for large datasets:
                Efficiency: Reading or writing small blocks requires less memory than dealing with the entire dataset
                    at once. This is especially beneficial when only a small portion of the data needs to be processed.
                Performance: For certain file formats and operations, working with optimal block sizes can significantly
                    improve performance. For example, if the block size matches the reading or processing window,
                        Pyramids can minimize disk access and data transfer.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data, If None the data of all bands will be read. Default is None
        window: [List]
            window to specify a block of data to read from the dataset. the window should be a list of 4 integers
            [offset_x, offset_y, window_columns, window_rows]. Default is None.

        Returns
        -------
        array : [array]
            array with all the values in the raster.

        Examples
        --------
        >>> dataset = Dataset.read_file("tests/data/geotiff/era5_land_monthly_averaged.tif")
        >>> arr = dataset.read_array(window=[0, 0, 5, 5])
        >>> print(arr.shape)
        >>> (5, 5)
        """
        pass

    @abstractmethod
    def _read_block(self, band: int, window=List[int]) -> np.ndarray:
        """Read block of data from the dataset.

        Parameters
        ----------
        band : int
            Band index.
        window: [List]
            window to specify a block of data to read from the dataset. the window should be a list of 4 integers
            [offset_x, offset_y, window_columns, window_rows]. Default is None.

        Returns
        -------
        np.ndarray[window[2], window[3]]
            array with the values of the block. the shape of the array is (window[2], window[3]), and the location of
            the block in the raster is (window[0], window[1]).
        """
        pass

    @abstractmethod
    def plot(
        self,
        band: int = None,
        exclude_value: Any = None,
        rgb: List[int] = None,
        surface_reflectance: int = 10000,
        cutoff: List = None,
        overview: bool = False,
        overview_index: int = 0,
        **kwargs,
    ):
        """Plot.

            - plot the values/overviews of a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0
        exclude_value: [Any]
            value to exclude from the plot. Default is None.
        rgb: [List]
            Default is [3,2,1]
        surface_reflectance: [int]
            Default is 10,000.
        cutoff: [List]
            clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
            and scale them back to between 0 and 1). Default is None.
        overview: [bool]
            True if you want to plot the overview. Default is False.
        overview_index: [int]
            index of the overview. Default is 0.
        **kwargs
            points : [array]
                3 column array with the first column as the value you want to display for the point, the second is the
                rows index of the point in the array, and the third column as the column index in the array.
                - the second and third column tells the location of the point in the array.
            point_color: [str]
                color.
            point_size: [Any]
                size of the point.
            pid_color: [str]
                the color of the annotation of the point. Default is blue.
            pid_size: [Any]
                size of the point annotation.
            figsize: [tuple], optional
                figure size. The default is (8,8).
            title: [str], optional
                title of the plot. The default is 'Total Discharge'.
            title_size: [integer], optional
                title size. The default is 15.
            orientation: [string], optional
                orientation of the color bar horizontal/vertical. The default is 'vertical'.
            rotation: [number], optional
                rotation of the color bar label. The default is -90.
            orientation: [string], optional
                orientation of the color bar horizontal/vertical. The default is 'vertical'.
            cbar_length: [float], optional
                ratio to control the height of the color bar. The default is 0.75.
            ticks_spacing: [integer], optional
                Spacing in the color bar ticks. The default is 2.
            cbar_label_size: integer, optional
                size of the color bar label. The default is 12.
            cbar_label: str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            color_scale: integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- color_scale 1 is the normal scale
                2- color_scale 2 is the power scale
                3- color_scale 3 is the SymLogNorm scale
                4- color_scale 4 is the PowerNorm scale
                5- color_scale 5 is the BoundaryNorm scale
                ------------------------------------------------------------------
                gamma: [float], optional
                    value needed for option 2. The default is 1./2.
                line_threshold: [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale: [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint: [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap: [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value: [bool]
                True if you want to display the values of the cells as a text
            num_size: integer, optional
                size of the numbers plotted on top of each cell. The default is 8.
            background_color_threshold: [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black, and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        pass

    @classmethod
    @abstractmethod
    def create_from_array(
        cls,
        arr: np.ndarray,
        geo: Tuple[float, float, float, float, float, float],
        bands_values: List = None,
        epsg: Union[str, int] = 4326,
        no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE,
        driver_type: str = "MEM",
        path: str = None,
        variable_name: str = None,
    ):
        """Create dataset from array.

            - Create_from_array method creates a `Dataset` from a given array and geotransform data.

        Parameters
        ----------
        arr: [np.ndarray]
            numpy array.
        geo : [Tuple]
            geotransform tuple [minimum lon/x, pixel-size, rotation, maximum lat/y, rotation, pixel-size].
        bands_values: [List]
            Name of the bands to be used in the netcdf file. Default is None,
        epsg: [integer]
            integer reference number to the new projection (https://epsg.io/)
                (default 3857 the reference no of WGS84 web mercator)
        no_data_value : Any, optional
            no data value to mask the cells out of the domain. The default is -9999.
        driver_type: [str] optional
            driver type ["GTiff", "MEM", "netcdf"]. Default is "MEM"
        path : [str]
            path to save the driver.
        variable_name: [str]
            name of the variable in the netcdf file. Default is None.

        Returns
        -------
        dst: [DataSet].
            Dataset object will be returned.
        """
        pass

    @abstractmethod
    def _get_crs(self) -> str:
        """Get coordinate reference system."""
        pass

    @abstractmethod
    def set_crs(self, crs: Optional = None, epsg: int = None):
        """Set Coordinates reference system.

            Set the Coordinate Reference System (CRS) of a

        Parameters
        ----------
        crs: [str]
            optional if epsg is specified
            WKT string.
            i.e. 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
            0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        epsg: [int]
            optional if crs is specified
            EPSG code specifying the projection.
        """
        # first change the projection of the gdal dataset object
        # second change the epsg attribute of the Dataset object
        if self.driver_type == "ascii":
            raise TypeError(
                "Setting CRS for ASCII file is not possible, you can save the files to a geotiff and then reset the crs"
            )
        else:
            if crs is not None:
                self.raster.SetProjection(crs)
                self._epsg = FeatureCollection.get_epsg_from_prj(crs)
            else:
                sr = AbstractDataset._create_sr_from_epsg(epsg)
                self.raster.SetProjection(sr.ExportToWkt())
                self._epsg = epsg

    @abstractmethod
    def to_crs(
        self,
        to_epsg: int,
        method: str = "nearest neighbor",
        maintain_alignment: int = False,
        inplace: bool = False,
    ) -> Union["AbstractDataset", None]:
        """To EPSG.

        to_epsg reprojects a raster to any projection
        (default the WGS84 web mercator projection, without resampling)

        Parameters
        ----------
        to_epsg: [integer]
            reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        method: [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "nearest neighbor" for nearest neighbor,"cubic" for cubic convolution,
            "bilinear" for bilinear
        maintain_alignment : [bool]
            True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.
        inplace: [bool]
            True to make changes inplace. Default is False.

        Returns
        -------
        Dataset:
            Dataset object, if inplace is True, the method returns None.

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> dataset = Dataset.read_file("path/raster_name.tif")
        >>> reprojected_dataset = dataset.to_crs(to_epsg=3857)
        """
        pass

    @abstractmethod
    def _get_epsg(self) -> int:
        """Get EPSG.

            This function reads the projection of a GEOGCS file or tiff file

        Returns
        -------
        epsg : [integer]
            epsg number
        """
        pass

    @staticmethod
    @abstractmethod
    def _create_sr_from_epsg(epsg: int = None) -> SpatialReference:
        """Create a spatial reference object from epsg number.

        https://gdal.org/tutorials/osr_api_tut.html

        Parameters
        ----------
        epsg: [int]
            epsg number.

        Returns
        -------
        SpatialReference object
        """
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(int(epsg))
        return sr

    @abstractmethod
    def _check_no_data_value(self, no_data_value: List):
        """Validate The no_data_value with the dtype of the object.

        Parameters
        ----------
        no_data_value

        Returns
        -------
        no_data_value:
            convert the no_data_value to comply with the dtype
        """
        # convert the no_data_value based on the dtype of each raster band.
        pass

    @abstractmethod
    def _set_no_data_value(
        self, no_data_value: Union[Any, list] = DEFAULT_NO_DATA_VALUE
    ):
        """Set the NoDataValue.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - used only when creating an empty driver.

            now the no_data_value is converted to the dtype of the raster bands and updated in the
            dataset attribute, gdal nodatavalue attribute, used to fill the raster band.
            from here you have to use the no_data_value stored in the no_data_value attribute as it is updated.

        Parameters
        ----------
        no_data_value: [numeric]
            no data value to fill the masked part of the array.
        """
        pass

    @abstractmethod
    def change_no_data_value(self, new_value: Any, old_value: Any = None):
        """Change No Data Value.

            - Set the no data value in all raster bands.
            - Fill the whole raster with the no_data_value.
            - Change the no_data_value in the array in all bands.

        Parameters
        ----------
        new_value: [numeric]
            no data value to set in the raster bands.

        old_value: [numeric]
            old no data value that is already in the raster bands.
        """
        pass

    @abstractmethod
    def to_file(self, path: str, band: int = 0) -> None:
        """Save dataset to disk.

            to_file a raster to a path, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Parameters
        ----------
        path: [string]
            a path including the name of the dataset whti the extention at the end (i.e. "data/cropped.tif").
        band: [int]
            band index, needed only in case of ascii drivers. Default is 0.

        Examples
        --------
        >>> dataset = Dataset.read_file("path/to/file/***.tif")
        >>> dataset.to_file("save_raster_test.tif")

        Notes
        -----
        The object will still refers to the dataset before saving. if you want to use the new saved dataset you have
        to read the file again.
        """
        pass

    @abstractmethod
    def crop(
        self,
        mask: Union[GeoDataFrame, FeatureCollection],
        touch: bool = True,
        inplace: bool = False,
    ) -> Union["AbstractDataset", None]:
        """Crop.

            Crop/Clip the Dataset object using a polygon/raster.

        Parameters
        ----------
        mask: [Polygon GeoDataFrame/Dataset]
            GeodataFrame with a polygon geometry, or a Dataset object.
        touch: [bool]
            To include the cells that touch the polygon not only those that lie entirely inside the polygon mask.
            Default is True.
        inplace: [bool]
            True to make the changes in place.

        Returns
        -------
        Dataset Object
        """
        pass

    @abstractmethod
    def extract(
        self,
        exclude_value: Any = None,
        feature: Union[FeatureCollection, GeoDataFrame] = None,
    ) -> np.ndarray:
        """Extract.

            - Extract method gets all the values in a raster, and excludes the values in the exclude_value parameter.
            - If the feature parameter is given, the raster will be clipped to the extent of the given feature and the
            values within the feature are extracted.

        Parameters
        ----------
        exclude_value: [Numeric]
            values you want to exclude from extracted values
        feature: [FeatureCollection/GeoDataFrame]
            vector file contains geometries you want to extract the values at their location. Default is None.
        """
        # Optimize: make the read_array return only the array for inside the mask feature, and not to read the whole
        #  raster
        pass

    @abstractmethod
    def overlay(
        self,
        classes_map,
        band: int = 0,
        exclude_value: Union[float, int] = None,
    ) -> Dict[List[float], List[float]]:
        """Overlay.

            overlay extracts all the values in raster file if you have two maps one with classes, and the other map
            contains any type of values, and you want to know the values in each class.

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object for the raster that have classes you want to overlay with the raster.
        band: [int]
            if the raster is multi-band raster choose the band you want to overlay with the classes map. Default is 0.
        exclude_value: [Numeric]
            values you want to exclude from extracted values.

        Returns
        -------
        Dictionary:
            dictionary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path.
        """
        pass

    @abstractmethod
    def create_overviews(
        self, resampling_method: str = "nearest", overview_levels: list = None
    ):
        """Create overviews for the dataset.

        Parameters
        ----------
        resampling_method : str, optional
            The resampling method used to create the overviews, by default "nearest"
            possible values are:
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE_MAGPHASE", "RMS",
                "BILINEAR".
        overview_levels : list, optional
            The overview levels, overview_levels are restricted to the typical power-of-two reduction factors.
            Default [2, 4, 8, 16, 32]

        Returns
        -------
        internal/external overviews:
            The overview (also known as pyramids) could be internal or external depending on the state you read
            the dataset with.
            - External (.ovr file):
                If the dataset is read with a`read_only=True` then the overviews' file will be created as an
                in the same directory of the dataset, with the same name of the dataset and .ovr extension.
            - Internal:
                If the dataset is read with a`read_only=False` then the overviews will be created internally in the
                dataset, and the dataset needs to be saved/flushed to save the new changes to disk.
        overview_count: [list]
            a list property attribute of the overviews for each band.
        """
        pass

    @abstractmethod
    def recreate_overviews(self, resampling_method: str = "nearest"):
        """Recreate overviews for the dataset.

        Parameters
        ----------
        resampling_method : str, optional
            The resampling method used to create the overviews, by default "nearest"
            possible values are:
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE_MAGPHASE", "RMS",
                "BILINEAR".

        Raises
        ------
        ValueError
            resampling_method should be one of {"NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS",
            "MODE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR"}
        ReadOnlyError
            If the overviews are internal and the Dataset is opened with a read only. Please read the dataset using
            read_only=False
        """
        pass

    @abstractmethod
    def get_overview(self, band: int = 0, overview_index: int = 0) -> gdal.Band:
        """Get an overview of a band.

        Parameters
        ----------
        band : int, optional
            The band index, by default 0
        overview_index: [int]
            index of the overview. Default is 0.

        Returns
        -------
        gdal.Band
            gdal band object
        """
        pass

    @abstractmethod
    def read_overview_array(
        self, band: int = None, overview_index: int = 0
    ) -> np.ndarray:
        """Read an overview array.

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data, If None, the data of all bands will be read. Default is None
        overview_index: [int]
            index of the overview. Default is 0.

        Returns
        -------
        array : [array]
            array with all the values in the raster.
        """
        pass
