"""Datacube module."""

import os
import re
from loguru import logger
import datetime as dt
import pandas as pd
from pathlib import Path
from typing import List, Union, Any, Callable, Dict
import numpy as np
from osgeo import gdal
from pyramids.dataset import Dataset
from pyramids._errors import DatasetNoFoundError
from pyramids.abstract_dataset import CATALOG
from pyramids._utils import import_cleopatra

try:
    from osgeo_utils import gdal_merge
except ModuleNotFoundError:  # pragma: no cover
    logger.warning(  # pragma: no cover
        "osgeo_utils module does not exist try install pip install osgeo-utils "
    )


class Datacube:
    """DataCube."""

    files: List[str]
    data: np.ndarray

    """
    files:
        list of geotiff files' names
    """

    def __init__(
        self,
        src: Dataset,
        time_length: int,
        files: List[str] = None,
    ):
        """Construct Datacube object."""
        self._base = src
        self._files = files
        self._time_length = time_length

        pass

    def __str__(self):
        """__str__."""
        message = f"""
            Files: {len(self.files)}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Dimension: {self.rows} * {self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

    def __repr__(self):
        """__repr__."""
        message = f"""
            Files: {len(self.files)}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Dimension: {self.rows} * {self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

    @property
    def base(self) -> Dataset:
        """base.

        Base Dataset
        """
        return self._base

    @property
    def files(self):
        """Files."""
        return self._files

    @property
    def time_length(self) -> int:
        """Length of the dataset."""
        return self._time_length

    @property
    def rows(self):
        """Number of rows."""
        return self._base.rows

    @property
    def shape(self):
        """Number of rows."""
        return self.time_length, self.rows, self.columns

    @property
    def columns(self):
        """Number of columns."""
        return self._base.columns

    @classmethod
    def create_cube(cls, src: Dataset, dataset_length: int):
        """Create Datacube.

            - Create Datacube from a sample raster and

        Parameters
        ----------
        src: [Dataset]
            RAster Object
        dataset_length: [int]
            length of the dataset.

        Returns
        -------
        Datacube object.
        """
        return cls(src, dataset_length)

    @classmethod
    def read_multiple_files(
        cls,
        path: Union[str, List[str]],
        with_order: bool = False,
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str = None,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        extension: str = ".tif",
    ):
        r"""read_multiple_files.

            - reads rasters from a folder and creates a 3d array with the same 2d dimensions of the first raster in
            the folder and length as the number of files.

        inside the folder.
        - All rasters should have the same dimensions
        - If you want to read the rasters with a certain order, then all raster file names should have a date that
            follows the same format (YYYY.MM .DD / YYYY-MM-DD or YYYY_MM_DD) (i.e. "MSWEP_1979.01.01.tif").

        Parameters
        ----------
        path:[str/list]
            path of the folder that contains all the rasters, ora list contains the paths of the rasters to read.
        with_order: [bool]
            True if the rasters names' follows a certain order, then the rasters' names should have a date that follows
            the same format (YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD).
            >>> "MSWEP_1979.01.01.tif"
            >>> "MSWEP_1979.01.02.tif"
            >>> ...
            >>> "MSWEP_1979.01.20.tif"
        regex_string: [str]
            a regex string that we can use to locate the date in the file names.Default is r"\d{4}.\d{
            2}.\d{2}".
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

        Returns
        -------
        DataCube:
            instance of the datacube class.

        Example
        -------
        >>> from pyramids.datacube import Datacube
        >>> raster_folder = "examples/GIS/data/raster-folder"
        >>> prec = Datacube.read_multiple_files(raster_folder)

        >>> import glob
        >>> search_criteria = "*.tif"
        >>> file_list = glob.glob(os.path.join(raster_folder, search_criteria))
        >>> prec = Datacube.read_multiple_files(file_list, with_order=False)
        """
        if not isinstance(path, str) and not isinstance(path, list):
            raise TypeError(f"path input should be string/list type, given{type(path)}")

        if isinstance(path, str):
            # check whither the path exists or not
            if not os.path.exists(path):
                raise FileNotFoundError("The path you have provided does not exist")
            # get a list of all files
            files = os.listdir(path)
            files = [i for i in files if i.endswith(extension)]
            # files = glob.glob(os.path.join(path, "*.tif"))
            # check whether there are files or not inside the folder
            if len(files) < 1:
                raise FileNotFoundError("The path you have provided is empty")
        else:
            files = path[:]

        # to sort the files in the same order as the first number in the name
        if with_order:
            match_str_fn = lambda x: re.search(regex_string, x)
            list_dates = list(map(match_str_fn, files))

            if None in list_dates:
                raise ValueError(
                    "The date format/separator given does not match the file names"
                )
            if date:
                if file_name_data_fmt is None:
                    raise ValueError(
                        f"To read the raster with a certain order (with_order = {with_order}, then you have to enter "
                        f"the value of the parameter file_name_data_fmt(given: {file_name_data_fmt})"
                    )
                fn = lambda x: dt.datetime.strptime(x.group(), file_name_data_fmt)
            else:
                fn = lambda x: int(x.group())
            list_dates = list(map(fn, list_dates))

            df = pd.DataFrame()
            df["files"] = files
            df["date"] = list_dates
            df.sort_values("date", inplace=True, ignore_index=True)
            files = df.loc[:, "files"].values

        if start is not None or end is not None:
            if date:
                start = dt.datetime.strptime(start, fmt)
                end = dt.datetime.strptime(end, fmt)

                files = (
                    df.loc[start <= df["date"], :]
                    .loc[df["date"] <= end, "files"]
                    .values
                )
            else:
                files = (
                    df.loc[start <= df["date"], :]
                    .loc[df["date"] <= end, "files"]
                    .values
                )

        if not isinstance(path, list):
            # add the path to all the files
            files = [f"{path}/{i}" for i in files]
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        sample = Dataset.read_file(files[0])

        return cls(sample, len(files), files)

    def open_datacube(self, band: int = 0):
        """open_datacube.

            Read values form the given bands as Arrays for all files

        Parameters
        ----------
        band: [int]
            index of the band you want to read default is 0.

        Returns
        -------
        Array
        """
        # check the given band number
        if not hasattr(self, "base"):
            raise ValueError(
                "please use the read_multiple_files method to get the files (tiff/ascii) in the"
                "dataset directory"
            )
        if band > self.base.band_count - 1:
            raise ValueError(
                f"the raster has only {self.base.band_count} check the given band number"
            )
        # fill the array with no_data_value data
        self._values = np.ones(
            (
                self.time_length,
                self.base.rows,
                self.base.columns,
            )
        )
        self._values[:, :, :] = np.nan

        for i, file_i in enumerate(self.files):
            # read the tif file
            raster_i = gdal.Open(f"{file_i}")
            self._values[i, :, :] = raster_i.GetRasterBand(band + 1).ReadAsArray()

    @property
    def values(self) -> np.ndarray:
        """Values.

        - The attribute where the dataset array is stored.
        - the 3D numpy array, [dataset length, rows, cols], [dataset length, lons, lats]
        """
        return self._values

    @values.setter
    def values(self, val):
        """Values.

        - setting the data (array) does not allow different dimension from the dimension that has been
        defined in creating the dataset.
        """
        # if the attribute is defined before check the dimension
        if hasattr(self, "values"):
            if self._values.shape != val.shape:
                raise ValueError(
                    f"The dimension of the new data: {val.shape}, differs from the dimension of the "
                    f"original dataset: {self._values.shape}, please redefine the base Dataset and "
                    f"dataset_length first"
                )

        self._values = val

    @values.deleter
    def values(self):
        self._values = None

    def __getitem__(self, key):
        """Getitem."""
        if not hasattr(self, "values"):
            raise AttributeError("Please use the read_dataset method to read the data")
        return self._values[key, :, :]

    def __setitem__(self, key, value: np.ndarray):
        """Setitem."""
        if not hasattr(self, "values"):
            raise AttributeError("Please use the read_dataset method to read the data")
        self._values[key, :, :] = value

    def __len__(self):
        """Length of the Datacube."""
        return self._values.shape[0]

    def __iter__(self):
        """Iterate over the Datacube."""
        return iter(self._values[:])

    def head(self, n: int = 5):
        """First 5 Datasets."""
        return self._values[:n, :, :]

    def tail(self, n: int = -5):
        """Last 5 Datasets."""
        return self._values[n:, :, :]

    def first(self):
        """First Dataset."""
        return self._values[0, :, :]

    def last(self):
        """Last Dataset."""
        return self._values[-1, :, :]

    def iloc(self, i):
        """iloc.

            - Access dataset array using index.

        Parameters
        ----------
        i: [int]
            index

        Returns
        -------
        Dataset:
            Dataset object.
        """
        if not hasattr(self, "values"):
            raise DatasetNoFoundError("please read the dataset first")
        arr = self._values[i, :, :]
        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.base.raster, 0)
        dst.GetRasterBand(1).WriteArray(arr)
        return Dataset(dst)

    def plot(self, band: int = 0, exclude_value: Any = None, **kwargs):
        """Read Array.

            - read the values stored in a given band.

        Parameters
        ----------
        band : [integer]
            the band you want to get its data. Default is 0
        exclude_value: [Any]
            value to exclude from the plot. Default is None.
        **kwargs
            points : [array]
                3 column array with the first column as the value you want to display for the point, the second is the
                rows index of the point in the array, and the third column as the column index in the array.
                the second and third column tells the location of the point in the array.
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
                gamma : [float], optional
                    value needed for option 2 . The default is 1./2..
                line_threshold : [float], optional
                    value needed for option 3. The default is 0.0001.
                line_scale : [float], optional
                    value needed for option 3. The default is 0.001.
                bounds: [List]
                    a list of number to be used as a discrete bounds for the color scale 4.Default is None,
                midpoint : [float], optional
                    value needed for option 5. The default is 0.
                ------------------------------------------------------------------
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            display_cell_value : [bool]
                True if you want to display the values of the cells as a text
            num_size : integer, optional
                size of the numbers plotted in top of each cells. The default is 8.
            background_color_threshold : [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.

        Returns
        -------
        axes: [figure axes].
            the axes of the matplotlib figure
        fig: [matplotlib figure object]
            the figure object
        """
        import_cleopatra(
            "The current funcrion uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/Serapieum-of-alex/cleopatra"
        )
        from cleopatra.array import Array

        data = self.values

        exclude_value = (
            [self.base.no_data_value[band], exclude_value]
            if exclude_value is not None
            else [self.base.no_data_value[band]]
        )

        cleo = Array(data, exclude_value=exclude_value)
        time = list(range(self.time_length))
        cleo.animate(time, **kwargs)
        return cleo

    def to_file(
        self, path: Union[str, List[str]], driver: str = "geotiff", band: int = 0
    ):
        """Save to geotiff format.

            saveRaster saves a raster to a path

        Parameters
        ----------
        path: [str/list]
            a path includng the name of the raster and extention.
            >>> path = "data/cropped.tif"
        driver: [str]
            driver = "geotiff".
        band: [int]
            band index, needed only in case of ascii drivers. Default is 1.

        Examples
        --------
        >>> raster_obj = Dataset.read_file("path/to/file/***.tif")
        >>> output_path = "examples/GIS/data/save_raster_test.tif"
        >>> raster_obj.to_file(output_path)
        """
        ext = CATALOG.get_extension(driver)

        if isinstance(path, str):
            if not Path(path).exists():
                Path(path).mkdir(parents=True, exist_ok=True)
            path = [f"{path}/{i}.{ext}" for i in range(self.time_length)]
        else:
            if not len(path) == self.time_length:
                raise ValueError(
                    f"Length of the given paths: {len(path)} does not equal number of rasters in the data cube: {self.time_length}"
                )
            if not Path(path[0]).parent.exists():
                Path(path[0]).parent.mkdir(parents=True, exist_ok=True)

        for i in range(self.time_length):
            src = self.iloc(i)
            src.to_file(path[i], band=band)

    def to_crs(
        self,
        to_epsg: int = 3857,
        method: str = "nearest neighbor",
        maintain_alignment: int = False,
    ):
        """to_epsg.

            - to_epsg reprojects a raster to any projection (default the WGS84 web mercator projection,
            without resampling) The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

        Parameters
        ----------
        to_epsg: [integer]
            reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        method: [String]
            resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "Nearest" for nearest neighbor,"cubic" for cubic convolution,
            "bilinear" for bilinear
        maintain_alignment : [bool]
            True to maintain the number of rows and columns of the raster the same after reprojection. Default is False.

        Returns
        -------
        raster:
            gdal dataset (you can read it by ReadAsArray)

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> src = Dataset.read_file("path/raster_name.tif")
        >>> projected_raster = src.to_crs(to_epsg=3857)
        """
        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.to_crs(
                to_epsg, method=method, maintain_alignment=maintain_alignment
            )
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (
                            self.time_length,
                            arr.shape[0],
                            arr.shape[1],
                        )
                    )
                    * np.nan
                )
            array[i, :, :] = arr

        self._values = array
        # use the last src as
        self._base = dst

    def crop(
        self, mask: Union[Dataset, str], inplace: bool = False, touch: bool = True
    ) -> Union[None, Dataset]:
        """cropAlignedFolder.

            cropAlignedFolder matches the location of nodata value from src raster to dst
            raster, Mask is where the NoDatavalue will be taken and the location of
            this value src_dir is path to the folder where rasters exist where we
            need to put the NoDataValue of the mask in RasterB at the same locations

        Parameters
        ----------
        mask : [Dataset]
            Dataset object of the mask raster to crop the rasters (to get the NoData value
            and it location in the array) Mask should include the name of the raster and the
            extension like "data/dem.tif", or you can read the mask raster using gdal and use
            is the first parameter to the function.
        inplace: [bool]
            True to make the changes in place.
        touch: [bool]
            to include the cells that touches the polygon not only those that lies entirely inside the polygon mask.
            Default is True.

        Returns
        -------
        new rasters have the values from rasters in B_input_path with the NoDataValue in the same
        locations as raster A.

        Examples
        --------
        >>> dem_path = "examples/GIS/data/acc4000.tif"
        >>> src_path = "examples/GIS/data/aligned_rasters/"
        >>> out_path = "examples/GIS/data/crop_aligned_folder/"
        >>> Datacube.crop(dem_path, src_path, out_path)
        """
        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.crop(mask, touch=touch)
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (self.time_length, arr.shape[0], arr.shape[1]),
                    )
                    * np.nan
                )

            array[i, :, :] = arr

        if inplace:
            self._values = array
            # use the last src as
            self._base = dst
        else:
            dataset = Datacube(dst, time_length=self.time_length)
            dataset._values = array
            return dataset

    # # TODO: merge ReprojectDataset and ProjectRaster they are almost the same
    # # TODO: still needs to be tested
    # @staticmethod
    # def to_epsg(
    #         src: gdal.Datacube,
    #         to_epsg: int = 3857,
    #         cell_size: int = [],
    #         method: str = "Nearest",
    #
    # ) -> gdal.Datacube:
    #     """to_epsg.
    #
    #         - to_epsg reprojects and resamples a folder of rasters to any projection
    #         (default the WGS84 web mercator projection, without resampling)
    #
    #     Parameters
    #     ----------
    #     src: [gdal dataset]
    #         gdal dataset object (src=gdal.Open("dem.tif"))
    #     to_epsg: [integer]
    #          reference number to the new projection (https://epsg.io/)
    #         (default 3857 the reference no of WGS84 web mercator )
    #     cell_size: [integer]
    #          number to resample the raster cell size to a new cell size
    #         (default empty so raster will not be resampled)
    #     method: [String]
    #         resampling technique default is "Nearest"
    #         https://gisgeography.com/raster-resampling/
    #         "Nearest" for nearest neighbor,"cubic" for cubic convolution,
    #         "bilinear" for bilinear
    #
    #     Returns
    #     -------
    #     raster: [gdal Datacube]
    #          a GDAL in-memory file object, where you can ReadAsArray etc.
    #     """
    #     if not isinstance(src, gdal.Datacube):
    #         raise TypeError(
    #             "src should be read using gdal (gdal dataset please read it using gdal"
    #             f" library) given {type(src)}"
    #         )
    #     if not isinstance(to_epsg, int):
    #         raise TypeError(
    #             "please enter correct integer number for to_epsg more information "
    #             f"https://epsg.io/, given {type(to_epsg)}"
    #         )
    #     if not isinstance(method, str):
    #         raise TypeError(
    #             "please enter correct method more information see " "docmentation "
    #         )
    #
    #     if cell_size:
    #         assert isinstance(cell_size, int) or isinstance(
    #             cell_size, float
    #         ), "please enter an integer or float cell size"
    #
    #     if method == "Nearest":
    #         method = gdal.GRA_NearestNeighbour
    #     elif method == "cubic":
    #         method = gdal.GRA_Cubic
    #     elif method == "bilinear":
    #         method = gdal.GRA_Bilinear
    #
    #     src_proj = src.GetProjection()
    #     src_gt = src.GetGeoTransform()
    #     src_x = src.RasterXSize
    #     src_y = src.RasterYSize
    #     dtype = src.GetRasterBand(1).DataType
    #     # spatial ref
    #     src_sr = osr.SpatialReference(wkt=src_proj)
    #     src_epsg = src_sr.GetAttrValue("AUTHORITY", 1)
    #
    #     # distination
    #     # spatial ref
    #     dst_epsg = osr.SpatialReference()
    #     dst_epsg.ImportFromEPSG(to_epsg)
    #     # transformation factors
    #     tx = osr.CoordinateTransformation(src_sr, dst_epsg)
    #
    #     # incase the source crs is GCS and longitude is in the west hemisphere gdal
    #     # reads longitude fron 0 to 360 and transformation factor wont work with valeus
    #     # greater than 180
    #     if src_epsg == "4326" and src_gt[0] > 180:
    #         lng_new = src_gt[0] - 360
    #         # transform the right upper corner point
    #         (ulx, uly, ulz) = tx.TransformPoint(lng_new, src_gt[3])
    #         # transform the right lower corner point
    #         (lrx, lry, lrz) = tx.TransformPoint(
    #             lng_new + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
    #         )
    #     else:
    #         # transform the right upper corner point
    #         (ulx, uly, ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
    #         # transform the right lower corner point
    #         (lrx, lry, lrz) = tx.TransformPoint(
    #             src_gt[0] + src_gt[1] * src_x, src_gt[3] + src_gt[5] * src_y
    #         )
    #
    #     if not cell_size:
    #         # the result raster has the same pixcel size as the source
    #         # check if the coordinate system is GCS convert the distance from angular to metric
    #         if src_epsg == "4326":
    #             coords_1 = (src_gt[3], src_gt[0])
    #             coords_2 = (src_gt[3], src_gt[0] + src_gt[1])
    #             #            pixel_spacing=geopy.distance.vincenty(coords_1, coords_2).m
    #             pixel_spacing = FeatureCollection.GCSDistance(coords_1, coords_2)
    #         else:
    #             pixel_spacing = src_gt[1]
    #     else:
    #         # if src_epsg.GetAttrValue('AUTHORITY', 1) != "4326":
    #         #     assert (cell_size > 1), "please enter cell size greater than 1"
    #         # if the user input a cell size resample the raster
    #         pixel_spacing = cell_size
    #
    #     # create a new raster
    #     cols = int(np.round(abs(lrx - ulx) / pixel_spacing))
    #     rows = int(np.round(abs(uly - lry) / pixel_spacing))
    #     dst = Dataset._create_dataset(cols, rows, 1, dtype, driver="MEM")
    #
    #     # new geotransform
    #     new_geo = (ulx, pixel_spacing, src_gt[2], uly, src_gt[4], -pixel_spacing)
    #     # set the geotransform
    #     dst.SetGeoTransform(new_geo)
    #     # set the projection
    #     dst.SetProjection(dst_epsg.ExportToWkt())
    #     # set the no data value
    #     no_data_value = src.GetRasterBand(1).GetNoDataValue()
    #     dst = Dataset._set_no_data_value(dst, no_data_value)
    #     # perform the projection & resampling
    #     gdal.ReprojectImage(
    #         src, dst, src_sr.ExportToWkt(), dst_epsg.ExportToWkt(), method
    #     )
    #
    #     return dst

    def align(self, alignment_src: Dataset):
        """matchDataAlignment.

        this function matches the coordinate system and the number of rows & columns
        between two rasters
        Raster A is the source of the coordinate system, no of rows and no of columns & cell size
        rasters_dir is path to the folder where Raster B exist where  Raster B is
        the source of data values in cells
        the result will be a raster with the same structure as RasterA but with
        values from RasterB using Nearest neighbor interpolation algorithm

        Parameters
        ----------
        alignment_src: [String]
            path to the spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns) alignment_src should include the name of the raster
            and the extension like "data/dem.tif"

        Returns
        -------
        new rasters:
            ٌRasters have the values from rasters in rasters_dir with the same
            cell size, no of rows & columns, coordinate system and alignment like raster A

        Examples
        --------
        >>> dem_path = "01GIS/inputs/4000/acc4000.tif"
        >>> prec_in_path = "02Precipitation/CHIRPS/Daily/"
        >>> prec_out_path = "02Precipitation/4km/"
        >>> Dataset.align(dem_path,prec_in_path,prec_out_path)
        """
        if not isinstance(alignment_src, Dataset):
            raise TypeError("alignment_src input should be a Dataset object")

        for i in range(self.time_length):
            src = self.iloc(i)
            dst = src.align(alignment_src)
            arr = dst.read_array()
            if i == 0:
                # create the array
                array = (
                    np.ones(
                        (self.time_length, arr.shape[0], arr.shape[1]),
                    )
                    * np.nan
                )

            array[i, :, :] = arr

        self._values = array
        # use the last src as
        self._base = dst

    @staticmethod
    def merge(
        src: List[str],
        dst: str,
        no_data_value: Union[float, int, str] = "0",
        init: Union[float, int, str] = "nan",
        n: Union[float, int, str] = "nan",
    ):
        """merge.

            merges group of rasters into one raster

        Parameters
        ----------
        src: List[str]
            list of the path to all input raster
        dst: [str]
            path to the output raster
        no_data_value: [float/int]
            Assign a specified nodata value to output bands.
        init: [float/int]
            Pre-initialize the output image bands with these values. However, it is not marked as the nodata value
            in the output file. If only one value is given, the same value is used in all the bands.
        n: [float/int]
            Ignore pixels from files being merged in with this pixel value.

        Returns
        -------
        None
        """
        # run the command
        # cmd = "gdal_merge.py -o merged_image_1.tif"
        # subprocess.call(cmd.split() + file_list)
        # vrt = gdal.BuildVRT("merged.vrt", file_list)
        # src = gdal.Translate("merged_image.tif", vrt)

        parameters = (
            ["", "-o", dst]
            + src
            + [
                "-co",
                "COMPRESS=LZW",
                "-init",
                str(init),
                "-a_nodata",
                str(no_data_value),
                "-n",
                str(n),
            ]
        )  # '-separate'
        gdal_merge.main(parameters)

    def apply(self, ufunc: Callable):
        """apply.

        apply a function on each raster in the datacube.

        Parameters
        ----------
        ufunc: [function]
            callable universal function ufunc (builtin or user defined)
            https://numpy.org/doc/stable/reference/ufuncs.html
            - To create an ufunc from a normal function
            (https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html)

        Returns
        -------
        new rasters will be saved to the save_to

        Examples
        --------
        >>> def func(val):
        >>>    return val%2
        >>> ufunc = np.frompyfunc(func, 1, 1)
        >>> dataset.apply(ufunc)
        """
        if not callable(ufunc):
            raise TypeError("The Second argument should be a function")
        arr = self.values
        no_data_value = self.base.no_data_value[0]
        # execute the function on each raster
        arr[~np.isclose(arr, no_data_value, rtol=0.001)] = ufunc(
            arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        )

    def overlay(
        self,
        classes_map,
        exclude_value: Union[float, int] = None,
    ) -> Dict[List[float], List[float]]:
        """Overlay.

        Parameters
        ----------
        classes_map: [Dataset]
            Dataset Object fpr the raster that have classes you want to overlay with the raster.
        exclude_value: [Numeric]
            values you want to exclude from extracted values.

        Returns
        -------
        Dictionary:
            dictionary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path.
        """
        values = {}
        for i in range(self.time_length):
            src = self.iloc(i)
            dict_i = src.overlay(classes_map, exclude_value)

            # these are the distinct values from the BaseMap which are keys in the
            # values dict with each one having a list of values
            classes = list(dict_i.keys())

            for class_i in classes:
                if class_i not in values.keys():
                    values[class_i] = list()

                values[class_i] = values[class_i] + dict_i[class_i]

        return values
