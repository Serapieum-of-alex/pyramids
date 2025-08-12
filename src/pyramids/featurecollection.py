"""
FeatureCollection.

ogr classes: https://gdal.org/java/org/gdal/ogr/package-summary.html
ogr tree: https://gdal.org/java/org/gdal/ogr/package-tree.html.

drivers available in geopandas
gpd.io.file._EXTENSION_TO_DRIVER
"""

import json
import os
import shutil
import tempfile
import uuid
import warnings
from numbers import Number
from typing import Any, Dict, Iterable, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, ogr, osr
from osgeo.ogr import DataSource
from pyproj import Proj, transform
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids._errors import DriverNotExistError
from pyramids._utils import (
    Catalog,
    ogr_ds_togdal_dataset,
    ogr_to_numpy_dtype,
)


CATALOG = Catalog(raster_driver=False)
MEMORY_FILE = "/vsimem/myjson.geojson"
gdal.UseExceptions()


class FeatureCollection:
    """FeatureCollection.

    FeatureCollection class contains different methods to deal with shapefiles

    Methods:
        1- GetXYCoords
        2- GetPointCoords
        3- GetLineCoords
        4- GetPolyCoords
        5- Explode
        6- MultiGeomHandler
        7- GetCoords
        8- XY
        9- CreatePolygon
        10- CreatePoint
        11- CombineGeometrics
        12- GCSDistance
        13- ReprojectPoints
        14- ReprojectPoints_2
        15- AddSpatialReference
        16- PolygonCenterPoint
        17- WriteShapefile
    """

    def __init__(self, gdf: Union[GeoDataFrame, DataSource]):
        """Create a FeatureCollection object."""
        # read the drivers catalog
        self._feature = gdf

    def __str__(self):
        """__str__."""
        message = f"""
            Feature: {self.feature}
        """
        # EPSG: {self.epsg}
        # Variables: {self.variables}
        # Number of Bands: {self.band_count}
        # Band names: {self.band_names}
        # Dimension: {self.rows * self.columns}
        # Mask: {self._no_data_value[0]}
        # Data type: {self.dtype[0]}
        return message

    @property
    def feature(self) -> Union[GeoDataFrame, DataSource]:
        """Geodataframe or DataSource."""
        return self._feature

    @property
    def epsg(self) -> int:
        """EPSG number."""
        return self._get_epsg()

    @property
    def total_bounds(self) -> List[Number]:
        """Bounding coordinates `min-x`, `min-y`, `max-x`, `maxy`."""
        if isinstance(self.feature, GeoDataFrame):
            bounds = self.feature.total_bounds.tolist()
        else:
            bounds = self.feature.GetLayer().GetExtent()
            bounds = [bounds[0], bounds[2], bounds[1], bounds[3]]
        return bounds

    @property
    def top_left_corner(self) -> List[Number]:
        """Top left corner coordinates."""
        if isinstance(self.feature, GeoDataFrame):
            bounds = self.feature.total_bounds.tolist()
        else:
            bounds = self.feature.GetLayer().GetExtent()

        bounds = [bounds[0], bounds[3]]
        return bounds

    @property
    def layers_count(self) -> Union[int, None]:
        """layers_count.

        Number of layers in a datasource.
        """
        if isinstance(self.feature, DataSource) or isinstance(
            self.feature, gdal.Dataset
        ):
            return self.feature.GetLayerCount()
        else:
            return None

    @property
    def layer_names(self) -> List[str]:
        """OGR object layers names'."""
        names = []
        if isinstance(self.feature, DataSource) or isinstance(
            self.feature, gdal.Dataset
        ):
            for i in range(self.layers_count):
                names.append(self.feature.GetLayer(i).GetLayerDefn().GetName())

        return names

    @property
    def column(self) -> List:
        """Column Names."""
        names = []
        if isinstance(self.feature, DataSource) or isinstance(
            self.feature, gdal.Dataset
        ):
            for i in range(self.layers_count):
                layer_dfn = self.feature.GetLayer(i).GetLayerDefn()
                cols = layer_dfn.GetFieldCount()
                names = names + [
                    layer_dfn.GetFieldDefn(j).GetName() for j in range(cols)
                ]
            names = names + ["geometry"]
        else:
            names = self.feature.columns.tolist()
        return names

    @property
    def file_name(self) -> str:
        """Get file name in case of the base object is an ogr.Datasource or gdal.Dataset."""
        if isinstance(self.feature, gdal.Dataset) or isinstance(
            self.feature, DataSource
        ):
            file_name = self.feature.GetFileList()[0]
        else:
            file_name = ""

        return file_name

    @property
    def dtypes(self) -> Dict[str, str]:
        """Data Type.

            - Get the data types of the columns in the vector file.

        Returns
        -------
            list of the data types (strings/numpy datatypes) of the columns in the vector file, except the geometry
            column.
        """
        if isinstance(self.feature, GeoDataFrame):
            dtypes = self.feature.dtypes.to_dict()
            # convert the dtype to string as it returns a dtype object in linux instead.
            dtypes = {key: str(value) for key, value in dtypes.items()}
        else:
            dtypes = []
            for i in range(self.layers_count):
                layer_dfn = self.feature.GetLayer(i).GetLayerDefn()
                cols = layer_dfn.GetFieldCount()
                dtypes = dtypes + [
                    ogr_to_numpy_dtype(layer_dfn.GetFieldDefn(j).GetType()).__name__
                    for j in range(cols)
                ]
            # the geometry column is not in the returned dictionary if the vector is DataSource
            dtypes = {col_i: type_i for col_i, type_i in zip(self.column, dtypes)}

        return dtypes

    @classmethod
    def read_file(cls, path: str):
        """Open a vector dataset using OGR or GeoPandas.

        Args:
            path (str): Path to vector file.

        Returns:
            FeatureCollection: A FeatureCollection wrapping the GeoDataFrame.
        """
        gdf = gpd.read_file(path)

        # update = False if read_only else True
        # ds = ogr.OpenShared(path, update=update)
        # # ds = gdal.OpenEx(path)
        return cls(gdf)

    @staticmethod
    def create_ds(driver: str = "geojson", path: str = None) -> Union[DataSource, None]:
        """Create OGR DataSource.

        Args:
            driver (str): Driver type ["GeoJSON", "memory"].
            path (str): Path to save the vector data.

        Returns:
            DataSource | None: Created OGR DataSource or None if inplace behavior applies elsewhere.
        """
        driver = driver.lower()
        gdal_name = CATALOG.get_gdal_name(driver)

        if gdal_name is None:
            raise DriverNotExistError(f"The given driver:{driver} is not suported.")

        if driver == "memory":
            path = "memData"

        ds = FeatureCollection._create_driver(gdal_name, path)
        return ds

    @staticmethod
    def _create_driver(driver: str, path: str):
        """Create Driver."""
        return ogr.GetDriverByName(driver).CreateDataSource(path)

    @staticmethod
    def _copy_driver_to_memory(ds: DataSource, name: str = "memory") -> DataSource:
        """Copy driver to a memory driver.

        Args:
            ds (DataSource): OGR datasource.
            name (str): Datasource name.

        Returns:
            DataSource: The copied in-memory OGR DataSource.
        """
        return ogr.GetDriverByName("Memory").CopyDataSource(ds, name)

    def to_file(self, path: str, driver: str = "geojson"):
        """Save FeatureCollection to disk.

        Currently saves OGR DataSource to disk.

        Args:
            path (str): Path to save the vector.
            driver (str): Driver type.

        Returns:
            None
        """
        driver_gdal_name = CATALOG.get_gdal_name(driver)
        if isinstance(self.feature, DataSource):
            ogr.GetDriverByName(driver_gdal_name).CopyDataSource(self.feature, path)
        else:
            self.feature.to_file(path, driver=driver_gdal_name)

    def _gdf_to_ds(
        self, inplace: bool = False, gdal_dataset=False
    ) -> Union[DataSource, None]:
        """Convert a GeoPandas GeoDataFrame into an OGR DataSource.

        Args:
            inplace (bool): Convert the GeoDataFrame to DataSource in place. Default is False.
            gdal_dataset (bool): True to convert the GeoDataFrame into a GDAL Dataset (the object created by reading the vector with gdal.OpenEx). Default is False.

        Returns:
            DataSource | FeatureCollection | None: OGR DataSource, or a FeatureCollection wrapper if not inplace.
        """
        if isinstance(self.feature, GeoDataFrame):
            gdf_json = json.loads(self.feature.to_json())
            geojson_str = json.dumps(gdf_json)
            # Use the /vsimem/ (Virtual File Systems) to write the GeoJSON string to memory
            gdal.FileFromMemBuffer(MEMORY_FILE, geojson_str)
            # Use OGR to open the GeoJSON from memory
            if not gdal_dataset:
                drv = ogr.GetDriverByName("GeoJSON")
                ds = drv.Open(MEMORY_FILE)
            else:
                ds = gdal.OpenEx(MEMORY_FILE)
        else:
            ds = self.feature

        if inplace:
            self.__init__(ds)
            ds = None
        else:
            ds = FeatureCollection(ds)

        return ds

    # def _gdf_to_ds_copy(self, inplace=False) -> DataSource:
    #     """Convert ogr DataSource object to a GeoDataFrame.
    #
    #     Returns
    #     -------
    #     ogr.DataSource
    #     """
    #     # Create a temporary directory for files.
    #     temp_dir = tempfile.mkdtemp()
    #     new_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.geojson")
    #     if isinstance(self.feature, GeoDataFrame):
    #         self.feature.to_file(new_vector_path)
    #         ds = ogr.Open(new_vector_path)
    #         # ds = FeatureCollection(ds)
    #         ds = FeatureCollection._copy_driver_to_memory(ds)
    #     else:
    #         ds = FeatureCollection._copy_driver_to_memory(self.feature)
    #
    #     if inplace:
    #         self.__init__(ds)
    #         ds = None
    #
    #     return ds

    def _ds_to_gdf_with_io(self, inplace: bool = False) -> GeoDataFrame:
        """Convert ogr DataSource object to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
        """
        # Create a temporary directory for files.
        temp_dir = tempfile.mkdtemp()
        new_vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.geojson")
        self.to_file(new_vector_path, driver="geojson")
        gdf = gpd.read_file(new_vector_path)

        shutil.rmtree(temp_dir, ignore_errors=True)
        if inplace:
            self.__init__(gdf)
            gdf = None

        return gdf

    def _ds_to_gdf_in_memory(self, inplace: bool = False) -> GeoDataFrame:
        """Convert ogr DataSource object to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
        """
        gdal_ds = ogr_ds_togdal_dataset(self.feature)
        layer_name = gdal_ds.GetLayer().GetName()  # self.layer_names[0]
        gdal.VectorTranslate(
            MEMORY_FILE,
            gdal_ds,
            SQLStatement=f"SELECT * FROM {layer_name}",
            layerName=layer_name,
        )
        # import fiona
        # from fiona import MemoryFile
        # f = MemoryFile(MEMORY_FILE)
        # f = fiona.Collection(MEMORY_FILE)

        # f = fiona.open(MEMORY_FILE, driver='geojson')
        # gdf = gpd.GeoDataFrame.from_features(f, crs=f.crs)
        gdf = gpd.read_file(MEMORY_FILE, layer=layer_name, driver="geojson")

        if inplace:
            self.__init__(gdf)
            gdf = None

        return gdf

    def _ds_to_gdf(self, inplace: bool = False) -> GeoDataFrame:
        """Convert ogr DataSource object to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
        """
        try:
            gdf = self._ds_to_gdf_in_memory(inplace=inplace)
        except:  # pragma: no cover
            # keep the exception unspecified and we want to catch fiona.errors.DriverError but we do not want to
            # explicitly import fiona here
            gdf = self._ds_to_gdf_with_io(inplace=inplace)

        return gdf

    def to_dataset(
        self,
        cell_size: Any = None,
        dataset=None,
        column_name: Union[str, List[str]] = None,
    ):
        """Covert a vector into raster.

            - The raster cell values will be taken from the column name given in the vector_filed in the vector file.
            - all the new raster geotransform data will be copied from the given raster.
            - raster and vector should have the same projection

        Args:
            cell_size (int | None): Cell size for the new raster. Optional if dataset is provided. Default is None.
            dataset (Dataset | None): Raster object to copy geotransform (projection, rows, columns, location) from. Optional if cell_size is provided. Default is None.
            column_name (str | List[str] | None): Column name(s) in the vector to burn values from. If None, all columns are considered as bands. Default is None.

        Returns:
            Dataset: Single-band raster with vector geometries burned.
        """
        from pyramids.dataset import Dataset

        if cell_size is None and dataset is None:
            raise ValueError("You have to enter either cell size of Dataset object")

        # Check EPSG are same, if not reproject vector.
        ds_epsg = self.epsg
        if dataset is not None:
            if dataset.epsg != ds_epsg:
                raise ValueError(
                    f"Dataset and vector are not the same EPSG. {dataset.epsg} != {ds_epsg}"
                )

        # TODO: this case
        if dataset is not None:
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    "The second parameter should be a Dataset object (check how to read a raster using the "
                    "Dataset module)"
                )
            # if the raster is given, the top left corner of the raster will be taken as the top left corner for
            # the rasterized polygon
            xmin, ymax = dataset.top_left_corner
            no_data_value = (
                dataset.no_data_value[0]
                if dataset.no_data_value[0] is not None
                else np.nan
            )
            rows = dataset.rows
            columns = dataset.columns
            cell_size = dataset.cell_size
        else:
            # if a raster is not given, the xmin and ymax will be taken as the top left corner for the rasterized
            # polygon.
            xmin, ymin, xmax, ymax = self.feature.total_bounds
            no_data_value = Dataset.default_no_data_value
            columns = int(np.ceil((xmax - xmin) / cell_size))
            rows = int(np.ceil((ymax - ymin) / cell_size))

        burn_values = None
        if column_name is None:
            column_name = self.column
            column_name.remove("geometry")

        if isinstance(column_name, list):
            numpy_dtype = self.dtypes[column_name[0]]
        else:
            numpy_dtype = self.dtypes[column_name]

        dtype = str(numpy_dtype)
        attribute = column_name

        # convert the vector to a gdal Dataset (vector but read by gdal.EX)
        vector_gdal_ex = self._gdf_to_ds(gdal_dataset=True)
        top_left_corner = (xmin, ymax)

        bands_count = 1 if not isinstance(attribute, list) else len(attribute)
        dataset_n = Dataset.create(
            cell_size,
            rows,
            columns,
            dtype,
            bands_count,
            top_left_corner,
            ds_epsg,
            no_data_value,
        )

        bands = list(range(1, bands_count + 1))
        # loop over bands
        for ind, band in enumerate(bands):
            rasterize_opts = gdal.RasterizeOptions(
                bands=[band],
                burnValues=burn_values,
                attribute=attribute[ind] if isinstance(attribute, list) else attribute,
                allTouched=True,
            )
            # if the second parameter to the Rasterize function is str, it will be read using gdal.OpenEX inside the
            # function, so if the second parameter is not str, it should be a dataset, if you try to use ogr.DataSource
            # it will give an error.
            # the second parameter can be given as a path, or read the vector using gdal.OpenEX and use it as a
            # second parameter.
            _ = gdal.Rasterize(
                dataset_n.raster, vector_gdal_ex.feature, options=rasterize_opts
            )

        return dataset_n

    @staticmethod
    def _get_ds_epsg(ds: DataSource):
        """Get EPSG for a given OGR DataSource.

        Args:
            ds (DataSource): OGR datasource (vector file read by OGR).

        Returns:
            int: EPSG number.
        """
        layer = ds.GetLayer(0)
        spatial_ref = layer.GetSpatialRef()
        spatial_ref.AutoIdentifyEPSG()
        epsg = int(spatial_ref.GetAuthorityCode(None))
        return epsg

    @staticmethod
    def _create_sr_from_proj(prj: str, string_type: str = None):
        """Create a spatial reference object from projection.

        Args:
            prj (str): Projection string, e.g.,
                >>> "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AXIS[\"Latitude\",NORTH],AXIS[\"Longitude\",EAST],AUTHORITY[\"EPSG\",\"4326\"]]"
            string_type (str): Type of the string ["ESRI wkt", "WKT", "PROj4"].
        """
        srs = osr.SpatialReference()

        if string_type is None:
            srs.ImportFromWkt(prj)
        elif prj.startswith("PROJCS") or prj.startswith("GEOGCS"):
            # ESRI well know text strings,
            srs.ImportFromESRI([prj])
        else:
            # Proj4 strings.
            srs.ImportFromProj4(prj)

        return srs

    @staticmethod
    def get_epsg_from_prj(prj: str) -> int:
        """Create spatial reference from the projection then auto identify the epsg using the osr object.

        Args:
            prj (str): Projection string.

        Returns:
            int: epsg number

        Examples:
            - Get EPSG from a dataset projection:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> src = Dataset.read_file("path/to/raster.tif")
              >>> prj = src.GetProjection()
              >>> epsg = FeatureCollection.get_epsg_from_prj(prj)

              ```
        """
        if prj != "":
            srs = FeatureCollection._create_sr_from_proj(prj)
            try:
                # if could not identify epsg use the authority code.
                response = srs.AutoIdentifyEPSG()
            except RuntimeError:
                response = 6

            if response == 0:
                epsg = int(srs.GetAuthorityCode(None))
            else:
                # the GetAuthorityCode failed to identify the epsg number https://gdal.org/doxygen/classOGRSpatialReference.html
                # srs.FindMatches()
                epsg = int(srs.GetAttrValue("AUTHORITY", 1))
        else:
            epsg = 4326
        return epsg

    @staticmethod
    def _get_gdf_epsg(gdf: GeoDataFrame):
        """Get epsg for a given geodataframe.

        Args:
            gdf (GeoDataFrame): Vector file read by geopandas.

        Returns:
            int: epsg number
        """
        return gdf.crs.to_epsg()

    def _get_epsg(self) -> int:
        """getEPSG.

        Returns:
            int: epsg number
        """
        vector_obj = self.feature
        if isinstance(vector_obj, ogr.DataSource):
            epsg = FeatureCollection._get_ds_epsg(vector_obj)
        elif isinstance(vector_obj, gpd.GeoDataFrame):
            epsg = FeatureCollection._get_gdf_epsg(vector_obj)
        else:
            raise ValueError(
                f"Unable to get EPSG from: {type(vector_obj)}, only ogr.Datasource and "
                "geopandas.GeoDataFrame are supported"
            )
        return epsg

    @staticmethod
    def _get_xy_coords(geometry, coord_type: str) -> List:
        """getXYCoords.

           Returns either x or y coordinates from geometry coordinate sequence.
           Used with LineString and Polygon geometries.

        Args:
            geometry (LineString): The geometry of a shapefile.
            coord_type (str): Either "x" or "y".

        Returns:
            array: Contains x coordinates or y coordinates of all edges of the shapefile
        """
        if coord_type == "x":
            coords = geometry.coords.xy[0].tolist()
        elif coord_type == "y":
            coords = geometry.coords.xy[1].tolist()
        else:
            raise ValueError("coord_type can only have a value of 'x' or 'y' ")

        return coords

    @staticmethod
    def _get_point_coords(geometry: Point, coord_type: str) -> Union[float, int]:
        """Get point coordinates for a Point geometry.

        Returns coordinates of a Point object.

        Args:
            geometry (Point): The geometry of a shapefile.
            coord_type (str): Either "x" or "y".

        Returns:
            float | int: The x or y coordinate of the Point according to coord_type.
        """
        if coord_type == "x":
            coord = geometry.x
        elif coord_type == "y":
            coord = geometry.y
        else:
            raise ValueError("coord_type can only have a value of 'x' or 'y' ")

        return coord

    @staticmethod
    def _get_line_coords(geometry: LineString, coord_type: str):
        """Get coordinates of a LineString object.

        Args:
            geometry (LineString): The geometry of a shapefile.
            coord_type (str): Either "x" or "y".

        Returns:
            list: Contains x or y coordinates of all edges of the shapefile.
        """
        return FeatureCollection._get_xy_coords(geometry, coord_type)

    @staticmethod
    def _get_poly_coords(geometry: Polygon, coord_type: str) -> List:
        """Get coordinates of a Polygon's exterior.

        Args:
            geometry (Polygon): The geometry of a shapefile.
            coord_type (str): Either "x" or "y".

        Returns:
            list: Contains x or y coordinates of all edges of the shapefile.
        """
        # convert the polygon into lines
        ext = geometry.exterior  # type = LinearRing

        return FeatureCollection._get_xy_coords(ext, coord_type)

    @staticmethod
    def _explode_multi_geometry(multi_polygon: MultiPolygon):
        """Explode a MultiPolygon into its Polygon parts.

        Args:
            multi_polygon (MultiPolygon): A MultiPolygon geometry.

        Returns:
            list: List of Polygon geometries.
        """
        # outdf = gpd.GeoDataFrame()
        # multdf = gpd.GeoDataFrame()
        # multdf["geometry"] = list(multi_polygon)
        # n_rows = len(multi_polygon)
        # multdf = multdf.append([multi_polygon] * n_rows, ignore_index=True)
        # for i, poly in enumerate(multi_polygon):
        #     multdf.loc[i, "geometry"] = poly
        return list(multi_polygon)

    @staticmethod
    def _explode_gdf(gdf: GeoDataFrame, geometry: str = "multipolygon"):
        """Explode Multi geometries into single geometries.

        Explodes MultiPolygon (or specified multi-geometry) into separate geometries per row.

        Args:
            gdf (GeoDataFrame): GeoDataFrame to explode.
            geometry (str): The multi-geometry type to explode. Default is "multipolygon".

        Returns:
            GeoDataFrame: A new GeoDataFrame with exploded geometries.
        """
        # explode the multi_polygon into polygon
        new_gdf = gpd.GeoDataFrame()
        to_drop = []
        for idx, row in gdf.iterrows():
            geom_type = row.geometry.geom_type.lower()
            if geom_type == geometry:
                # get number of the polygons inside the multipolygon class
                n_rows = len(row.geometry.geoms)
                new_gdf = gpd.GeoDataFrame(pd.concat([new_gdf] + [row] * n_rows))
                new_gdf.reset_index(drop=True, inplace=True)
                new_gdf.columns = row.index.values
                # for each rows assign each polygon
                for geom in range(n_rows):
                    new_gdf.loc[geom, "geometry"] = row.geometry.geoms[geom]
                to_drop.append(idx)

        # drop the exploded rows
        gdf.drop(labels=to_drop, axis=0, inplace=True)
        # concatinate the exploded rows
        new_gdf = gpd.GeoDataFrame(pd.concat([gdf] + [new_gdf]))
        new_gdf.reset_index(drop=True, inplace=True)
        new_gdf.columns = gdf.columns
        return new_gdf

    @staticmethod
    def _multi_geom_handler(
        multi_geometry: Union[MultiPolygon, MultiPoint, MultiLineString],
        coord_type: str,
        geom_type: str,
    ):
        """Handle multi-geometries by merging coordinates.

        Function for handling multi-geometries (MultiPoint, MultiLineString, MultiPolygon).
        Returns a list of coordinates where all parts are merged into a single list; individual geometries are separated with np.nan.

        Args:
            multi_geometry (MultiPolygon | MultiPoint | MultiLineString): The geometry of a shapefile.
            coord_type (str): Either "x" or "y".
            geom_type (str): "MultiPoint" or "MultiLineString" or "MultiPolygon".

        Returns:
            list: Contains x or y coordinates of all edges of the shapefile.
        """
        coord_arrays = []
        geom_type = geom_type.lower()
        if geom_type == "multipoint" or geom_type == "multilinestring":
            for i, part in enumerate(multi_geometry.geoms):
                if geom_type == "multipoint":
                    vals = FeatureCollection._get_point_coords(part, coord_type)
                    coord_arrays.append(vals)
                elif geom_type == "multilinestring":
                    vals = FeatureCollection._get_line_coords(part, coord_type)
                    coord_arrays.append(vals)
        elif geom_type == "multipolygon":
            for i, part in enumerate(multi_geometry.geoms):
                # multi_2_single = FeatureCollection._explode(part) if part.type.startswith("MULTI") else part
                vals = FeatureCollection._get_poly_coords(part, coord_type)
                coord_arrays.append(vals)

        return coord_arrays

    @staticmethod
    def _get_coords(row, geom_col: str, coord_type: str):
        """Get coordinates ('x' or 'y') of a geometry row.

        Returns coordinates for Point, LineString, or Polygon as a list. Can also handle Multi geometries (not MultiPolygon) appropriately.

        Args:
            row (pd.Series): A whole row of the GeoDataFrame.
            geom_col (str): Name of the column where the geometry is stored in the dataframe.
            coord_type (str): "x" or "y" to choose which coordinate to get.

        Returns:
            list | int: Coordinates or -9999 for multipolygon to mark for removal.
        """
        # get geometry object
        geom = row[geom_col]
        # check the geometry type
        gtype = geom.geom_type.lower()
        # "Normal" geometries
        if gtype == "point":
            return FeatureCollection._get_point_coords(geom, coord_type)
        elif gtype == "linestring":
            return list(FeatureCollection._get_line_coords(geom, coord_type))
        elif gtype == "polygon":
            return list(FeatureCollection._get_poly_coords(geom, coord_type))
        elif gtype == "multipolygon":
            # the multipolygon geometry row will be deleted in the xy method
            return -9999
        elif gtype == "geometrycollection":
            return FeatureCollection._geometry_collection(geom, coord_type)
        # Multi geometries
        else:
            return FeatureCollection._multi_geom_handler(geom, coord_type, gtype)

    def xy(self) -> None:
        """Compute x and y coordinates of all vertices.

        Processes the geometry column of the GeoDataFrame and returns the x and y coordinates of all the vertices.

        Returns:
            None
        """
        # explode the gdf if the Geometry of type MultiPolygon
        gdf = self._explode_gdf(self._feature, geometry="multipolygon")
        gdf = self._explode_gdf(gdf, geometry="geometrycollection")
        self._feature = gdf

        # get the x & y coordinates of the exploded multi_polygons
        self._feature["x"] = self._feature.apply(
            self._get_coords, geom_col="geometry", coord_type="x", axis=1
        )
        self._feature["y"] = self._feature.apply(
            self._get_coords, geom_col="geometry", coord_type="y", axis=1
        )

        to_delete = np.where(self._feature["x"] == -9999)[0]
        self._feature.drop(to_delete, inplace=True)
        self._feature.reset_index(drop=True, inplace=True)

    @staticmethod
    def create_polygon(
        coords: List[Tuple[float, float]], wkt: bool = False
    ) -> Union[str, Polygon]:
        """Create a polygon geometry from coordinates.

        Args:
            coords (List[Tuple[float, float]]): List of (x, y) tuples.
            wkt (bool): True to return Well-Known Text (WKT) string; False to return a Shapely Polygon object.

        Returns:
            str | Polygon: WKT string if wkt is True; otherwise a Shapely Polygon object.

        Examples:
            - Create a WKT polygon from coordinates and print it:

              ```python
              >>> coordinates = [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]
              >>> feature_collection = FeatureCollection.create_polygon(coordinates, wkt=True)
              >>> print(feature_collection)
              'POLYGON ((-106.64 24, -106.49 24.05, -106.49 24.01, -106.49 23.98, -106.64 24))'

              ```

            - Create a Shapely Polygon and assign it to a GeoDataFrame:

              ```python
              >>> new_geometry = gpd.GeoDataFrame()
              >>> new_geometry.loc[0,'geometry'] = FeatureCollection.create_polygon(coordinates, wkt=False)

              ```
        """
        poly = Polygon(coords)
        if wkt:
            return poly.wkt
        else:
            return poly

    @staticmethod
    def create_point(
        coords: Iterable[Tuple[float]], epsg: int = None
    ) -> Union[List[Point], GeoDataFrame]:
        """Create Shapely Point objects from coordinate tuples.

        Args:
            coords (Iterable[Tuple[float]]): List of tuples [(x1, y1), (x2, y2)] or [(lon1, lat1), (lon2, lat1)].
            epsg (int): EPSG number for coordinates. If provided, returns a GeoDataFrame wrapped as FeatureCollection.

        Returns:
            list | FeatureCollection: List of Shapely Point objects, or FeatureCollection if epsg is provided.

        Examples:
            - Create points and assign to a GeoDataFrame:

              ```python
              >>> coordinates = [(24.95, 60.16), (24.95, 60.16), (24.95, 60.17), (24.95, 60.16)]
              >>> point_list = FeatureCollection.create_point(coordinates)
              >>> new_geometry = gpd.GeoDataFrame()
              >>> new_geometry.loc[:, 'geometry'] = point_list

              ```
        """
        points = list(map(Point, coords))

        if epsg is not None:
            points = gpd.GeoDataFrame(columns=["geometry"], data=points, crs=epsg)
            points = FeatureCollection(points)

        return points

    def concate(self, gdf: GeoDataFrame, inplace: bool = False):
        """Concatenate two shapefiles into one object.

        Args:
            gdf (GeoDataFrame): GeoDataFrame containing the geometries to combine.
            inplace (bool): If True, modifies the current object in place. Default is False.

        Returns:
            GeoDataFrame | None: New combined GeoDataFrame, or None if inplace is True.

        Examples:
            - Concatenate two GeoDataFrames:

              ```python
              >>> subbasins = FeatureCollection.read_file("sub-basins.shp")
              >>> new_sub = gpd.read_file("new-sub-basins.shp")
              >>> all_subs = subbasins.concate(new_sub, new_sub, inplace=False)

              ```
        """
        # concatenate the second shapefile into the first shapefile
        new_gdf = gpd.GeoDataFrame(pd.concat([self.feature, gdf]))
        # re-index the data frame
        new_gdf.index = [i for i in range(len(new_gdf))]
        # take the spatial reference of the first geodataframe
        new_gdf.crs = self.feature.crs
        if inplace:
            self.__init__(new_gdf)
            new_gdf = None

        return new_gdf

    # @staticmethod
    # def gcs_distance(coords_1: tuple, coords_2: tuple):
    #     """GCS_distance.
    #
    #     this function calculates the distance between two points that have
    #     geographic coordinate system
    #
    #     parameters
    #     ----------
    #     coords_1: [Tuple]
    #         tuple of (long, lat) of the first point
    #     coords_2: [Tuple]
    #         tuple of (long, lat) of the second point
    #
    #     Returns
    #     -------
    #     distance between the two points
    #
    #     Examples
    #     --------
    #     >>> point_1 = (52.22, 21.01)
    #     >>> point_2 = (52.40, 16.92)
    #     >>> distance = FeatureCollection.gcs_distance(point_1, point_2)
    #     """
    #     import_error_msg = f"The triggered function requires geopy package to be install, please install is manually"
    #     import_geopy(import_error_msg)
    #     import geopy.distance as distance
    #     dist = distance.vincenty(coords_1, coords_2).m
    #
    #     return dist

    @staticmethod
    def reproject_points(
        lat: list,
        lon: list,
        from_epsg: int = 4326,
        to_epsg: int = 3857,
        precision: int = 6,
    ):
        """reproject_points.

        this function change the projection of the coordinates from a coordinate system
        to another (default from GCS to web mercator used by google maps)

        Parameters
        ----------
        lat: [list]
            list of latitudes of the points
        lon: [list]
            list of longitude of the points
        from_epsg: [integer]
            reference number to the projection of the points (https://epsg.io/)
        to_epsg: [integer]
            reference number to the new projection of the points (https://epsg.io/)
        precision: [integer]
            number of decimal places

        Returns
        -------
        y: [list]
            list of y coordinates of the points
        x: [list]
            list of x coordinates of the points

        Examples
        --------
        # from web mercator to GCS WGS64:
        >>> x_coords = [-8418583.96378159, -8404716.499972705]
        >>> y_coords = [529374.3212213353, 529374.3212213353]
        >>>  longs, lats = FeatureCollection.reproject_points(y_coords, x_coords, from_epsg=3857, to_epsg=4326)
        """
        # Proj gives a future warning however the from_epsg argument to the functiuon
        # is correct the following couple of code lines are to disable the warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            from_epsg = "epsg:" + str(from_epsg)
            inproj = Proj(init=from_epsg)  # GCS geographic coordinate system
            to_epsg = "epsg:" + str(to_epsg)
            outproj = Proj(init=to_epsg)  # WGS84 web mercator

        x = np.ones(len(lat)) * np.nan
        y = np.ones(len(lat)) * np.nan

        for i in range(len(lat)):
            x[i], y[i] = np.round(
                transform(inproj, outproj, lon[i], lat[i], always_xy=True), precision
            )

        return y, x

    @staticmethod
    def reproject_points2(
        lat: list, lng: list, from_epsg: int = 4326, to_epsg: int = 3857
    ):
        """reproject_points.

        this function change the projection of the coordinates from a coordinate system
        to another (default from GCS to web mercator used by google maps)

        PArameters
        ----------
        lat: [list]
            list of latitudes of the points
        lng: [list]
            list of longitude of the points
        from_epsg: [int]
            integer reference number to the projection of the points (https://epsg.io/)
        to_epsg: [int]
            integer reference number to the new projection of the points (https://epsg.io/)

        Returns
        -------
        x: [list]
            list of x coordinates of the points
        y: [list]
            list of y coordinates of the points

        Examples
        --------
        # from web mercator to GCS WGS64:
        >>> x_coords = [-8418583.96378159, -8404716.499972705]
        >>> y_coords = [529374.3212213353, 529374.3212213353]
        >>> longs, lats = FeatureCollection.reproject_points2(y_coords, x_coords, from_epsg=3857, to_epsg=4326)
        """
        source = osr.SpatialReference()
        source.ImportFromEPSG(from_epsg)

        target = osr.SpatialReference()
        target.ImportFromEPSG(to_epsg)

        transform = osr.CoordinateTransformation(source, target)
        x = []
        y = []
        for i in range(len(lat)):
            point = ogr.CreateGeometryFromWkt(
                "POINT (" + str(lng[i]) + " " + str(lat[i]) + ")"
            )
            point.Transform(transform)
            x.append(point.GetPoints()[0][0])
            y.append(point.GetPoints()[0][1])
        return x, y

    def center_point(
        self,
    ):
        """PolygonCenterPoint.

        PolygonCenterPoint function takes the a geodata frame of polygons and and
        returns the center of each polygon

        Returns
        -------
        saveIng the shapefile or CenterPointDataFrame :
            If you choose True in the "save" input the function will save the
            shapefile in the given "savePath"
            If you choose False in the "save" input the function will return a
            [geodataframe] dataframe containing CenterPoint DataFrame
            you can save it as a shapefile using
            CenterPointDataFrame.to_file("Anyname.shp")


        Examples
        --------
        Return a geodata frame
        >>> sub_basins = gpd.read_file("inputs/sub_basins.shp")
        >>> CenterPointDataFrame = FeatureCollection.polygon_center_point(sub_basins, save=False)
        save a shapefile
        >>> sub_basins = gpd.read_file("Inputs/sub_basins.shp")
        >>> FeatureCollection.center_point(sub_basins, save=True, save_path="centerpoint.shp")
        """
        # get the X, Y coordinates of the points of the polygons and the multipolygons
        self.xy()
        poly = self.feature
        # calculate the average X & Y coordinate for each geometry in the shapefile
        for i, row_i in poly.iterrows():
            poly.loc[i, "avg_x"] = np.mean(row_i["x"])
            poly.loc[i, "avg_y"] = np.mean(row_i["y"])

        coords_list = zip(poly["avg_x"].tolist(), poly["avg_y"].tolist())
        poly["center_point"] = FeatureCollection.create_point(coords_list)

        return poly
