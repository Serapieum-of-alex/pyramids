"""
ogr classes: https://gdal.org/java/org/gdal/ogr/package-summary.html
ogr tree: https://gdal.org/java/org/gdal/ogr/package-tree.html.

drivers available in geopandas
gpd.io.file._EXTENSION_TO_DRIVER
"""
import os
import tempfile
import uuid
import json
import shutil
import warnings
from typing import List, Tuple, Union

import geopandas as gpd
import geopy.distance as distance
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr, osr, gdal
from osgeo.ogr import DataSource
from pyproj import Proj, transform
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multilinestring import MultiLineString
from pyramids.utils import Catalog
from pyramids.errors import DriverNotExistError

CATALOG = Catalog(raster_driver=False)


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

    def __init__(self, gdf: GeoDataFrame):
        # read the drivers catalog
        self._feature = gdf

    def __str__(self):
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
    def feature(self):
        """GeoDataFrame or DataSource"""
        return self._feature

    @property
    def epsg(self):
        """EPSG number"""
        return self._get_epsg()

    @property
    def bounds(self):
        """bounding coordinates"""
        return list(self.feature.bounds.values[0])

    @classmethod
    def read_file(cls, path: str):
        """Open a vector dataset using OGR or GeoPandas.

        Parameters
        ----------
        path : str
            Path to vector file.

        Returns
        -------
        GeoDataFrame if ``with_geopandas`` else OGR datsource.
        """
        gdf = gpd.read_file(path)

        # update = False if read_only else True
        # ds = ogr.OpenShared(path, update=update)
        # # ds = gdal.OpenEx(path)
        return cls(gdf)

    @staticmethod
    def create_ds(driver: str = "geojson", path: str = None) -> Union[DataSource, None]:
        """Create ogr DataSource.

        Parameters
        ----------
        driver: [str]
            driver type ["GeoJSON", "memory"]
        path: [str]
            path to save the vector driver.

        Returns
        -------
        ogr DataSource
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
        """Create Driver"""
        return ogr.GetDriverByName(driver).CreateDataSource(path)

    @staticmethod
    def _copy_driver_to_memory(ds: DataSource, name: str = "memory") -> DataSource:
        """copyDriverToMemory.

            copy driver to a memory driver

        Parameters
        ----------
        ds: [ogr.DataSource]
            ogr datasource
        name: [str]
            datasource name

        Returns
        -------
        ogr.DataSource
        """
        return ogr.GetDriverByName("Memory").CopyDataSource(ds, name)

    def to_file(self, path: str, driver: str = "geojson"):
        """Save FeatureCollection to disk.

            Currently saves ogr DataSource to disk

        Parameters
        ----------
        path: [path]
            path to save the vector
        driver: [str]
            driver type

        Returns
        -------
        None
        """
        driver_gdal_name = CATALOG.get_gdal_name(driver)
        if isinstance(self.feature, DataSource):
            ogr.GetDriverByName(driver_gdal_name).CopyDataSource(self.feature, path)
        else:
            self.feature.to_file(path, driver=driver_gdal_name)

    def _gdf_to_ds(self, inplace: bool = False) -> Union[DataSource, None]:
        """convert a geopandas geodataframe into ogr DataSource.

        Parameters
        ----------
        inplace: [bool]
            convert the geodataframe to datasource inplace. Default is False.

        Returns
        -------
        ogr.DataSource
        """
        # if isinstance(self.feature, GeoDataFrame):
        ds = ogr.Open(self.feature.to_json())

        if inplace:
            self.__init__(ds)
            ds = None
        return ds

    # def _gdf_to_ds(self, inplace=False) -> DataSource:
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
    #         ds = FeatureCollection.read_file(new_vector_path, engine="ogr")
    #         ds = FeatureCollection._copy_driver_to_memory(ds.feature)
    #     else:
    #         ds = FeatureCollection._copy_driver_to_memory(self.feature)
    #
    #     if inplace:
    #         self.__init__(ds, engine="ogr")
    #         ds = None
    #
    #     return ds

    def _ds_to_gdf(self, inplace: bool = False) -> GeoDataFrame:
        """Convert ogr DataSource object to a GeoDataFrame.

        Returns
        -------
        GeoDataFrame
        """
        # # TODO: not complete yet the function needs to take an ogr.DataSource and then write it to disk and then read
        # #  it using the gdal.OpenEx as below
        # # but this way if i write the vector to disk i can just read it ysing geopandas as df directly.
        # # https://gis.stackexchange.com/questions/227737/python-gdal-ogr-2-x-read-vectors-with-gdal-openex-or-ogr-open
        #
        # # read the vector using gdal not ogr
        # ds = gdal.OpenEx(path)  # , gdal.OF_READONLY
        # layer = ds.GetLayer(0)
        # layer_name = layer.GetName()
        # mempath = "/vsimem/test.geojson"
        # # convert the vector read as a gdal dataset to memory
        # # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.VectorTranslateOptions
        # gdal.VectorTranslate(mempath, ds)  # , SQLStatement=f"SELECT * FROM {layer_name}", layerName=layer_name
        # # reading the memory file using fiona
        # f = fiona.open(mempath, driver='geojson')
        # gdf = gpd.GeoDataFrame.from_features(f, crs=f.crs)

        # till i manage to do the above way just write the ogr.DataSource to disk and then read it using geopandas

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

    def to_dataset(
        self,
        cell_size: int = None,
        src=None,
        vector_field=None,
    ) -> Union[None, gdal.Dataset]:
        """Covert a vector into raster.

            - The raster cell values will be taken from the column name given in the vector_filed in the vector file.
            - all the new raster geotransform data will be copied from the given raster.
            - raster and vector should have the same projection

        Parameters
        ----------
        cell_size: [int]
            cell size.
        src : [gdal Dataset]
            Raster object, the raster will only be used as a source for the geotransform (
            projection, rows, columns, location) data to be copied to the rasterized vector.
        vector_field : str or None
            Name of a field in the vector to burn values from. If None, all vector
            features are burned with a constant value of 1.

        Returns
        -------
        gdal.Dataset
            Single band raster with vector geometries burned.
        """
        from pyramids.dataset import Dataset

        if cell_size is None and src is None:
            raise ValueError("You have to enter either cell size of Dataset object")

        # Check EPSG are same, if not reproject vector.
        ds_epsg = self.epsg
        if src is not None:
            if src.epsg != ds_epsg:
                # TODO: reproject the vector to the raster projection instead of raising an error.
                raise ValueError(
                    f"Dataset and vector are not the same EPSG. {src.epsg} != {ds_epsg}"
                )

        if src is not None:
            if not isinstance(src, Dataset):
                raise TypeError(
                    "The second parameter should be a Dataset object (check how to read a raster using the "
                    "Dataset module)"
                )
            # if the raster is given the top left corner of the raster will be taken as the top left corner for
            # the rasterized polygon
            xmin, _, _, ymax, _, _ = src.geotransform
        else:
            # if a raster is not given the xmin and ymax will be taken as the top left corner for the rasterized
            # polygon.
            xmin, ymin, xmax, ymax = self.feature.bounds.values[0]

        if vector_field is None:
            # Use a constant value for all features.
            burn_values = [1]
            attribute = None
            dtype = 5
        else:
            # Use the values given in the vector field.
            burn_values = None
            attribute = vector_field
            dtype = 7

        if isinstance(src, Dataset):
            no_data_value = src.no_data_value[0]
            rows = src.rows
            columns = src.columns
            cell_size = src.cell_size
        else:
            no_data_value = Dataset.default_no_data_value
            columns = int(np.ceil((xmax - xmin) / cell_size))
            rows = int(np.ceil((ymax - ymin) / cell_size))

        # save the geodataframe to disk and get the path
        temp_dir = tempfile.mkdtemp()
        vector_path = os.path.join(temp_dir, f"{uuid.uuid1()}.geojson")
        self.feature.to_file(vector_path)

        top_left_coords = (xmin, ymax)
        # TODO: enable later multi bands
        bands = 1
        src = Dataset.create_driver_from_scratch(
            cell_size,
            rows,
            columns,
            dtype,
            bands,
            top_left_coords,
            ds_epsg,
            no_data_value,
        )

        rasterize_opts = gdal.RasterizeOptions(
            bands=[1], burnValues=burn_values, attribute=attribute, allTouched=True
        )
        # the second parameter to the Rasterize function if str is read using gdal.OpenEX inside the function,
        # so the second parameter if not str should be a dataset, if you try to use ogr.DataSource it will give an error
        # for future trial to remove writing the vector to disk and enter the second parameter as a path, try to find
        # a way to convert the ogr.DataSource or GeoDataFrame into a similar object to the object resulting from
        # gdal.OpenEx which is a dataset
        _ = gdal.Rasterize(src.raster, vector_path, options=rasterize_opts)
        shutil.rmtree(temp_dir, ignore_errors=True)

        return src

    @staticmethod
    def _get_ds_epsg(ds: DataSource):
        """Get epsg for a given ogr Datasource.

        Parameters
        ----------
        ds: [DataSource]
            ogr datasource (vector file read by ogr)

        Returns
        -------
        int:
            epsg number
        """
        layer = ds.GetLayer(0)
        spatial_ref = layer.GetSpatialRef()
        spatial_ref.AutoIdentifyEPSG()
        epsg = int(spatial_ref.GetAuthorityCode(None))
        return epsg

    @staticmethod
    def _create_sr_from_proj(prj: str, string_type: str = None):
        """Create spatial reference object from projection.

        Parameters
        ----------
        prj: [str]
            projection string
            >>> "GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]"
        string_type: [str]
            type of the string ["ESRI wkt", "WKT", "PROj4"]

        Returns
        -------
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
    def get_epsg_from_Prj(prj: str) -> int:
        """create spatial reference from the projection then auto identify the epsg using the osr object.

        Parameters
        ----------
        prj: [str]

        Returns
        -------
        int
            epsg number

        Examples
        --------
        >>> from pyramids.dataset import Dataset
        >>> src = Dataset.read_file("path/to/raster.tif")
        >>> prj = src.GetProjection()
        >>> epsg = FeatureCollection.get_epsg_from_Prj(prj)
        """
        if prj != "":
            srs = FeatureCollection._create_sr_from_proj(prj)
            response = srs.AutoIdentifyEPSG()
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

        Parameters
        ----------
        gdf: [GeoDataFrame]
            vector file read by geopandas

        Returns
        -------
        int:
            epsg number
        """
        return gdf.crs.to_epsg()

    def _get_epsg(self) -> int:
        """getEPSG.

        Returns
        -------
        int:
            epsg number
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

           Returns either x or y coordinates from  geometry coordinate sequence.
           Used with LineString and Polygon geometries.

        Parameters
        ----------
        geometry: [LineString Geometry]
             the geometry of a shpefile
        coord_type: [string]
            either "x" or "y"

        Returns
        -------
        array:
            contains x coordinates or y coordinates of all edges of the shapefile
        """
        if coord_type == "x":
            coords = geometry.coords.xy[0].tolist()
        elif coord_type == "y":
            coords = geometry.coords.xy[1].tolist()

        return coords

    @staticmethod
    def _get_point_coords(geometry: Point, coord_type: str) -> Union[float, int]:
        """GetPointCoords.

        Returns Coordinates of Point object.

        parameters
        ----------
        geometry: [Shapely Point object]
            the geometry of a shpefile
        coord_type: [string]
            either "x" or "y"

        Returns
        -------
        array:
            contains x coordinates or y coordinates of all edges of the shapefile
        """
        if coord_type == "x":
            coord = geometry.x
        if coord_type == "y":
            coord = geometry.y

        return coord

    @staticmethod
    def _get_line_coords(geometry: LineString, coord_type: str):
        """getLineCoords.

        Returns Coordinates of Linestring object.

        parameters
        ----------
        geometry: [Shapely Linestring object]
             the geometry of a shpefile
        coord_type: [string]
            either "x" or "y"

        Returns
        -------
        array:
            contains x coordinates or y coordinates of all edges of the shapefile
        """
        return FeatureCollection._get_xy_coords(geometry, coord_type)

    @staticmethod
    def _get_poly_coords(geometry: Polygon, coord_type: str) -> List:
        """getPolyCoords.

            Returns Coordinates of Polygon using the Exterior of the Polygon.

        parameters
        ----------
        geometry: [Shapely polygon object]
            the geometry of a shpefile
        coord_type: [string]
             either "x" or "y"

        Returns
        -------
        array:
            contains x coordinates or y coordinates of all edges of the shapefile
        """
        # convert the polygon into lines
        ext = geometry.exterior  # type = LinearRing

        return FeatureCollection._get_xy_coords(ext, coord_type)

    @staticmethod
    def _explode(multi_polygon: MultiPolygon):
        """Explode.

        explode function converts the multipolygon into a polygons

        Parameters
        ----------
        multi_polygon: [data frame series]
            the dataframe rows that its geometry type is Multipolygon

        Returns
        -------
        outdf: [dataframe]
            the dataframe of the created polygons
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
    def _multi_geom_handler(
        multi_geometry: Union[MultiPolygon, MultiPoint, MultiLineString],
        coord_type: str,
        geom_type: str,
    ):
        """multiGeomHandler.

        Function for handling multi-geometries. Can be MultiPoint, MultiLineString or MultiPolygon.
        Returns a list of coordinates where all parts of Multi-geometries are merged into a single list.
        Individual geometries are separated with np.nan which is how Bokeh wants them.
        # Bokeh documentation regarding the Multi-geometry issues can be found here (it is an open issue)
        # https://github.com/bokeh/bokeh/issues/2321

        parameters
        ----------
        multi_geometry :[geometry]
            the geometry of a shpefile
        coord_type : [string]
            "string" either "x" or "y"
        geom_type : [string]
            "MultiPoint" or "MultiLineString" or "MultiPolygon"
        Returns
        -------
        array: [ndarray]
            contains x coordinates or y coordinates of all edges of the shapefile
        """
        coord_arrays = []
        geom_type = geom_type.lower()
        if geom_type == "multipoint" or geom_type == "multilinestring":
            for i, part in enumerate(multi_geometry):
                if geom_type == "multipoint":
                    vals = FeatureCollection._get_point_coords(part, coord_type)
                    coord_arrays.append(vals)
                elif geom_type == "multilinestring":
                    vals = FeatureCollection._get_line_coords(part, coord_type)
                    coord_arrays.append(vals)
        elif geom_type == "multipolygon":
            for i, part in enumerate(multi_geometry):
                # multi_2_single = FeatureCollection._explode(part) if part.type.startswith("MULTI") else part
                vals = FeatureCollection._get_poly_coords(part, coord_type)
                coord_arrays.append(vals)
        # for multigeometres that has one geometry inside 'MULTIPOINT (1 1)' the returned value is single not a list
        return coord_arrays

    @staticmethod
    def _get_coords(row, geom_col: str, coord_type: str):
        """getCoords.

        Returns coordinates ('x' or 'y') of a geometry (Point, LineString or Polygon)
        as a list (if geometry is Points, LineString or Polygon). Can handle also
        MultiGeometries but not MultiPolygon.

        parameters
        ----------
        row: [dataframe]
            a whole rwo of the dataframe
        geom_col: [string]
            name of the column where the geometry is stored in the dataframe
        coord_type: [string]
            "X" or "Y" choose which coordinate toy want to get from
            the function

        Returns
        -------
        array:
             contains x coordinates or y coordinates of all edges of the shapefile
        """
        # get geometry object
        geom = row[geom_col]
        # check the geometry type
        gtype = geom.geom_type
        # "Normal" geometries
        if gtype == "Point":
            return FeatureCollection._get_point_coords(geom, coord_type)
        elif gtype == "LineString":
            return list(FeatureCollection._get_line_coords(geom, coord_type))
        elif gtype == "Polygon":
            return list(FeatureCollection._get_poly_coords(geom, coord_type))
        elif gtype == "MultiPolygon":
            return 999
        # Multi geometries
        else:
            return FeatureCollection._multi_geom_handler(geom, coord_type, gtype)

    @staticmethod
    def get_features(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them."""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]

    def xy(self):
        """XY.

        XY function takes a geodataframe and process the geometry column and return
        the x and y coordinates of all the votrices

        Returns
        -------
        x :[dataframe column]
            column contains the x coordinates of all the votices of the geometry
            object in each rows
        y :[dataframe column]
            column contains the y coordinates of all the votices of the geometry
            object in each rows
        """
        # get the x & y coordinates for all types of geometries except multi_polygon
        # self._feature["x"] = self._feature.apply(
        #     self._get_coords, geom_col="geometry", coord_type="x", axis=1
        # )
        # self._feature["y"] = self._feature.apply(
        #     self._get_coords, geom_col="geometry", coord_type="y", axis=1
        # )
        # if the Geometry of type MultiPolygon
        # explode the multi_polygon into polygon
        for idx, row in self._feature.iterrows():
            if isinstance(row.geometry, MultiPolygon):
                # create a new geodataframe
                multdf = gpd.GeoDataFrame()  # columns=indf.columns
                # get number of the polygons inside the multipolygon class
                recs = len(row.geometry)
                multdf = multdf.append([row] * recs, ignore_index=True)
                # for each rows assign each polygon
                for geom in range(recs):
                    multdf.loc[geom, "geometry"] = row.geometry[geom]
                self._feature = self._feature.append(multdf, ignore_index=True)

        # get the x & y coordinates of the exploded multi_polygons
        self._feature["x"] = self._feature.apply(
            self._get_coords, geom_col="geometry", coord_type="x", axis=1
        )
        self._feature["y"] = self._feature.apply(
            self._get_coords, geom_col="geometry", coord_type="y", axis=1
        )

        to_delete = np.where(self._feature["x"] == 999)[0]
        self._feature = self._feature.drop(to_delete)

    @staticmethod
    def create_polygon(coords: List[Tuple[float, float]], wkt: bool = False):
        """Create a polygon Geometry.

        Create a polygon from coordinates

        parameters
        ----------
        coords: [List]
            list of tuples [(x1,y1),(x2,y2)]
        wkt: [bool]
            True if you want to create the WellKnownText only not the shapely polygon object

        Returns
        -------
        - if wkt is True the function returns a string of the polygon and its coordinates as
        a WellKnownText,
        - if wkt is False the function returns Shapely Polygon object you can assign it
        to a GeoPandas GeoDataFrame directly

        Examples
        --------
        >>> coordinates = [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]
        >>> FeatureCollection.create_polygon(coordinates, wkt=True)
        it will give
        >>> 'POLYGON ((24.95 60.16 0,24.95 60.16 0,24.95 60.17 0,24.95 60.16 0))'
        while
        >>> new_geometry = gpd.GeoDataFrame()
        >>> new_geometry.loc[0,'geometry'] = FeatureCollection.create_polygon(coordinates, wkt=False)
        """
        poly = Polygon(coords)
        if wkt:
            return poly.wkt
        else:

            return poly

    @staticmethod
    def create_point(coords: List[Tuple[float]]) -> List[Point]:
        """CreatePoint.

        CreatePoint takes a list of tuples of coordinates and convert it into
        a list of Shapely point object

        parameters
        ----------
        coords : [List]
            list of tuples [(x1,y1),(x2,y2)] or [(long1,lat1),(long2,lat1)]

        Returns
        -------
        points: [List]
            list of Shaply point objects [Point,Point]

        Examples
        --------
        >>> coordinates = [(24.95, 60.16), (24.95, 60.16), (24.95, 60.17), (24.95, 60.16)]
        >>> point_list = FeatureCollection.create_point(coordinates)
        # to assign these objects to a geopandas dataframe
        >>> new_geometry = gpd.GeoDataFrame()
        >>> new_geometry.loc[:, 'geometry'] = point_list
        """
        points = list(map(Point, coords))

        return points

    @staticmethod
    def combine_geometrics(
        path1: str, path2: str, save: bool = False, save_path: str = None
    ):
        """CombineGeometrics.

        CombineGeometrics reads two shapefiles and combine them into one
        shapefile

        Parameters
        ----------
        path1: [String]
            a path includng the name of the shapefile and extention like
            path="data/subbasins.shp"

        path2: [String]
            a path includng the name of the shapefile and extention like
            path="data/subbasins.shp"
        save: [Boolen]
            True if you want to save the result shapefile in a certain
            path "SavePath"
        save_path: [String]
            a path includng the name of the shapefile and extention like
            path="data/subbasins.shp"

        Returns
        -------
        SaveIng the shapefile or NewGeoDataFrame :
            If you choose True in the "save" input the function will save the
            shapefile in the given "SavePath"
            If you choose False in the "save" input the function will return a
            [geodataframe] dataframe containing both input shapefiles
            you can save it as a shapefile using
            NewDataFrame.to_file("Anyname.shp")

        Examples
        --------
        Return a geodata frame
        >>> shape_file1 = "Inputs/RIM_sub.shp"
        >>> shape_file2 = "Inputs/addSubs.shp"
        >>> NewDataFrame = FeatureCollection.combine_geometrics(shape_file1, shape_file2, save=False)
        save a shapefile
        >>> shape_file1 = "Inputs/RIM_sub.shp"
        >>> shape_file2 = "Inputs/addSubs.shp"
        >>> FeatureCollection.combine_geometrics(shape_file1, shape_file2, save=True, save_path="AllBasins.shp")
        """
        assert type(path1) == str, "path1 input should be string type"
        assert type(path2) == str, "path2 input should be string type"
        assert type(save) == bool, "SavePath input should be string type"

        # input values
        ext = path1[-4:]
        assert ext == ".shp", "please add the extension at the end of the path1"
        ext = path2[-4:]
        assert ext == ".shp", "please add the extension at the end of the path2"
        if save:
            assert type(save_path) == str, "SavePath input should be string type"
            ext = save_path[-4:]
            assert ext == ".shp", "please add the extension at the end of the SavePath"

        # read shapefiles
        GeoDataFrame1 = gpd.read_file(path1)
        GeoDataFrame2 = gpd.read_file(path2)

        # concatenate the second shapefile into the first shapefile
        NewGeoDataFrame = gpd.GeoDataFrame(pd.concat([GeoDataFrame1, GeoDataFrame2]))
        # re-index the data frame
        NewGeoDataFrame.index = [i for i in range(len(NewGeoDataFrame))]
        # take the spatial reference of the first geodataframe
        NewGeoDataFrame.crs = GeoDataFrame1.crs
        if save:
            NewGeoDataFrame.to_file(save_path)
        else:
            return NewGeoDataFrame

    @staticmethod
    def gcs_distance(coords_1: tuple, coords_2: tuple):
        """GCS_distance.

        this function calculates the distance between two points that have
        geographic coordinate system

        parameters
        ----------
        coords_1: [Tuple]
            tuple of (long, lat) of the first point
        coords_2: [Tuple]
            tuple of (long, lat) of the second point

        Returns
        -------
        distance between the two points

        Examples
        --------
        >>> point_1 = (52.22, 21.01)
        >>> point_2 = (52.40, 16.92)
        >>> distance = FeatureCollection.gcs_distance(point_1, point_2)
        """
        dist = distance.vincenty(coords_1, coords_2).m

        return dist

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

    # @staticmethod
    # def addSpatialReference(gdf: GeoDataFrame, epsg: int):
    #     """AddSpatialReference.
    #
    #     AddSpatialReference takes GeoPandas DataFrame and set the coordinate system
    #     based on the given epsg input
    #
    #     Parameters
    #     ----------
    #     gdf: [GeoDataFrame]
    #         geopandas dataframe
    #     epsg: [integer]
    #         EPSG stands for European Petroleum Survey Group and is an organization
    #         that maintains a geodetic parameter database with standard codes,
    #         the EPSG codes, for coordinate systems, datums, spheroids, units
    #         and such alike (https://epsg.io/) default value is [None].
    #
    #     Returns
    #     -------
    #     gdf: [GeoDataFrame]
    #         the same input geopandas dataframe but with spatial reference
    #
    #     Examples
    #     --------
    #     >>> NewGeometry = gpd.GeoDataFrame()
    #     >>> coordinates = [(24.950899, 60.169158), (24.953492, 60.169158),
    #     >>>                 (24.953510, 60.170104), (24.950958, 60.169990)]
    #     >>> NewGeometry.loc[0,'geometry'] = FeatureCollection.createPolygon(coordinates,2)
    #     # adding spatial reference system
    #     >>> NewGeometry.crs = from_epsg(4326)
    #     # to check the spatial reference
    #     >>> NewGeometry.crs
    #     >>> {'init': 'epsg:4326', 'no_defs': True}
    #     """
    #     # from fiona.crs import from_epsg
    #     gdf.crs = from_epsg(epsg)
    #
    #     return gdf

    @staticmethod
    def polygon_center_point(
        poly: GeoDataFrame, save: bool = False, save_path: str = None
    ):
        """PolygonCenterPoint.

        PolygonCenterPoint function takes the a geodata frame of polygons and and
        returns the center of each polygon

        Parameters
        ----------
        poly: [GeoDataFrame]
            GeoDataframe containing
            all the polygons you want to get the center point
        save: [Boolen]
            True if you want to save the result shapefile in a certain
            path "savePath"
        save_path: [String]
            a path includng the name of the shapefile and extention like
            path="data/subbasins.shp"

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
        >>> FeatureCollection.polygon_center_point(sub_basins, save=True, save_path="centerpoint.shp")
        """
        assert (
            type(poly) == gpd.geopandas.geodataframe.GeoDataFrame
        ), "poly input should be GeoDataFrame type"
        assert type(save) == bool, "savePath input should be string type"

        # input values
        if save:
            assert type(save_path) == str, "savePath input should be string type"
            ext = save_path[-4:]
            assert ext == ".shp", "please add the extension at the end of the savePath"

        # get the X, Y coordinates of the points of the polygons and the multipolygons
        poly = FeatureCollection.xy(poly)

        # re-index the data frame
        poly.index = [i for i in range(len(poly))]
        # calculate the average X & Y coordinate for each geometry in the shapefile
        for i in range(len(poly)):
            poly.loc[i, "AvgX"] = np.mean(poly.loc[i, "x"])
            poly.loc[i, "AvgY"] = np.mean(poly.loc[i, "y"])

        # create a new geopandas dataframe of points that is in the middle of each
        # sub-basin
        poly = poly.drop(["geometry", "x", "y"], axis=1)

        MiddlePointdf = gpd.GeoDataFrame()
        #    MiddlePointdf = poly

        MiddlePointdf["geometry"] = None
        # create a list of tuples of the coordinates (x,y) or (long, lat)
        # of the points
        CoordinatesList = zip(poly["AvgX"].tolist(), poly["AvgY"].tolist())
        PointsList = FeatureCollection.create_point(CoordinatesList)
        # set the spatial reference
        MiddlePointdf["geometry"] = PointsList
        MiddlePointdf.crs = poly.crs
        MiddlePointdf[poly.columns.tolist()] = poly[poly.columns.tolist()]

        if save:
            MiddlePointdf.to_file(save_path)
        else:
            return MiddlePointdf

    @staticmethod
    def write_shapefile(poly, path: str):
        """write_shapefile.

        this function takes a polygon geometry and creates a ashapefile and save it
        (https://gis.stackexchange.com/a/52708/8104)

        parameters
        ----------
        poly: [shapely object]
            polygon, point, or lines or multi
        path:
            string, of the path and name of the shapefile

        Returns
        -------
        saving the shapefile to the path
        """
        # Now convert it to a shapefile with OGR
        driver = ogr.GetDriverByName("Esri Shapefile")
        ds = driver.CreateDataSource(path)
        layer = ds.CreateLayer("", None, ogr.wkbPolygon)

        # Add one attribute
        layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        defn = layer.GetLayerDefn()

        ## If there are multiple geometries, put the "for" loop here

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField("id", 123)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(poly)
        # geom = ogr.CreateGeometryFromWkb(poly.wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

        # save and close everything
        ds = layer = feat = geom = None

    def listAttributes(self):
        """Print Attributes List."""
        print("\n")
        print(
            f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")
