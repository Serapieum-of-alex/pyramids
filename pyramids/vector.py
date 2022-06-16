"""
Created on Sun Jul 01 17:07:40 2018

@author: Mostafa
"""

import json
import warnings

import geopandas as gpd
import geopy.distance as distance
import numpy as np
import pandas as pd
from fiona.crs import from_epsg
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr, osr
from pyproj import Proj, transform
from shapely.geometry import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon


class Vector:
    """Vector

    Vector class contains different methods to deal with shapefiles

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

    def __init__(self):
        pass

    @staticmethod
    def getXYCoords(geometry, coord_type: str):
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
            return geometry.coords.xy[0]
        elif coord_type == "y":
            return geometry.coords.xy[1]


    @staticmethod
    def getPointCoords(geometry, coord_type: str):
        """GetPointCoords

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
            return geometry.x
        if coord_type == "y":
            return geometry.y


    @staticmethod
    def getLineCoords(geometry, coord_type: str):
        """getLineCoords

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
        return Vector.getXYCoords(geometry, coord_type)


    @staticmethod
    def getPolyCoords(geometry, coord_type: str):
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

        return Vector.getXYCoords(ext, coord_type)


    @staticmethod
    def explode(dataframe_row):
        """Explode.

        explode function converts the multipolygon into a polygons

        Parameters
        ----------
        dataframe_row: [data frame series]
            the dataframe row that its geometry type is Multipolygon

        Returns
        -------
        outdf: [dataframe]
            the dataframe of the created polygons
        """
        row = dataframe_row
        outdf = gpd.GeoDataFrame()
        multdf = gpd.GeoDataFrame()
        recs = len(row)
        multdf = multdf.append([row] * recs, ignore_index=True)
        for geom in range(recs):
            multdf.loc[geom, "geometry"] = row.geometry[geom]
        outdf = outdf.append(multdf, ignore_index=True)


    @staticmethod
    def multiGeomHandler(multi_geometry, coord_type: str, geom_type: str):
        """multiGeomHandler

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
        if geom_type == "MultiPoint" or geom_type == "MultiLineString":
            for i, part in enumerate(multi_geometry):
                # On the first part of the Multi-geometry initialize the coord_array (np.array)
                if i == 0:
                    if geom_type == "MultiPoint":
                        coord_arrays = Vector.getPointCoords(
                            part, coord_type
                        )
                    elif geom_type == "MultiLineString":
                        coord_arrays = Vector.getLineCoords(part, coord_type)
                else:
                    if geom_type == "MultiPoint":
                        coord_arrays = np.concatenate(
                            [coord_arrays, Vector.getPointCoords(part, coord_type)]
                        )
                    elif geom_type == "MultiLineString":
                        coord_arrays = np.concatenate(
                            [coord_arrays, Vector.getLineCoords(part, coord_type)]
                        )

        elif geom_type == "MultiPolygon":
            for i, part in enumerate(multi_geometry):
                if i == 0:
                    multi_2_single = Vector.explode(multi_geometry)
                    for j in range(len(multi_2_single)):
                        if j == 0:
                            coord_arrays = Vector.getPolyCoords(
                                multi_2_single[j], coord_type
                            )
                        else:
                            coord_arrays = np.concatenate(
                                [
                                    coord_arrays,
                                    Vector.getPolyCoords(multi_2_single[j], coord_type),
                                ]
                            )
                else:
                    # explode the multipolygon into polygons
                    multi_2_single = Vector.explode(part)
                    for j in range(len(multi_2_single)):
                        coord_arrays = np.concatenate(
                            [
                                coord_arrays,
                                Vector.getPolyCoords(multi_2_single[j], coord_type),
                            ]
                        )
            # return the coordinates
            return coord_arrays

    @staticmethod
    def getCoords(row, geom_col: str, coord_type: str):
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
            return Vector.getPointCoords(geom, coord_type)
        elif gtype == "LineString":
            return list(Vector.getLineCoords(geom, coord_type))
        elif gtype == "Polygon":
            return list(Vector.getPolyCoords(geom, coord_type))
        elif gtype == "MultiPolygon":
            return 999
        # Multi geometries
        else:
            return list(Vector.multiGeomHandler(geom, coord_type, gtype))

    @staticmethod
    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a
        manner that rasterio wants them"""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]

    @staticmethod
    def XY(input_dataframe):
        """XY.

        XY function takes a geodataframe and process the geometry column and return
        the x and y coordinates of all the votrices

        parameters
        ----------
        input_dataframe:[geodataframe]
            geodataframe contains the Shapely geometry object in a column name
            "geometry"
        Returns
        -------
        x :[dataframe column]
            column contains the x coordinates of all the votices of the geometry
            object in each row
        y :[dataframe column]
            column contains the y coordinates of all the votices of the geometry
            object in each row
        """
        # get the x & y coordinates for all types of geometries except multi_polygon
        input_dataframe["x"] = input_dataframe.apply(
            Vector.getCoords, geom_col="geometry", coord_type="x", axis=1
        )
        input_dataframe["y"] = input_dataframe.apply(
            Vector.getCoords, geom_col="geometry", coord_type="y", axis=1
        )

        # if the Geometry of type MultiPolygon
        # explode the multi_polygon into polygon
        for idx, row in input_dataframe.iterrows():
            if type(row.geometry) == MultiPolygon:
                # create a new geodataframe
                multdf = gpd.GeoDataFrame()  # columns=indf.columns
                # get number of the polygons inside the multipolygon class
                recs = len(row.geometry)
                multdf = multdf.append([row] * recs, ignore_index=True)
                # for each row assign each polygon
                for geom in range(recs):
                    multdf.loc[geom, "geometry"] = row.geometry[geom]
                input_dataframe = input_dataframe.append(multdf, ignore_index=True)

        # get the x & y coordinates of the exploded multi_polygons
        input_dataframe["x"] = input_dataframe.apply(
            Vector.getCoords, geom_col="geometry", coord_type="x", axis=1
        )
        input_dataframe["y"] = input_dataframe.apply(
            Vector.getCoords, geom_col="geometry", coord_type="y", axis=1
        )

        to_delete = np.where(input_dataframe["x"] == 999)[0]
        input_dataframe = input_dataframe.drop(to_delete)

        return input_dataframe

    @staticmethod
    def createPolygon(coords, geom_type: int = 1):
        """create_polygon.

        this function creates a polygon from coordinates

        parameters
        ----------
        coords: [List]
            list of tuples [(x1,y1),(x2,y2)]
        geom_type: [Integer]
            1 to return a polygon in the form of WellKnownText, 2 to return a
            polygon as an object

        Returns
        -------
        Type 1 returns a string of the polygon and its coordinates as
        a WellKnownText, Type 2 returns Shapely Polygon object you can assign it
        to a GeoPandas GeoDataFrame directly

        Examples
        --------
        >>> coordinates = [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]
        >>> Vector.createPolygon(coordinates, 1)
        it will give
        >>> 'POLYGON ((24.95 60.16 0,24.95 60.16 0,24.95 60.17 0,24.95 60.16 0))'
        while
        >>> new_geometry = gpd.GeoDataFrame()
        >>> new_geometry.loc[0,'geometry'] = Vector.createPolygon(coordinates, 2)
        """
        if geom_type == 1:
            # create a ring
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for coord in coords:
                ring.AddPoint(np.double(coord[0]), np.double(coord[1]))

            # Create polygon
            poly = ogr.Geometry(ogr.wkbPolygon)

            poly.AddGeometry(ring)
            return poly.ExportToWkt()
        else:
            poly = Polygon(coords)
            return poly


    @staticmethod
    def createPoint(coords: list):
        """CreatePoint

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
        >>> point_list = Vector.createPoint(coordinates)
        # to assign these objects to a geopandas dataframe
        >>> new_geometry = gpd.GeoDataFrame()
        >>> new_geometry.loc[:, 'geometry'] = point_list
        """
        points = list()
        for i in range(len(coords)):
            points.append(Point(coords[i]))

        return points

    @staticmethod
    def combineGeometrics(path1: str, path2: str, save: bool=False, save_path: str=None):
        """CombineGeometrics

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
        >>> NewDataFrame = Vector.combineGeometrics(shape_file1, shape_file2, save=False)
        save a shapefile
        >>> shape_file1 = "Inputs/RIM_sub.shp"
        >>> shape_file2 = "Inputs/addSubs.shp"
        >>> Vector.combineGeometrics(shape_file1, shape_file2, save=True, save_path="AllBasins.shp")
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
    def GCSDistance(coords_1: tuple, coords_2: tuple):
        """GCS_distance

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
        >>> distance = Vector.GCSDistance(point_1, point_2)
        """
        dist = distance.vincenty(coords_1, coords_2).m

        return dist

    @staticmethod
    def reprojectPoints(lat: list, lon: list, from_epsg: int=4326, to_epsg: int=3857, precision: int=6):
        """reproject_points

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
        >>>  longs, lats = Vector.reprojectPoints(y_coords, x_coords, from_epsg=3857, to_epsg=4326)
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
    def reprojectPoints2(lat: list, lng: list, from_epsg: int=4326, to_epsg: int=3857):
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
        >>> longs, lats = Vector.reprojectPoints2(y_coords, x_coords, from_epsg=3857, to_epsg=4326)
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

    @staticmethod
    def addSpatialReference(gdf: GeoDataFrame, epsg: int):
        """AddSpatialReference.

        AddSpatialReference takes GeoPandas DataFrame and set the coordinate system
        based on the given epsg input

        Parameters
        ----------
        gdf: [GeoDataFrame]
            geopandas dataframe
        epsg: [integer]
            EPSG stands for European Petroleum Survey Group and is an organization
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units
            and such alike (https://epsg.io/) default value is [None].

        Returns
        -------
        gdf: [GeoDataFrame]
            the same input geopandas dataframe but with spatial reference

        Examples
        --------
        >>> NewGeometry = gpd.GeoDataFrame()
        >>> coordinates = [(24.950899, 60.169158), (24.953492, 60.169158),
        >>>                 (24.953510, 60.170104), (24.950958, 60.169990)]
        >>> NewGeometry.loc[0,'geometry'] = Vector.createPolygon(coordinates,2)
        # adding spatial reference system
        >>> NewGeometry.crs = from_epsg(4326)
        # to check the spatial reference
        >>> NewGeometry.crs
        >>> {'init': 'epsg:4326', 'no_defs': True}
        """

        gdf.crs = from_epsg(epsg)

        return gdf

    @staticmethod
    def polygonCenterPoint(poly: GeoDataFrame, save: bool=False, save_path: str=None):
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
        >>> CenterPointDataFrame = Vector.polygonCenterPoint(sub_basins, save=False)
        save a shapefile
        >>> sub_basins = gpd.read_file("Inputs/sub_basins.shp")
        >>> Vector.polygonCenterPoint(sub_basins, save=True, save_path="centerpoint.shp")
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
        poly = Vector.XY(poly)

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
        CoordinatesList = zip(
            poly["AvgX"].tolist(), poly["AvgY"].tolist()
        )
        PointsList = Vector.createPoint(CoordinatesList)
        # set the spatial reference
        MiddlePointdf["geometry"] = PointsList
        MiddlePointdf.crs = poly.crs
        MiddlePointdf[poly.columns.tolist()] = poly[
            poly.columns.tolist()
        ]

        if save:
            MiddlePointdf.to_file(save_path)
        else:
            return MiddlePointdf


    @staticmethod
    def writeShapefile(poly, path: str):
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
        """
        Print Attributes List
        """
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
