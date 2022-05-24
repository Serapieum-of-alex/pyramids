"""
Created on Sun Jul 01 17:07:40 2018

@author: Mostafa
"""

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from fiona.crs import from_epsg
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
    def getXYCoords(geometry, coord_type):
        """getXYCoords.

        Returns either x or y coordinates from  geometry coordinate sequence.
         Used with LineString and Polygon geometries.

         inputs
         ------
             1- geometry:
                  [Geometry object of type LineString] the geometry of a shpefile
             2- coord_type:
                 [string] either "x" or "y"

         outpus
         ------
             1-array:
                 contains x coordinates or y coordinates of all edges of the shapefile
        """
        if coord_type == "x":
            return geometry.coords.xy[0]
        elif coord_type == "y":
            return geometry.coords.xy[1]

    @staticmethod
    def getPointCoords(geometry, coord_type):
        """GetPointCoords

        Returns Coordinates of Point object.
        inputs
        ------
            1- geometry:
                [Shapely Point object] the geometry of a shpefile
            2- coord_type:
                [string] either "x" or "y"
        outpus
        ------
            1-array:
                contains x coordinates or y coordinates of all edges of the shapefile
        """
        if coord_type == "x":
            return geometry.x
        if coord_type == "y":
            return geometry.y

    @staticmethod
    def getLineCoords(geometry, coord_type):
        """getLineCoords

        Returns Coordinates of Linestring object.

        inputs
        ------
            1- geometry:
                 [Shapely Linestring object] the geometry of a shpefile
            2- coord_type:
                [string] either "x" or "y"

        outpus
        ------
            1-array:
                contains x coordinates or y coordinates of all edges of the shapefile
        """
        return Vector.getXYCoords(geometry, coord_type)

    @staticmethod
    def getPolyCoords(geometry, coord_type):
        """getPolyCoords.

        Returns Coordinates of Polygon using the Exterior of the Polygon.
        inputs
        ------
            1- geometry:
             [Shapely polygon object] the geometry of a shpefile
            2- coord_type:
                 [string] either "x" or "y"

        outpus
        ------
            1-array:
                contains x coordinates or y coordinates of all edges of the shapefile
        """
        # convert the polygon into lines
        ext = geometry.exterior  # type = LinearRing

        return Vector.getXYCoords(ext, coord_type)

    @staticmethod
    def explode(dataframe_row):
        """Explode.

        explode function converts the multipolygon into a polygons
        inputs
        ------
            1- dataframe_row: (data frame series)
                the dataframe row that its geometry type is Multipolygon
        outputs
        -------
            1- outdf
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
    def multiGeomHandler(multi_geometry, coord_type, geom_type):
        """multiGeomHandler

        Function for handling multi-geometries. Can be MultiPoint, MultiLineString or MultiPolygon.
        Returns a list of coordinates where all parts of Multi-geometries are merged into a single list.
        Individual geometries are separated with np.nan which is how Bokeh wants them.
        # Bokeh documentation regarding the Multi-geometry issues can be found here (it is an open issue)
        # https://github.com/bokeh/bokeh/issues/2321

        inputs
        ------
            1- multi_geometry :[geometry]
                the geometry of a shpefile
            2- coord_type : [string]
                "string" either "x" or "y"
            3- geom_type : [string]
                "MultiPoint" or "MultiLineString" or "MultiPolygon"
        outpus
        ------
            1-array: [ndarray]
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
    def getCoords(row, geom_col, coord_type):
        """getCoords.

        Returns coordinates ('x' or 'y') of a geometry (Point, LineString or Polygon)
        as a list (if geometry is Points, LineString or Polygon). Can handle also
        MultiGeometries but not MultiPolygon.

        inputs
            1- row:
                [dataframe] a whole rwo of the dataframe
            2- geom_col"
                [string] name of the column where the geometry is stored in the dataframe
            3- coord_type:
                [string] "X" or "Y" choose which coordinate toy want to get from
                the function

        outpus:
            1-array:
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

    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a
        manner that rasterio wants them"""
        return [json.loads(gdf.to_json())["features"][0]["geometry"]]

    @staticmethod
    def XY(input_dataframe):
        """XY.

        XY function takes a geodataframe and process the geometry column and return
        the x and y coordinates of all the votrices

        inputs
            1- input_dataframe:[geodataframe]
                geodataframe contains the Shapely geometry object in a column name
                "geometry"
        Output:
            1-x :[dataframe column]
                column contains the x coordinates of all the votices of the geometry
                object in each row
            2-y :[dataframe column]
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
    def createPolygon(coords, Type=1):
        """create_polygon.

        this function creates a polygon from coordinates

        inputs
        ----------
            coords :
                [List] list of tuples [(x1,y1),(x2,y2)]
            Type :
                [Integer] 1 to return a polygon in the form of WellKnownText, 2 to return a
                polygon as an object

        outputs:
        ----------
            Type 1 returns a string of the polygon and its coordinates as
            a WellKnownText, Type 2 returns Shapely Polygon object you can assign it
            to a GeoPandas GeoDataFrame directly


        Example:
        ----------
            coords = [(-106.6472953, 24.0370137), (-106.4933356, 24.05293569), (-106.4941789, 24.01969175), (-106.4927777, 23.98804445)]
            GIS.CreatePolygon(coords,1)
            it will give
            'POLYGON ((24.950899 60.169158 0,24.953492 60.169158 0,24.95351 60.170104 0,24.950958 60.16999 0))'
            while
            NewGeometry = gpd.GeoDataFrame()
            NewGeometry.loc[0,'geometry'] = GIS.CreatePolygon(coordinates,2)
            then
            NewGeometry.loc[0,'geometry']
            will draw an object
        """
        if Type == 1:
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
    def createPoint(coords):
        """CreatePoint

        CreatePoint takes a list of tuples of coordinates and convert it into
        a list of Shapely point object

        inputs
        ----------
            1-coords : [List]
                list of tuples [(x1,y1),(x2,y2)] or [(long1,lat1),(long2,lat1)]

        Outputs:
        ----------
            1-points : [List]
                list of Shaply point objects [Point,Point]

        examples:
        ----------
            coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
            PointList = GIS.CreatePoint(coordinates)
            # to assign these objects to a geopandas dataframe
            # NopreviousGeoms is the number of geometries already exists in the
            # geopandas dataframe
            NopreviousGeoms = 5
            for i in range(NopreviousGeoms,NopreviousGeoms+len(PointList)):
                NewGeometry.loc[i,'geometry'] = PointList[i-NopreviousGeoms]
        """
        points = list()
        for i in range(len(coords)):
            points.append(Point(coords[i]))

        return points

    @staticmethod
    def combineGeometrics(Path1, Path2, Save=False, SavePath=None):
        """CombineGeometrics

        CombineGeometrics reads two shapefiles and combine them into one
        shapefile

        inputs
        ----------
            1-Path1:
                [String] a path includng the name of the shapefile and extention like
                path="data/subbasins.shp"

            2-Path2:
                [String] a path includng the name of the shapefile and extention like
                path="data/subbasins.shp"
            3-Save:
                [Boolen] True if you want to save the result shapefile in a certain
                path "SavePath"
            3-SavePath:
                [String] a path includng the name of the shapefile and extention like
                path="data/subbasins.shp"

        Output:
        ----------
            1-SaveIng the shapefile or NewGeoDataFrame :
                If you choose True in the "Save" input the function will save the
                shapefile in the given "SavePath"
                If you choose False in the "Save" input the function will return a
                [geodataframe] dataframe containing both input shapefiles
                you can save it as a shapefile using
                NewDataFrame.to_file("Anyname.shp")

        Example:
        ----------
            1- Return a geodata frame
                RIMSubPath = "Inputs/RIM_sub.shp"
                AddSubsPath = "Inputs/addSubs.shp"
                NewDataFrame = GIS.CombineGeometrics(RIMSubPath,AddSubsPath, Save=False)
            2- Save a shapefile
                RIMSubPath = "Inputs/RIM_sub.shp"
                AddSubsPath = "Inputs/addSubs.shp"
                GIS.CombineGeometrics(RIMSubPath,AddSubsPath, Save=True, SavePath = "AllBasins.shp")
        """
        assert type(Path1) == str, "Path1 input should be string type"
        assert type(Path2) == str, "Path2 input should be string type"
        assert type(Save) == bool, "SavePath input should be string type"

        # input values
        ext = Path1[-4:]
        assert ext == ".shp", "please add the extension at the end of the Path1"
        ext = Path2[-4:]
        assert ext == ".shp", "please add the extension at the end of the Path2"
        if Save == True:
            assert type(SavePath) == str, "SavePath input should be string type"
            ext = SavePath[-4:]
            assert ext == ".shp", "please add the extension at the end of the SavePath"

        # read shapefiles
        GeoDataFrame1 = gpd.read_file(Path1)
        GeoDataFrame2 = gpd.read_file(Path2)

        # concatenate the second shapefile into the first shapefile
        NewGeoDataFrame = gpd.GeoDataFrame(pd.concat([GeoDataFrame1, GeoDataFrame2]))
        # re-index the data frame
        NewGeoDataFrame.index = [i for i in range(len(NewGeoDataFrame))]
        # take the spatial reference of the first geodataframe
        NewGeoDataFrame.crs = GeoDataFrame1.crs
        if Save == True:
            NewGeoDataFrame.to_file(SavePath)
        else:
            return NewGeoDataFrame

    @staticmethod
    def GCSDistance(coords_1, coords_2):
        """GCS_distance

        this function calculates the distance between two points that have
        geographic coordinate system

        inputs
        ----------
            1-coord_1:
                tuple of (long, lat) of the first point
            2- coord_2:
                tuple of (long, lat) of the second point

        Output:
        ----------
            1-distance between the two points

        example:
        -------
            coords_1 = (52.2296756, 21.0122287)
            coords_2 = (52.406374, 16.9251681)

        """
        import geopy.distance

        dist = geopy.distance.vincenty(coords_1, coords_2).m

        return dist

    @staticmethod
    def reprojectPoints(lat, lon, from_epsg=4326, to_epsg=3857, precision=6):
        """reproject_points

        this function change the projection of the coordinates from a coordinate system
        to another (default from GCS to web mercator used by google maps)

        inputs
        ----------
            1- lat: [list]
                list of latitudes of the points
            2- lon: [list]
                list of longitude of the points
            3- from_epsg: [integer]
                reference number to the projection of the points (https://epsg.io/)
            4- to_epsg: [integer]
                reference number to the new projection of the points (https://epsg.io/)
            5- precision: [integer]
                number of decimal places
        outputs:
        ----------
            1-y:
                list of y coordinates of the points
            2-x:
                list of x coordinates of the points

        Ex:
        ----------
            # from web mercator to GCS WGS64:
            x=[-8418583.96378159, -8404716.499972705], y=[529374.3212213353, 529374.3212213353]
            from_epsg = 3857, to_epsg = 4326
            longs, lats=reproject_points(y,x,from_epsg="3857", to_epsg="4326")
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
    def reprojectPoints_2(lat, lng, from_epsg=4326, to_epsg=3857):
        """reproject_points.

        this function change the projection of the coordinates from a coordinate system
        to another (default from GCS to web mercator used by google maps)

        inputs
        ----------
            1- lat:
                list of latitudes of the points
            2- lng:
                list of longitude of the points
            3- from_epsg:
                integer reference number to the projection of the points (https://epsg.io/)
            4- to_epsg:
                integer reference number to the new projection of the points (https://epsg.io/)

        outputs:
        ----------
            1-x:
                list of x coordinates of the points
            2-y:
                list of y coordinates of the points

        Ex:
        ----------
            # from web mercator to GCS WGS64:
            x=[-8418583.96378159, -8404716.499972705], y=[529374.3212213353, 529374.3212213353]
            from_epsg = 3857, to_epsg = 4326
            longs, lats=reproject_points(y,x,from_epsg="3857", to_epsg="4326")
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
    def addSpatialReference(GpdDF, epsg):
        """AddSpatialReference.

        AddSpatialReference takes GeoPandas DataFrame and set the coordinate system
        based on the given epsg input

        inputs
            1-GpdDF:
                [geopandas.geodataframe.GeoDataFrame] geopandas dataframe
            2-epsg:
                [integer] EPSG stands for European Petroleum Survey Group and is an organization
                that maintains a geodetic parameter database with standard codes,
                the EPSG codes, for coordinate systems, datums, spheroids, units
                and such alike (https://epsg.io/) default value is [None].

        Outputs:
            1-GpdDF:
                [geopandas.geodataframe.GeoDataFrame] the same input geopandas
                dataframe but with spatial reference

        examples:
            NewGeometry = gpd.GeoDataFrame()
            coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
            NewGeometry.loc[0,'geometry'] = GIS.CreatePolygon(coordinates,2)
            # adding spatial reference system
            NewGeometry.crs = from_epsg(4326)
            # to check the spatial reference
            NewGeometry.crs
            the you will get
            {'init': 'epsg:4326', 'no_defs': True}
        """

        GpdDF.crs = from_epsg(epsg)

        return GpdDF

    @staticmethod
    def polygonCenterPoint(PolygonDataFrame, Save=False, SavePath=None):
        """PolygonCenterPoint.

        PolygonCenterPoint function takes the a geodata frame of polygons and and
        returns the center of each polygon

        inputs
            1-PolygonDataFrame:
                [geopandas.geodataframe.GeoDataFrame] GeoDataframe containing
                all the polygons you want to get the center point
            3-Save:
                [Boolen] True if you want to save the result shapefile in a certain
                path "SavePath"
            3-SavePath:
                [String] a path includng the name of the shapefile and extention like
                path="data/subbasins.shp"
        Outputs:
            1-SaveIng the shapefile or CenterPointDataFrame :
                If you choose True in the "Save" input the function will save the
                shapefile in the given "SavePath"
                If you choose False in the "Save" input the function will return a
                [geodataframe] dataframe containing CenterPoint DataFrame
                you can save it as a shapefile using
                CenterPointDataFrame.to_file("Anyname.shp")


        Example:
            1- Return a geodata frame
                RIMSubPath = "Inputs/RIM_sub.shp"
                RIMSub = gpd.read_file(RIMSubPath)
                CenterPointDataFrame = GIS.PolygonCenterPoint(RIMSub, Save=False)
            2- Save a shapefile
                RIMSubPath = "Inputs/RIM_sub.shp"
                RIMSub = gpd.read_file(RIMSubPath)
                GIS.PolygonCenterPoint(RIMSub, Save=True, SavePath = "centerpoint.shp")

        """
        assert (
            type(PolygonDataFrame) == gpd.geopandas.geodataframe.GeoDataFrame
        ), "PolygonDataFrame input should be GeoDataFrame type"
        assert type(Save) == bool, "SavePath input should be string type"

        # input values
        if Save == True:
            assert type(SavePath) == str, "SavePath input should be string type"
            ext = SavePath[-4:]
            assert ext == ".shp", "please add the extension at the end of the SavePath"

        # get the X, Y coordinates of the points of the polygons and the multipolygons
        PolygonDataFrame = Vector.XY(PolygonDataFrame)

        # re-index the data frame
        PolygonDataFrame.index = [i for i in range(len(PolygonDataFrame))]
        # calculate the average X & Y coordinate for each geometry in the shapefile
        for i in range(len(PolygonDataFrame)):
            PolygonDataFrame.loc[i, "AvgX"] = np.mean(PolygonDataFrame.loc[i, "x"])
            PolygonDataFrame.loc[i, "AvgY"] = np.mean(PolygonDataFrame.loc[i, "y"])

        # create a new geopandas dataframe of points that is in the middle of each
        # sub-basin
        PolygonDataFrame = PolygonDataFrame.drop(["geometry", "x", "y"], axis=1)

        MiddlePointdf = gpd.GeoDataFrame()
        #    MiddlePointdf = PolygonDataFrame

        MiddlePointdf["geometry"] = None
        # create a list of tuples of the coordinates (x,y) or (long, lat)
        # of the points
        CoordinatesList = zip(
            PolygonDataFrame["AvgX"].tolist(), PolygonDataFrame["AvgY"].tolist()
        )
        PointsList = Vector.createPoint(CoordinatesList)
        # set the spatial reference
        MiddlePointdf["geometry"] = PointsList
        MiddlePointdf.crs = PolygonDataFrame.crs
        MiddlePointdf[PolygonDataFrame.columns.tolist()] = PolygonDataFrame[
            PolygonDataFrame.columns.tolist()
        ]

        if Save == True:
            MiddlePointdf.to_file(SavePath)
        else:
            return MiddlePointdf

    @staticmethod
    def writeShapefile(poly, out_shp):
        """write_shapefile.

        this function takes a polygon geometry and creates a ashapefile and save it
        (https://gis.stackexchange.com/a/52708/8104)

        inputs
        ----------
            1-geometry:
                polygon, point, or lines or multi
            2-path:
                string, of the path and name of the shapefile

        outputs:
        ----------
            1-saving the shapefile to the path
        """
        # Now convert it to a shapefile with OGR
        driver = ogr.GetDriverByName("Esri Shapefile")
        ds = driver.CreateDataSource(out_shp)
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

        # Save and close everything
        ds = layer = feat = geom = None

    def listAttributes(self):
        """
        Print Attributes List
        """

        print("\n")
        print(
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")
