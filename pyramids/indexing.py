from typing import Dict, Set, Tuple, Union

import h3
from geopandas.geodataframe import GeoDataFrame
from pandas import DataFrame
from pandas.core.series import Series
from shapely.geometry import Polygon


class H3:
    def __init__(self):
        pass

    @staticmethod
    def geometryToIndex(lat, lon, resolution):
        """Get Hexagon index.

            get the index of the hexagon that the coordinates lie inside at a certain resolution level.

        Parameters
        ----------
        lat: float
            Latitude or y coordinate
        lon: float
            longitude or x coordinate
        resolution: [int]
            resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

        Returns
        -------
        str:
            hexadecimal number
        """
        return h3.geo_to_h3(lat, lon, resolution)

    @staticmethod
    def getIndex(gdf: GeoDataFrame, resolution: int) -> Series:
        """Get Hexagon index.

            get the index of the hexagon that the coordinates lie inside at a certain resolution level.

        Parameters
        ----------
        gdf: [GeoDataFrame]
            GeoDataFrame
        resolution: [int]
            resolution level (0, 15), 0 is the the very coarse, and 15 is the very fine resolution.

        Returns
        -------
        Pandas Series:
            pandas series with a column with the h3 hex index.

        Examples
        --------
        >>> import geopandas as gpd
        >>> gdf = gpd.read_file("file.geojson")
        >>> resolution = 3
        >>> gdf["h3_index"] = H3.getIndex(gdf)
        """
        fn = lambda gdf_row: H3.geometryToIndex(
            gdf_row.geometry.y, gdf_row.geometry.x, resolution
        )
        return gdf.apply(fn, axis=1)

    @staticmethod
    def getCoords(hex_index: str) -> Tuple:
        """Get the coordinates.

            returns the coordinates from the hexdecimal string

        Parameters
        ----------
        hex_index: [str]
            hexagon index (hexadecimal format)

        Returns
        -------
        Tuple:
            (lat, lon)
        """
        return h3.h3_to_geo(hex_index)

    @staticmethod
    def indexToPolygon(hex_index: str) -> Polygon:
        """Get Polygon.

            return the polygon corresponding to the given hexagon index

        Parameters
        ----------
        hex_index: [str]
            hexagon index (hexadecimal format)

        Returns
        -------
        Shapely Polygon

        Examples
        --------
        >>> def add_geometry(row):
        >>>     return H3.indexToPolygon(row['hex_index'])
        >>> gdf['geometry'] = gdf.apply(add_geometry, axis=1)
        """
        return Polygon(h3.h3_to_geo_boundary(hex_index, geo_json=True))

    # def getGeometry():
    @staticmethod
    def getGeometry(gdf: Union[GeoDataFrame, DataFrame], index_col: str) -> Series:
        """Get the Hexagon polygon geometry form a hexagon index.

        Parameters
        ----------
        gdf: [GeoDataFrame]
            geodataframe with a column filled with hexagon index

        index_col: [str]
            column where the hexagon index is stored

        Returns
        -------
        Pandas Series
            polygon geometries corespondint to the hexagon index.
        """
        fn = lambda gdf_row: H3.indexToPolygon(gdf_row[index_col])
        return gdf.apply(fn, axis=1)

    @staticmethod
    def getParent(hex_index: str) -> Set[str]:
        """Get the parent hexagon.

            return the parent hexagon index corresponding to the given index

        Parameters
        ----------
        hex_index: [str]
            hexagon index (hexadecimal format)

        Returns
        -------
        get the index of the parent hexagon
        """
        return h3.h3_to_parent(hex_index)

    @staticmethod
    def getChild(hex_index: str) -> Set[str]:
        """Get the child hexagon.

            return the parent hexagon index corresponding to the given index

        Parameters
        ----------
        hex_index: [str]
            hexagon index (hexadecimal format)

        Returns
        -------
        get the index of the child hexagon
        """
        return h3.h3_to_children(hex_index)

    @staticmethod
    def getAttributes(hex_index: str) -> Dict[str, str]:
        """Get hexagon attributes.

            returns some attributes for a given hexagon

        Parameters
        ----------
        hex_index: [str]
            hexagon index (hexadecimal format)

        Returns
        -------
        Dict:
            keys ["cordinates", "boundary", "parent", "children"]
        >>> {
        >>>     "cordinates": (-89.83, -157.30),
        >>>     "boundary": 'POLYGON ((177.60 -89.79, 170.21 -89.88, -140.29 -89.91, -126.50 -89.82, -142.19 -89.75,'
        >>>                     '-162.75 -89.740, 177.60191 -89.790))',
        >>>     "parent": "84f2939ffffffff",
        >>>     "children": {
        >>>                     '86f29394fffffff', '86f293977ffffff', '86f293957ffffff', '86f29396fffffff',
        >>>                     '86f29395fffffff', '86f293967ffffff', '86f293947ffffff'
        >>>                 },
        >>> }
        """
        return {
            "cordinates": H3.getCoords(hex_index),
            "boundary": H3.getPolygon(hex_index).wkt,
            "parent": H3.getParent(hex_index),
            "children": H3.getChild(hex_index),
        }

    @staticmethod
    def aggregate(gdf: Union[DataFrame, GeoDataFrame], col_name: str) -> GeoDataFrame:
        """Get number of geometries in each hexagon.

        Parameters
        ----------
        gdf: [GeoDataFrame]
            input DataFrame
        col_name: [str]
            column where the hexdecimal index is stored.

        Returns
        -------
        GeoDataFrame:
            GeoDataFrame with extra column "count" containing the count of geometries in each hexagon.
        """
        return (
            gdf.groupby([col_name])[col_name]
            .agg("count")
            .to_frame("count")
            .reset_index()
        )
