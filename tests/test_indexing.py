from typing import Tuple

from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import Polygon

from pyramids.indexing import H3


def test_geometry_to_index(
    lat_lon: Tuple[float, float],
    h3_resolution: int,
    hex_index: str,
):
    hex = H3.geometryToIndex(lat_lon[0], lat_lon[1], h3_resolution)
    assert hex == hex_index


def test_get_index(point_gdf: GeoDataFrame, h3_resolution: int):
    point_gdf["h3"] = H3.getIndex(point_gdf, h3_resolution)
    assert all(point_gdf["h3"] == point_gdf["h3"])


def test_index_to_polygon(
    hex_index: str,
    hex_8503262bfffffff_res5_polygon: Polygon,
):
    geom = H3.indexToPolygon(hex_index)
    assert geom.almost_equals(hex_8503262bfffffff_res5_polygon)


def test_get_geometry(point_gdf: GeoDataFrame, index_column: str):
    geom = H3.getGeometry(point_gdf, index_column)
    assert all("Polygon" == geom.geometry.geom_type)


def test_aggregate(point_gdf: GeoDataFrame, index_column: str):
    aggregated_hex = H3.aggregate(point_gdf, index_column)
    assert aggregated_hex.loc[0, "count"] == 4
