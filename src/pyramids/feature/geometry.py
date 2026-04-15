"""Shape factories and coordinate-extraction helpers for :mod:`pyramids.feature`.

ARC-10 moved these helpers off :class:`pyramids.feature.FeatureCollection`
into this module so that the collection class can focus on per-feature
behavior rather than static geometry utilities. The FeatureCollection
class keeps thin static-method delegates for back-compat: callers that
wrote ``FeatureCollection.create_polygon(...)`` or
``FeatureCollection._get_coords(...)`` continue to work unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon


def get_xy_coords(geometry: Any, coord_type: str) -> list:
    """Return x or y coords from a LineString / Polygon boundary.

    Args:
        geometry: Any geometry exposing ``.coords.xy`` (LineString,
            LinearRing, …).
        coord_type (str): ``"x"`` or ``"y"``.

    Returns:
        list: Coordinate values.

    Raises:
        ValueError: If ``coord_type`` is not ``"x"`` or ``"y"``.
    """
    if coord_type == "x":
        return list(geometry.coords.xy[0].tolist())
    if coord_type == "y":
        return list(geometry.coords.xy[1].tolist())
    raise ValueError("coord_type can only have a value of 'x' or 'y'")


def get_point_coords(geometry: Point, coord_type: str) -> float | int:
    """Return the x or y coordinate of a shapely :class:`Point`.

    Args:
        geometry (Point): A shapely Point.
        coord_type (str): ``"x"`` or ``"y"``.

    Returns:
        float | int: The requested coordinate.

    Raises:
        ValueError: If ``coord_type`` is not ``"x"`` or ``"y"``.
    """
    if coord_type == "x":
        return float(geometry.x)
    if coord_type == "y":
        return float(geometry.y)
    raise ValueError("coord_type can only have a value of 'x' or 'y'")


def get_line_coords(geometry: LineString, coord_type: str) -> list:
    """Return x or y coordinates of a :class:`LineString`."""
    return get_xy_coords(geometry, coord_type)


def get_poly_coords(geometry: Polygon, coord_type: str) -> list:
    """Return x or y coordinates of a :class:`Polygon` exterior."""
    return get_xy_coords(geometry.exterior, coord_type)


def explode_gdf(
    gdf: GeoDataFrame, geometry: str = "multipolygon"
) -> GeoDataFrame:
    """Explode multi-geometries into per-row single geometries.

    Rows whose geometry type matches ``geometry`` are expanded so that
    each child geometry becomes its own row.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to process.
        geometry (str): The geometry type to explode
            (``"multipolygon"`` or ``"geometrycollection"``).

    Returns:
        GeoDataFrame: A new GeoDataFrame with exploded rows first and
        the preserved (non-matching) rows after.
    """
    new_gdf = gpd.GeoDataFrame()
    to_drop: list[int] = []
    for idx, row in gdf.iterrows():
        geom_type = row.geometry.geom_type.lower()
        if geom_type == geometry:
            n_rows = len(row.geometry.geoms)
            new_gdf = gpd.GeoDataFrame(pd.concat([new_gdf] + [row] * n_rows))
            new_gdf.reset_index(drop=True, inplace=True)
            new_gdf.columns = row.index.values
            for geom in range(n_rows):
                new_gdf.loc[geom, "geometry"] = row.geometry.geoms[geom]
            to_drop.append(idx)

    gdf.drop(labels=to_drop, axis=0, inplace=True)
    new_gdf = gpd.GeoDataFrame(pd.concat([gdf] + [new_gdf]))
    new_gdf.reset_index(drop=True, inplace=True)
    new_gdf.columns = gdf.columns
    return new_gdf


def multi_geom_handler(
    multi_geometry: MultiPolygon | MultiPoint | MultiLineString,
    coord_type: str,
    geom_type: str,
) -> list:
    """Extract per-part coordinates from a multi-geometry.

    Args:
        multi_geometry: A shapely ``MultiPoint`` / ``MultiLineString`` /
            ``MultiPolygon`` instance.
        coord_type (str): ``"x"`` or ``"y"``.
        geom_type (str): One of ``"multipoint"``, ``"multilinestring"``,
            ``"multipolygon"`` (case-insensitive).

    Returns:
        list: A list of per-part coordinate sequences.
    """
    coord_arrays: list[Any] = []
    geom_type = geom_type.lower()
    if geom_type in ("multipoint", "multilinestring"):
        for part in multi_geometry.geoms:
            if geom_type == "multipoint":
                coord_arrays.append(get_point_coords(part, coord_type))
            else:
                coord_arrays.append(get_line_coords(part, coord_type))
    elif geom_type == "multipolygon":
        for part in multi_geometry.geoms:
            coord_arrays.append(get_poly_coords(part, coord_type))
    return coord_arrays


def geometry_collection_coords(geom: Any, coord_type: str) -> list[Any]:
    """Extract coords from every sub-geometry of a GeometryCollection.

    Args:
        geom: A shapely ``GeometryCollection``.
        coord_type (str): ``"x"`` or ``"y"``.

    Returns:
        list: Merged coordinates from Point / LineString / Polygon
        sub-geometries, in iteration order.
    """
    coords: list[Any] = []
    for sub_geom in geom.geoms:
        gtype = sub_geom.geom_type.lower()
        if gtype == "point":
            coords.append(get_point_coords(sub_geom, coord_type))
        elif gtype == "linestring":
            coords.extend(get_line_coords(sub_geom, coord_type))
        elif gtype == "polygon":
            coords.extend(get_poly_coords(sub_geom, coord_type))
    return coords


def get_coords(row: Any, geom_col: str, coord_type: str) -> Any:
    """Return coordinates for a row, dispatching by geometry type.

    Returns ``-9999`` for MultiPolygon rows as a sentinel so the caller
    (currently :meth:`FeatureCollection.xy`) can drop them after the
    per-row ``.apply()``.

    Args:
        row (pandas.Series): A row of the GeoDataFrame.
        geom_col (str): Name of the geometry column.
        coord_type (str): ``"x"`` or ``"y"``.

    Returns:
        Any: Coordinates as ``list`` / ``float`` / ``int``, or the
        sentinel ``-9999`` for MultiPolygon rows.
    """
    geom = row[geom_col]
    gtype = geom.geom_type.lower()
    if gtype == "point":
        return get_point_coords(geom, coord_type)
    if gtype == "linestring":
        return list(get_line_coords(geom, coord_type))
    if gtype == "polygon":
        return list(get_poly_coords(geom, coord_type))
    if gtype == "multipolygon":
        return -9999
    if gtype == "geometrycollection":
        return geometry_collection_coords(geom, coord_type)
    return multi_geom_handler(geom, coord_type, gtype)


def create_polygon(
    coords: list[tuple[float, float]], wkt: bool = False
) -> str | Polygon:
    """Build a :class:`shapely.Polygon` (or its WKT) from coordinates.

    Args:
        coords (list[tuple[float, float]]): Sequence of ``(x, y)``
            tuples forming the ring.
        wkt (bool): Return WKT string form instead of a shapely
            ``Polygon``. Default ``False``.

    Returns:
        str | Polygon: WKT if ``wkt=True``, else the ``Polygon``.
    """
    poly = Polygon(coords)
    return poly.wkt if wkt else poly


def create_point(
    coords: Iterable[tuple[float, ...]], epsg: int | None = None
) -> list[Point] | GeoDataFrame:
    """Build shapely ``Point`` objects (or wrap them in a GeoDataFrame).

    Args:
        coords: Iterable of ``(x, y)`` tuples.
        epsg (int | None): When given, return a GeoDataFrame with the
            supplied CRS; otherwise return a plain ``list[Point]``.

    Returns:
        list[Point] | GeoDataFrame: A list of shapely Points, or a
        GeoDataFrame if ``epsg`` is provided.
    """
    points = list(map(Point, coords))
    if epsg is not None:
        return gpd.GeoDataFrame(
            columns=["geometry"], data=points, crs=epsg
        )
    return points
