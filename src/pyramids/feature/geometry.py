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
    # D-L6 / M2: single-return via lazy axis-index lookup. The previous
    # D-L6 version eagerly called ``geometry.coords.xy`` at the top,
    # which materialised the full coordinate array before the axis
    # check — wasteful on large rings and fatal on geometries whose
    # ``.coords.xy`` raises (e.g. empty geometries). Defer the
    # attribute access until after the coord_type check.
    if coord_type not in ("x", "y"):
        raise ValueError("coord_type can only have a value of 'x' or 'y'")
    idx = 0 if coord_type == "x" else 1
    return list(geometry.coords.xy[idx].tolist())


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
    # D-L6 / M2: single-return via lazy attribute dispatch. The
    # original D-L6 built ``{"x": geometry.x, "y": geometry.y}``
    # which eagerly evaluated BOTH attributes on every call — fatal
    # for empty shapely Points where ``.x`` / ``.y`` raise
    # ``GEOSException`` before the coord_type guard can fire. Defer
    # attribute access until after the string check.
    if coord_type not in ("x", "y"):
        raise ValueError("coord_type can only have a value of 'x' or 'y'")
    value = geometry.x if coord_type == "x" else geometry.y
    return float(value)


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

    D-H1: the input ``gdf`` is **not** mutated. Earlier versions
    silently dropped the exploded rows from the caller's frame via
    ``inplace=True`` — a caller that kept a handle to the input saw
    its data change underneath them. The function now snapshots the
    input up-front and returns a new frame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to process. Not mutated.
        geometry (str): The geometry type to explode
            (``"multipolygon"`` or ``"geometrycollection"``).

    Returns:
        GeoDataFrame: A new GeoDataFrame with exploded rows first and
        the preserved (non-matching) rows after.
    """
    # D-H1: work against a copy so the caller's frame is untouched.
    gdf = gdf.copy()
    # D-L2: accumulate per-child rows into a Python list and do ONE
    # final ``pd.concat`` at the end. The old implementation rebuilt
    # ``new_gdf`` on every iteration via ``pd.concat([new_gdf, row]*n)``,
    # producing O(N²) copying on multi-geometry-heavy frames.
    exploded_rows: list = []
    to_drop: list[int] = []
    for idx, row in gdf.iterrows():
        geom_type = row.geometry.geom_type.lower()
        if geom_type == geometry:
            for child in row.geometry.geoms:
                new_row = row.copy()
                new_row["geometry"] = child
                exploded_rows.append(new_row)
            to_drop.append(idx)

    gdf = gdf.drop(labels=to_drop, axis=0)
    if exploded_rows:
        exploded_gdf = gpd.GeoDataFrame(exploded_rows).reset_index(drop=True)
        result = gpd.GeoDataFrame(pd.concat([gdf, exploded_gdf]))
    else:
        result = gdf
    result.reset_index(drop=True, inplace=True)
    return result


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

    ARC-9: the previous implementation returned the magic value
    ``-9999`` for ``MultiPolygon`` rows as a sentinel that the caller
    (:meth:`FeatureCollection.xy`) filtered out afterwards. That
    conflated real coordinate values of ``-9999`` (plausible in
    several projected CRSes) with the "unhandled geometry" signal and
    could silently drop valid rows. The sentinel is gone; callers
    must explode MultiPolygon rows *before* calling this function
    (:func:`explode_gdf` does exactly that and is always invoked from
    :meth:`FeatureCollection.xy`).

    Args:
        row (pandas.Series): A row of the GeoDataFrame.
        geom_col (str): Name of the geometry column.
        coord_type (str): ``"x"`` or ``"y"``.

    Returns:
        Any: Coordinates as ``list`` / ``float`` / ``int``.

    Raises:
        InvalidGeometryError: If the geometry is a ``MultiPolygon``
            (explode the frame with :func:`explode_gdf` first so that
            each row is a single Polygon — this replaces the old
            ``-9999`` sentinel), or if the geometry is **empty**
            (``geom.is_empty``). C22 turned the silent ``(nan, nan)``
            return into an explicit error so callers that need to
            tolerate empty inputs filter them out beforehand.
    """
    geom = row[geom_col]
    # C22: an empty shapely geometry previously surfaced as
    # ``(nan, nan)`` via ``geom.x`` / ``geom.y`` (Points) or raised an
    # opaque GEOSException. Raise a typed error so callers know to
    # drop empty rows before calling this helper.
    if geom.is_empty:
        from pyramids.base._errors import InvalidGeometryError

        raise InvalidGeometryError(
            "get_coords received an empty geometry. Empty geometries "
            "have no coordinates; filter them out of the GeoDataFrame "
            "(e.g. ``gdf[~gdf.geometry.is_empty]``) before extracting "
            "coordinates."
        )
    gtype = geom.geom_type.lower()
    if gtype == "point":
        return get_point_coords(geom, coord_type)
    if gtype == "linestring":
        return list(get_line_coords(geom, coord_type))
    if gtype == "polygon":
        return list(get_poly_coords(geom, coord_type))
    if gtype == "multipolygon":
        from pyramids.base._errors import InvalidGeometryError

        raise InvalidGeometryError(
            "get_coords does not accept MultiPolygon rows — explode the "
            "GeoDataFrame with explode_gdf(gdf, 'multipolygon') first "
            "(ARC-9: previously returned the -9999 sentinel, which "
            "silently clashed with real coordinate values)."
        )
    if gtype == "geometrycollection":
        return geometry_collection_coords(geom, coord_type)
    return multi_geom_handler(geom, coord_type, gtype)


def create_polygon(coords: list[tuple[float, float]]) -> Polygon:
    """Build a :class:`shapely.Polygon` from a sequence of (x, y) tuples.

    ARC-15: the return type is now unconditional — always a
    ``Polygon``. For the WKT string form use :func:`polygon_wkt`.

    C21: validates that the ring has at least 3 distinct-capable
    vertices. Shapely itself accepts 2-vertex input and produces an
    invalid polygon; raising here surfaces the user error at the
    point of origin.

    Args:
        coords (list[tuple[float, float]]): Sequence of ``(x, y)``
            tuples forming the ring. Must have at least 3 vertices.

    Returns:
        Polygon: A shapely ``Polygon``.

    Raises:
        InvalidGeometryError: If ``coords`` has fewer than 3 vertices.
    """
    if len(coords) < 3:
        from pyramids.base._errors import InvalidGeometryError

        raise InvalidGeometryError(
            f"create_polygon requires at least 3 vertices; got "
            f"{len(coords)}. A valid polygon ring needs three distinct "
            f"corners."
        )
    return Polygon(coords)


def polygon_wkt(coords: list[tuple[float, float]]) -> str:
    """Build a WKT string for a polygon from (x, y) tuples (ARC-15).

    Args:
        coords (list[tuple[float, float]]): Ring coordinates.

    Returns:
        str: Well-Known Text representation of the polygon.
    """
    return Polygon(coords).wkt


def create_points(coords: Iterable[tuple[float, ...]]) -> list[Point]:
    """Build a list of shapely ``Point`` objects from (x, y) tuples.

    ARC-15: the return type is now unconditional — always a
    ``list[Point]``. For the GeoDataFrame wrapper use
    :func:`point_collection`.

    Args:
        coords: Iterable of ``(x, y)`` tuples.

    Returns:
        list[Point]: The constructed shapely Points.
    """
    return list(map(Point, coords))


def point_collection(
    coords: Iterable[tuple[float, ...]], crs: Any
) -> GeoDataFrame:
    """Build a :class:`GeoDataFrame` of points with a given CRS (ARC-15).

    Args:
        coords: Iterable of ``(x, y)`` tuples.
        crs: A CRS accepted by :class:`geopandas.GeoDataFrame` (EPSG
            int, WKT, Proj string, etc.).

    Returns:
        GeoDataFrame: A GeoDataFrame with a single ``geometry`` column.
    """
    pts = create_points(coords)
    return gpd.GeoDataFrame(columns=["geometry"], data=pts, crs=crs)


