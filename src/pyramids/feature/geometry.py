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

    gdf = gdf.drop(labels=to_drop, axis=0)
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


# Legacy polymorphic signatures kept as deprecated shims ---------------


def create_polygon_legacy(
    coords: list[tuple[float, float]], wkt: bool = False
) -> str | Polygon:
    """Deprecated ARC-15 shim. Use :func:`create_polygon` or :func:`polygon_wkt`.

    Returns a WKT string when ``wkt=True``, otherwise a
    :class:`shapely.Polygon`. The polymorphic return type is the
    reason ARC-15 split this into two functions; new code should
    pick the right one explicitly.
    """
    import warnings as _w

    _w.warn(
        "create_polygon(coords, wkt=True) is deprecated (ARC-15). "
        "Use polygon_wkt(coords) for a WKT string, or "
        "create_polygon(coords) for a Polygon.",
        DeprecationWarning,
        stacklevel=2,
    )
    return polygon_wkt(coords) if wkt else create_polygon(coords)


def create_point_legacy(
    coords: Iterable[tuple[float, ...]], epsg: int | None = None
) -> list[Point] | GeoDataFrame:
    """Deprecated ARC-15 shim. Use :func:`create_points` or :func:`point_collection`.

    When ``epsg`` is ``None`` returns a ``list[Point]``; when given,
    returns a ``GeoDataFrame``. New code should pick the right one
    explicitly.
    """
    import warnings as _w

    _w.warn(
        "create_point(coords, epsg=...) with polymorphic return is "
        "deprecated (ARC-15). Use create_points(coords) for the list "
        "or point_collection(coords, crs=...) for a GeoDataFrame.",
        DeprecationWarning,
        stacklevel=2,
    )
    if epsg is not None:
        return point_collection(coords, crs=epsg)
    return create_points(coords)
