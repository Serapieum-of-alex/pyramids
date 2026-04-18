"""Zonal statistics over a :class:`Dataset` + polygon
:class:`FeatureCollection`.

DASK-25 first cut: single-pass rasterize of every polygon into an
integer label grid, then numpy-based per-label reductions. Follows
xvec's ``method="rasterize"`` pattern (single rasterize, then
groupby). Optional ``method="exactextract"`` path delegates to the
``exactextract`` pybind11 library when installed.

Scope kept narrow:

* Supported stats: ``mean``, ``sum``, ``min``, ``max``, ``count``,
  ``std``, ``var``.
* One band at a time (caller selects via ``band=``).
* Raster and vector must share a CRS — caller reprojects first if
  not. (A later enhancement can auto-align.)
* Output is a ``pandas.DataFrame`` indexed by the FeatureCollection
  row index, with one column per stat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature import FeatureCollection


_STAT_FUNCS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "var": np.nanvar,
    "count": lambda vals: float(np.sum(~np.isnan(vals))),
}


_EXACTEXTRACT_IMPORT_ERROR = (
    "method='exactextract' requires the optional 'exactextract' "
    "dependency. Install with: pip install 'pyramids-gis[zonal]'"
)


def _rasterize_labels(ds: "Dataset", fc: "FeatureCollection") -> np.ndarray:
    """Rasterize ``fc`` into an integer label array shaped like ``ds``.

    Label values are 0-based feature-index integers. Pixels not
    covered by any polygon get -1.

    H4: the FeatureCollection's CRS is attached to the OGR layer so
    GDAL's ``RasterizeLayer`` reprojects coordinates when the vector
    and raster CRSes disagree. A mismatch that ``pyproj`` considers
    incompatible raises :class:`ValueError` early rather than silently
    producing a mis-aligned label grid.
    """
    from osgeo import gdal, ogr, osr

    fc_crs = getattr(fc, "crs", None)
    ds_epsg = int(ds.epsg) if ds.epsg else None
    if fc_crs is not None and ds_epsg is not None:
        fc_epsg = fc_crs.to_epsg()
        if fc_epsg is not None and fc_epsg != ds_epsg:
            raise ValueError(
                f"zonal_stats: FeatureCollection CRS (EPSG:{fc_epsg}) does "
                f"not match Dataset CRS (EPSG:{ds_epsg}). Reproject the "
                "FeatureCollection via fc.to_crs(ds.epsg) first."
            )

    srs = osr.SpatialReference()
    if fc_crs is not None:
        srs.ImportFromWkt(fc_crs.to_wkt())
    elif ds_epsg is not None:
        srs.ImportFromEPSG(ds_epsg)
    mem_driver = ogr.GetDriverByName("Memory")
    ds_vec = mem_driver.CreateDataSource("zonal_mem")
    layer = ds_vec.CreateLayer("features", srs=srs, geom_type=ogr.wkbPolygon)
    id_field = ogr.FieldDefn("pid", ogr.OFTInteger)
    layer.CreateField(id_field)
    for idx, geom in enumerate(fc.geometry):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("pid", int(idx))
        ogr_geom = ogr.CreateGeometryFromWkb(geom.wkb)
        feat.SetGeometry(ogr_geom)
        layer.CreateFeature(feat)
        feat = None

    mem_drv = gdal.GetDriverByName("MEM")
    label_ds = mem_drv.Create("", ds.columns, ds.rows, 1, gdal.GDT_Int32)
    label_ds.SetGeoTransform(ds.geotransform)
    label_ds.SetProjection(ds.raster.GetProjection())
    label_ds.GetRasterBand(1).Fill(-1)
    gdal.RasterizeLayer(
        label_ds, [1], layer,
        options=["ATTRIBUTE=pid", "ALL_TOUCHED=FALSE"],
    )
    labels = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None
    ds_vec = None
    return labels


def _rasterize_zonal_stats(
    ds: "Dataset",
    fc: "FeatureCollection",
    stats: Sequence[str],
    band: int,
    no_data: float | None,
) -> pd.DataFrame:
    """Compute stats via single-rasterize + numpy groupby per label."""
    raster = np.asarray(ds.read_array(band=band), dtype=np.float64)
    if no_data is not None:
        raster = np.where(raster == no_data, np.nan, raster)
    labels = _rasterize_labels(ds, fc)
    n_features = len(fc)
    rows: list[dict[str, float]] = []
    for pid in range(n_features):
        mask = labels == pid
        vals = raster[mask]
        rows.append({stat: _apply_stat(stat, vals) for stat in stats})
    return pd.DataFrame(rows, index=fc.index)


def _apply_stat(stat: str, vals: np.ndarray) -> float:
    """Safe per-polygon stat — empty cohort or all-NaN returns NaN."""
    if vals.size == 0:
        return float("nan")
    valid = vals[~np.isnan(vals)] if stat != "count" else vals
    if stat != "count" and valid.size == 0:
        return float("nan")
    try:
        func = _STAT_FUNCS[stat]
    except KeyError as exc:
        raise ValueError(
            f"unknown stat {stat!r}; supported: {sorted(_STAT_FUNCS)}"
        ) from exc
    return float(func(vals))


def _exactextract_zonal_stats(
    ds: "Dataset",
    fc: "FeatureCollection",
    stats: Sequence[str],
) -> pd.DataFrame:
    """Delegate to :mod:`exactextract` for area-weighted zonal stats."""
    try:
        from exactextract import exact_extract
    except ImportError as exc:
        raise ImportError(_EXACTEXTRACT_IMPORT_ERROR) from exc
    result = exact_extract(
        rast=ds.raster, vec=fc, ops=list(stats), output="pandas",
    )
    return result


def zonal_stats(
    ds: "Dataset",
    fc: "FeatureCollection",
    *,
    stats: Sequence[str] = ("mean",),
    method: str = "rasterize",
    band: int = 0,
) -> pd.DataFrame:
    """Compute zonal statistics of ``ds`` over polygons in ``fc``.

    Args:
        ds: The source :class:`~pyramids.dataset.Dataset`.
        fc: A :class:`~pyramids.feature.FeatureCollection` of polygons.
            CRS must match ``ds`` — reproject first if needed.
        stats: Statistics to compute per polygon. One or more of
            ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``,
            ``"var"``, ``"count"``.
        method: ``"rasterize"`` (default) uses a single-pass GDAL
            rasterize then numpy groupby per label — fast and
            accurate down to one-pixel polygons. ``"exactextract"``
            delegates to the :mod:`exactextract` library for
            area-weighted stats useful when polygons are smaller
            than a pixel.
        band: Zero-based band index on ``ds``. Default 0.

    Returns:
        pandas.DataFrame: Indexed by ``fc.index``; one column per
        stat.

    Raises:
        ValueError: Unknown ``stat`` name or unknown ``method``.
        ImportError: ``method="exactextract"`` without exactextract
            installed.
    """
    if method == "exactextract":
        return _exactextract_zonal_stats(ds, fc, stats)
    if method != "rasterize":
        raise ValueError(
            f"method must be 'rasterize' or 'exactextract', got {method!r}"
        )
    no_data_list = ds.no_data_value
    no_data = no_data_list[band] if no_data_list else None
    return _rasterize_zonal_stats(ds, fc, stats, band, no_data)


__all__ = ["zonal_stats"]
