"""Lazy vector collection — subclasses :class:`dask_geopandas.GeoDataFrame`.

DASK-40b: separate class (ARC-29 implemented). A LazyFeatureCollection IS a
:class:`dask_geopandas.GeoDataFrame`; it is NOT a
:class:`~pyramids.feature.FeatureCollection` or :class:`geopandas.GeoDataFrame`.
Both classes satisfy the :class:`~pyramids.base.protocols.SpatialObject`
protocol so consumers accept either via ``isinstance(x, SpatialObject)``.

Importing this module requires ``dask-geopandas`` to be installed (the
``[parquet-lazy]`` extra). The eager ``FeatureCollection`` is unaffected:
its readers only import this module from inside the ``backend='dask'``
branch, so minimal installs never evaluate this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask_geopandas

if TYPE_CHECKING:
    from pathlib import Path

    from pyramids.feature.collection import FeatureCollection


class LazyFeatureCollection(dask_geopandas.GeoDataFrame):
    """Lazy vector collection backed by a dask_geopandas graph.

    Subclasses :class:`dask_geopandas.GeoDataFrame` — every method of that
    parent is available unchanged (``to_crs``, ``clip``, ``sjoin``,
    ``.npartitions``, ``.spatial_partitions``). Overrides: ``compute`` /
    ``persist`` return pyramids-native wrappers, ``spatial_shuffle``
    re-wraps the shuffled result, ``to_file`` / ``plot`` raise with
    actionable messages, and ``read_file`` is a classmethod delegating
    back to the eager reader so the
    :class:`~pyramids.base.protocols.SpatialObject` protocol is satisfied.

    Construct via ``FeatureCollection.read_file(..., backend='dask')`` or
    ``FeatureCollection.read_parquet(..., backend='dask')``. Direct
    construction is discouraged; use
    :meth:`LazyFeatureCollection._from_dask_gdf` if you must wrap a
    pre-built dask-geopandas frame.

    Note:
        On installs without the ``[parquet-lazy]`` extra, importing this
        class from :mod:`pyramids.feature` returns ``None`` (sentinel).
        Consumer code using ``isinstance`` must guard explicitly:
        ``if LazyFeatureCollection is not None and isinstance(x, LazyFeatureCollection): ...``.
    """

    @classmethod
    def _from_dask_gdf(cls, dask_gdf: Any) -> LazyFeatureCollection:
        """Wrap an existing ``dask_geopandas.GeoDataFrame`` as a LazyFC.

        Uses class-swap (``copy()`` + ``__class__`` assignment) rather than
        a four-arg reconstruction, because ``(dask, _name, _meta, divisions)``
        is private dask.dataframe API that shifted between dask-expr and
        legacy. Class-swap relies only on Python's object model and
        survives upstream churn.
        """
        result = dask_gdf.copy()
        result.__class__ = cls
        return result

    @classmethod
    def read_file(
        cls,
        path: str | Path,
        *,
        npartitions: int | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> LazyFeatureCollection:
        """Construct a LazyFC from a vector file.

        Satisfies the :class:`~pyramids.base.protocols.SpatialObject`
        protocol's ``read_file`` classmethod requirement. Delegates to
        :meth:`FeatureCollection.read_file` with ``backend='dask'``.

        The import of :class:`FeatureCollection` is inline by
        necessity — :mod:`pyramids.feature.collection` imports this module
        from inside its dask branch, so a top-of-file import here would
        form a cycle. Per ``CLAUDE.md``, circular-import breaks are an
        accepted exception to the no-inline-imports rule.
        """
        from pyramids.feature.collection import FeatureCollection

        result = FeatureCollection.read_file(
            path,
            backend="dask",
            npartitions=npartitions,
            chunksize=chunksize,
            **kwargs,
        )
        return result

    @property
    def epsg(self) -> int | None:
        """EPSG code of the CRS, or ``None`` when the CRS is unset."""
        crs = self.crs
        result = None if crs is None else crs.to_epsg()
        return result

    @property
    def top_left_corner(self) -> list[float]:
        """Top-left corner ``[xmin, ymax]`` of the total bounds.

        :attr:`dask_geopandas.GeoDataFrame.total_bounds` returns a lazy
        dask Scalar — this property calls ``.compute()`` on that single
        Scalar (cheap, O(partitions) — no materialization of the full
        frame) so :class:`~pyramids.base.protocols.SpatialObject`
        consumers get concrete numbers.
        """
        bounds = list(self.total_bounds.compute())
        result = [bounds[0], bounds[3]]
        return result

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """No lazy plot path; materialize first.

        Raises:
            NotImplementedError: Always. Call ``.compute().plot(...)`` to
                materialize before plotting.
        """
        raise NotImplementedError(
            "LazyFeatureCollection.plot is not supported — call "
            ".compute().plot(...) to materialize first."
        )

    def compute(self, **kwargs: Any) -> FeatureCollection:
        """Materialize the graph and return an eager :class:`FeatureCollection`.

        The ``from pyramids.feature.collection import FeatureCollection``
        import below is inline by necessity — ``collection.py`` imports
        this module from inside its dask branch, so a top-of-file import
        here would form a cycle. Per ``CLAUDE.md``, circular-import
        breaks are the second accepted exception to the no-inline-imports
        rule.

        Args:
            **kwargs: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.compute` (e.g.
                ``scheduler="processes"``).

        Returns:
            FeatureCollection: an eager wrapper around the materialized
            :class:`geopandas.GeoDataFrame`.
        """
        from pyramids.feature.collection import FeatureCollection

        gdf = super().compute(**kwargs)
        result = FeatureCollection(gdf)
        return result

    def persist(self, **kwargs: Any) -> LazyFeatureCollection:
        """Persist partitions in worker memory; keep the FC lazy.

        Args:
            **kwargs: Forwarded to
                :meth:`dask_geopandas.GeoDataFrame.persist`.

        Returns:
            LazyFeatureCollection: A new lazy FC whose graph has been
            materialized into worker memory.
        """
        persisted = super().persist(**kwargs)
        result = type(self)._from_dask_gdf(persisted)
        return result

    def spatial_shuffle(
        self,
        by: str = "hilbert",
        level: int = 16,
        *,
        npartitions: int | None = None,
        divisions: Any = None,
    ) -> LazyFeatureCollection:
        """Spatially sort rows via a space-filling curve.

        Thin override of :meth:`dask_geopandas.GeoDataFrame.spatial_shuffle`
        that rewraps the result as a :class:`LazyFeatureCollection`.
        Populates :attr:`spatial_partitions`, enabling partition-pruned
        :meth:`sjoin` on the result.

        Args:
            by: One of ``"hilbert"``, ``"morton"``, or ``"geohash"``.
            level: Curve resolution; default 16.
            npartitions: Optional output partition count.
            divisions: Optional explicit division boundaries.

        Returns:
            LazyFeatureCollection: A new lazy FC with ``spatial_partitions``
            populated.
        """
        kwargs: dict[str, Any] = {}
        if npartitions is not None:
            kwargs["npartitions"] = npartitions
        if divisions is not None:
            kwargs["divisions"] = divisions
        shuffled = super().spatial_shuffle(by=by, level=level, **kwargs)
        result = type(self)._from_dask_gdf(shuffled)
        return result

    def to_file(self, path: Any, *args: Any, **kwargs: Any) -> None:
        """No lazy OGR write path exists; raise with an actionable message.

        Raises:
            NotImplementedError: Always. dask-geopandas has no lazy OGR
                write path for GeoJSON / Shapefile / GeoPackage. Call
                ``.compute().to_file(path)`` to materialize first, or
                use ``to_parquet`` which IS lazy.
        """
        raise NotImplementedError(
            "LazyFeatureCollection.to_file is not supported — "
            "dask-geopandas has no lazy OGR write path. Call "
            ".compute().to_file(path) to materialize first, or "
            "write via read_parquet + fc.to_parquet(path) which IS lazy."
        )

    def __repr__(self) -> str:
        """Pyramids-branded repr; avoids the generic Dask DataFrame one."""
        return (
            f"LazyFeatureCollection(npartitions={self.npartitions}, "
            f"columns={list(self.columns)}, crs={self.crs})"
        )
