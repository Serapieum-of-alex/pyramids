"""The capability facades `UgridDataset` exposes, as a mixin.

The algorithms live in focused modules — :mod:`~pyramids.netcdf.ugrid.graph` (topology and
graph operations), :mod:`~pyramids.netcdf.ugrid.fill` (spatial gap filling),
:mod:`~pyramids.netcdf.ugrid.vector_ops` (mesh↔vector), :mod:`~pyramids.netcdf.ugrid.regrid`
(sampling and regridding) and :mod:`~pyramids.netcdf.ugrid.interop_xarray` (xarray / zarr /
dask hand-off). Each takes a :class:`~pyramids.netcdf.ugrid.mesh.Mesh2d` plus plain arrays, so
none of them imports ``ugrid.dataset`` and the import graph stays acyclic.

This mixin is the user-facing half: it resolves a variable by name, hands the kernel the mesh
and the values, and wraps the answer back into a dataset / variable / GeoDataFrame. Keeping the
facades here rather than in ``dataset.py`` keeps that module about the data model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.netcdf.ugrid import fill as _fill
from pyramids.netcdf.ugrid import graph as _graph
from pyramids.netcdf.ugrid import interop_xarray as _interop
from pyramids.netcdf.ugrid import regrid as _regrid
from pyramids.netcdf.ugrid import vector_ops as _vector_ops

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyramids.netcdf.ugrid.models import MeshVariable


def _numeric_stamps(stamps: Any) -> np.ndarray:
    """Time coordinates as float64, or a written error when they are not numeric.

    Args:
        stamps: The time coordinate values.

    Returns:
        np.ndarray: The coordinates as float64.

    Raises:
        ValueError: The coordinates are date-like or otherwise non-numeric, which the
            interpolation cannot measure along. A bare ``TypeError`` about `Timestamp` from
            inside numpy is not an answer.
    """
    try:
        return np.asarray(stamps, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "interp_time needs numeric time coordinates to interpolate between; this "
            "dataset's cannot be read as numbers. Convert them to numbers first."
        ) from exc


class _UgridFacades:
    """Capability facades mixed into :class:`~pyramids.netcdf.ugrid.dataset.UgridDataset`."""

    def _facade_variable(self, operation: str, variable_name: str | None) -> Any:
        """The variable ``operation`` acts on, defaulting to the only data variable.

        Most of these capabilities act on one field. Naming it is optional when the dataset
        carries exactly one, which is the common case for a derived mesh; with several, the
        name is required rather than guessed.

        Args:
            operation: The calling method's name, for the message.
            variable_name: The requested name, or ``None`` to take the only one.

        Returns:
            MeshVariable: The resolved variable.

        Raises:
            KeyError: The named variable is not in this dataset.
            ValueError: ``None`` was given but the dataset does not hold exactly one variable,
                or the resolved variable has no loaded data.
        """
        names = list(self._data_variables)  # type: ignore[attr-defined]
        if variable_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"{operation}: this dataset holds {len(names)} data variable(s) "
                    f"{names}; name the one to use."
                )
            variable_name = names[0]
        if variable_name not in names:
            raise KeyError(variable_name)
        var = self._data_variables[variable_name]  # type: ignore[attr-defined]
        if var.data is None:
            raise ValueError(
                f"{operation}: variable {variable_name!r} has no loaded data."
            )
        return var

    def _rewrap(self, var: MeshVariable, values: np.ndarray) -> Any:
        """A new dataset on the same mesh, with ``var`` replaced by ``values``."""
        new_vars = dict(self._data_variables)  # type: ignore[attr-defined]
        new_vars[var.name] = var.with_data(np.asarray(values))
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def connected_components(self) -> np.ndarray:
        """Label the mesh's connected face components.

        Returns:
            numpy.ndarray: One component label per face.
        """
        return _graph.connected_components(self._mesh)  # type: ignore[attr-defined]

    def exterior_edges(self) -> np.ndarray:
        """The indices of the edges on the mesh boundary.

        Returns:
            numpy.ndarray: Boundary edge indices.
        """
        return _graph.exterior_edges(self._mesh)  # type: ignore[attr-defined]

    def exterior_faces(self) -> np.ndarray:
        """The indices of the faces touching the mesh boundary.

        Returns:
            numpy.ndarray: Boundary face indices.
        """
        return _graph.exterior_faces(self._mesh)  # type: ignore[attr-defined]

    def bounding_polygon(self) -> Any:
        """The mesh footprint as one dissolved polygon, holes preserved.

        Returns:
            shapely.geometry.base.BaseGeometry: The (multi)polygon of the mesh outline.
        """
        return _graph.bounding_polygon(self._mesh)  # type: ignore[attr-defined]

    def circumcenters(self) -> np.ndarray:
        """The circumcentre of every face (triangular meshes only).

        Returns:
            numpy.ndarray: ``(n_face, 2)`` circumcentre coordinates.

        Raises:
            ValueError: The mesh is not triangular.
        """
        return _graph.circumcenters(self._mesh)  # type: ignore[attr-defined]

    def voronoi(self) -> Any:
        """The Voronoi tesselation of the mesh nodes (the mesh dual).

        Returns:
            scipy.spatial.Voronoi: The tesselation.
        """
        return _graph.voronoi(self._mesh)  # type: ignore[attr-defined]

    def sparse_adjacency(self, kind: str = "face") -> Any:
        """The mesh adjacency as a sparse matrix.

        Args:
            kind: ``"face"`` (faces sharing an edge) or ``"node"`` (nodes sharing an edge).

        Returns:
            scipy.sparse.csr_matrix: The symmetric adjacency.
        """
        return _graph.sparse_adjacency(self._mesh, kind)  # type: ignore[attr-defined]

    def binary_dilation(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Grow a boolean face mask over its neighbours.

        Args:
            variable_name: The boolean face variable to dilate, or ``None`` for the only one.
            iterations: How many times to grow. Defaults to 1.
            border_value: What lies outside the mesh. Defaults to False.

        Returns:
            UgridDataset: A new dataset with the dilated mask.
        """
        var = self._facade_variable("binary_dilation", variable_name)
        grown = _graph.binary_dilation(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data).astype(bool),
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )
        return self._rewrap(var, grown)

    def binary_erosion(
        self,
        variable_name: str | None = None,
        *,
        iterations: int = 1,
        border_value: bool = False,
        fixed: np.ndarray | None = None,
    ) -> Any:
        """Shrink a boolean face mask away from its boundary.

        Args:
            variable_name: The boolean face variable to erode, or ``None`` for the only one.
            iterations: How many times to shrink. Defaults to 1.
            border_value: What lies outside the mesh. Defaults to False, which erodes the
                boundary; pass True to erode only the interior.

        Returns:
            UgridDataset: A new dataset with the eroded mask.
        """
        var = self._facade_variable("binary_erosion", variable_name)
        shrunk = _graph.binary_erosion(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data).astype(bool),
            iterations=iterations,
            border_value=border_value,
            fixed=fixed,
        )
        return self._rewrap(var, shrunk)

    def to_location(self, location: str, variable_name: str | None = None) -> Any:
        """Remap a variable onto another facet of the mesh.

        Averages the values of the contributing elements, so a face field asked for on the
        nodes becomes each node's mean of the faces around it.

        Args:
            location: The facet to remap onto — ``"node"``, ``"edge"`` or ``"face"``.
            variable_name: The variable to remap, or ``None`` for the only one.

        Returns:
            UgridDataset: A new dataset whose variable sits on ``location``.

        Raises:
            ValueError: An unknown ``location``, or a remap the kernel does not offer.
        """
        var = self._facade_variable("to_location", variable_name)
        remap = {"node": _graph.to_node, "edge": _graph.to_edge, "face": _graph.to_face}
        if location not in remap:
            raise ValueError(
                f"to_location: location must be one of {sorted(remap)}, got {location!r}."
            )
        if var.location == location:
            # Already there. The averaging kernels only accept the two *other* facets as a
            # source, so without this an identity remap failed with "source must be 'node' or
            # 'edge'" -- a confusing answer to a request that is trivially already satisfied.
            return self._rebuild(dict(self._data_variables))  # type: ignore[attr-defined]
        values = remap[location](
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            source=var.location,
        )
        moved = var.with_data(np.asarray(values))
        moved.location = location
        # An element with no contributor (an orphan node, say) averages to NaN, so the remapped
        # variable has to declare NaN as its no-data value -- the same reasoning the regrid facade
        # already applies. Otherwise `to_file` writes those NaNs under the old `_FillValue`.
        moved.nodata = float("nan")
        if moved.dimensions:
            # The element axis is the last dimension, by the mesh-variable invariant. Renaming it
            # to the new facet matters because a file-read variable carries real dimension names
            # and the xarray bridge prefers them: leaving the old facet's name on an axis that now
            # has the new facet's length makes the export fail on conflicting dimension sizes.
            mesh_name = self.mesh_name  # type: ignore[attr-defined]
            moved.dimensions = moved.dimensions[:-1] + (
                f"{mesh_name}_n{location.capitalize()}s",
            )
        new_vars = dict(self._data_variables)  # type: ignore[attr-defined]
        new_vars[var.name] = moved
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def fill_na_spatial(
        self, variable_name: str | None = None, *, max_distance: float | None = None
    ) -> Any:
        """Fill each gap from the nearest valid element in space.

        Where :meth:`interpolate_na` fills along time, this fills across the mesh: a gap takes
        the value of the closest element that has one.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            max_distance: Leave a gap unfilled when no valid element is this close. ``None``
                (the default) is unbounded.

        Returns:
            UgridDataset: A new dataset with the gaps filled.
        """
        var = self._facade_variable("fill_na_spatial", variable_name)
        filled = _fill.fill_na_spatial(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            var.location,
            max_distance=max_distance,
        )
        return self._rewrap(var, filled)

    def laplace_interpolate(
        self,
        variable_name: str | None = None,
        *,
        use_weights: bool = True,
        direct_solve: bool = False,
    ) -> Any:
        """Fill the interior gaps by solving Laplace's equation over the mesh.

        The valid values are the boundary condition and each gap becomes the smooth solution
        between them — a better hole-filler than nearest when the field varies smoothly.

        Args:
            variable_name: The variable to fill, or ``None`` for the only one.
            use_weights: Weight the graph Laplacian by inverse distance. Defaults to True.
            direct_solve: Solve directly rather than iteratively. Defaults to False.

        Returns:
            UgridDataset: A new dataset with the interior gaps filled.
        """
        var = self._facade_variable("laplace_interpolate", variable_name)
        filled = _fill.laplace_interpolate(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            var.location,
            use_weights=use_weights,
            direct_solve=direct_solve,
        )
        return self._rewrap(var, filled)

    def polygonize(
        self, variable_name: str | None = None, *, decimals: int | None = None
    ) -> Any:
        """Dissolve adjacent faces of equal value into polygons.

        Args:
            variable_name: The face variable to dissolve by, or ``None`` for the only one.
            decimals: Round to this many decimals before grouping, for approximate equality.

        Returns:
            geopandas.GeoDataFrame: One row per merged region, with its value and geometry.
        """
        var = self._facade_variable("polygonize", variable_name)
        return _vector_ops.polygonize(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            decimals=decimals,
            crs=self.crs,  # type: ignore[attr-defined]
        )

    def sample_line(self, line: Any, variable_name: str | None = None) -> Any:
        """Sample a variable along a line, ordered by distance along it.

        Args:
            line: A shapely ``LineString``, or a sequence of coordinates.
            variable_name: The variable to sample, or ``None`` for the only one.

        Returns:
            tuple: ``(distances, values, face_indices)`` for the faces the line crosses.
        """
        var = self._facade_variable("sample_line", variable_name)
        return _vector_ops.sample_line(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            line,
        )

    def burn(
        self,
        geometry: Any,
        value: float,
        *,
        name: str = "burned",
        fill: float = float("nan"),
        all_touched: bool = True,
    ) -> Any:
        """Burn vector geometry onto the faces as a new variable.

        The mesh analogue of rasterising a vector layer: every face the geometry covers takes
        ``value``, the rest take ``fill``.

        Args:
            geometry: A shapely geometry, GeoDataFrame, FeatureCollection, or iterable of them.
            value: The value burned into the covered faces.
            name: The new variable's name. Defaults to ``"burned"``.
            fill: The value for the faces the geometry misses. Defaults to NaN.
            all_touched: Burn every face the geometry touches, rather than only those whose
                centroid falls inside. Defaults to True.

        Returns:
            UgridDataset: A new dataset carrying the burned variable.
        """
        values = _vector_ops.burn(
            self._mesh,  # type: ignore[attr-defined]
            geometry,
            value,
            fill=fill,
            all_touched=all_touched,
        )
        # Declare the fill as the variable's no-data value, so an untouched face is
        # distinguishable from data once written. The default fill is NaN.
        return self.with_variable(  # type: ignore[attr-defined]
            name, np.asarray(values), location="face", nodata=float(fill)
        )

    def rasterize_like(
        self,
        reference: Any,
        variable_name: str | None = None,
        *,
        method: str = "nearest",
        nodata: float = -9999.0,
    ) -> Any:
        """Interpolate a variable onto an existing raster's grid.

        Where :meth:`to_dataset` builds its own grid from a cell size, this snaps to the
        geotransform, shape and CRS of a raster you already have, so the result is
        co-registered with it.

        Args:
            reference: The raster whose grid to match (a pyramids ``Dataset`` / ``NetCDF``).
            variable_name: The variable to interpolate, or ``None`` for the only one.
            method: ``"nearest"``, ``"linear"`` or ``"idw"``. Defaults to ``"nearest"``.
            nodata: The value for cells the mesh does not cover.

        Returns:
            Dataset: A raster on ``reference``'s grid.
        """
        var = self._facade_variable("rasterize_like", variable_name)
        return _regrid.rasterize_like(
            self._mesh,  # type: ignore[attr-defined]
            np.asarray(var.data),
            reference,
            location=var.location,
            method=method,
            nodata=nodata,
            source_epsg=self.epsg,  # type: ignore[attr-defined]
        )

    def interp_time(self, new_times: Any, *, method: str = "linear") -> Any:
        """Interpolate every temporal variable onto new time coordinates.

        Distinct from :meth:`interpolate_na`, which fills gaps on the existing axis: this
        *relabels* the axis, resampling the series onto the times you ask for.

        Args:
            new_times: The time coordinates to interpolate onto.
            method: A ``scipy.interpolate.interp1d`` kind. Defaults to ``"linear"``.

        Returns:
            UgridDataset: A new dataset on the requested time coordinates.

        Raises:
            ValueError: No variable has a time dimension.
        """
        stamps = self._time_coords("interp_time")  # type: ignore[attr-defined]
        targets = list(np.atleast_1d(np.asarray(new_times)))
        new_vars: dict[str, Any] = {}
        for name, var in self._data_variables.items():  # type: ignore[attr-defined]
            if not var.has_time:
                new_vars[name] = var
                continue
            moved = _regrid.interp_time(
                _numeric_stamps(stamps),
                np.asarray(var.data),
                np.asarray(targets, dtype="float64"),
                method=method,
                axis=cast("int", var.time_index),
            )
            new_var = var.with_data(np.asarray(moved))
            new_var.attributes = {**var.attributes, "time_values": targets}
            new_vars[name] = new_var
        return self._rebuild(new_vars)  # type: ignore[attr-defined]

    def regrid(
        self, target: Any, *, method: str = "overlap", variable_name: str | None = None
    ) -> Any:
        """Regrid a variable onto another mesh.

        Args:
            target: The destination ``UgridDataset`` (its mesh is the target).
            method: ``"overlap"`` (area-weighted mean), ``"conservative"`` (integral
                preserving), ``"centroid"`` or ``"barycentric"``. Defaults to ``"overlap"``.
            variable_name: The variable to regrid, or ``None`` for the only one.

        Returns:
            UgridDataset: ``target``'s mesh carrying the regridded variable.
        """
        var = self._facade_variable("regrid", variable_name)
        regridder = _regrid.Regridder(
            self._mesh,  # type: ignore[attr-defined]
            target.mesh,
            method=method,
            source_location=var.location,
        )
        values = regridder.regrid(np.asarray(var.data))
        # A target element the source does not cover comes back NaN, so the regridded variable
        # has to declare NaN as its no-data value -- otherwise `to_file` writes those NaNs with
        # no `_FillValue` and a reader cannot tell them from data.
        return target.with_variable(
            var.name, np.asarray(values), location="face", nodata=float("nan")
        )

    def to_xarray(self) -> Any:
        """Hand the mesh and its variables to xarray, in the UGRID-CF encoding.

        xarray is a **peer dependency**, imported only here (as the raster
        :meth:`pyramids.netcdf.NetCDF.to_xarray` does), so nothing here imports it at module
        scope. Note the parent ``pyramids.netcdf`` package does currently pull xarray and dask
        at import time for unrelated reasons, so that cost is not avoided in practice yet.

        Returns:
            xarray.Dataset: The mesh's UGRID-CF variables plus the data variables.

        Raises:
            OptionalPackageDoesNotExist: xarray is not installed.
        """
        return _interop.to_xarray(
            self._mesh,  # type: ignore[attr-defined]
            self._data_variables,  # type: ignore[attr-defined]
            # The dataset's own topology name, not the default: exporting a mesh called
            # anything else wrote its topology as "mesh2d" while the data variables kept
            # their own element dimension, leaving two unrelated dimensions in one file.
            mesh_name=self.mesh_name,  # type: ignore[attr-defined]
            crs_wkt=self.crs_wkt,  # type: ignore[attr-defined]
        )

    def to_zarr(self, store: Any, **kwargs: Any) -> Any:
        """Write the mesh and its variables to a zarr store, through xarray.

        Args:
            store: The zarr store (a path or a mapping).
            **kwargs: Passed to ``xarray.Dataset.to_zarr``.

        Returns:
            The value ``xarray.Dataset.to_zarr`` returns.

        Raises:
            OptionalPackageDoesNotExist: xarray or zarr is not installed.
        """
        return _interop.to_zarr(
            self._mesh,  # type: ignore[attr-defined]
            self._data_variables,  # type: ignore[attr-defined]
            store,
            mesh_name=self.mesh_name,  # type: ignore[attr-defined]
            crs_wkt=self.crs_wkt,  # type: ignore[attr-defined]
            **kwargs,
        )
