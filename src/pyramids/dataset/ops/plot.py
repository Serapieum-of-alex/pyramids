"""Plot mixin for the Dataset class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import DataFrame

from pyramids.base._utils import import_cleopatra

if TYPE_CHECKING:
    from cleopatra.array_glyph import ArrayGlyph


class PlotMixin:

    def plot(
        self,
        band: int | None = None,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        overview: bool | None = False,
        overview_index: int | None = 0,
        percentile: int | None = None,
        **kwargs: Any,
    ) -> ArrayGlyph:
        """Plot the values/overviews of a given band.

        The plot function uses the `cleopatra` as a backend to plot the raster data, for more information check
        [ArrayGlyph](https://serapieum-of-alex.github.io/cleopatra/latest/api/array-glyph-class/#cleopatra.array_glyph.ArrayGlyph.plot).

        Args:
            band (int, optional):
                The band you want to get its data. Default is 0.
            exclude_value (Any, optional):
                Value to exclude from the plot. Default is None.
            rgb (List[int], optional):
                The indices of the red, green, and blue bands in the `Dataset`. the `rgb` parameter can be a list of
                three values, or a list of four values if the alpha band is also included.
                The `plot` method will check if the rgb bands are defined in the `Dataset`, if all the three bands (
                red, green, blue)) are defined, the method will use them to plot the real image, if not the rgb bands
                will be considered as [2,1,0] as the default order for sentinel tif files.
            surface_reflectance (int, optional):
                Surface reflectance value for normalizing satellite data, by default None.
                Typically 10000 for Sentinel-2 data.
            cutoff (List, optional):
                clip the range of pixel values for each band. (take only the pixel values from 0 to the value of the cutoff
                and scale them back to between 0 and 1). Default is None.
            overview (bool, optional):
                True if you want to plot the overview. Default is False.
            overview_index (int, optional):
                Index of the overview. Default is 0.
            percentile: int
                The percentile value to be used for scaling.
        kwargs:
                | Parameter                   | Type                | Description |
                |-----------------------------|---------------------|-------------|
                | `points`                    | array               | 3 column array with the first column as the value to display for the point, the second as the row index, and the third as the column index in the array. The second and third columns tell the location of the point. |
                | `point_color`               | str                 | Color of the point. |
                | `point_size`                | Any                 | Size of the point. |
                | `pid_color`                 | str                 | Color of the annotation of the point. Default is blue. |
                | `pid_size`                  | Any                 | Size of the point annotation. |
                | `figsize`                   | tuple, optional     | Figure size. Default is `(8, 8)`. |
                | `title`                     | str, optional       | Title of the plot. Default is `'Total Discharge'`. |
                | `title_size`                | int, optional       | Title size. Default is `15`. |
                | `orientation`               | str, optional       | Orientation of the color bar (`horizontal` or `vertical`). Default is `'vertical'`. |
                | `rotation`                  | number, optional    | Rotation of the color bar label. Default is `-90`. |
                | `cbar_length`               | float, optional     | Ratio to control the height of the color bar. Default is `0.75`. |
                | `ticks_spacing`             | int, optional       | Spacing between color bar ticks. Default is `2`. |
                | `cbar_label_size`           | int, optional       | Size of the color bar label. Default is `12`. |
                | `cbar_label`                | str, optional       | Label of the color bar. Default is `'Discharge m\u00b3/s'`. |
                | `color_scale`               | int, optional       | Scale mode for colors. Options: 1 = normal, 2 = power, 3 = SymLogNorm, 4 = PowerNorm, 5 = BoundaryNorm. Default is `1`. |
                | `gamma`                     | float, optional     | Value needed for color scale option 2. Default is `1/2`. |
                | `line_threshold`            | float, optional     | Value needed for color scale option 3. Default is `0.0001`. |
                | `line_scale`                | float, optional     | Value needed for color scale option 3. Default is `0.001`. |
                | `bounds`                    | list, optional      | Discrete bounds for color scale option 4. Default is `None`. |
                | `midpoint`                  | float, optional     | Value needed for color scale option 5. Default is `0`. |
                | `cmap`                      | str, optional       | Color map style. Default is `'coolwarm_r'`. |
                | `display_cell_value`        | bool, optional      | Whether to display cell values as text. |
                | `num_size`                  | int, optional       | Size of numbers plotted on top of each cell. Default is `8`. |
                | `background_color_threshold`| float or int, optional | Threshold for deciding text color over cells: if value > threshold -> black text; else white text. If `None`, max value / 2 is used. Default is `None`. |

        Returns:
            ArrayGlyph:
                ArrayGlyph object. For more details of the ArrayGlyph object check the [ArrayGlyph](https://serapieum-of-alex.github.io/cleopatra/latest/api/array-glyph-class/).


        Examples:
            - Plot a certain band:

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326)
              >>> dataset.plot(band=0)
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            - plot using power scale.

              ```python
              >>> dataset.plot(band=0, color_scale="power")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            - plot using SymLogNorm scale.

              ```python
              >>> dataset.plot(band=0, color_scale="sym-lognorm")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            - plot using PowerNorm scale.

              ```python
              >>> dataset.plot(band=0, color_scale="boundary-norm", bounds=[0, 0.2, 0.4, 0.6, 0.8, 1])
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```

            - plot using BoundaryNorm scale.

              ```python
              >>> dataset.plot(band=0, color_scale="midpoint")
              (<Figure size 800x800 with 2 Axes>, <Axes: >)

              ```
        """
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/serapieum-org/cleopatra"
        )
        from cleopatra.array_glyph import ArrayGlyph

        no_data_value = [np.nan if i is None else i for i in self.no_data_value]
        if overview:
            arr = self.read_overview_array(
                band=band,
                overview_index=overview_index if overview_index is not None else 0,
            )
        else:
            arr = self.read_array(band=band)
        # if the raster has three bands or more.
        if self.band_count >= 3:
            if band is None:
                if rgb is None:
                    rgb_candidate: list[int | None] = [
                        self.get_band_by_color("red"),
                        self.get_band_by_color("green"),
                        self.get_band_by_color("blue"),
                    ]
                    if None in rgb_candidate:
                        rgb = [2, 1, 0]
                    else:
                        rgb = [int(v) for v in rgb_candidate if v is not None]
                # first make the band index the first band in the rgb list (red band)
                band = rgb[0]
        # elif self.band_count == 1:
        #     band = 0
        else:
            if band is None:
                band = 0

        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )

        cleo = ArrayGlyph(
            arr,
            exclude_value=exclude_value,
            extent=self.bbox,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            percentile=percentile,
            **kwargs,
        )
        cleo.plot(**kwargs)
        return cleo

    @staticmethod
    def _process_color_table(color_table: DataFrame) -> DataFrame:
        import_cleopatra(
            "The current function uses cleopatra package to for plotting, please install it manually, for more info"
            " check https://github.com/serapieum-org/cleopatra"
        )
        from cleopatra.colors import Colors

        # if the color_table does not contain the red, green, and blue columns, assume it has one column with
        # the color as hex and then, convert the color to rgb.
        if all(elem in color_table.columns for elem in ["red", "green", "blue"]):
            color_df = color_table.loc[:, ["values", "red", "green", "blue"]]
        elif "color" in color_table.columns:
            color = Colors(color_table["color"].tolist())
            color_rgb = color.to_rgb(normalized=False)
            color_df = DataFrame(columns=["values"])
            color_df["values"] = color_table["values"].to_list()
            color_df.loc[:, ["red", "green", "blue"]] = color_rgb
        else:
            raise ValueError(
                f"color_table must contain either red, green, blue, or color columns. given columns are: "
                f"{color_table.columns}"
            )

        if "alpha" not in color_table.columns:
            color_df.loc[:, "alpha"] = 255
        else:
            color_df.loc[:, "alpha"] = color_table["alpha"]

        return color_df
