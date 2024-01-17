"""DEM module."""
from typing import Any, Dict, Tuple
import numpy as np
from osgeo import gdal

from pyramids.dataset import Dataset
import sys

sys.setrecursionlimit(5000)

INDEX = [0, 1, 2, 3, 4, 5, 6, 7]
X_INDEX = [0, -1, -1, -1, 0, 1, 1, 1]
Y_INDEX = [1, 1, 0, -1, -1, -1, 0, 1]


class DEM(Dataset):
    """DEM class.

    DEM contains methods to deal with the digital elevation model (DEM) and generate the flow direction based on
    the D8 method and process the DEM.
    """

    def __init__(self, src: gdal.Dataset):
        super().__init__(src)

    @property
    def values(self):
        """values.

        The Values property retrieves the values of the raster and replaces the no data values with np.nan
        """
        values = self.read_array(band=0).astype(np.float32)
        # get the value stores in no data value cells
        no_val = self.no_data_value[0]
        values[np.isclose(values, no_val, rtol=0.00001)] = np.nan
        return values

    def fill_sinks(self, inplace: bool = False) -> Tuple[Any, Dataset]:
        """fill_sinks.

        Parameters
        ----------
        inplace: [bool]
            Default is False.

        Returns
        -------
        [numpy array]
            DEM after filling sinks.
        """
        elev = self.values

        elev_sinkless = np.copy(elev)
        for i in range(1, self.rows - 1):
            for j in range(1, self.columns - 1):
                # Get elevation of surrounding cells
                f = elev[i - 1 : i + 2, j - 1 : j + 2].flatten()
                # Exclude the center cell
                f[4] = np.nan
                min_f = np.nanmin(f)
                if elev_sinkless[i, j] < min_f:
                    elev_sinkless[i, j] = min_f + 0.1

        src = self.dataset_like(self, elev_sinkless)
        if inplace:
            self.__init__(src.raster)
        else:
            return src

    def _get_8_direction_slopes(self) -> np.ndarray:
        """compute_slopes.


        Returns
        -------
        flow_direction: [numpy array]
            flow direction array. The array contains values [0, 1, 2, 3, 4, 5, 6, 7] referring to the 8 directions,
            where 0 is the top cell, 1 is the top left cell, 2 is the left cell, 3 is the bottom left cell, 4 is the
            bottom cell, 5 is the bottom right cell, 6 is the right cell, and 7 is the top right cell.
        slopes: [numpy array]
            The slope array.
        """
        elev = self.values
        cell_size = self.cell_size
        dist2 = cell_size * np.sqrt(2)
        distances = [
            cell_size,
            dist2,
            cell_size,
            dist2,
            cell_size,
            dist2,
            cell_size,
            dist2,
        ]
        rows, cols = elev.shape
        slopes = np.full((rows, cols, 8), np.nan, dtype=np.float32)

        # padding = 2
        # pad_1 = padding - 1
        # Create a padded elevation array for boundary conditions
        padded_elev = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
        padded_elev[1:-1, 1:-1] = elev

        # Calculate elevation differences using slicing
        diff_right = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, 2:]
        diff_top_right = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 2:]
        diff_top = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 1:-1]
        diff_top_left = padded_elev[1:-1, 1:-1] - padded_elev[:-2, :-2]
        diff_left = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, :-2]
        diff_bottom_left = padded_elev[1:-1, 1:-1] - padded_elev[2:, :-2]
        diff_bottom = padded_elev[1:-1, 1:-1] - padded_elev[2:, 1:-1]
        diff_bottom_right = padded_elev[1:-1, 1:-1] - padded_elev[2:, 2:]

        # Calculate slopes
        slopes[:, :, 0] = diff_bottom / distances[0]
        slopes[:, :, 1] = diff_bottom_left / distances[1]
        slopes[:, :, 2] = diff_left / distances[2]
        slopes[:, :, 3] = diff_top_left / distances[3]
        slopes[:, :, 4] = diff_top / distances[4]
        slopes[:, :, 5] = diff_top_right / distances[5]
        slopes[:, :, 6] = diff_right / distances[6]
        slopes[:, :, 7] = diff_bottom_right / distances[7]

        return slopes

    def slope(self):
        """slope."""
        slope = self._get_8_direction_slopes()
        max_slope = np.nanmax(slope, axis=2)

        src = self.dataset_like(self, max_slope)
        return src

    def flow_direction(self) -> np.ndarray:
        """flow_direction.

        Returns
        -------
        flow_direction: [numpy array]
            flow direction array. The array contains values [0, 1, 2, 3, 4, 5, 6, 7] referring to the 8 directions,
            where 0 is the bottom cell, 1 is the bottom left cell, 2 is the left cell, 3 is the top left cell,
            4 is the top cell, 5 is the top right cell, 6 is the right cell, and 7 is the bottom right cell.
        """
        elev = self.values
        slopes = self._get_8_direction_slopes()
        # Create a mask for non-NaN cells in the elevation array
        mask = ~np.isnan(elev)

        # Create a mask for cells with at least one non-NaN slope
        valid_mask = ~np.all(np.isnan(slopes), axis=2)

        # Combine masks to identify cells where calculations should be done
        valid_cells_mask = mask & valid_mask

        # Initialize the flow_direction array with NaN values
        flow_direction = np.full(elev.shape, np.nan)

        # Apply np.nanargmax only where the mask is True to get the index of the maximum slope
        # hence, the flow direction.
        flow_direction[valid_cells_mask] = np.nanargmax(
            slopes[valid_cells_mask], axis=1
        )

        return flow_direction

    def convert_flow_direction_to_cell_indices(self):
        """D8 method generates flow direction raster from DEM and fills sinks.

        Returns
        -------
        flow_direction_cell: [numpy array]
            with the same dimensions of the raster and 2 layers
            first layer for rows index and second rows for column index
        """
        no_columns = self.columns
        no_rows = self.rows

        flow_direction = self.flow_direction()

        # convert index of the flow direction to the index of the cell
        flow_direction_cell = np.ones((no_rows, no_columns, 2)) * np.nan

        for i in range(no_rows):
            for j in range(no_columns):
                if not np.isnan(flow_direction[i, j]):
                    ind = int(flow_direction[i, j])
                    flow_direction_cell[i, j, 0] = i + X_INDEX[ind]
                    flow_direction_cell[i, j, 1] = j + Y_INDEX[ind]

        return flow_direction_cell

    def flow_direction_index(self) -> np.ndarray:
        """flow_direction_index.

            flow_direction_index takes flow direction raster and converts codes for the 8 directions
            (1,2,4,8,16,32,64,128) into indices of the Downstream cell.

        flow_direct:
            [gdal.dataset] flow direction raster obtained from catchment delineation
            it only contains values [1,2,4,8,16,32,64,128]

        Returns
        -------
        [numpy array]:
            with the same dimensions of the raster and 2 layers
            first layer for rows index and second rows for column index
        """
        # check flow direction input raster
        no_val = self.no_data_value[0]
        cols = self.columns
        rows = self.rows

        fd = self.read_array(band=0)
        fd_val = np.unique(fd[~np.isclose(fd, no_val, rtol=0.00001)])
        fd_should = [1, 2, 4, 8, 16, 32, 64, 128]
        if not all(fd_val[i] in fd_should for i in range(len(fd_val))):
            raise ValueError(
                "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "
            )

        fd_cell = np.ones((rows, cols, 2)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if fd[i, j] == 1:
                    # index of the rows
                    fd_cell[i, j, 0] = i
                    # index of the column
                    fd_cell[i, j, 1] = j + 1
                elif fd[i, j] == 128:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j + 1
                elif fd[i, j] == 64:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j
                elif fd[i, j] == 32:
                    fd_cell[i, j, 0] = i - 1
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 16:
                    fd_cell[i, j, 0] = i
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 8:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j - 1
                elif fd[i, j] == 4:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j
                elif fd[i, j] == 2:
                    fd_cell[i, j, 0] = i + 1
                    fd_cell[i, j, 1] = j + 1

        return fd_cell

    def flow_direction_table(self) -> Dict:
        """Flow Direction Table.

            - flow_direction_table takes flow direction indices created by FlowDirectِِIndex function and creates a
            dictionary with the cells' indices as a key and indices of directly upstream cells as values
            (list of tuples).


            flow_direct:
                [gdal.dataset] flow direction raster obtained from catchment delineation
                it only contains values [1,2,4,8,16,32,64,128]

        Returns
        -------
        flowAccTable:
            [Dict] dictionary with the cells indices as a key and indices of directly
            upstream cells as values (list of tuples)
        """
        flow_direction_index = self.flow_direction_index()

        rows = self.rows
        cols = self.columns

        cell_i = []
        cell_j = []
        celli_content = []
        cellj_content = []
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(flow_direction_index[i, j, 0]):
                    # store the indexes of not empty cells and the indexes stored inside these cells
                    cell_i.append(i)
                    cell_j.append(j)
                    # store the index of the receiving cells
                    celli_content.append(flow_direction_index[i, j, 0])
                    cellj_content.append(flow_direction_index[i, j, 1])

        flow_acc_table = {}
        # for each cell store the directly giving cells
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(flow_direction_index[i, j, 0]):
                    # get the indexes of the cell and use it as a key in a dictionary
                    name = str(i) + "," + str(j)
                    flow_acc_table[name] = []
                    for k in range(len(celli_content)):
                        # search if any cell are giving this cell
                        if i == celli_content[k] and j == cellj_content[k]:
                            flow_acc_table[name].append((cell_i[k], cell_j[k]))

        return flow_acc_table

    @staticmethod
    def delete_basins(basins: gdal.Dataset, path: str):
        """Delete Basins

            delete_basins deletes all the basins in a basin raster created when delineating a catchment and leaves
            only the first basin which is the biggest basin in the raster.

        Parameters
        ----------
        basins: [gdal.dataset]
            raster you create during delineation of the catchment
            values of its cells are the number of the basin it belongs to
        path: [str]
             path you want to save the resulted raster to it should include
            the extension ".tif"

        Returns
        -------
        raster with only one basin (the basin that its name is 1)
        """
        if not isinstance(path, str):
            raise TypeError(f"path: {path} input should be string type")
        if not isinstance(basins, gdal.Dataset):
            raise TypeError(
                "basins raster should be read using gdal (gdal dataset please read it using gdal library)"
            )

        # get number of rows
        rows = basins.RasterYSize
        # get number of columns
        cols = basins.RasterXSize
        # array
        basins_a = basins.ReadAsArray()
        # no data value
        no_val = np.float32(basins.GetRasterBand(1).GetNoDataValue())
        # get number of basins and there names
        basins_val = list(
            set(
                [
                    int(basins_a[i, j])
                    for i in range(rows)
                    for j in range(cols)
                    if basins_a[i, j] != no_val
                ]
            )
        )

        # keep the first basin and delete the others by filling their cells by nodata value
        for i in range(rows):
            for j in range(cols):
                if basins_a[i, j] != no_val and basins_a[i, j] != basins_val[0]:
                    basins_a[i, j] = no_val

        Dataset.dataset_like(basins, basins_a, path)
