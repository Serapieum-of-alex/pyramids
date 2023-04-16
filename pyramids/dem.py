from typing import Dict
import numpy as np
from osgeo import gdal

from pyramids.dataset import Dataset
import sys

sys.setrecursionlimit(5000)


class DEM(Dataset):
    """GISCatchment class contains methods to deal with the MED and generate the flow direction based on the D8 method and process the DEM.

    Methods:
        1- D8
        2- FlowDirectIndex
        3- FlowDirecTable
        4- DeleteBasins
        5- NearestCell
        6- GroupNeighbours
        7- Cluster
        8- ListAttributes
    """

    def __init__(self, src: gdal.Dataset):
        super().__init__(src)

    def D8(self):
        """D8 method generate flow direction raster from DEM and fill sinks.

        Returns
        -------
        flow_direction_cell: [numpy array]
            with the same dimensions of the raster and 2 layers
            first layer for rows index and second rows for column index
        elev_sinkless: [numpy array]
            DEM after filling sinks
        """
        cellsize = self.cell_size
        dist2 = cellsize * np.sqrt(2)
        no_columns = self.columns
        no_rows = self.rows

        elev = self.read_array(band=0)
        # get the value stores in novalue cells
        dem_no_val = self.no_data_value[0]
        elev = elev.astype(np.float32)
        elev[np.isclose(elev, dem_no_val, rtol=0.00001)] = np.nan

        slopes = np.ones((no_rows, no_columns, 9)) * np.nan
        distances = [cellsize, dist2, cellsize, dist2, cellsize, dist2, cellsize, dist2]

        # filling sinks
        elev_sinkless = elev
        for i in range(1, no_rows - 1):
            for j in range(1, no_columns - 1):
                # get elevation of surrounding cells
                f = [
                    elev[i - 1, j],
                    elev[i - 1, j - 1],
                    elev[i, j - 1],
                    elev[i + 1, j - 1],
                    elev[i + 1, j],
                    elev[i + 1, j + 1],
                    elev[i, j + 1],
                    elev[i - 1, j + 1],
                ]
                if elev[i, j] < min(f):
                    elev_sinkless[i, j] = min(f) + 0.1

        flow_direction = np.ones((no_rows, no_columns)) * np.nan

        for i in range(1, no_rows - 1):
            for j in range(1, no_columns - 1):
                # calculate only if cell in elev is not nan
                if not np.isnan(elev[i, j]):
                    # calculate slope
                    # slope with cell to the right
                    slopes[i, j, 0] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j + 1]
                    ) / distances[0]
                    # slope with cell to the top right
                    slopes[i, j, 1] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j + 1]
                    ) / distances[1]
                    # slope with cell to the top
                    slopes[i, j, 2] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j]
                    ) / distances[2]
                    # slope with cell to the top left
                    slopes[i, j, 3] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j - 1]
                    ) / distances[3]
                    # slope with cell to the left
                    slopes[i, j, 4] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j - 1]
                    ) / distances[4]
                    # slope with cell to the bottom left
                    slopes[i, j, 5] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j - 1]
                    ) / distances[5]
                    # slope with cell to the bottom
                    slopes[i, j, 6] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j]
                    ) / distances[6]
                    # slope with cell to the bottom right
                    slopes[i, j, 7] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j + 1]
                    ) / distances[7]
                    # get the flow direction index
                    flow_direction[i, j] = np.where(
                        slopes[i, j, :] == np.nanmax(slopes[i, j, :])
                    )[0][0]
                    slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # first rows without corners
        for i in [0]:
            for j in range(1, no_columns - 1):  # all columns
                if not np.isnan(elev[i, j]):
                    # slope with cell to the right
                    slopes[i, j, 0] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j + 1]
                    ) / distances[0]
                    # slope with cell to the left
                    slopes[i, j, 4] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j - 1]
                    ) / distances[4]
                    # slope with cell to the bottom left
                    slopes[i, j, 5] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j - 1]
                    ) / distances[5]
                    # slope with cell to the bottom
                    slopes[i, j, 6] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j]
                    ) / distances[6]
                    # slope with cell to the bottom right
                    slopes[i, j, 7] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j + 1]
                    ) / distances[7]

                    flow_direction[i, j] = np.where(
                        slopes[i, j, :] == np.nanmax(slopes[i, j, :])
                    )[0][0]
                    slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # last rows without corners
        for i in [no_rows - 1]:
            for j in range(1, no_columns - 1):  # all columns
                if not np.isnan(elev[i, j]):
                    # slope with cell to the right
                    slopes[i, j, 0] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j + 1]
                    ) / distances[0]
                    # slope with cell to the top right
                    slopes[i, j, 1] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j + 1]
                    ) / distances[1]
                    # slope with cell to the top
                    slopes[i, j, 2] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j]
                    ) / distances[2]
                    # slope with cell to the top left
                    slopes[i, j, 3] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j - 1]
                    ) / distances[3]
                    # slope with cell to the left
                    slopes[i, j, 4] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j - 1]
                    ) / distances[4]

                    flow_direction[i, j] = np.where(
                        slopes[i, j, :] == np.nanmax(slopes[i, j, :])
                    )[0][0]
                    slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # top left corner
        i = 0
        j = 0
        if not np.isnan(elev[i, j]):
            # slope with cell to the left
            slopes[i, j, 0] = (
                elev_sinkless[i, j] - elev_sinkless[i, j + 1]
            ) / distances[0]
            # slope with cell to the bottom
            slopes[i, j, 6] = (
                elev_sinkless[i, j] - elev_sinkless[i + 1, j]
            ) / distances[6]
            # slope with cell to the bottom right
            slopes[i, j, 7] = (
                elev_sinkless[i, j] - elev_sinkless[i + 1, j + 1]
            ) / distances[7]

            flow_direction[i, j] = np.where(
                slopes[i, j, :] == np.nanmax(slopes[i, j, :])
            )[0][0]
            slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # top right corner
        i = 0
        j = no_columns - 1
        if not np.isnan(elev[i, j]):
            # slope with cell to the left
            slopes[i, j, 4] = (
                elev_sinkless[i, j] - elev_sinkless[i, j - 1]
            ) / distances[4]
            # slope with cell to the bottom left
            slopes[i, j, 5] = (
                elev_sinkless[i, j] - elev_sinkless[i + 1, j - 1]
            ) / distances[5]
            # slope with cell to the bott
            slopes[i, j, 6] = (
                elev_sinkless[i, j] - elev_sinkless[i + 1, j]
            ) / distances[6]

            flow_direction[i, j] = np.where(
                slopes[i, j, :] == np.nanmax(slopes[i, j, :])
            )[0][0]
            slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # bottom left corner
        i = no_rows - 1
        j = 0
        if not np.isnan(elev[i, j]):
            # slope with cell to the right
            slopes[i, j, 0] = (
                elev_sinkless[i, j] - elev_sinkless[i, j + 1]
            ) / distances[0]
            # slope with cell to the top right
            slopes[i, j, 1] = (
                elev_sinkless[i, j] - elev_sinkless[i - 1, j + 1]
            ) / distances[1]
            # slope with cell to the top
            slopes[i, j, 2] = (
                elev_sinkless[i, j] - elev_sinkless[i - 1, j]
            ) / distances[2]

            flow_direction[i, j] = np.where(
                slopes[i, j, :] == np.nanmax(slopes[i, j, :])
            )[0][0]
            slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # bottom right
        i = no_rows - 1
        j = no_columns - 1
        if not np.isnan(elev[i, j]):
            slopes[i, j, 2] = (
                elev_sinkless[i, j] - elev_sinkless[i - 1, j]
            ) / distances[
                2
            ]  # slope with cell to the top
            slopes[i, j, 3] = (
                elev_sinkless[i, j] - elev_sinkless[i - 1, j - 1]
            ) / distances[
                3
            ]  # slope with cell to the top left
            slopes[i, j, 4] = (
                elev_sinkless[i, j] - elev_sinkless[i, j - 1]
            ) / distances[
                4
            ]  # slope with cell to the left

            flow_direction[i, j] = np.where(
                slopes[i, j, :] == np.nanmax(slopes[i, j, :])
            )[0][0]
            slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # first column
        for i in range(1, no_rows - 1):
            for j in [0]:
                if not np.isnan(elev[i, j]):
                    slopes[i, j, 0] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j + 1]
                    ) / distances[
                        0
                    ]  # slope with cell to the right
                    slopes[i, j, 1] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j + 1]
                    ) / distances[
                        1
                    ]  # slope with cell to the top right
                    slopes[i, j, 2] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j]
                    ) / distances[
                        2
                    ]  # slope with cell to the top
                    slopes[i, j, 6] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j]
                    ) / distances[
                        6
                    ]  # slope with cell to the bottom
                    slopes[i, j, 7] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j + 1]
                    ) / distances[
                        7
                    ]  # slope with cell to the bottom right
                    # get the flow direction index

                    flow_direction[i, j] = np.where(
                        slopes[i, j, :] == np.nanmax(slopes[i, j, :])
                    )[0][0]
                    slopes[i, j, 8] = np.nanmax(slopes[i, j, :])

        # last column
        for i in range(1, no_rows - 1):
            for j in [no_columns - 1]:
                if not np.isnan(elev[i, j]):
                    slopes[i, j, 2] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j]
                    ) / distances[
                        2
                    ]  # slope with cell to the top
                    slopes[i, j, 3] = (
                        elev_sinkless[i, j] - elev_sinkless[i - 1, j - 1]
                    ) / distances[
                        3
                    ]  # slope with cell to the top left
                    slopes[i, j, 4] = (
                        elev_sinkless[i, j] - elev_sinkless[i, j - 1]
                    ) / distances[
                        4
                    ]  # slope with cell to the left
                    slopes[i, j, 5] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j - 1]
                    ) / distances[
                        5
                    ]  # slope with cell to the bottom left
                    slopes[i, j, 6] = (
                        elev_sinkless[i, j] - elev_sinkless[i + 1, j]
                    ) / distances[
                        6
                    ]  # slope with cell to the bottom
                    # get the flow direction index

                    flow_direction[i, j] = np.where(
                        slopes[i, j, :] == np.nanmax(slopes[i, j, :])
                    )[0][0]
                    slopes[i, j, 8] = np.nanmax(slopes[i, j, :])
        #        print(str(i)+","+str(j))

        flow_direction_cell = np.ones((no_rows, no_columns, 2)) * np.nan
        # for i in range(1,no_rows-1):
        #    for j in range(1,no_columns-1):
        for i in range(no_rows):
            for j in range(no_columns):
                if flow_direction[i, j] == 0:
                    flow_direction_cell[i, j, 0] = i  # index of the rows
                    flow_direction_cell[i, j, 1] = j + 1  # index of the column
                elif flow_direction[i, j] == 1:
                    flow_direction_cell[i, j, 0] = i - 1
                    flow_direction_cell[i, j, 1] = j + 1
                elif flow_direction[i, j] == 2:
                    flow_direction_cell[i, j, 0] = i - 1
                    flow_direction_cell[i, j, 1] = j
                elif flow_direction[i, j] == 3:
                    flow_direction_cell[i, j, 0] = i - 1
                    flow_direction_cell[i, j, 1] = j - 1
                elif flow_direction[i, j] == 4:
                    flow_direction_cell[i, j, 0] = i
                    flow_direction_cell[i, j, 1] = j - 1
                elif flow_direction[i, j] == 5:
                    flow_direction_cell[i, j, 0] = i + 1
                    flow_direction_cell[i, j, 1] = j - 1
                elif flow_direction[i, j] == 6:
                    flow_direction_cell[i, j, 0] = i + 1
                    flow_direction_cell[i, j, 1] = j
                elif flow_direction[i, j] == 7:
                    flow_direction_cell[i, j, 0] = i + 1
                    flow_direction_cell[i, j, 1] = j + 1

        return flow_direction_cell, elev_sinkless

    def flowDirectionIndex(self) -> np.ndarray:
        """this function takes flow firection raster and convert codes for the 8 directions (1,2,4,8,16,32,64,128) into indices of the Downstream cell.

        flow_direct:
            [gdal.dataset] flow direction raster obtained from catchment delineation
            it only contains values [1,2,4,8,16,32,64,128]

        Returns
        -------
        fd_indices:
            [numpy array] with the same dimensions of the raster and 2 layers
            first layer for rows index and second rows for column index

        Example:
        ----------
            fd=gdal.Open("Flowdir.tif")
            fd_indices=FlowDirectِِIndex(fd)
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
                    fd_cell[i, j, 0] = i  # index of the rows
                    fd_cell[i, j, 1] = j + 1  # index of the column
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

    def flowDirectionTable(self) -> Dict:
        """Flow Direction Table.
            - This function takes flow direction indices created by FlowDirectِِIndex function and create a
            dictionary with the cells indices as a key and  indices of directly upstream cells as values (list of tuples)


            flow_direct:
                [gdal.dataset] flow direction raster obtained from catchment delineation
                it only contains values [1,2,4,8,16,32,64,128]

        Returns
        -------
        flowAccTable:
            [Dict] dictionary with the cells indices as a key and indices of directly
            upstream cells as values (list of tuples)
        """
        FDI = self.flowDirectionIndex()

        rows = self.rows
        cols = self.columns

        celli = []
        cellj = []
        celli_content = []
        cellj_content = []
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(FDI[i, j, 0]):
                    # store the indexes of not empty cells and the indexes stored inside these cells
                    celli.append(i)
                    cellj.append(j)
                    # store the index of the receiving cells
                    celli_content.append(FDI[i, j, 0])
                    cellj_content.append(FDI[i, j, 1])

        flow_acc_table = {}
        # for each cell store the directly giving cells
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(FDI[i, j, 0]):
                    # get the indexes of the cell and use it as a key in a dictionary
                    name = str(i) + "," + str(j)
                    flow_acc_table[name] = []
                    for k in range(len(celli_content)):
                        # search if any cell are giving this cell
                        if i == celli_content[k] and j == cellj_content[k]:
                            flow_acc_table[name].append((celli[k], cellj[k]))

        return flow_acc_table

    @staticmethod
    def deleteBasins(basins, pathout):
        """Delete Basins

            - this function deletes all the basins in a basin raster created when delineating a catchment and leave
            only the first basin which is the biggest basin in the raster.

        Parameters
        ----------
        basins: [gdal.dataset]
            raster you create during delineation of a catchment
            values of its cells are the number of the basin it belongs to
        pathout: [str]
             path you want to save the resulted raster to it should include
            the extension ".tif"

        Returns
        -------
        raster with only one basin (the basin that its name is 1 )
        """
        assert type(pathout) == str, "A_path input should be string type"
        assert (
            type(basins) == gdal.Dataset
        ), "basins raster should be read using gdal (gdal dataset please read it using gdal library) "

        # get number of rows
        rows = basins.RasterYSize
        # get number of columns
        cols = basins.RasterXSize
        # array
        basins_A = basins.ReadAsArray()
        # no data value
        no_val = np.float32(basins.GetRasterBand(1).GetNoDataValue())
        # get number of basins and there names
        basins_val = list(
            set(
                [
                    int(basins_A[i, j])
                    for i in range(rows)
                    for j in range(cols)
                    if basins_A[i, j] != no_val
                ]
            )
        )

        # keep the first basin and delete the others by filling their cells by nodata value
        for i in range(rows):
            for j in range(cols):
                if basins_A[i, j] != no_val and basins_A[i, j] != basins_val[0]:
                    basins_A[i, j] = no_val

        Dataset.dataset_like(basins, basins_A, pathout)

    def listAttributes(self):
        """Print Attributes List."""

        print("\n")
        print(
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")
