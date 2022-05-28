"""
Created on Sat May 26 04:52:15 2018

@author: Mostafa
"""

import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo.gdal import Dataset
from pandas import DataFrame

from pyramids.raster import Raster as raster

# from osgeo import gdalconst
# import osr


class Catchment:
    """
    GISCatchment class contains methods to deal with the MED and generate the
    flow direction based on the D8 method and process the DEM

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

    def __init__(self):
        pass

    @staticmethod
    def D8(DEM):
        """
        D8 method generate flow direction raster from DEM and fill sinks

        inputs:
        ----------
            1-Raster:
                [Gdal object] DEM

        Outputs:
        ----------
            1- flow_direction_cell:
                [numpy array] with the same dimensions of the raster and 2 layers
                first layer for row index and second row for column index
            2-elev_sinkless:
                [numpy array] DEM after filling sinks
        """
        #        DEM=self.DEM

        gt = DEM.GetGeoTransform()
        cellsize = gt[1]
        dist2 = cellsize * np.sqrt(2)
        no_columns = DEM.RasterXSize
        no_rows = DEM.RasterYSize

        elev = DEM.ReadAsArray()
        # get the value stores in novalue cells
        dem_no_val = np.float32(DEM.GetRasterBand(1).GetNoDataValue())
        elev[elev == dem_no_val] = np.nan

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

        # first row without corners
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

        # last row without corners
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
                    flow_direction_cell[i, j, 0] = i  # index of the row
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

    @staticmethod
    def flowDirectionIndex(flow_direct):
        """
        this function takes flow firection raster and convert codes for the 8 directions
        (1,2,4,8,16,32,64,128) into indices of the Downstream cell

        inputs:
        ----------
            1- flow_direct:
                [gdal.dataset] flow direction raster obtained from catchment delineation
                it only contains values [1,2,4,8,16,32,64,128]

        output:
        ----------
            1-fd_indices:
                [numpy array] with the same dimensions of the raster and 2 layers
                first layer for row index and second row for column index

        Example:
        ----------
            fd=gdal.Open("Flowdir.tif")
            fd_indices=FlowDirectِِIndex(fd)
        """
        # input data validation
        # data type
        assert (
            type(flow_direct) == gdal.Dataset
        ), "src should be read using gdal (gdal dataset please read it using gdal library) "

        # check flow direction input raster
        no_val = np.float32(flow_direct.GetRasterBand(1).GetNoDataValue())
        cols = flow_direct.RasterXSize
        rows = flow_direct.RasterYSize

        fd = flow_direct.ReadAsArray()
        fd_val = [
            int(fd[i, j])
            for i in range(rows)
            for j in range(cols)
            if fd[i, j] != no_val
        ]
        fd_val = list(set(fd_val))
        fd_should = [1, 2, 4, 8, 16, 32, 64, 128]
        assert all(
            fd_val[i] in fd_should for i in range(len(fd_val))
        ), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "

        #    fd=np.float32(flow_direct.ReadAsArray())
        #    fd[fd==no_val]=np.nan
        fd_cell = np.ones((rows, cols, 2)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if fd[i, j] == 1:
                    fd_cell[i, j, 0] = i  # index of the row
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

    @staticmethod
    def flowDirectionTable(flow_direct):
        """
        this function takes flow direction indices created by FlowDirectِِIndex function
        and create a dictionary with the cells indices as a key and  indices of directly
        upstream cells as values (list of tuples)

        Inputs:
        ----------
            1- flow_direct:
                [gdal.dataset] flow direction raster obtained from catchment delineation
                it only contains values [1,2,4,8,16,32,64,128]

        Outputs:
        ----------
            1-flowAccTable:
                [Dict] dictionary with the cells indices as a key and indices of directly
                upstream cells as values (list of tuples)

        Example:
        ----------

        """
        # input data validation
        # validation is inside FlowDirectِِIndex
        FDI = Catchment.flowDirectionIndex(flow_direct)

        rows = flow_direct.RasterYSize
        cols = flow_direct.RasterXSize

        celli = []
        cellj = []
        celli_content = []
        cellj_content = []
        for i in range(rows):  # rows
            for j in range(cols):  # columns
                if not np.isnan(FDI[i, j, 0]):
                    # store the indexes of not empty cells and the indexes stored inside these cells
                    celli.append(i)
                    cellj.append(j)
                    # store the index of the receiving cells
                    celli_content.append(FDI[i, j, 0])
                    cellj_content.append(FDI[i, j, 1])

        flow_acc_table = {}
        # for each cell store the directly giving cells
        for i in range(rows):  # rows
            for j in range(cols):  # columns
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
        """
        this function deletes all the basins in a basin raster created when delineating
        a catchment and leave only the first basin which is the biggest basin in the raster

        Inputs:
        ----------
            1- basins:
                [gdal.dataset] raster you create during delineation of a catchment
                values of its cells are the number of the basin it belongs to
            2- pathout:
                [String] path you want to save the resulted raster to it should include
                the extension ".tif"
        Outputs:
        ----------
            1- raster with only one basin (the basin that its name is 1 )

        Example:
        ----------
            basins=gdal.Open("Data/basins.tif")
            pathout="mask.tif"
            DeleteBasins(basins,pathout)
        """
        # input data validation
        # data type
        assert type(pathout) == str, "A_path input should be string type"
        assert (
            type(basins) == gdal.Dataset
        ), "basins raster should be read using gdal (gdal dataset please read it using gdal library) "

        # input values
        # check wether the user wrote the extension of the raster or not
        ext = pathout[-4:]
        assert ext == ".tif", "please add the extension at the end of the path input"

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

        raster.rasterLike(basins, basins_A, pathout)

    @staticmethod
    def nearestCell(
            Raster: Dataset,
            StCoord: DataFrame,
    ) -> DataFrame:
        """NearestCell.

            this function calculates the the indices (row, col) of nearest cell in a given
            raster to a station
            coordinate system of the raster has to be projected to be able to calculate
            the distance

        Parameters
        ----------
        Raster: [gdal.dataset]
            raster to get the spatial information (coordinates of each cell)
        StCoord: [Dataframe]
            dataframe with two columns "x", "y" contains the coordinates
            of each station

        Returns
        -------
        StCoord:the same input dataframe with two extra columns "cellx","celly"

        Examples
        --------
        >>> soil_type = gdal.Open("DEM.tif")
        >>> data = dict(id = [0,1,2,3], x = [1,2,3,6], y = [5,4,7,8])
        >>> stations = pd.DataFrame(data)
        >>> coordinates = stations[['id','x','y']][:]
        >>> coordinates.loc[:,["cell_row","cell_col"]] = Catchment.nearestCell(Raster, StCoord).values
        """
        if not isinstance(Raster, gdal.Dataset):
            raise TypeError ("raster should be read using gdal (gdal dataset please read it using gdal library) ")

        if not isinstance(StCoord, DataFrame):
            raise TypeError(f"please check StCoord input it should be pandas dataframe - given {type(StCoord)}")

        # check if the user has stored the coordinates in the dataframe with the right names or not
        if not "x" in StCoord.columns:
            raise ValueError ("please check the StCoord parameter it should contain a column with the x coordinates"
                             "and the column name should be 'x'")
        if not "y" in StCoord.columns:
            raise ValueError ("please check the StCoord parameter it should contain a column with the y coordinates"
                             "and the column name should be 'y'")

        StCoord["cell_row"] = np.nan
        StCoord["cell_col"] = np.nan

        rows = Raster.RasterYSize
        cols = Raster.RasterXSize
        geo_trans = Raster.GetGeoTransform()
        # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        # X_coordinate= upperleft corner x+ index* cell size+celsize/2
        coox = np.ones((rows, cols))
        cooy = np.ones((rows, cols))
        for i in range(rows):  # iteration by row
            for j in range(cols):  # iteration by column
                coox[i, j] = (
                    geo_trans[0] + geo_trans[1] / 2 + j * geo_trans[1]
                )  # calculate x
                cooy[i, j] = (
                    geo_trans[3] + geo_trans[5] / 2 + i * geo_trans[5]
                )  # calculate y

        Dist = np.ones((rows, cols))
        for no in range(len(StCoord["x"])):
            # calculate the distance from the station to all cells
            for i in range(rows):  # iteration by row
                for j in range(cols):  # iteration by column
                    Dist[i, j] = np.sqrt(
                        np.power((StCoord.loc[StCoord.index[no], "x"] - coox[i, j]), 2)
                        + np.power(
                            (StCoord.loc[StCoord.index[no], "y"] - cooy[i, j]), 2
                        )
                    )

            StCoord.loc[no, "cell_row"], StCoord.loc[no, "cell_col"] = np.where(
                Dist == np.min(Dist)
            )

        return StCoord.loc[:, ["cell_row", "cell_col"]]

    @staticmethod
    def groupNeighbours(
        array, i, j, lowervalue, uppervalue, position, values, count, cluster
    ):
        # bottom cell
        if (
            lowervalue <= array[i + 1, j] <= uppervalue
            and cluster[i + 1, j] == 0
            and i + 1 < array.shape[0]
        ):
            position.append([i + 1, j])
            values.append(array[i + 1, j])
            cluster[i + 1, j] = count
            Catchment.groupNeighbours(
                array,
                i + 1,
                j,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # bottom right
        if (
            j + 1 < array.shape[1]
            and i + 1 < array.shape[0]
            and lowervalue <= array[i + 1, j + 1] <= uppervalue
            and cluster[i + 1, j + 1] == 0
        ):
            position.append([i + 1, j + 1])
            values.append(array[i + 1, j + 1])
            cluster[i + 1, j + 1] = count
            Catchment.groupNeighbours(
                array,
                i + 1,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # right
        if (
            j + 1 < array.shape[1]
            and lowervalue <= array[i, j + 1] <= uppervalue
            and cluster[i, j + 1] == 0
        ):
            position.append([i, j + 1])
            values.append(array[i, j + 1])
            cluster[i, j + 1] = count
            Catchment.groupNeighbours(
                array,
                i,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # upper right
        if (
            j + 1 < array.shape[1]
            and i - 1 >= 0
            and lowervalue <= array[i - 1, j + 1] <= uppervalue
            and cluster[i - 1, j + 1] == 0
        ):
            position.append([i - 1, j + 1])
            values.append(array[i - 1, j + 1])
            cluster[i - 1, j + 1] = count
            Catchment.groupNeighbours(
                array,
                i - 1,
                j + 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # top
        if (
            i - 1 >= 0
            and lowervalue <= array[i - 1, j] <= uppervalue
            and cluster[i - 1, j] == 0
        ):
            position.append([i - 1, j])
            values.append(array[i - 1, j])
            cluster[i - 1, j] = count
            Catchment.groupNeighbours(
                array,
                i - 1,
                j,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # top left
        if (
            i - 1 >= 0
            and j - 1 >= 0
            and lowervalue <= array[i - 1, j - 1] <= uppervalue
            and cluster[i - 1, j - 1] == 0
        ):
            position.append([i - 1, j - 1])
            values.append(array[i - 1, j - 1])
            cluster[i - 1, j - 1] = count
            Catchment.groupNeighbours(
                array,
                i - 1,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # left
        if (
            j - 1 >= 0
            and lowervalue <= array[i, j - 1] <= uppervalue
            and cluster[i, j - 1] == 0
        ):
            position.append([i, j - 1])
            values.append(array[i, j - 1])
            cluster[i, j - 1] = count
            Catchment.groupNeighbours(
                array,
                i,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )
        # bottom left
        if (
            j - 1 >= 0
            and i + 1 < array.shape[0]
            and lowervalue <= array[i + 1, j - 1] <= uppervalue
            and cluster[i + 1, j - 1] == 0
        ):
            position.append([i + 1, j - 1])
            values.append(array[i + 1, j - 1])
            cluster[i + 1, j - 1] = count
            Catchment.groupNeighbours(
                array,
                i + 1,
                j - 1,
                lowervalue,
                uppervalue,
                position,
                values,
                count,
                cluster,
            )

    @staticmethod
    def cluster(Data, LowerValue, UpperValue):
        """
        Cluster method group all the connected values between two numbers in
        a raster in clusters

        Parameters
        ----------
        Data : [array]
            numpt array of the daya in the raster.
        LowerValue : [numeric]
            lower bound of the cluster.
        UpperValue : [numeric]
            upper bound of the cluster.

        Returns
        -------
        cluster : [array]
            array contains integer numbers representing the number of the cluster.
        count : [integer]
            number of the clusters in the array.
        position : [list]
            list contains two indeces [x,y] for the position of each value .
        values : [numeric]
            the values stored in each cell in the cluster .

        """
        position = []
        values = []
        count = 1
        cluster = np.zeros(shape=(Data.shape[0], Data.shape[1]))

        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):

                if LowerValue <= Data[i, j] <= UpperValue and cluster[i, j] == 0:
                    Catchment.groupNeighbours(
                        Data,
                        i,
                        j,
                        LowerValue,
                        UpperValue,
                        position,
                        values,
                        count,
                        cluster,
                    )
                    if cluster[i, j] == 0:
                        position.append([i, j])
                        values.append(Data[i, j])
                        cluster[i, j] = count
                    count = count + 1

        return cluster, count, position, values

    def listAttributes(self):
        """
        Print Attributes List
        """

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
