from typing import Tuple, Union, List

import numpy as np


def _get_indeces(
    arr: np.ndarray, mask_val: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the array indeces for the non-zero cells.

    Parameters
    ----------
    arr: [np.ndarray]
        2D array with values you need to get the indexes of the cells that are filled with these values
    mask_val: [int]
        if you need to locate only a certain value, and not all values in the array

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - first array is the x index
        - second row is the y index
    """
    # Use the arr to get the indices of the non-zero pixels.
    if mask_val:
        (i, j) = (arr == mask_val).nonzero()
    else:
        (i, j) = arr.nonzero()

    return i, j


def get_pixels(arr, mask, mask_val=None):
    """Get pixels from a raster (with optional mask).

    Parameters
    ----------
    arr : [np.ndarray]
        Array of raster data in the form [bands][y][x].
    mask : [np.ndarray]
        Array (2D) of zeroes to mask data.(from the rastarizing the vector)
    mask_val : int
        Value of the data pixels in the mask. Default: non-zero.

    Returns
    -------
    np.ndarray
        Array of non-masked data.
    """
    if mask is None:
        return arr

    i, j = _get_indeces(mask, mask_val)
    # get the coresponding values to the indeces from the array
    vals = arr[i, j] if arr.ndim == 2 else arr[:, i, j]
    return vals


def _get_indices2(arr: np.ndarray, mask: List) -> List[Tuple[int, int]]:
    """Get Indeces

        - get the indeces of array cells after filtering the values based on two mask values

    Parameters
    ----------
    arr: [np.ndarray]
        numpy array
    mask: [list]
        currently the mask list should contain only two values.

    Returns
    -------

    """
    # get the position of cells that is not zeros
    if mask is not None:
        if len(mask) > 1:
            mask = np.logical_and(
                ~np.isclose(arr, mask[0], rtol=0.001),
                ~np.isclose(arr, mask[1], rtol=0.001),
            )
        else:
            mask = ~np.isclose(arr, mask[0], rtol=0.001)

        rows = np.where(mask)[0]
        cols = np.where(mask)[1]

        ind = list(zip(rows, cols))
    else:
        rows = arr.shape[0]
        cols = arr.shape[1]
        ind = [(i, j) for i in range(rows) for j in range(cols)]

    return ind


def _get_pixels2(arr: np.ndarray, mask: List) -> List:
    """Get pixels from a raster (with optional mask).

    Parameters
    ----------
    arr : [np.ndarray]
        Array of raster data in the form [y][x].
    mask : [np.ndarray]
        Array (2D) of zeroes to mask data.(from the rastarizing the vector)

    Returns
    -------
    np.ndarray
        Array of non-masked data.
    """
    ind = _get_indices2(arr, mask)

    fn = lambda x: arr[x[0], x[1]]
    values = list(map(fn, ind))
    return values


def locate_values(values: np.ndarray, grid: np.ndarray):
    """Locate values in an array

        locate a value in array, each point has to values (resembling the x & y coordinates), the values array
        is the grid that we are trying to locate our coordinates in, with the first column being the x
        coordinate, and the second column being the y coordinates.

    Parameters
    ----------
    values: [array]
        array with a dimension (any, 2), each row has two values x & y coordinates.
        array([[454795, 503143],
               [443847, 481850],
               [454044, 481189]])
    grid: [array]
        array with a dimension (any, 2), resembling the x & y coordinates (first and second columns
        respectively).
        - The first column is the x coordinates starting from left to righ (west to east), so the first value is the min
        - The second column is the y coordinates starting from top to bottom (north to south), so the first value is
        the max
        np.array([[434968, 518007],
                   [438968, 514007],
                   [442968, 510007],
                   [446968, 506007],
                   [450968, 502007],
                   [454968, 498007],
                   [458968, 494007],
                   [462968, 490007],
                   [466968, 486007],
                   [470968, 482007],
                   [474968, 478007],
                   [478968, 474007],
                   [482968, 470007],
                   [486968, 466007]])

    Returns
    -------
    array:
        array with a shape (any, 2), for the row, column indices in the array.
        array([[ 5,  4],
               [ 2,  9],
               [ 5,  9]])
    """
    x_grid = grid[:, 0]
    y_grid = grid[:, 1]

    def find(point_i):
        x_ind = np.abs(point_i[0] - x_grid).argmin()
        y_ind = np.abs(point_i[1] - y_grid).argmin()
        return x_ind, y_ind

    indices = np.array(list(map(find, values)))

    return indices
