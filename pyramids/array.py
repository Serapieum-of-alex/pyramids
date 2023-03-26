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
