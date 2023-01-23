from typing import Tuple, Union

import numpy as np


def _getIndeces(
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


def getPixels(arr, mask, mask_val=None):
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

    i, j = _getIndeces(mask, mask_val)
    # get the coresponding values to the indeces from the array
    vals = arr[i, j] if arr.ndim == 2 else arr[:, i, j]
    return vals
