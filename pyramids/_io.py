import os
import zipfile
import tarfile
import gzip
import numpy as np
import warnings
from osgeo import gdal

from pyramids._errors import FileFormatNotSupported

gdal.UseExceptions()

COMPRESSED_FILES_EXTENSIONS = [".zip", ".gz", ".tar"]
DOES_NOT_SUPPORT_INTERNAL = [".gz"]


def _is_zip(path: str):
    return path.endswith(".zip") or path.__contains__(".zip")


def _is_gzip(path: str):
    return path.endswith(".gz") or path.__contains__(".gz")


def _is_tar(path: str):
    return path.endswith(".tar.gz") or path.__contains__(".tar")


def _get_zip_path(path: str, file_i: int = 0):
    """Get Zip Path.

    Parameters
    ----------
    path: [str]
        path to the zip file.
    file_i: [int]
        index to the file inside the compressed file you want to read.

    Returns
    -------
    str path to gdal to read the zipped file.

    Examples
    --------
    - Internal Zip file path (one/multiple files inside the compressed file):
        if the path contains a zip but does not end with zip (compressed-file-name.zip/1.asc), so the path contains
        the internal path inside the zip file, so just ad
    >>> rdir = "tests/data/virtual-file-system"
    >>> path = _get_zip_path(f"{rdir}/multiple_compressed_files.zip/1.asc")
    >>> print(path)
    >>> "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc"
    - Only the Zip file path (one/multiple files inside the compressed file):
        If you provide the name of the zip file with multiple files inside it, it will return the path to the first
        file.
    >>> path = _get_zip_path(f"{rdir}/multiple_compressed_files.zip")
    >>> print(path)
    >>> "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc"
    - Zip file path and an index (one/multiple files inside the compressed file):
        if you provide the path to the zip file and an index to the file inside the compressed file you want to read
    >>> path = _get_zip_path("compressed-file-name.zip", file_i=1)
    >>> print(path)
    >>> "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/2.asc"
    """
    # get a list of files inside the compressed file
    if path.__contains__(".zip") and not path.endswith(".zip"):
        vsi_path = f"/vsizip/{path}"
    else:
        file_list = zipfile.ZipFile(path).namelist()
        vsi_path = f"/vsizip/{path}/{file_list[file_i]}"
    return vsi_path


def _get_gzip_path(path: str, file_i: int = 0):
    """Get Zip Path.

        - check if the given path contains a .gz in it
        - if the path contains a gz but does not end with gz (xxxx.gz/1.asc), so the path contains the
        internal path inside the gz file, so just add the prefix
        - anything else just add the prefix.

    Parameters
    ----------
    path: [str]
        path to the zip file.

    Returns
    -------
    str path to gdal to read the zipped file.
    """
    # get list of files inside the compressed file
    warnings.warn(
        "gzip compressed files does not support getting internal file list, if the compressed file contains more than "
        "one file error will be given, you have to provide the internal path (i.e. "
        "path/file-name.gz/internal-file.ext)"
    )
    if path.__contains__(".gz") and not path.endswith(".gz"):
        vsi_path = f"/vsigzip/{path}"
    else:
        try:
            file_list = tarfile.open(path).getnames()
            vsi_path = f"/vsigzip/{path}/{file_list[file_i]}"
        except tarfile.ReadError:
            # if the tarfile.open() does not give a getnames() method, it means the file contains one file
            # so return the path of the main file
            vsi_path = f"/vsigzip/{path}"
    return vsi_path


def _get_tar_path(path: str):
    """Get Zip Path.

        - check if the given path contains a .gz in it
        - if the path contains a gz but does not end with gz (xxxx.gz/1.asc), so the path contains the
        internal path inside the gz file, so just add the prefix
        - anything else just add the prefix.

    Parameters
    ----------
    path: [str]
        path to the zip file.

    Returns
    -------
    str path to gdal to read the zipped file.
    """
    # get list of files inside the compressed file
    if path.__contains__(".tar") and not path.endswith(".tar"):
        vsi_path = f"/vsitar/{path}"
    else:
        vsi_path = f"/vsitar/{path}"
    return vsi_path


def _parse_path(path: str, file_i: int = 0) -> str:
    """Parse Path.

    Parameters
    ----------
    path: [str]
        path to the file.
    file_i: [int]
        index to the file inside the compressed file you want to read, if the compressed file has only one file
        inside it will read this file, if multiple files are compressed, it will return the first file.

    Returns
    -------
    file path: [str]
        path to the file to read.
    """
    if _is_zip(path):
        new_path = _get_zip_path(path, file_i=file_i)
    elif _is_tar(path):
        new_path = _get_tar_path(path)
    elif _is_gzip(path):
        new_path = _get_gzip_path(path, file_i=file_i)
    else:
        new_path = path
    return new_path


def extract_from_gz(input_file: str, output_file: str, delete=False):
    """Extract data from the zip/.gz files, save the data.

    Parameters
    ----------
    input_file : [str]
        zipped file name.
    output_file : [str]
        directory where the unzipped data must be
                            stored.
    delete : [bool]
        True if you want to delete the zipped file after the extracting the data
    Returns
    -------
    None.
    """
    with gzip.GzipFile(input_file, "rb") as zf:
        content = zf.read()
        save_file_content = open(output_file, "wb")
        save_file_content.write(content)

    save_file_content.close()
    zf.close()

    if delete:
        os.remove(input_file)


def read_file(
    path: str,
    read_only: bool = True,
    open_as_multi_dimensional: bool = False,
    file_i: int = 0,
):
    """Open file.

        - for geotiff and ASCII files.

    Parameters
    ----------
    path : [str]
        Path of file to open (works for ascii, geotiff).
    read_only : [bool]
        File mode, set to `False` to open in "update" mode.
    open_as_multi_dimensional: [bool]
        Default is False.
    file_i: [int] default is 0
        index to the file inside the compressed file you want to read, if the compressed file have only one file

    Returns
    -------
    GDAL dataset
    """
    if not isinstance(path, str):
        raise TypeError(
            f"the path parameter should be of string type, given: {type(path)}"
        )
    path = _parse_path(path, file_i=file_i)
    access = gdal.GA_ReadOnly if read_only else gdal.GA_Update
    try:
        # get the file extension
        # file_extension = path.split(".")[-1].lower()
        # Example criteria for using gdal.OpenEx with OF_MULTIDIM_RASTER for complex multi-dimensional formats
        if (
            open_as_multi_dimensional
        ):  # file_extension in ["hdf", "h5", "nc", "nc4", "grib", "grib2", "jp2"]:
            # Use OpenEx with the OF_MULTIDIM_RASTER flag for formats that often require handling of multi-dimensional
            # data
            src = gdal.OpenEx(path, access | gdal.OF_MULTIDIM_RASTER)
        else:
            # Use OpenShared for potentially frequently accessed raster files
            src = gdal.OpenShared(path, access)
    except Exception as e:
        if str(e).__contains__(" not recognized as a supported file format."):
            if any(path.endswith(i) for i in COMPRESSED_FILES_EXTENSIONS):
                raise FileFormatNotSupported(
                    "File format is not supported if you provided a gzip compressed file with multiple internal "
                    "files. Currently, it is not supported to read gzip files with multiple compressed internal "
                    "files"
                )
            else:
                raise e
        elif any(path.__contains__(i) for i in DOES_NOT_SUPPORT_INTERNAL) and not any(
            path.endswith(i) for i in DOES_NOT_SUPPORT_INTERNAL
        ):
            raise FileFormatNotSupported(
                "File format is not supported, if you provided a gzip/7z compressed file with multiple internal "
                "files. Currently it is not supported to read gzip/7z files with multiple compressed internal "
                "files"
            )
        elif str(e).__contains__(" No such file or directory"):
            raise FileNotFoundError(f"{path} you entered does not exist")
        else:
            raise e
    # if src is None:
    #     raise ValueError(
    #         f"The raster path: {path} you enter gives a None gdal Object check the read premission, maybe "
    #         f"the raster is being used by other software"
    #     )
    return src


def insert_space(inp):
    """Insert space between the ascii file values."""
    return str(inp) + "  "


def to_ascii(
    arr: np.ndarray, cell_size: int, xmin, ymin, no_data_value, path: str
) -> None:
    """Write raster into ascii file.

        to_ascii reads writes the raster to disk into an ascii format.

    Parameters
    ----------
    arr: [np.ndarray]
        array you want to write to disk.
    cell_size: [int]
        cell size.
    xmin: [float]
        x coordinate of the lower left corner.
    ymin: [float]
        y coordinate of the lower left corner.
    no_data_value: [numeric]
        no data vlaue.
    path: [str]
        name of the ASCII file you want to convert and the name
        should include the extension ".asc"
    """
    if not isinstance(path, str):
        raise TypeError("path input should be string type")

    if os.path.exists(path):
        raise FileExistsError(
            f"There is a file with the same path you have provided: {path}"
        )
    rows = arr.shape[0]
    columns = arr.shape[1]
    # y_lower_side = geotransform[3] - rows * cell_size
    # write the the ASCII file details
    File = open(path, "w")
    File.write("ncols         " + str(columns) + "\n")
    File.write("nrows         " + str(rows) + "\n")
    File.write("xllcorner     " + str(xmin) + "\n")
    File.write("yllcorner     " + str(ymin) + "\n")
    File.write("cellsize      " + str(cell_size) + "\n")
    File.write("NODATA_value  " + str(no_data_value) + "\n")
    # write the array
    for i in range(rows):
        File.writelines(list(map(insert_space, arr[i, :])))
        File.write("\n")

    File.close()
