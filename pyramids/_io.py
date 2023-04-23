import os
import zipfile
import gzip
import numpy as np
import warnings
from osgeo import gdal

from pyramids._errors import FileFormatNoSupported

gdal.UseExceptions()

COMPRESSED_FILES_EXTENSIONS = [".zip", ".gz", ".tar"]
DOES_NOT_SUPPORT_INTERNAL = [".gz"]


def _is_zip(path: str):
    return path.endswith(".zip") or path.__contains__(".zip")


def _is_gzip(path: str):
    return path.endswith(".gz") or path.__contains__(".gz")


def _is_tar(path: str):
    return path.endswith(".tar") or path.__contains__(".tar")


def _get_zip_path(path: str, file_i: int = 0):
    """Get Zip Path.

            - check if the given path contains a .zip in it
            - if the path contains a zip but does not end with zip (xxxx.zip/1.asc), so the path contains the
    4        internal path inside the zip file, so just add the prefix
            - anything else get the list of the files inside the zip and return the full path to first file.

        Parameters
        ----------
        path: [str]
            path to the zip file.
        file_i: [int]
            index to the file inside the compressed file you want to read.

        Returns
        -------
        str path to gdal to read the zipped file.
    """
    # get list of files inside the compressed file
    if path.__contains__(".zip") and not path.endswith(".zip"):
        vsi_path = f"/vsizip/{path}"
    else:
        zfile = zipfile.ZipFile(path)
        file_list = zfile.namelist()
        vsi_path = f"/vsizip/{path}/{file_list[file_i]}"
    return vsi_path


def _get_gzip_path(path: str):
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
    """Parse Path


    Parameters
    ----------
    path: [str]
        path to the file.
    file_i: [int]
        index to the file inside the compressed file you want to read, if the compressed file have only one file
        inside it will read this file, if multiple files are compressed, it will return the first file.

    Returns
    -------
    file path
    """
    if _is_zip(path):
        new_path = _get_zip_path(path, file_i=file_i)
    elif _is_gzip(path):
        new_path = _get_gzip_path(path)
    elif _is_tar(path):
        new_path = _get_tar_path(path)
    else:
        new_path = path
    return new_path


def extract_from_gz(input_file: str, output_file: str, delete=False):
    """ExtractFromGZ method extract data from the zip/.gz files, save the data.

    Parameters
    ----------
    input_file : [str]
        zipped file name .
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


def read_file(path: str, read_only: bool = True):
    """Open file.

        - for a geotiff and ASCII files.

    Parameters
    ----------
    path : [str]
        Path of file to open(works for ascii, geotiff).
    read_only : [bool]
        File mode, set to False to open in "update" mode.

    Returns
    -------
    GDAL dataset
    """
    if not isinstance(path, str):
        raise TypeError(
            f"the path parameter should be of string type, given: {type(path)}"
        )
    path = _parse_path(path)
    access = gdal.GA_ReadOnly if read_only else gdal.GA_Update
    try:
        src = gdal.OpenShared(path, access)
    except Exception as e:
        if str(e).__contains__(" not recognized as a supported file format."):
            if any(path.endswith(i) for i in COMPRESSED_FILES_EXTENSIONS):
                raise FileFormatNoSupported(
                    "File format is not supported, if you provided a gzip compressed file with multiple internal "
                    "files. Currently it is not supported to read gzip files with multiple compressed internal "
                    "files"
                )
        elif any(path.__contains__(i) for i in DOES_NOT_SUPPORT_INTERNAL) and not any(
            path.endswith(i) for i in DOES_NOT_SUPPORT_INTERNAL
        ):
            raise FileFormatNoSupported(
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


def stringSpace(inp):
    return str(inp) + "  "


def to_ascii(
    arr: np.ndarray, cell_size: int, xmin, ymin, no_data_value, path: str
) -> None:
    """write raster into ascii file.

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
        File.writelines(list(map(stringSpace, arr[i, :])))
        File.write("\n")

    File.close()
