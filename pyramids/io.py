import os
import zipfile
import gzip
from osgeo import gdal


def _is_zip(path: str):
    return path.endswith(".zip") or path.__contains__(".zip")

def _is_gzip(path: str):
    return path.endswith(".gz") or path.__contains__(".gz")

def _get_zip_path(path: str, file_i: int = 0):
    """Get Zip Path.

        - check if the given path contains a .zip in it
        - if the path contains a zip but does not end with zip (xxxx.zip/1.asc), so the path contains the
        internal path inside the zip file, so just add the prefix
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
    file_i: [int]
        index to the file inside the compressed file you want to read.

    Returns
    -------
    str path to gdal to read the zipped file.
    """
    # get list of files inside the compressed file
    if path.__contains__(".gz") and not path.endswith(".gz"):
        vsi_path = f"/vsigzip/{path}"
    else:
        vsi_path = f"/vsigzip/{path}"
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
        new_path =  _get_zip_path(path, file_i=file_i)
    elif _is_gzip(path):
        new_path = _get_gzip_path(path, file_i=file_i)
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
    path = _parse_path(path)
    access = gdal.GA_ReadOnly if read_only else gdal.GA_Update
    src = gdal.OpenShared(path, access)
    if src is None:
        raise ValueError(
            f"The raster path: {path} you enter gives a None gdal Object check the read premission, maybe "
            f"the raster is being used by other software"
        )
    return src