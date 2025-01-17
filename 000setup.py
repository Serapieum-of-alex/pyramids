
from setuptools import setup
import subprocess
import requests
import zipfile
from io import BytesIO
import os
from pathlib import Path
import urllib.request
INCLUDE_DIR = Path("C:/Program Files/GDAL/include")


def pre_install_gdal():
    # Download and install GDAL MSI
    download_url_3 = "https://download.gisinternals.com/sdk/downloads/release-1930-x64-gdal-3-10-0-mapserver-8-2-2/gdal-3.10.0-1930-x64-core.msi"
    # this msi has
    # download_url_3 = "https://download.gisinternals.com/sdk/downloads/release-1930-x64-dev.zip"
    download_path_3 = "core.msi"
    urllib.request.urlretrieve(download_url_3, download_path_3)

    # custom_dir = r"C:\\Program Files\\GDAL"
    subprocess.run(
        [
            "msiexec",
            "/i",
            download_path_3,
            # f"INSTALLDIR={custom_dir}",
            "/quiet",
            "/norestart"
        ],
        check=True,
    )

    # Update environment variables
    os.environ["PATH"] += os.pathsep + r"C:\\Program Files\\GDAL"
    os.environ["GDAL_DATA"] = r"C:\\Program Files\\GDAL\\gdal-data"

# def download_and_extract_headers(version, output_dir = INCLUDE_DIR):
#     """
#     Downloads the GDAL source code as a zip file, extracts only `gdal.h` and `cpl_string.h`,
#     and saves them to the specified output directory.
#
#     Parameters:
#     - version (str): The GDAL version tag or branch (e.g., 'v3.10.0', 'release/3.10.0').
#     - output_dir (str): The directory to save the extracted files.
#
#     Returns:
#     - None
#     """
#     # Construct the URL to the zip file of the specified version
#     zip_url = f"https://github.com/OSGeo/gdal/archive/{version}.zip"
#
#     # Ensure the output directory exists
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir()
#
#     try:
#         print(f"Downloading GDAL source code from {zip_url}...")
#         response = requests.get(zip_url, stream=True)
#         response.raise_for_status()
#         version = version[1:]
#         # Read the zip file from the response
#         with zipfile.ZipFile(BytesIO(response.content)) as gdal_zip:
#             # File paths to extract
#
#             needed_files = [
#                 f"gdal-{version}/gcore/gdal.h",
#                 f"gdal-{version}/port/cpl_string.h",
#                 f"gdal-{version}/port/cpl_port.h",
#                 f"gdal-{version}/port/cpl_error.h",
#                 f"gdal-{version}/port/cpl_progress.h",
#                 f"gdal-{version}/port/cpl_conv.h",
#                 f"gdal-{version}/port/cpl_virtualmem.h",
#                 f"gdal-{version}/port/cpl_vsi.h",
#                 f"gdal-{version}/port/cpl_minixml.h",
#                 f"gdal-{version}/port/cpl_multiproc.h",
#                 f"gdal-{version}/port/cpl_http.h",
#                 f"gdal-{version}/ogr/ogr_api.h",
#                 f"gdal-{version}/ogr/ogr_srs_api.h",
#             ]
#             for file_path in needed_files:
#                 try:
#                     print(f"Extracting {file_path}...")
#                     file_data = gdal_zip.read(file_path)
#
#                     # Save the file to the output directory
#                     file_name = os.path.basename(file_path)
#                     with open(os.path.join(output_dir, file_name), "wb") as output_file:
#                         output_file.write(file_data)
#
#                     print(f"Saved {file_name} to {output_dir}")
#                 except KeyError:
#                     print(f"File {file_path} not found in the archive.")
#     except requests.RequestException as e:
#         print(f"Error downloading GDAL source code: {e}")

def download_and_extract_headers(version, output_dir=INCLUDE_DIR):
    """
    Downloads the GDAL source code as a zip file, extracts required header files,
    and saves them to the specified output directory. If certain files are missing
    (e.g., gdal_version.h or cpl_config.h), generates placeholder files.

    Parameters:
    - version (str): The GDAL version tag or branch (e.g., 'v3.10.0', 'release/3.10.0').
    - output_dir (str): The directory to save the extracted files.

    Returns:
    - None
    """
    # GDAL GitHub repository zip file URL
    zip_url = f"https://github.com/OSGeo/gdal/archive/{version}.zip"

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        print(f"Downloading GDAL source code from {zip_url}...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()

        # Read the zip file from the response
        with zipfile.ZipFile(BytesIO(response.content)) as gdal_zip:
            archive_files = gdal_zip.namelist()

            # Extract all .h files from port and ogr directories
            for file_path in archive_files:
                if file_path.endswith(".h"): #and ("/port/" in file_path or "/ogr/" in file_path)
                    try:
                        print(f"Extracting {file_path}...")
                        file_data = gdal_zip.read(file_path)

                        # Save the file to the output directory
                        output_path = os.path.join(output_dir, os.path.basename(file_path))
                        with open(output_path, "wb") as output_file:
                            output_file.write(file_data)

                        print(f"Saved {file_path} to {output_dir}")
                    except KeyError:
                        print(f"File {file_path} not found in the archive. Skipping.")

        # Generate placeholder files for gdal_version.h and cpl_config.h
        generate_placeholder_files(output_dir)

    except requests.RequestException as e:
        print(f"Error downloading GDAL source code: {e}")


def set_env_vars():
    gdal_lib_dir = r"C:\Program Files\GDAL\lib"  # Optional: Directory for GDAL libraries

    existing_include = os.getenv("INCLUDE", None)
    # Update environment variables for the build process
    os.environ["INCLUDE"] = str(INCLUDE_DIR) if existing_include is None else existing_include + ";" + os.environ.get("INCLUDE")
    existing_lib = os.getenv("LIB", None)
    os.environ["LIB"] = gdal_lib_dir if existing_lib is None else gdal_lib_dir + ";" + os.environ.get("LIB")
    os.environ["CFLAGS"] = f"-I{INCLUDE_DIR}"


def generate_placeholder_files(output_dir):
    """
    Generate placeholder files for gdal_version.h and cpl_config.h.

    Parameters:
    - output_dir (str): The directory to save the placeholder files.

    Returns:
    - None
    """
    import platform
    sizeof_unsigned_long = 8 if platform.architecture()[0] == "64bit" else 4
    sizeof_voidp = 8 if platform.architecture()[0] == "64bit" else 4

    placeholders = {
        "gdal_version.h": f"""
#ifndef GDAL_VERSION_H
#define GDAL_VERSION_H

#define GDAL_VERSION_MAJOR 3
#define GDAL_VERSION_MINOR 10
#define GDAL_VERSION_REV 0
#define GDAL_RELEASE_DATE 20211020
#define GDAL_RELEASE_NAME \"3.10.0\"

#endif /* GDAL_VERSION_H */
        """,
        "cpl_config.h": f"""
#ifndef CPL_CONFIG_H
#define CPL_CONFIG_H

/* Placeholder configuration */
#define SIZEOF_UNSIGNED_LONG {sizeof_unsigned_long}
#define SIZEOF_VOIDP {sizeof_voidp}

#endif /* CPL_CONFIG_H */
        """
    }

    for file_name, content in placeholders.items():
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w") as file:
            file.write(content)
        print(f"Generated placeholder for {file_name} at {output_path}")

def build_with_gdal(output_dir, source_file):
    """
    Compile and link a test program with gdal_i.lib.

    Parameters:
    - output_dir (str): The directory containing GDAL headers and libraries.
    - source_file (str): The path to the source file to compile.

    Returns:
    - None
    """
    gdal_lib_dir = os.path.join(output_dir, "lib")
    gdal_include_dir = os.path.join(output_dir, "include")
    gdal_import_lib = os.path.join(gdal_lib_dir, "gdal_i.lib")

    if not os.path.exists(gdal_import_lib):
        print(f"Error: {gdal_import_lib} not found.")
        return

    try:
        compile_cmd = [
            "cl",
            f"/I{gdal_include_dir}",
            source_file,
            f"/link",
            f"/LIBPATH:{gdal_lib_dir}",
            "gdal_i.lib",
        ]

        print("Compiling and linking with GDAL...")
        subprocess.run(compile_cmd, check=True)
        print("Build successful.")

    except subprocess.CalledProcessError as e:
        print(f"Error during build: {e}")

#%%
pre_install_gdal()
download_and_extract_headers(version="v3.10.0")
generate_placeholder_files(output_dir=INCLUDE_DIR)

setup()

