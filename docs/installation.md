# Installation

## Dependencies

### Required dependencies

- Python (3.11 or later)
- numpy (1.21 or later)
- GDAL (3.9.0 or later)
- pandas (2 or later)
- geopandas (0.12.2 or later)
- Shapely (1.8.4 or later)
- pyproj (3.4 or later)
- PyYAML (6.0 or later)

### Optional dependencies

- cleopatra (0.3.4 or later)

## Stable release

Please install pyramids in a virtual environment so that its requirements don't tamper with your system Python.

### conda

The easiest way to install pyramids is using the conda package manager. pyramids is available in the conda-forge channel. To install, run:

```console
conda install -c conda-forge pyramids
```

If this works, it will install pyramids with all dependencies including Python and GDAL, and you can skip the rest of the installation instructions.

## Installing Python and GDAL dependencies

The main dependencies for pyramids are an installation of Python 3.9+ and GDAL.

### Installing Python

We recommend using the Anaconda Distribution for Python 3:
https://www.anaconda.com/download/

Ensure python, pip, and conda are on your PATH.

Note: There is no hard requirement for Anaconda specifically, but conda often makes installing dependencies easier.

## Install as a conda environment

The easiest and most robust way to install pyramids is in a separate conda environment. In the root repository directory there is an environment.yml file listing all dependencies (use from a specific release for stability).

Create the environment:

```console
conda env create -f environment.yml
```

Activate it:

```console
conda activate pyramids
```

Install pyramids from PyPI (specific release example):

```console
pip install pyramids-gis=={release}
```

## From sources

The sources for pyramids can be downloaded from the GitHub repo.

Clone the repository:

```console
git clone https://github.com/MAfarrag/pyramids
```

Or download the tarball:

```console
curl -OJL https://github.com/MAfarrag/pyramids/tarball/main
```

Install from source:

```console
python -m pip install .
```

To install directly from GitHub (HEAD of main branch):

```console
pip install git+https://github.com/MAfarrag/pyramids.git
```

Or from a specific release:

```console
pip install git+https://github.com/MAfarrag/pyramids.git@{release}
```

Now you should be able to start Python and run:

```python
import pyramids
```

More details on conda environments:
https://conda.io/docs/user-guide/tasks/manage-environments.html

If you plan to contribute to development, clone the repository and do an editable install:

```console
git clone https://github.com/MAfarrag/pyramids.git
cd pyramids
conda activate pyramids
pip install -e .
```

Alternatively, download a zip archive and test the latest version:
https://github.com/MAfarrag/pyramids/archive/master.zip

## Install using pip

Besides the recommended conda setup, you can also install with pip. For harder dependencies, use conda first:

```console
conda install numpy gdal
```

Then install a release with pip:

```console
pip install pyramids-gis=={release}
```

## Check if the installation is successful

Go to the examples directory and run the following command:

```console
python -m pyramids.*******
```

This should run without errors.

> Note
>
> This documentation was generated on |today|
>
> Documentation for the development version:
> https://pyramids-gis.readthedocs.org/en/latest/
>
> Documentation for the stable version:
> https://pyramids-gis.readthedocs.org/en/stable/
