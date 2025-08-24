# syntax=docker/dockerfile:1.7
ARG ENV_NAME=default

FROM ghcr.io/prefix-dev/pixi:bookworm-slim AS build

ARG ENV_NAME
ENV PIXI_ENV_DIR=/app/.pixi/envs/${ENV_NAME}

WORKDIR /app

COPY pyproject.toml pixi.lock README.md ./
# Ensure non-editable install: set any 'editable = true' to 'editable = false' in pyproject.toml
RUN sed -i -E 's/(\beditable[[:space:]]*=[[:space:]]*)true/\1false/g' pyproject.toml
COPY src ./src


RUN pixi install -e "${ENV_NAME}"


FROM debian:bookworm-slim AS production

ARG ENV_NAME
ENV PIXI_ENV_DIR=/app/.pixi/envs/${ENV_NAME}
ENV VENV_DIR=/opt/venv

# Copy the ready-to-run environment directly to the neutral prefix
COPY --from=build "${PIXI_ENV_DIR}" "${VENV_DIR}"

ENV PATH="${VENV_DIR}/bin:${PATH}"

ENV LD_LIBRARY_PATH="${VENV_DIR}/lib:${LD_LIBRARY_PATH}"
ENV PROJ_LIB="${VENV_DIR}/share/proj"
ENV GDAL_DATA="${VENV_DIR}/share/gdal"
ENV PYTHONNOUSERSITE=1

# Verify installation of key dependencies and package import
RUN ${VENV_DIR}/bin/python - <<'PY'
import os, sys, importlib
from osgeo import gdal
import shapely, pyproj
importlib.import_module('pyramids')
print('Python executable:', sys.executable)
assert sys.executable.startswith(os.environ.get('VENV_DIR', '/opt/venv')), (sys.executable, os.environ.get('VENV_DIR'))
print('GDAL version:', gdal.__version__)
print('pyramids-gis installed, Python=' + sys.version.split()[0])
PY

# Simple default command prints package version to confirm install
CMD ["python", "-c", "import importlib, sys; importlib.import_module('pyramids'); print('pyramids installed,Python='+sys.version.split()[0])"]
