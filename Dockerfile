# syntax=docker/dockerfile:1.7
FROM ghcr.io/prefix-dev/pixi:bookworm-slim AS build

ARG ENV_NAME=default
ENV ENV_NAME=${ENV_NAME}
ENV PIXI_ENV_DIR=/app/.pixi/envs/${ENV_NAME}

WORKDIR /app

COPY pyproject.toml pixi.lock README.md ./
# Ensure non-editable install: set any 'editable = true' to 'editable = false' in pyproject.toml
RUN sed -i -E 's/(\beditable[[:space:]]*=[[:space:]]*)true/\1false/g' pyproject.toml
COPY src ./src


RUN pixi install -e "${ENV_NAME}"


FROM debian:bookworm-slim AS production

ARG ENV_NAME=default
ENV ENV_NAME=${ENV_NAME}
ENV PIXI_ENV_DIR=/app/.pixi/envs/${ENV_NAME}

WORKDIR /app
# Copy the ready-to-run environment
COPY --from=build "${PIXI_ENV_DIR}" "${PIXI_ENV_DIR}"

ENV PATH="${PIXI_ENV_DIR}/bin:${PATH}"

ENV LD_LIBRARY_PATH="${PIXI_ENV_DIR}/lib:${LD_LIBRARY_PATH}"
ENV PROJ_LIB="${PIXI_ENV_DIR}/share/proj"
ENV GDAL_DATA="${PIXI_ENV_DIR}/share/gdal"
ENV PYTHONNOUSERSITE=1

# Verify installation of key dependencies and package import
RUN ${PIXI_ENV_DIR}/bin/python - <<'PY'
from osgeo import gdal
import shapely, pyproj, importlib, sys
importlib.import_module('pyramids')
print('GDAL version:', gdal.__version__)
print('pyramids-gis installed, Python=' + sys.version.split()[0])
PY

# Simple default command prints package version to confirm install
CMD ["${PIXI_ENV_DIR}/bin/python", "-c", "import importlib, sys; importlib.import_module('pyramids'); print('pyramids installed,Python='+sys.version.split()[0])"]
