# ADR-0006: Docker image

- Status: Accepted
- Date: 2025-08-24

## Context
We need a reproducible, hermetic container that runs the pyramids package across platforms without requiring local
installations of GDAL/PROJ/NetCDF or compiler toolchains. The image should:
- Be fast to rebuild via Docker layer caching.
- Bundle all native geospatial dependencies inside the image/runtime environment.
- Be CI-friendly and cross-platform (supporting amd64/arm64 builds via Buildx when needed).
- Avoid ad-hoc apt installs in the final image and avoid conda complexity; rely on a locked environment.

## Decision
- Use a multi-stage Dockerfile driven by Pixi environments:
  - build stage: `FROM ghcr.io/prefix-dev/pixi:bookworm-slim`
    - Build arg `ENV_NAME` (defaults to `default`) selects which Pixi environment to build.
    - Environment path: `/app/.pixi/envs/${ENV_NAME}` (exported as `PIXI_ENV_DIR`).
    - Copy `pyproject.toml`, `pixi.lock`, and `README.md` first for caching; then `src/`.
    - Ensure non-editable install by rewriting any `editable = true` to `editable = false` in `pyproject.toml` during build.
    - Run `pixi install -e "${ENV_NAME}"` to resolve and build the environment per `pixi.lock`.
  - production stage: `FROM debian:bookworm-slim`
    - Copy only the built environment from the build stage into a neutral prefix `VENV_DIR=/opt/venv`.
    - Set runtime environment variables so native deps resolve correctly:
      - `PATH="${VENV_DIR}/bin:${PATH}`
      - `LD_LIBRARY_PATH="${VENV_DIR}/lib:${LD_LIBRARY_PATH}`
      - `PROJ_LIB="${VENV_DIR}/share/proj"`, `GDAL_DATA="${VENV_DIR}/share/gdal"`, `PYTHONNOUSERSITE=1`
    - Verify at build time that `pyramids`, `osgeo.gdal`, `shapely`, and `pyproj` import and that Python runs from `${VENV_DIR}`.
    - Default `CMD` uses `${VENV_DIR}/bin/python` to import `pyramids` and print the Python version as a sanity check.
- Python version and native library versions are pinned by `pixi.lock` (not by a Docker `PYTHON_VERSION` build arg).
- No additional `apt-get` installs are performed in the final image; all binaries and data files come from the bundled environment.
- No non-root user or explicit `WORKDIR` is set in the current design; revisit if required for security or UX.

## Consequences
- Reproducible builds and runs across developer machines and CI without local GDAL/PROJ installs.
- Hermetic runtime: GDAL/PROJ/GEOS, etc., come from the copied environment; env vars ensure correct resolution.
- Good layer caching: dependency resolution happens in the build stage and is cached until `pyproject.toml`/`pixi.lock` change; source changes rebuild quickly.
- Users switch environments with `--build-arg ENV_NAME=<name>`; Python version comes from the selected Pixi env.
- Image remains moderately sized because we copy only the resolved environment into a slim Debian base; no extra system packages are added.
- Absence of a non-root user means the container runs as root by default; consider adding an unprivileged user in a future revision if needed.
