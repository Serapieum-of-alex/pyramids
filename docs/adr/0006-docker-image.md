# ADR-0006: Docker image

- Status: Accepted
- Date: 2025-08-16

## Context
We want a reproducible, contributor-friendly container that can build and run the pyramids package across platforms 
without forcing local GDAL/PROJ/NetCDF installations. The image should:
- Be reasonably small and fast to build using layer caching.
- Support Python 3.12 by default but allow other patch/minor versions when needed.
- Include the minimum native libraries commonly required by geospatial Python wheels and tools.
- Avoid environment managers that complicate CI (e.g., conda) for the base image.

## Decision
- Base image: `python:${PYTHON_VERSION}-slim` with a build arg `PYTHON_VERSION` defaulting to 3.12.
- Install a minimal set of native packages useful for geospatial stacks: `gdal-bin`, `libgdal-dev`, `libgeos-dev`, 
 `proj-bin`, `libproj-dev`, `libnetcdf-dev`, `libhdf4-0-alt`, plus common runtime libs (`libxml2`, `libsqlite3-0`, 
  `libcurl4`, `ca-certificates`) and build tools (`build-essential`, `git`).
- Copy `pyproject.toml` and `README.md` first to leverage Docker layer caching; copy `src/` afterward.
- Use PEP 517/518 build flow and install via `pip install .` to produce a wheel and install the package from `pyproject.toml`.
- Create and use an unprivileged user (`appuser`, uid 1000) for runtime.
- Set the working directory to `/workspace` at runtime to support volume mounts for local projects/data.
- Provide a simple default `CMD` that verifies installation by printing the package version.
- Keep a single-stage Dockerfile (no separate builder stage) to reduce complexity; revisit if size becomes an issue.
- Document common usage in comments within the Dockerfile (build with `--build-arg PYTHON_VERSION=3.12`, run with a volume mount).

## Consequences
- Reproducible builds across developer machines and CI without requiring local GDAL/PROJ installations.
- Image size remains moderate because we start from `-slim` and install only necessary native libs; still larger than pure Python due to geospatial deps.
- Layer caching makes incremental builds fast when only source changes (since dependency layers are above source layers).
- Users can override the Python version with a build arg for compatibility testing.
- Non-root default user improves security; some volumes may need adjusted permissions on host systems.
- If heavier native stacks (e.g., full GDAL CLI usage) or conda-only packages are needed, we can introduce an alternative `-conda` or `-tools` image in a future ADR.
