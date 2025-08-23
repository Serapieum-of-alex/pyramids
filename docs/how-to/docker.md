# Docker image and GitHub Container Registry (GHCR)

This guide explains everything you need to work with the provided Dockerfile:
- What the Dockerfile does and its stages
- How to build the image locally
- How to run the container
- How to push the image to GitHub Container Registry (GHCR)
- How to delete images from GHCR when needed

> Requirements
> - Docker installed (Docker Desktop on Windows/macOS, or Docker Engine on Linux)
> - Optional: GitHub CLI (`gh`) for advanced GHCR management
> - Optional for pushing/deleting: a GitHub Personal Access Token (classic) with at least `read:packages` and `write:packages` scopes. For deleting, also `delete:packages`.

---

## Understand the Dockerfile

Path: `./Dockerfile`

The Dockerfile is multi-stage and optimized to build a ready-to-run environment using [Pixi]. It has two stages:

1) build (FROM `ghcr.io/prefix-dev/pixi:bookworm-slim`)
   - Installs the project into an isolated Pixi environment.
   - Honors `ENV_NAME` so you can choose which Pixi environment to build (defaults to `default`).
   - Caches Pixi downloads for faster rebuilds.

2) production (FROM `debian:bookworm-slim`)
   - Copies only the built Pixi environment into a small runtime image.
   - Sets environment variables so GDAL/PROJ, etc. work out of the box.
   - Default `CMD` simply verifies the package import and prints Python version.

Key build arguments and environment variables:
- `ARG ENV_NAME=default` → set with `--build-arg ENV_NAME=<name>` to pick a Pixi environment.
- `ENV PIXI_ENV_DIR=/app/.pixi/envs/${ENV_NAME}` → where the runtime lives.

If you need multiple platforms (e.g., Apple Silicon/arm64 vs. amd64), you can use Buildx (see below).


## Docker runtime environment variables

- LD_LIBRARY_PATH
    - Purpose: Ensures the dynamic linker can find shared libraries (e.g., GDAL, PROJ, GEOS) bundled inside the environment embedded in the image.
    - Effect: Points the loader to the environment’s lib directory first so extensions depending on native libraries resolve correctly at runtime.

- PROJ_LIB
    - Purpose: Tells PROJ where to find its datum shift grids and CRS resource files.
    - Effect: Enables coordinate transformations and reprojections that require PROJ’s data files.

- GDAL_DATA
    - Purpose: Points GDAL to its data directory containing coordinate system definitions, driver metadata, and supporting resources.
    - Effect: Ensures GDAL utilities and Python bindings can locate EPSG definitions and other essential data.

- PYTHONNOUSERSITE=1
    - Purpose: Prevents Python from loading packages from the user’s site-packages directory.
    - Effect: Produces a hermetic runtime by using only the packages inside the image’s environment, avoiding accidental contamination from host-level Python packages.

---

## Build the image

Typical local build (PowerShell or any shell):

```powershell
# In the repo root
$IMAGE = "pyramids"
# Build using the default Pixi environment
docker build -t $IMAGE:latest .
```

Choose a Pixi environment with `ENV_NAME` (if you have multiple, e.g., `py311`):

```powershell
docker build --build-arg ENV_NAME=default -t pyramids:default .
# or
# docker build --build-arg ENV_NAME=py311 -t pyramids:py311 .
```

Tag with a version too:

```powershell
$VERSION = "0.1.0"
docker build -t pyramids:latest -t pyramids:$VERSION .
```

Multi-arch build (optional) with Buildx:

```powershell
# Create and use a builder once (if needed)
docker buildx create --use --name multi
# Build for linux/amd64 (default on most PCs) and/or linux/arm64 (Apple Silicon)
docker buildx build --platform linux/amd64 -t pyramids:latest .
# For arm64 as well:
# docker buildx build --platform linux/amd64,linux/arm64 -t pyramids:latest .
```

---

## Run the container

Basic run:

```powershell
docker run --rm pyramids:latest
```

Interactive shell inside the container:

```powershell
docker run --rm -it pyramids:latest bash
```

Mount a local folder (e.g., to access data):

```powershell
# Replace C:\data with your real path
docker run --rm -it -v C:\data:/data pyramids:latest bash
```

Note: Environment variables like `GDAL_DATA` and `PROJ_LIB` are already set inside the image. The Pixi environment is in the PATH by default.

---

## Push to GitHub Container Registry (GHCR)

Image naming convention for GHCR: `ghcr.io/<owner>/<repo>[:tag]`.

For this repository, a convenient image name at release time is:

```
ghcr.io/<lowercased-owner>/<lowercased-repo>
```

Example (adjust owner/repo):

```powershell
$OWNER = "Serapieum-of-alex"
$REPO  = "pyramids"
$IMAGE = ("ghcr.io/{0}/{1}" -f $OWNER.ToLower(), $REPO.ToLower())
$VERSION = "0.1.0"

# Tag the local image to GHCR
docker tag pyramids:latest  $IMAGE:latest
docker tag pyramids:latest  $IMAGE:$VERSION

# Login to GHCR (use a PAT with write:packages)
# On Windows PowerShell, you can be prompted for the PAT:
# Create an env var GHCR_PAT temporarily if you prefer non-interactive login
# $env:GHCR_PAT = "<YOUR_PAT>"
# echo $env:GHCR_PAT | docker login ghcr.io -u <your-github-username> --password-stdin

docker login ghcr.io -u <your-github-username>
# Paste PAT when prompted

# Push
docker push $IMAGE:$VERSION
docker push $IMAGE:latest
```

Automated publish on GitHub Release
- This repository includes a workflow: `.github/workflows/docker-release.yml`.
- When a GitHub Release is created, it:
  - Builds the Docker image from `Dockerfile`.
  - Tags it with the release version (and `latest` if not a pre-release) using `docker/metadata-action`.
  - Pushes to `ghcr.io/<owner>/<repo>` using the repository’s `GITHUB_TOKEN`.

To trigger: create a new release (or use the "Run workflow" button for manual dispatch if enabled).

---

## Delete images from GHCR

There are two common approaches: web UI and `gh` CLI.

A) Web UI
1. Go to your GitHub org/user → Packages → Find the container package named after the repo (e.g., `pyramids`).
2. Open the package → Versions → Delete the version(s) you want. You may need `delete:packages` scope.

B) GitHub CLI (`gh`)

The API endpoints differ for user vs organization. Examples below assume the image name is `pyramids` under organization `Serapieum-of-alex`.

List versions (organization):

```powershell
$ORG = "Serapieum-of-alex"
$PKG = "pyramids"  # package name in GHCR equals the lowercased repo name by default

gh api -H "Accept: application/vnd.github+json" `
  "/orgs/$ORG/packages/container/$PKG/versions" | `
  ConvertFrom-Json | Select-Object id,name,metadata
```

Delete a specific version by ID (organization):

```powershell
$VERSION_ID = 123456

gh api -X DELETE -H "Accept: application/vnd.github+json" `
  "/orgs/$ORG/packages/container/$PKG/versions/$VERSION_ID"
```

For user-owned packages, replace `/orgs/{org}` with `/user`:

```powershell
$PKG = "pyramids"

# List
gh api -H "Accept: application/vnd.github+json" "/user/packages/container/$PKG/versions"

# Delete
$VERSION_ID = 123456
gh api -X DELETE -H "Accept: application/vnd.github+json" "/user/packages/container/$PKG/versions/$VERSION_ID"
```

Notes:
- You must authenticate `gh` (run `gh auth login`) with a token that has `read:packages`, `write:packages`, and `delete:packages` to delete.
- Deleting "versions" removes specific tags. Deleting the whole package is also possible via the UI if you need a full reset.

---

## Troubleshooting

- denied: permission: Check you’re logged into GHCR and your token has the right scopes.
- image name invalid: Ensure the name is lowercase for GHCR (owner and repo must be lowercase in the full image reference).
- multi-arch build fails: Use `docker buildx` and ensure QEMU emulation is enabled if cross-building.
- slow builds: The Dockerfile uses Pixi cache; make sure Docker BuildKit is enabled (default in recent Docker releases).

---

## References
- GitHub Container Registry: https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- docker/build-push-action: https://github.com/docker/build-push-action
- docker/login-action: https://github.com/docker/login-action
- docker/metadata-action: https://github.com/docker/metadata-action
- Pixi: https://pixi.sh
