# Release Process — Platform Wheels + sdist

This document covers the release flow once Phase 6 automation is in place
(PyPI trusted publishing, build-wheels.yml publish job enabled).

Prerequisites: Phase 1–5 complete, PyPI trusted publishing configured
(see [`pypi-trusted-publishing.md`](./pypi-trusted-publishing.md) once written).

## Regular release (patch / minor)

### 1. Merge everything to `main`

All target changes for the release should land on `main`. CI on `main`
should be green on:
- `tests.yml` (main test suite against conda GDAL)
- `wheel-test.yml` (pure-Python wheel + conda GDAL)
- `build-wheels.yml` (platform wheels across Linux/macOS/Windows)

### 2. Bump version via commitizen

```bash
pixi run -e dev cz-bump
```

This will:
1. Determine the next version from conventional commits since last tag
2. Update `pyproject.toml` `version = "..."`
3. Create a tag `<major>.<minor>.<patch>`
4. Commit and push the tag

**Do NOT** manually edit `pyproject.toml`'s version field — commitizen
owns it.

### 3. Create GitHub release

Either via:

```bash
gh release create <version> --generate-notes
```

Or via the GitHub web UI. Tagging a release triggers `build-wheels.yml`
which:

1. Builds sdist (pure Python)
2. Builds platform wheels for Linux × {cp311, cp312, cp313}
3. Builds platform wheels for macOS × {x86_64, arm64} × {cp311, cp312, cp313}
4. Builds platform wheels for Windows × {cp311, cp312, cp313}
5. Runs the test suite against each wheel in a clean env
6. If all tests pass, uploads sdist + 15 wheels to PyPI via Trusted
   Publishing (no API tokens needed)

Total expected duration: 30–45 minutes for all platforms in parallel.

### 4. Verify on PyPI

```bash
# Wait ~1-2 minutes for PyPI to update its index, then:
pip index versions pyramids-gis
# Should show the new version

# Install in a clean env:
docker run --rm python:3.12-slim bash -c \
    "apt-get update -qq && apt-get install -y -qq libexpat1 && \
     pip install pyramids-gis==<version> && \
     python -c 'from pyramids import Dataset; from osgeo import gdal; print(gdal.__version__)'"
```

### 5. Conda-forge feedstock

The feedstock auto-detects new PyPI releases and opens a PR within a
few hours. Typically no manual action needed unless:
- Added a new runtime dep → update `recipe/meta.yaml` in the PR
- GDAL version constraint changed → same
- Bumped Python version support → update skip list

Approve the feedstock PR once it's green.

## Pre-release validation (recommended for minor / major bumps)

Before creating the real GitHub release, stage via test.pypi.org:

```bash
# 1. Trigger build-wheels.yml manually on main:
gh workflow run build-wheels.yml \
    --ref main \
    -f upload-to-pypi=false

# 2. Once green, download all artifacts:
RUN_ID=$(gh run list --workflow=build-wheels.yml --limit=1 --json databaseId --jq '.[0].databaseId')
gh run download $RUN_ID --dir /tmp/release-staging

# 3. Upload to test.pypi.org:
twine upload --repository testpypi /tmp/release-staging/wheels-*/*.whl \
                                   /tmp/release-staging/sdist/*.tar.gz

# 4. Test install from test.pypi.org:
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pyramids-gis

# 5. If good, proceed with the real release (step 2-3 above).
```

Enable the publish job in `build-wheels.yml` when ready by uncommenting
the `publish:` job.

## Emergency yank procedure

If a broken wheel is discovered after shipping:

### Step 1 — Yank on PyPI

```bash
pip yank pyramids-gis==<broken-version> \
    --reason "Broken <describe issue>; use <version> instead"
```

This marks the version as yanked:
- `pip install pyramids-gis` skips it
- `pip install pyramids-gis==<broken-version>` still works (explicit opt-in)
- Prevents new users from hitting the bug

**Do NOT delete the release on PyPI.** PyPI permanently blocks deleted
version numbers — you can never re-upload `X.Y.Z` even if the original
is deleted.

### Step 2 — Ship a patch release immediately

```bash
git checkout main
# Fix the bug ...
git add -A && git commit -m "fix: <description>"
pixi run -e dev cz-bump   # bumps to <broken-version>+1 patch
git push --follow-tags
# Then trigger release (step 3 above)
```

Target: ship the fix within **24 hours** of discovering a broken wheel.

### Step 3 — Communicate

1. Pin a GitHub issue explaining the bug, the fix version, and the yank.
2. Post to any active support channels (Slack, Discord, etc.).
3. Update the CHANGELOG with both the broken version (marked yanked)
   and the fix version.
4. If CVE-worthy (security), coordinate with Anthropic's security
   channels before disclosure — not applicable to pyramids today.

## Rollback: revert to pure-Python wheel

If the bundled-GDAL approach causes systematic issues (e.g., manylinux
policy changes, conda-forge breakage), we can fall back to the pre-Phase-1
pure-Python wheel:

1. Revert `build-wheels.yml` to an earlier commit (pre-Phase 1).
2. Revert `pyproject.toml` to remove `[tool.cibuildwheel]` and the
   `wheel-build` pixi env.
3. Remove `setup.py` (BinaryDistribution override).
4. Release a new patch version with just the sdist → pip installs fall
   back to "requires system GDAL" behavior.
5. Document the regression in the CHANGELOG.

Keep the `ci/` scripts archived; they represent real work and may be
useful later.

## Release checklist (copy into GitHub release draft)

```markdown
Pre-release
- [ ] `tests.yml` green on main
- [ ] `wheel-test.yml` green on main
- [ ] `build-wheels.yml` green on main (test.pypi upload optional)
- [ ] Test install from test.pypi.org in fresh container (Linux, Windows, macOS)
- [ ] Wheel sizes all under 100 MB (report in CI notice annotations)

Release
- [ ] `pixi run -e dev cz-bump`
- [ ] `git push --follow-tags`
- [ ] `gh release create <version> --generate-notes`

Post-release
- [ ] PyPI index shows new version (wait ~2 min)
- [ ] Smoke-test `pip install pyramids-gis==<version>` in clean Docker
- [ ] Conda-forge feedstock PR merged
- [ ] CHANGELOG updated with user-facing highlights
```
