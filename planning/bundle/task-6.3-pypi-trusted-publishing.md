# Task 6.3 — Wire `build-wheels.yml` to PyPI Publishing

## Status

**Partially Solved (2026-04-13).** The workflow wiring is in place
using the existing `PYPI_PUBLISH` API token (same secret as
`pypi-release.yml`). The publish job remains **commented out** until
the user explicitly approves enabling it.

Trusted Publishing (OIDC) is deferred — documented below as a future
upgrade path but not implemented.

## Decision made — token-based, not OIDC (for now)

The user chose to keep the existing token-based flow because:

- The `PYPI_PUBLISH` secret already exists and works
- No PyPI web UI config required today
- Matches the pattern of the existing `pypi-release.yml` workflow
- Upgrade to Trusted Publishing stays ~10 minutes of work when wanted

## What was implemented

### 1. `build-wheels.yml` — draft `publish:` job

Added (commented out) near the bottom of the file:

```yaml
publish:
  needs: [build-sdist, build-linux-wheels, build-macos-wheels,
          build-windows-wheels, test-wheels]
  if: >-
    github.event_name == 'release' ||
    (github.event_name == 'workflow_dispatch' && inputs.upload-to-pypi)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v5
      with: { fetch-depth: 0 }
    - name: Download all build artifacts (sdist + all platform wheels)
      uses: actions/download-artifact@v4
      with: { pattern: "*", merge-multiple: true, path: dist/ }
    - name: Set up Pixi
      uses: serapeum-org/github-actions/actions/python-setup/pixi@pixi/v1
      with: { environments: dev, activate-environment: dev, verify-lock: 'false' }
    - name: Publish pre-built wheels + sdist to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_PUBLISH }}
      run: |
        pixi run -e dev twine upload --non-interactive dist/*
```

Key differences from the `pypi-release.yml` custom action:

- **No `python -m build` step.** Wheels + sdist are already produced
  by the `build-*-wheels` / `build-sdist` jobs earlier in the workflow.
  The custom action insists on running `python -m build` which would
  produce an empty/broken wheel in this context (cibuildwheel's
  `CIBW_BEFORE_BUILD` side effects aren't present).
- **Same twine pattern.** Uses `pixi run -e dev twine upload` with
  `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` from the same
  `PYPI_PUBLISH` secret.

### 2. `pypi-release.yml` — demoted to emergency fallback

The legacy `pypi-release.yml`:

- Auto-trigger via `workflow_run` is **disabled** (commented out)
- Kept available via `workflow_dispatch` for sdist-only emergency
  releases (e.g., if the platform wheel matrix is broken but
  conda-forge needs a new sdist)
- Header comment explains the new flow

This prevents double-publishing — `build-wheels.yml` is the single
canonical release entry point.

## How to enable the publish job

When ready (after testing with test.pypi.org or a dry-release):

1. Uncomment the `publish:` job in `build-wheels.yml`.
2. Create a release: `pixi run -e dev cz-bump && gh release create <ver> --generate-notes`.
3. Watch the workflow — the publish job runs after all builds + tests
   pass, downloads all artifacts, and uploads them to PyPI using the
   `PYPI_PUBLISH` secret.

## Risks of token-based publishing (understood and accepted)

| Risk | Mitigation |
|------|------------|
| Token leaks via logs / env exposure | GitHub masks secrets in logs; `TWINE_PASSWORD` is an env var not a CLI arg |
| Token lifetime (no auto-rotation) | Manual rotation yearly; set calendar reminder |
| Anyone with repo write access can publish | GitHub Environments with required reviewers (optional — can be added later) |
| Audit trail limited to GitHub's logs | Enable GitHub audit log retention; PyPI project history shows upload times |

If any of these become a problem, switch to Trusted Publishing (see
§"Future upgrade: Trusted Publishing" below).

## Future upgrade: Trusted Publishing (OIDC)

When you're ready to eliminate the token:

### 1. One-time setup on pypi.org

Sign in → project Settings → Publishing → add Trusted Publisher:

- **Owner:** `Serapieum-of-alex`
- **Repository name:** `pyramids`
- **Workflow name:** `build-wheels.yml`
- **Environment name:** (empty, or `release` for extra gating)

### 2. Replace the publish step in `build-wheels.yml`

Swap the `run: pixi run ... twine upload` step for:

```yaml
permissions:
  id-token: write   # Required for OIDC

steps:
  ...
  - name: Publish to PyPI (Trusted Publishing)
    uses: pypa/gh-action-pypi-publish@release/v1
```

No token needed. `pypa/gh-action-pypi-publish` handles the OIDC handshake.

### 3. Remove `PYPI_PUBLISH` secret from the repo

Once trusted publishing is confirmed working, delete the secret to
reduce attack surface. `pypi-release.yml` emergency fallback workflow
would also need updating (or deletion).

## What Claude did

- Uncommented + rewrote the `publish:` job in `build-wheels.yml`
  (still commented out behind `#` for safety)
- Demoted `pypi-release.yml` to emergency-only (disabled auto-trigger)
- Updated this document to reflect the decision + path

## What Claude will NOT do without explicit user approval

- Uncomment the `publish:` job (enable auto-publish)
- Run `gh workflow run build-wheels.yml -f upload-to-pypi=true`
- Push the branch with the enabled publish job to `main`

These all require the user's explicit go-ahead per the standing "do
not publish without approve" instruction.

## Deliverables checklist

- [x] (Claude) Draft `publish:` job in `build-wheels.yml` with token auth
- [x] (Claude) Demote `pypi-release.yml` to emergency fallback
- [x] (Claude) Document decision + future upgrade path
- [ ] (User) Test with test.pypi.org first (optional but recommended)
- [ ] (User) Uncomment the `publish:` job when ready
- [ ] (User) Cut first real release via the new flow
- [ ] (Future) Migrate to Trusted Publishing when convenient

## What this task does

Currently `.github/workflows/build-wheels.yml` builds and tests wheels
for Linux, macOS, and Windows on every release / workflow_dispatch —
but doesn't publish them. The `publish:` job exists in the file, fully
drafted, but is commented out.

Task 6.3 enables automated publishing to PyPI via **Trusted Publishing**
(OIDC-based authentication), so creating a GitHub release triggers the
full pipeline: build → test → publish to PyPI, with no human in the
loop after release creation.

## What "Trusted Publishing" is (vs the legacy API-token flow)

The traditional PyPI publish flow uses a long-lived **API token** stored
as a GitHub secret. That has several problems:

- Tokens leak (logs, accidentally committed, phishing)
- No rotation enforcement — a single leaked token gives indefinite
  upload rights
- No audit trail — PyPI can't tell which GitHub workflow used the token
- If the token owner leaves the project, rotating it breaks CI

**Trusted Publishing** replaces the token with **OpenID Connect (OIDC)**:

- **No secrets in GitHub** — no API token, no rotation concerns, no leaks
- **Scoped to specific workflows** — PyPI will only accept uploads from
  the exact `<owner>/<repo>/.github/workflows/<file>.yml` you configure.
  An attacker with write access to a different workflow can't publish.
- **Audit trail** — every publish is logged on PyPI with the triggering
  commit SHA, workflow run URL, and OIDC claim details.
- **Works without touching project settings every release** — set it up
  once, forever.

Reference: <https://docs.pypi.org/trusted-publishers/>.

## How it works end-to-end

### 1. One-time setup on pypi.org (web UI, user action)

You do this once, from a browser logged into your PyPI account with
2FA:

1. Sign in to <https://pypi.org> with the account that owns
   `pyramids-gis`.
2. Go to the project page → **Manage** → **Publishing**.
3. Click **Add a new publisher** → choose **GitHub**.
4. Fill in:
   - **Owner:** `Serapieum-of-alex`
   - **Repository name:** `pyramids`
   - **Workflow name:** `build-wheels.yml`
   - **Environment name:** (leave empty or set to `release` — see §4)
5. Save.

PyPI records this as a "trusted publisher" and will accept uploads from
GitHub workflows that match the exact owner/repo/workflow combination.

### 2. Workflow side (code change)

In `.github/workflows/build-wheels.yml`, uncomment the `publish:` job
and update its `needs:` list to reference all current build jobs.

The key bits:

```yaml
publish:
  needs: [build-sdist, build-linux-wheels, build-macos-wheels, build-windows-wheels, test-wheels]
  if: >-
    github.event_name == 'release' ||
    (github.event_name == 'workflow_dispatch' && inputs.upload-to-pypi)
  runs-on: ubuntu-latest
  permissions:
    # Required for Trusted Publishing — lets the job mint an OIDC token
    # that identifies the workflow run to PyPI.
    id-token: write
  steps:
    - name: Download all artifacts
      uses: actions/download-artifact@v4
      with:
        pattern: "*"           # sdist + wheels-linux-* + wheels-macos-* + wheels-windows-*
        merge-multiple: true
        path: dist/
    - name: Publish to PyPI
      # This action auto-negotiates the OIDC token via id-token: write.
      uses: pypa/gh-action-pypi-publish@release/v1
```

### 3. On a GitHub release (automatic)

After (1) and (2) are in place:

1. Developer: `pixi run -e dev cz-bump` → pushes a version tag
2. Developer: `gh release create <version> --generate-notes`
3. GitHub Actions triggers `build-wheels.yml` on the `release` event
4. cibuildwheel builds sdist + 9 wheels (3 OSes × 3 Python versions)
5. `test-wheels` job runs pytest in clean envs for all 9 wheels
6. If all green, `publish:` job runs:
   - Downloads all artifacts (sdist + 9 wheels)
   - Requests an OIDC token from GitHub Actions
   - `pypa/gh-action-pypi-publish` uploads to PyPI with the OIDC token
   - PyPI verifies token matches the configured Trusted Publisher
   - Upload proceeds — no password, no API token
7. Within ~1–2 minutes, `pip install pyramids-gis==<version>` works

## Why this is gated on user approval

Three reasons we haven't flipped the switch:

### (a) PyPI web UI config is user-only

It requires interactive login with 2FA to the PyPI account that owns
`pyramids-gis`. Claude can't do this.

### (b) First Trusted Publisher use is a "pending publisher"

PyPI treats the first-ever trusted-publisher upload as a "pending
publisher"— it requires the first upload to succeed before the trust
relationship is confirmed. That upload attaches a real version to PyPI
permanently. It's fine to use it, but not a reversible config step.

### (c) Standing instruction: never publish without approval

User instruction from earlier: "do not publish anything in github or
pypi without my approve". Uncommenting the `publish:` job makes future
releases auto-publish. That's a de-facto authorization, needs explicit
go-ahead.

## Recommended rollout plan

Order of operations when the user is ready:

### Step 1 — Test.PyPI dry-run (safe, reversible)

Before configuring production PyPI:

1. Create a Trusted Publisher on **test.pypi.org** (separate account,
   same flow as real PyPI).
2. Add a test-publish job to `build-wheels.yml`:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
3. Trigger via `gh workflow run build-wheels.yml -f upload-to-pypi=true`.
4. Verify install from test.pypi works:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       pyramids-gis
   ```

### Step 2 — Add GitHub Environment protection (optional, recommended)

Create a GitHub environment called `release` in the repo settings:

- Settings → Environments → New environment → `release`
- Add a **required reviewer** (e.g., a maintainer)
- Check **prevent self-review**

Then reference it in the publish job:

```yaml
publish:
  environment:
    name: release
    url: https://pypi.org/project/pyramids-gis/
```

Effect: every PyPI publish requires a human click-through in the
GitHub UI. Belt-and-suspenders even if the OIDC config is correct.

### Step 3 — Configure real PyPI Trusted Publisher

Do the pypi.org web UI config from §1 above.

### Step 4 — Enable the publish job

Uncomment the `publish:` job in `build-wheels.yml` with the correct
`needs:` list (all 4 build jobs + test-wheels).

### Step 5 — Dry-release for validation

Cut a pre-release tag like `v0.13.0rc1`, push, and watch the workflow.
Verify:
- All wheels build (Linux + macOS + Windows, cp311/312/313)
- All tests pass
- Artifacts download into the publish job
- OIDC exchange succeeds
- Upload to PyPI completes
- `pip install pyramids-gis==0.13.0rc1` works

If step 5 is green, future real releases are just `cz bump` + `gh release
create` and they ship automatically.

## What Claude can do without user approval (preparatory work)

- Update the **drafted** `publish:` job in `build-wheels.yml` with
  current job names and keep it commented out
- Add a header comment explaining how to enable it
- Add a GitHub Environment reference (commented out)
- Add a test.pypi.org variant (commented out)
- Create a separate draft `publish-test.yml` workflow for the dry-run step

None of these actually publish anything — they just prepare the wiring
so the "enable it" step is a minimal diff reviewed by the user.

## What Claude absolutely cannot do

- Log into pypi.org and configure a Trusted Publisher
- Uncomment the `publish:` job without explicit user go-ahead
- Push the branch with the enabled publish job to `main`
- Trigger `gh workflow run build-wheels.yml -f upload-to-pypi=true`

These all require the user's explicit approval per the standing
"do not publish without approve" instruction.

## Estimated effort

- **PyPI web UI config** (user): 5 minutes
- **Workflow changes** (Claude, preparatory): 30 minutes
- **Test.PyPI dry run** (Claude + user verification): 15 minutes
- **Flip to production** (user approval + Claude edit): 10 minutes
- **First real release via the new pipeline**: same as today's release,
  just triggers automatically

Total: **~1 hour of real work** spread across the two actors.

## Outcome when complete

- `pip install pyramids-gis` works on Linux, macOS, Windows without any
  pre-installed GDAL
- Each `gh release create` ships all 9 wheels + sdist automatically
- No API tokens to manage or rotate
- Audit trail of every publish on both GitHub Actions and PyPI
- Conda-forge feedstock continues to auto-detect new PyPI sdists

## Deliverables checklist

- [ ] (User) Trusted Publisher configured on pypi.org
- [ ] (User) Trusted Publisher configured on test.pypi.org (optional)
- [ ] (User) GitHub Environment `release` with required reviewer (optional)
- [ ] (Claude) Uncommented + updated `publish:` job in `build-wheels.yml`
- [ ] (Claude) Wheel-size CI gate (done in 6.1)
- [ ] (User + Claude) First dry-release cut (`v0.13.0rc1` or similar)
- [ ] (User + Claude) First production release via auto-publish flow
