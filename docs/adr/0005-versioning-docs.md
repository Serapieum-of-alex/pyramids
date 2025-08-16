# ADR-0005: Versioned documentation strategy

- Status: Accepted
- Date: 2025-08-16

## Context
We want readers to access documentation matching their installed version and preview changes on PRs.

## Decision
- Use `mike` to publish versioned docs to GitHub Pages.
- Track `main` as default; publish releases under their tag and alias `latest`.
- Optionally publish PR previews under a `develop` alias.

## Consequences
- Contributors can update docs alongside code and see previews.
- Users can browse historical docs aligned with releases.
