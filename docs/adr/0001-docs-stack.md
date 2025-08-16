# ADR-0001: Documentation stack

- Status: Accepted
- Date: 2025-08-16

## Context
We need a robust, contributor-friendly documentation stack that supports API reference, diagrams, tutorials, and versioned publishing.

## Decision
- Use MkDocs with the Material theme as the primary docs framework.
- Use mkdocstrings[python] for auto-generating API reference from docstrings.
- Prefer Mermaid fenced blocks for diagrams; allow PlantUML if complexity requires it.
- Use mike for versioned documentation on GitHub Pages.

## Consequences
- Contributors document code with Google-style docstrings.
- Diagrams render client-side in the site; heavy diagrams can be split or pre-rendered if needed.
- CI builds the docs on pushes and releases, publishing with mike.
