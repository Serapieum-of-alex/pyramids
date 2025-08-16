# ADR-0002: Type checking approach

- Status: Accepted
- Date: 2025-08-16

## Context
The project benefits from static analysis to improve reliability and developer experience, but runtime dependencies (e.g., GDAL) and large legacy surfaces make full coverage challenging.

## Decision
- Adopt gradual typing with standard Python type hints.
- Allow optional external checkers (mypy or pyright) for contributors; not enforced in CI yet.
- Prefer Google-style docstrings with type annotations to improve mkdocstrings output.

## Consequences
- Type hints will be added progressively, focusing on public APIs first.
- We can later enable strictness per-module and add a type coverage report.
