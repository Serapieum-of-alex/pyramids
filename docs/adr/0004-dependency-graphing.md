# ADR-0004: Dependency graphing approach

- Status: Accepted
- Date: 2025-08-16

## Context
We want visibility into internal module dependencies, cycles, and layering to guide refactoring and documentation.

## Decision
- Represent a high-level dependency graph using Mermaid in the docs (modules only).
- Provide a simple, deterministic diagram generation step that exports Mermaid sources to artifacts for CI publishing.
- Optionally evaluate pydeps/pyan/grimp later for automated graphs; keep manual graph for now to avoid heavy dependencies.

## Consequences
- Readers have a clear mental model of package layering.
- Automated cycle detection can be added in future CI without blocking current workflows.
