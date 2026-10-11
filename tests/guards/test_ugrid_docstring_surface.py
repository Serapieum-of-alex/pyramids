"""Guard meta-test: a `UgridDataset` method documents the parameters it actually takes.

A parameter added to a signature without a matching `Args:` entry is invisible: the method
works, every behaviour test passes, and the rendered reference page simply never mentions
the knob. This branch added `groupby` to `reduce`, `dim` to the fill family and `fixed` to
both morphology facades, and each arrived undocumented.

Static reflection over the live class, so it needs no docs build and no optional dependency.
"""

from __future__ import annotations

import inspect
import re

import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_ARG_LINE = re.compile(r"\s*\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:")


def _documented_parameters(doc: str) -> set[str]:
    """The parameter names a Google-style `Args:` section lists.

    Args:
        doc: The de-indented docstring to read.

    Returns:
        set[str]: The names at the `Args:` block's own indentation level; continuation lines
        of a description sit deeper and are ignored.
    """
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return set()
    indent = len(lines[start]) - len(lines[start].lstrip())
    names: set[str] = set()
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        here = len(line) - len(line.lstrip())
        if here <= indent:
            break
        match = _ARG_LINE.match(line)
        if match and here == indent + 4:
            names.add(match.group(1))
    return names


def _declared_parameters(member) -> set[str]:
    """The names a callable takes, minus `self`/`cls` and the `*args`/`**kwargs` catch-alls."""
    signature = inspect.signature(member)
    return {
        name
        for name, spec in signature.parameters.items()
        if name not in ("self", "cls")
        and spec.kind not in (spec.VAR_POSITIONAL, spec.VAR_KEYWORD)
    }


def _public_callables(cls: type) -> list[tuple[str, object]]:
    """Every public callable attribute of `cls`, inherited members included."""
    found = []
    for name, member in inspect.getmembers(cls, callable):
        if name.startswith("_"):
            continue
        try:
            inspect.signature(member)
        except (TypeError, ValueError):
            continue
        found.append((name, member))
    return found


_CALLABLES = _public_callables(UgridDataset)


class TestUgridDatasetDocstringSurface:
    """Every public `UgridDataset` entry point against its own signature."""

    def test_the_sweep_sees_the_whole_surface(self):
        """The sweep below is only as good as the surface it walks.

        Test scenario:
            `UgridDataset` is assembled from a mixin, so a sweep that forgot
            `inspect.getmembers`' inherited members would silently check a fraction of the
            API and pass. Pin a floor and two names that only the mixin supplies.
        """
        names = {name for name, _ in _CALLABLES}

        assert len(names) >= 90, (
            f"expected at least 90 public callables, got {len(names)}"
        )
        assert {"regrid", "binary_dilation"} <= names, (
            "the sweep must reach the mixin's facades, not only the class body"
        )

    @pytest.mark.parametrize("name,member", _CALLABLES, ids=[n for n, _ in _CALLABLES])
    def test_every_parameter_is_documented(self, name, member):
        """A parameter in the signature has an `Args:` entry.

        Args:
            name: The attribute name on `UgridDataset`.
            member: The callable itself.

        Test scenario:
            Six entry points failed this when it was written -- `reduce(groupby=)`,
            `ffill(dim=)`, `bfill(dim=)`, `dropna(dim=)`, `binary_dilation(fixed=)` and
            `binary_erosion(fixed=)` -- each a parameter this branch added to a signature
            whose docstring was never extended to match.
        """
        doc = inspect.getdoc(member) or ""
        missing = sorted(_declared_parameters(member) - _documented_parameters(doc))

        assert not missing, (
            f"UgridDataset.{name} takes {missing} but its docstring's Args section does "
            f"not list them"
        )
