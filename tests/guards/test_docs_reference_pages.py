"""Guard meta-test: a reference page renders the methods the class actually offers.

`UgridDataset` is assembled from a mixin, so *every* one of its ~20 spatial, morphology,
interop and regrid methods is an inherited member rather than one defined in the class
body. mkdocstrings renders only a class's own members unless the page asks for the
inherited ones, so a page that leaves `inherited_members` unset documents a `UgridDataset`
with none of those methods on it — and nothing in a behaviour test notices, because the
methods work perfectly; only the rendered page is empty.

This is a static check on the page's mkdocstrings options plus the live class, so it needs
no docs build and no optional dependency.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_REPO_ROOT = Path(__file__).parents[2]
_UGRID_DATASET_PAGE = _REPO_ROOT / "docs" / "reference" / "netcdf" / "ugrid" / "dataset.md"


def _inherited_public_methods(cls: type) -> list[str]:
    """Public methods `cls` offers but does not define in its own body."""
    return sorted(
        name
        for name, _ in inspect.getmembers(cls, callable)
        if not name.startswith("_") and name not in vars(cls)
    )


def _mkdocstrings_options(page: Path, identifier: str) -> dict:
    """The options block of the `::: <identifier>` directive on `page`.

    Args:
        page: The markdown page to read.
        identifier: The dotted path the directive documents.

    Returns:
        dict: The parsed `options:` mapping, or `{}` when the directive carries none.
    """
    lines = page.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"::: {identifier}")
    block = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith((" ", "\t")):
            break
        block.append(line)
    parsed = yaml.safe_load("\n".join(block)) or {}
    return parsed.get("options") or {}


class TestUgridDatasetReferencePage:
    """The `UgridDataset` reference page against the class it documents."""

    def test_every_public_method_is_inherited(self):
        """The premise: the class body defines none of the methods users call.

        Test scenario:
            Asserting the fix without this is brittle — if the facades were ever moved onto
            `UgridDataset` itself the `inherited_members` requirement below would stop being
            about anything. Both assertions together say *why* the option is needed.
        """
        inherited = _inherited_public_methods(UgridDataset)

        assert len(inherited) >= 20, (
            f"expected the facade mixin to supply at least 20 public methods, got "
            f"{len(inherited)}: {inherited}"
        )
        assert "regrid" in inherited, "regrid should come from the facade mixin"
        assert "to_xarray" in inherited, "to_xarray should come from the facade mixin"

    def test_page_renders_inherited_members(self):
        """The page asks mkdocstrings for the inherited members.

        Test scenario:
            With `inherited_members` unset (mkdocstrings' default is False) the rendered
            page carried the class docstring and nothing else — no `regrid`, no `to_xarray`,
            no `polygonize`. The option is what puts the mixin's methods on the page.
        """
        options = _mkdocstrings_options(_UGRID_DATASET_PAGE, "pyramids.netcdf.ugrid.UgridDataset")
        inherited_members = options.get("inherited_members")

        assert inherited_members, (
            "docs/reference/netcdf/ugrid/dataset.md must set `inherited_members` -- every "
            "public UgridDataset method comes from the _UgridFacades mixin, so the page "
            f"renders none of them without it. Current options: {sorted(options)}"
        )
