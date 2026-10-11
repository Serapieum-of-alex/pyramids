"""The `pyramids.stac` package surface matches the tour its docstring gives.

`pyramids.stac.__doc__` is the structured tour of the subpackage — "Search and
discovery", "Reading assets", "Item metadata" — and is what a reader meets
first, in the API docs and in `help(pyramids.stac)`. A name exported without a
line there is undiscoverable except by reading `__all__`, which is how sixteen
of them accumulated at once.
"""

from __future__ import annotations

import builtins
import importlib
import inspect

import pytest

import pyramids.stac as stac_package
from pyramids.stac import _windowed

pytestmark = pytest.mark.core

_CROSS_REFERENCE_MARKERS = (":func:`", ":class:`", ":meth:`", ":data:`", ":attr:`")
# A field whose default is None is present but falsy, so absence needs its own marker.
_MISSING = object()


class TestExportedNamesAreDocumented:
    """Every exported name appears in the module docstring, and vice versa.

    Test scenario:
        `__all__` and the docstring are maintained by hand in the same file, so
        they drift in both directions: a new export that nobody wrote a line
        for, and a line left behind by a name that was renamed or removed.
    """

    def test_every_exported_name_is_in_the_docstring(self):
        """No public name is missing from the tour."""
        doc = stac_package.__doc__ or ""
        missing = sorted(name for name in stac_package.__all__ if name not in doc)
        assert missing == [], (
            f"{len(missing)} exported name(s) are absent from the "
            f"pyramids.stac module docstring: {missing}"
        )

    def test_every_exported_name_is_importable(self):
        """`__all__` names nothing the package does not actually expose."""
        absent = sorted(
            name for name in stac_package.__all__ if not hasattr(stac_package, name)
        )
        assert absent == [], f"__all__ names unexported attribute(s): {absent}"

    def test_the_docstring_references_no_unexported_name(self):
        """A `:func:`/`:class:` reference in the tour resolves to an export.

        Test scenario:
            A renamed export leaves a stale cross-reference behind, which
            renders as a broken link in the docs rather than failing anywhere.
        """
        doc = stac_package.__doc__ or ""
        referenced = {
            fragment.split("`", 1)[0]
            for marker in (":func:`", ":class:`", ":meth:`")
            for fragment in doc.split(marker)[1:]
        }
        local = {name for name in referenced if "." not in name}
        unexported = sorted(local - set(stac_package.__all__))
        assert unexported == [], (
            f"the docstring cross-references name(s) the package does not "
            f"export: {unexported}"
        )


class TestTheWindowedSurfaceIsReachable:
    """`_windowed` declares a public surface; the package must carry all of it.

    Test scenario:
        `pyramids.stac._windowed` is an all-public module — four reads, the
        default window CRS and the envelope helper — so its `__all__` *is* the
        item-level windowed API. `STAC_WINDOW_CRS` and `geometry_bounds` were
        left out of the package's re-export while the exported reads kept
        cross-referencing them from their own docstrings, so a reader following
        `read_item_part`'s `bbox_crs` text had to import a private module to get
        at the name it pointed to (L6). Pinning the whole `__all__` rather than
        the two names catches the next read helper added with a public constant
        of its own.
    """

    def test_every_windowed_public_name_is_re_exported(self):
        """Nothing in `_windowed.__all__` is reachable only from the private module."""
        missing = sorted(set(_windowed.__all__) - set(stac_package.__all__))
        assert missing == [], (
            f"pyramids.stac does not re-export {missing} from _windowed, so the "
            "name is reachable only through the private module"
        )

    def test_the_re_exports_are_the_same_objects(self):
        """The package name binds the module's object, not a copy of it."""
        mismatched = [
            name
            for name in _windowed.__all__
            if getattr(stac_package, name) is not getattr(_windowed, name)
        ]
        assert mismatched == [], f"re-exported name(s) rebound elsewhere: {mismatched}"

    def test_the_windowed_reads_reference_only_reachable_names(self):
        """Each exported read's docstring points at names the package exports.

        Test scenario:
            This is the defect itself, stated as a rule: `read_item_part`'s
            `bbox_crs` documents ``:data:`STAC_WINDOW_CRS``` and
            `read_item_feature` documents ``:func:`geometry_bounds```, so those
            references were dangling for anyone importing from the package.
        """
        unreachable: dict[str, list[str]] = {}
        for name in _windowed.__all__:
            doc = getattr(getattr(_windowed, name), "__doc__", None) or ""
            referenced = {
                fragment.split("`", 1)[0].lstrip("~")
                for marker in _CROSS_REFERENCE_MARKERS
                for fragment in doc.split(marker)[1:]
            }
            local = {ref for ref in referenced if "." not in ref}
            gap = sorted(local - set(stac_package.__all__))
            if gap:
                unreachable[name] = gap
        assert unreachable == {}, (
            f"windowed docstring(s) cross-reference unexported name(s): {unreachable}"
        )


class TestPublicDocstringsReferenceReachableNames:
    """Every exported symbol's docstring points only at names the package exports.

    Test scenario:
        `TestTheWindowedSurfaceIsReachable` states this rule for `_windowed`
        alone, which is why it caught `STAC_WINDOW_CRS` but not the same defect
        in `_item`: `asset_alternate_href` cross-referenced `:func:`asset_field``
        and `preferred_asset_href`'s See Also pointed at
        `:func:`preferred_asset_source``, neither of which the package exported.
        Stating the rule over the whole surface — every exported name and every
        public method of an exported class — closes the class rather than the
        two instances, so the next helper referenced out of a private module
        fails here instead of rendering as a dead link.
    """

    @staticmethod
    def _bare_references(doc: str) -> set[str]:
        """Return the unqualified names a docstring cross-references."""
        referenced = {
            fragment.split("`", 1)[0].lstrip("~")
            for marker in _CROSS_REFERENCE_MARKERS
            for fragment in doc.split(marker)[1:]
        }
        return {
            ref for ref in referenced if "." not in ref and not hasattr(builtins, ref)
        }

    def test_no_exported_docstring_references_an_unexported_name(self):
        """A bare `:func:`/`:class:` role in a public docstring resolves to an export."""
        exported = set(stac_package.__all__)
        unreachable: dict[str, list[str]] = {}
        for name in exported:
            obj = getattr(stac_package, name)
            targets = {name: inspect.getdoc(obj) or ""}
            if inspect.isclass(obj):
                targets.update(
                    {
                        f"{name}.{member_name}": inspect.getdoc(member) or ""
                        for member_name, member in vars(obj).items()
                        if not member_name.startswith("_") and callable(member)
                    }
                )
            for label, doc in targets.items():
                gap = sorted(self._bare_references(doc) - exported)
                if gap:
                    unreachable[label] = gap
        assert unreachable == {}, (
            f"public docstring(s) cross-reference name(s) pyramids.stac does "
            f"not export: {unreachable}"
        )


class TestQualifiedReferencesResolve:
    """Every dotted cross-reference in a public docstring names something real.

    Test scenario:
        A fully-qualified role looks authoritative and is never executed, so a
        wrong one survives review: `read_item_feature` told the reader to
        rasterize a geometry with
        `:meth:`pyramids.feature.FeatureCollection.to_dataset``, a method that
        does not exist on any pyramids class — the burn-in entry point is
        `Dataset.from_features`. Resolution is by import and `getattr`, so the
        check fails on a name that was never run rather than on a style rule.
        A reference into a private module (`pyramids.base._errors.StacError`,
        which nothing public re-exports) is accurate and passes.
    """

    @staticmethod
    def _resolves(ref: str) -> bool:
        """Say whether a dotted reference can be imported and walked."""
        parts = ref.split(".")
        roots = []
        for split in range(len(parts) - 1, 0, -1):
            try:
                roots.append(
                    (importlib.import_module(".".join(parts[:split])), parts[split:])
                )
            except ImportError:
                continue
            break
        if parts[0] in stac_package.__all__:
            roots.append((getattr(stac_package, parts[0]), parts[1:]))
        found = False
        for root, attrs in roots:
            target = root
            for attr in attrs:
                target = getattr(target, attr, _MISSING)
                if target is _MISSING:
                    break
            if target is not _MISSING:
                found = True
                break
        return found

    def test_every_dotted_reference_in_a_public_docstring_resolves(self):
        """No public docstring cites a module, class or member that is not there."""
        dangling: dict[str, list[str]] = {}
        for name in stac_package.__all__:
            obj = getattr(stac_package, name)
            targets = {name: inspect.getdoc(obj) or ""}
            if inspect.isclass(obj):
                targets.update(
                    {
                        f"{name}.{member_name}": inspect.getdoc(member) or ""
                        for member_name, member in vars(obj).items()
                        if not member_name.startswith("_") and callable(member)
                    }
                )
            for label, doc in targets.items():
                referenced = {
                    fragment.split("`", 1)[0].lstrip("~")
                    for marker in _CROSS_REFERENCE_MARKERS
                    for fragment in doc.split(marker)[1:]
                }
                gap = sorted(
                    ref for ref in referenced if "." in ref and not self._resolves(ref)
                )
                if gap:
                    dangling[label] = gap
        assert dangling == {}, (
            f"public docstring(s) cite name(s) that do not resolve: {dangling}"
        )
