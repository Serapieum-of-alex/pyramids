"""A reusable conformance kit for ``pyramids.raster_drivers`` plugins.

A downstream distribution that registers a raster driver runs these checks against its driver
*class* in its own test suite — under any test runner, since the kit raises plain
:class:`AssertionError` and needs no pytest. It verifies the same contract
:mod:`pyramids.dataset._drivers` enforces at discovery (so a driver that conforms here is never
silently skipped at runtime), plus the behavioural promises that static validation cannot see:
:meth:`~pyramids.dataset._drivers.RasterDriver.can_open` is callable on the *class* (never needing
an instance), returns a ``bool``, and never raises on a hostile input.

Usage in a plugin's tests::

    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms
    from my_plugin import FciDriver

    def test_driver_conforms():
        assert_raster_driver_conforms(FciDriver)
"""

from __future__ import annotations

from typing import Any

from pyramids.dataset._drivers import _validate

__all__ = ["check_raster_driver", "assert_raster_driver_conforms"]

#: Hostile inputs ``can_open`` must survive without raising — an empty string, an unknown path, a
#: non-string, and ``None`` all have to answer ``False`` (or at least a ``bool``), never raise.
_PROBES: tuple[Any, ...] = ("", "no-such-source-xyz", 123, None)


def check_raster_driver(driver: Any) -> list[str]:
    """Return every way a raster driver class departs from the contract.

    Args:
        driver: The driver class to check (the class itself, not an instance).

    Returns:
        A list of human-readable problems; empty when the driver conforms. The first entry, if
        any, is the static-claims problem that discovery itself would reject on.

    Examples:
        - A well-formed driver class has no problems:
            ```python
            >>> from pyramids.dataset.driver_conformance import check_raster_driver
            >>> class FciDriver:
            ...     name, long_name, api_version = "fci", "EUMETSAT FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".nat")
            ...     def read(self, source, **kw): ...
            >>> check_raster_driver(FciDriver)
            []

            ```
        - A missing capability is reported:
            ```python
            >>> from pyramids.dataset.driver_conformance import check_raster_driver
            >>> class Bad:
            ...     name, long_name, api_version = "bad", "Bad", 1
            ...     suffixes, schemes = (".x",), ()
            ...     capabilities, extra = frozenset(), None
            ...     @classmethod
            ...     def can_open(cls, source): return False
            ...     def read(self, source, **kw): ...
            >>> check_raster_driver(Bad)[0]
            "does not declare the required 'read' capability"

            ```
    """
    problems: list[str] = []
    static_problem = _validate(driver)
    if static_problem is not None:
        problems.append(static_problem)

    can_open = getattr(driver, "can_open", None)
    if callable(can_open):
        problems.extend(_check_can_open(can_open))
    return problems


def _check_can_open(can_open: Any) -> list[str]:
    """Probe ``can_open`` with hostile inputs; it must return a ``bool`` and never raise.

    Args:
        can_open: The driver class's ``can_open`` classmethod.

    Returns:
        A list with at most one problem (the first probe that misbehaves).
    """
    problems: list[str] = []
    for probe in _PROBES:
        try:
            result = can_open(probe)
        except Exception as error:  # noqa: BLE001 - the whole point is that it must not raise
            problems.append(
                f"can_open({probe!r}) raised {error!r}; can_open must never raise, it must "
                f"answer False for anything it does not recognise"
            )
            break
        if not isinstance(result, bool):
            problems.append(
                f"can_open({probe!r}) returned {result!r}; can_open must return a bool"
            )
            break
    return problems


def assert_raster_driver_conforms(driver: Any) -> None:
    """Raise :class:`AssertionError` when a raster driver class violates the contract.

    Args:
        driver: The driver class to check.

    Raises:
        AssertionError: The driver departs from the contract; the message lists every problem.

    Examples:
        - A conforming driver passes silently:
            ```python
            >>> from pyramids.dataset.driver_conformance import assert_raster_driver_conforms
            >>> class FciDriver:
            ...     name, long_name, api_version = "fci", "EUMETSAT FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".nat")
            ...     def read(self, source, **kw): ...
            >>> assert_raster_driver_conforms(FciDriver)

            ```
    """
    problems = check_raster_driver(driver)
    if problems:
        raise AssertionError(
            "raster driver does not conform to pyramids.raster_drivers:\n- "
            + "\n- ".join(problems)
        )
