"""Discovery of third-party mesh file-format drivers.

pyramids reads and writes UGRID NetCDF itself. Every other unstructured-mesh format — 2DM,
Selafin, DFSU, XMDF, PLY and the rest — is read by a separate distribution that registers a
driver into the ``pyramids.mesh_drivers`` entry-point group. This module is the reading side of
that contract: it resolves the group, validates what it finds, and hands the drivers to
:meth:`UgridDataset.read_file`, :meth:`UgridDataset.attach_data` and
:meth:`UgridDataset.to_file`.

The dependency runs one way. A plugin imports pyramids' mesh types and builds them; pyramids
never imports the plugin, and knows only the group name. So a bare pyramids install behaves
exactly as it did before this module existed.

Three properties matter enough to state outright, because each is load-bearing:

- **Discovery is lazy and cached.** ``importlib.metadata.entry_points`` walks every installed
  distribution, so it must not run on the UGRID path. Callers reach this module only after a
  non-NetCDF suffix has already been established.
- **Every failure is a warning, never an exception.** A plugin that raises on import, declares a
  future API version or collides on a name is skipped. One broken third-party package cannot stop
  a NetCDF file from opening.
- **Entry points load individually.** Resolving the group reads names and metadata only; a
  driver module is imported when its suffix is wanted. A missing optional dependency therefore
  costs its own driver and nothing else.
"""

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pyramids.base._errors import FileFormatNotSupportedError

if TYPE_CHECKING:  # pragma: no cover - import cycle, types only
    from pyramids.netcdf.ugrid.dataset import UgridDataset
    from pyramids.netcdf.ugrid.models import MeshVariable
    from pyramids.netcdf.ugrid.network import Network1d

__all__ = [
    "CAPABILITIES",
    "RESERVED_NAMES",
    "ENTRY_POINT_GROUP",
    "MESH_DRIVER_API_VERSION",
    "MeshDriver",
    "UnsupportedMeshFormatError",
    "clear_driver_cache",
    "drivers_for_suffix",
    "mesh_drivers",
]

#: The contract version a driver must declare. Bumped only on a breaking change to
#: :class:`MeshDriver`'s members or to the constructors a driver builds. A driver declaring
#: anything else is skipped with a warning, so an old plugin never breaks a pyramids upgrade.
MESH_DRIVER_API_VERSION = 1

#: The entry-point group a plugin registers into. Part of the published contract.
ENTRY_POINT_GROUP = "pyramids.mesh_drivers"

#: Names a plugin may not claim. `ugrid` identifies pyramids' own reader in `info()`
#: listings and in `driver=`, so a plugin taking it would make both ambiguous.
RESERVED_NAMES = frozenset({"ugrid", "netcdf"})

#: Every capability token a driver may declare. A driver naming anything else is skipped: an
#: unknown token means the plugin was written against a different contract than it claims.
CAPABILITIES = frozenset(
    {
        "read_mesh",
        "read_data",
        "needs_mesh",
        "write_mesh",
        "write_data",
        "lazy_data",
        "multi_mesh",
    }
)


class UnsupportedMeshFormatError(FileFormatNotSupportedError):
    """No installed driver reads this file.

    Subclasses the package's existing :class:`~pyramids.base._errors.FileFormatNotSupportedError`
    so ``except FileFormatNotSupportedError`` keeps catching an unreadable mesh, as it did when
    GDAL was the only reader.
    """


@runtime_checkable
class MeshDriver(Protocol):
    """A third-party reader/writer for one mesh file format.

    A driver is a class, not a function, because :meth:`can_open` matters: several of these
    formats share ``.dat``, ``.mesh`` and ``.out``, so a suffix alone cannot identify one. The
    registry instantiates the class with no arguments.

    A driver declares **exactly one** of ``read_mesh`` or ``needs_mesh``. ``read_mesh`` means the
    file carries a topology and :meth:`read` builds it. ``needs_mesh`` means the file carries
    values only — an SMS ``.dat``, an XMDF or an XDMF — and :meth:`read_data` attaches them to a
    mesh the caller already has.

    Attributes:
        name: Short unique identifier, and the value accepted by ``driver=``.
        long_name: Human-readable format name, shown by :meth:`UgridDataset.info`.
        suffixes: Lowercase, dot-prefixed file suffixes this driver claims.
        api_version: Must equal :data:`MESH_DRIVER_API_VERSION`.
        capabilities: A subset of :data:`CAPABILITIES`.
    """

    name: str
    long_name: str
    suffixes: tuple[str, ...]
    api_version: int
    capabilities: frozenset[str]

    def can_open(self, path: str) -> bool:
        """Report whether this driver recognises the file.

        Sniff the content; never trust the suffix alone. Must not raise for any input — an empty
        file, a directory, or a file of another format all answer ``False`` — and must not read
        the whole file.

        Args:
            path: The file to inspect.

        Returns:
            True when :meth:`read` or :meth:`read_data` can handle this file.
        """
        ...  # pragma: no cover - protocol

    def read(
        self, path: str, *, mesh_name: str | None = None, lazy: bool = True
    ) -> UgridDataset | Network1d:
        """Read a file that carries its own topology.

        Raises for a ``needs_mesh`` format, which has no topology to build a mesh from and is
        reached through :meth:`read_data` instead.

        Args:
            path: The file to read.
            mesh_name: Which topology to read, for a multi-mesh file. ``None`` takes the first.
            lazy: Defer reading variable values until first access.

        Returns:
            A dataset, or a :class:`Network1d` for a file whose topology dimension is 1.
        """
        ...  # pragma: no cover - protocol

    def read_data(
        self, path: str, dataset: UgridDataset, *, lazy: bool = True
    ) -> dict[str, MeshVariable]:
        """Read a results-only file's variables, for a mesh supplied by the caller.

        Required only when ``needs_mesh`` is in :attr:`capabilities`.

        Return **only the variables read from the file**; :meth:`UgridDataset.attach_data`
        merges them onto the mesh. That division is what makes a name the dataset already
        defines an unambiguous collision, and it means a driver cannot drop the dataset's own
        variables by rebuilding the mapping.

        ``dataset`` is passed in so the driver can check the file's declared element count
        against it. That check is the only safeguard there is: nothing in a bare results file
        ties it to the right mesh, so an unchecked attach yields a plausible, wrong result. A
        mismatch is reported by raising :class:`ValueError`.

        Args:
            path: The results file to read.
            dataset: The mesh the values will attach to, for the element-count check.
            lazy: Defer reading variable values until first access.

        Returns:
            The variables read from the file, keyed by name.
        """
        ...  # pragma: no cover - protocol

    def mesh_names(self, path: str) -> list[str]:
        """List the topology names in a file.

        Args:
            path: The file to inspect.

        Returns:
            One name per topology; a single-mesh format returns a one-item list.
        """
        ...  # pragma: no cover - protocol

    def write(self, dataset: UgridDataset, path: str, **options: Any) -> None:
        """Write a dataset in this driver's format.

        Required only when ``write_mesh`` is in :attr:`capabilities`.

        Args:
            dataset: The dataset to write.
            path: The destination file.
            **options: Format-specific settings, passed through by pyramids uninterpreted.
        """
        ...  # pragma: no cover - protocol


#: Resolved drivers, keyed by name. ``None`` until the first miss on the NetCDF path forces
#: discovery; :func:`clear_driver_cache` resets it for tests.
_CACHE: dict[str, MeshDriver] | None = None


def _validate(driver: Any) -> str | None:
    """Check a driver against the contract.

    Args:
        driver: The instantiated driver to check.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    problem: str | None = None
    required = ("name", "long_name", "suffixes", "api_version", "capabilities")
    missing = [attribute for attribute in required if not hasattr(driver, attribute)]
    name = getattr(driver, "name", None)
    raw_capabilities = getattr(driver, "capabilities", None)
    raw_suffixes = getattr(driver, "suffixes", None)

    # Every attribute below belongs to a third party, so each is type-checked before it is
    # used. `_validate` runs inside `_instantiate`'s guarded block as well, but a message
    # naming the offending attribute is far more useful to a plugin author than the
    # `AttributeError` that coercing it would produce.
    if missing:
        problem = f"missing attribute(s) {missing}"
    elif not isinstance(name, str) or not name.strip():
        problem = f"declares name {name!r}; a driver needs a non-blank string name"
    elif name.strip().lower() in RESERVED_NAMES:
        problem = (
            f"declares the reserved name {name!r}; {sorted(RESERVED_NAMES)} identify "
            f"pyramids' built-in reader in listings and in `driver=`"
        )
    elif not isinstance(getattr(driver, "long_name", None), str):
        problem = "declares a non-string long_name"
    elif getattr(driver, "api_version", None) != MESH_DRIVER_API_VERSION:
        problem = (
            f"declares api_version {getattr(driver, 'api_version', None)!r}, but this pyramids "
            f"speaks version {MESH_DRIVER_API_VERSION}"
        )
    elif not isinstance(raw_capabilities, (set, frozenset, tuple, list)):
        problem = (
            f"declares capabilities {raw_capabilities!r}; expected a set of tokens from "
            f"{sorted(CAPABILITIES)}"
        )
    elif not isinstance(raw_suffixes, (tuple, list)):
        problem = (
            f"declares suffixes {raw_suffixes!r}; expected a tuple of lowercase, "
            f"dot-prefixed strings, e.g. ('.slf',)"
        )
    else:
        problem = _validate_tokens(driver, set(raw_capabilities), tuple(raw_suffixes))
    return problem


def _validate_tokens(driver: Any, capabilities: set, suffixes: tuple) -> str | None:
    """Check the capability and suffix values, and the methods they promise.

    Split from :func:`_validate` so each function stays within the project's complexity
    budget; it runs only once the two collections are known to be the right shape.

    Args:
        driver: The instantiated driver to check.
        capabilities: Its capability tokens, already known to be a collection.
        suffixes: Its suffixes, already known to be a sequence.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    problem: str | None = None
    unknown = sorted(token for token in capabilities if token not in CAPABILITIES)
    owns_topology = "read_mesh" in capabilities
    needs_mesh = "needs_mesh" in capabilities

    if unknown:
        problem = f"declares unknown capability token(s) {unknown}"
    elif owns_topology and needs_mesh:
        problem = (
            "declares both 'read_mesh' and 'needs_mesh'; a driver declares exactly one"
        )
    elif not owns_topology and not needs_mesh:
        problem = "declares neither 'read_mesh' nor 'needs_mesh'; a driver declares exactly one"
    elif not suffixes:
        problem = "declares no suffixes"
    else:
        malformed = [
            suffix
            for suffix in suffixes
            if not isinstance(suffix, str)
            or not suffix.startswith(".")
            or suffix != suffix.lower()
        ]
        if malformed:
            problem = (
                f"declares malformed suffix(es) {malformed}; each must be a lowercase, "
                f"dot-prefixed string, e.g. '.slf'"
            )
        else:
            problem = _validate_methods(driver, owns_topology, needs_mesh, capabilities)
    return problem


def _validate_methods(
    driver: Any, owns_topology: bool, needs_mesh: bool, capabilities: set
) -> str | None:
    """Check that a driver implements the methods its capabilities promise.

    Catching this at registration rather than at first use matters for the error the user
    sees: a missing method surfacing later is reported by the dispatch wrapper as a *format*
    failure, which sends them to inspect their file instead of the plugin.

    Args:
        driver: The instantiated driver to check.
        owns_topology: Whether it declares ``read_mesh``.
        needs_mesh: Whether it declares ``needs_mesh``.
        capabilities: Its capability tokens.

    Returns:
        A message naming the first missing method, or ``None`` when all are present.
    """
    expected = ["can_open"]
    if owns_topology:
        expected.append("read")
    if needs_mesh:
        expected.append("read_data")
    if "write_mesh" in capabilities:
        expected.append("write")

    absent = [
        method for method in expected if not callable(getattr(driver, method, None))
    ]
    problem: str | None = None
    if absent:
        problem = (
            f"does not implement {absent}, which its capabilities "
            f"{sorted(capabilities)} promise"
        )
    return problem


def _instantiate(entry_point: EntryPoint) -> MeshDriver | None:
    """Load and construct one driver, or warn and give up.

    A plugin that raises while importing — most often because an optional dependency of its own
    is absent — costs itself and nothing else.

    Args:
        entry_point: The entry point to resolve.

    Returns:
        The constructed driver, or ``None`` when it could not be loaded or is invalid.
    """
    driver: MeshDriver | None = None
    problem: str | None = None
    # Loading AND validation both sit inside the guard. `_validate` reads attributes a
    # plugin author controls and has to coerce some of them, so running it outside meant a
    # `suffixes = (1,)` typo raised `AttributeError` straight out of `read_file` -- breaking
    # every non-NetCDF read, and contradicting this module's "every failure is a warning"
    # guarantee.
    try:
        driver = entry_point.load()()
        problem = _validate(driver)
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"Mesh driver {entry_point.name!r} ({entry_point.value}) could not be loaded and "
            f"was skipped: {error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        driver = None
    else:
        if problem is not None:
            warnings.warn(
                f"Mesh driver {entry_point.name!r} ({entry_point.value}) {problem}; it was "
                f"skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            driver = None
    return driver


def mesh_drivers() -> dict[str, MeshDriver]:
    """Return the installed mesh drivers, discovered once and cached.

    Discovery walks every installed distribution, so it runs on the first call only — and callers
    reach this function only after a non-NetCDF suffix has been established, so a UGRID read never
    pays for it.

    Returns:
        Driver instances keyed by :attr:`MeshDriver.name`, in registration order.

    Examples:
        With no plugin installed the mapping is empty, and every pyramids read works as before:

            >>> from pyramids.netcdf.ugrid._drivers import mesh_drivers
            >>> isinstance(mesh_drivers(), dict)
            True
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    if _CACHE is None:
        resolved: dict[str, MeshDriver] = {}
        for entry_point in entry_points(group=ENTRY_POINT_GROUP):
            driver = _instantiate(entry_point)
            if driver is None:
                continue
            if driver.name in resolved:
                warnings.warn(
                    f"Mesh driver name {driver.name!r} is registered twice; keeping the first "
                    f"and skipping {entry_point.value}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            resolved[driver.name] = driver
        _CACHE = resolved
    return _CACHE


def drivers_for_suffix(suffix: str) -> list[MeshDriver]:
    """Return the drivers claiming a file suffix, in registration order.

    Args:
        suffix: A file suffix, with or without its leading dot, in any case.

    Returns:
        Every driver declaring that suffix. Empty when none does.

    Examples:
        - A suffix no installed driver claims answers an empty list, which is what a bare
          pyramids install returns for every mesh format but UGRID NetCDF:

            ```python
            >>> from pyramids.netcdf.ugrid._drivers import drivers_for_suffix
            >>> drivers_for_suffix(".slf")
            []

            ```
        - The dot is optional and the case is ignored, so a suffix taken straight from a
          user-supplied filename needs no normalising first:

            ```python
            >>> from pyramids.netcdf.ugrid._drivers import drivers_for_suffix
            >>> drivers_for_suffix("SLF") == drivers_for_suffix(".slf")
            True

            ```
    """
    wanted = suffix.lower()
    if wanted and not wanted.startswith("."):
        wanted = f".{wanted}"
    result = [driver for driver in mesh_drivers().values() if wanted in driver.suffixes]
    return result


def clear_driver_cache() -> None:
    """Discard the discovery cache so the next lookup re-reads the entry points.

    For tests. Production code has no reason to call it: the installed set cannot change inside a
    running process.
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    _CACHE = None


def describe_drivers() -> list[str]:
    """Render one line per installed driver, for :meth:`UgridDataset.info`.

    Returns:
        A line per driver — name, long name, suffixes and capabilities — plus a line for the
        built-in UGRID NetCDF reader, which is not a plugin and so is not in the registry.

    Examples:
        - On a bare install the only line is the built-in reader, so the listing is never
          empty and :meth:`UgridDataset.info` always has something to show:

            ```python
            >>> from pyramids.netcdf.ugrid._drivers import describe_drivers
            >>> lines = describe_drivers()
            >>> len(lines)
            1
            >>> lines[0].split()[0]
            'ugrid'

            ```
        - Each line names the suffixes that reach that driver, which is what makes the
          listing useful when a file will not open:

            ```python
            >>> from pyramids.netcdf.ugrid._drivers import describe_drivers
            >>> '.nc' in describe_drivers()[0]
            True

            ```
    """
    lines = [
        "ugrid (built-in)  UGRID NetCDF  .nc .nc4 .cdf  [read_mesh read_data write_mesh]"
    ]
    for driver in mesh_drivers().values():
        suffixes = " ".join(driver.suffixes)
        capabilities = " ".join(sorted(driver.capabilities))
        lines.append(f"{driver.name}  {driver.long_name}  {suffixes}  [{capabilities}]")
    return lines


def unsupported_format_message(path: str, suffix: str) -> str:
    """Build the error text for a file no installed driver reads.

    Args:
        path: The file that could not be read.
        suffix: Its suffix, as detected.

    Returns:
        A message naming the file, what is installed, and how to install more.
    """
    installed = mesh_drivers()
    if installed:
        catalogue = ", ".join(
            f"{driver.name} ({' '.join(driver.suffixes)})"
            for driver in installed.values()
        )
    else:
        catalogue = "none"
    shown = suffix or "(no suffix)"
    result = (
        f"No installed driver reads {shown!r} ({path}).\n"
        f"  Built-in        : ugrid (.nc .nc4 .cdf)\n"
        f"  Installed drivers: {catalogue}\n"
        f"  `pip install pyramids-mesh` adds readers for 2DM, Selafin, DFSU, XMDF, PLY and more."
    )
    return result
