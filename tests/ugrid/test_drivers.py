"""The mesh driver registry and dispatch, exercised with no plugin installed.

Every test here builds its own fake driver. Two fixtures make that possible: `registered`
injects drivers straight into the registry cache, and `fake_entry_points` monkeypatches
`importlib.metadata.entry_points` with in-memory `EntryPoint` objects so real discovery runs
without a distribution on disk.

The negative cases matter more than the positive ones. A plugin that raises on import, declares
a future API version or collides on a name must be skipped with a warning, because the
alternative is a broken third-party package stopping a NetCDF file from opening.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import DriverNotExistError, FileFormatNotSupportedError
from pyramids.netcdf.ugrid import _drivers
from pyramids.netcdf.ugrid._drivers import (
    CAPABILITIES,
    ENTRY_POINT_GROUP,
    MESH_DRIVER_API_VERSION,
    UnsupportedMeshFormatError,
    clear_driver_cache,
    describe_drivers,
    drivers_for_suffix,
    mesh_drivers,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable
from pyramids.netcdf.ugrid.network import Network1d


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the discovery cache around every test.

    Without this the cache leaks between tests and `test_netcdf_read_does_no_discovery` passes
    for the wrong reason -- an already-populated cache is never refilled, so the assertion that
    it stayed empty would hold even if the NetCDF path did consult the registry.
    """
    clear_driver_cache()
    yield
    clear_driver_cache()


def _square_mesh() -> UgridDataset:
    """Build the smallest valid dataset a fake driver can return.

    Returns:
        A one-quad mesh carrying a single face-located variable.
    """
    faces = Connectivity(
        data=np.array([[0, 1, 2, 3]]),
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )
    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        face_node_connectivity=faces,
    )
    variable = MeshVariable(
        name="depth",
        location="face",
        mesh_name="mesh2d",
        shape=(1,),
        _data=np.array([2.5]),
    )
    return UgridDataset(
        mesh=mesh,
        data_variables={"depth": variable},
        global_attributes={"Conventions": "UGRID-1.0"},
    )


class FakeDriver:
    """A minimal driver that reads one made-up format."""

    name = "fake"
    long_name = "Fake Mesh Format"
    suffixes = (".fake",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "write_mesh", "write_data"})

    def __init__(self) -> None:
        """Record what the driver was asked to do, so tests can assert on it."""
        self.read_calls: list[tuple[str, str | None]] = []
        self.write_calls: list[tuple[str, dict]] = []
        self.sniffed: list[str] = []
        self.returns: object | None = None

    def can_open(self, path: str) -> bool:
        """Accept any path this driver is offered.

        Args:
            path: The file to inspect.

        Returns:
            True, always; tests that need a veto use `VetoDriver`.
        """
        self.sniffed.append(path)
        return True

    def read(self, path, *, mesh_name=None, lazy=True):
        """Return the prepared dataset.

        Args:
            path: The file to read.
            mesh_name: Which topology to read.
            lazy: Defer reading values.

        Returns:
            Whatever `returns` was set to, or a one-quad mesh.
        """
        self.read_calls.append((path, mesh_name))
        return _square_mesh() if self.returns is None else self.returns

    def read_data(self, path, dataset, *, lazy=True):
        """Attach nothing; `needs_mesh` drivers have their own fake.

        Args:
            path: The results file.
            dataset: The mesh to attach to.
            lazy: Defer reading values.

        Returns:
            The dataset unchanged.
        """
        return dataset

    def mesh_names(self, path: str) -> list[str]:
        """Report two topologies, so multi-mesh routing is observable.

        Args:
            path: The file to inspect.

        Returns:
            Two names.
        """
        return ["first", "second"]

    def write(self, dataset, path, **options) -> None:
        """Record the write instead of producing a file.

        Args:
            dataset: The dataset to write.
            path: The destination.
            **options: Format-specific settings.
        """
        self.write_calls.append((path, options))


class VetoDriver(FakeDriver):
    """A driver that claims the suffix but refuses the file."""

    name = "veto"
    long_name = "Veto Format"
    suffixes = (".fake",)

    def can_open(self, path: str) -> bool:
        """Refuse every file.

        Args:
            path: The file to inspect.

        Returns:
            False, always.
        """
        self.sniffed.append(path)
        return False


class AttachDriver(FakeDriver):
    """A results-only driver: values, no topology."""

    name = "attach"
    long_name = "Attachable Results"
    suffixes = (".res",)
    capabilities = frozenset({"needs_mesh", "read_data"})

    def __init__(self, declared_elements: int | None = None) -> None:
        """Set up the driver.

        Args:
            declared_elements: The element count the file claims, so a mismatch can be tested.
                ``None`` matches whatever mesh it is given.
        """
        super().__init__()
        self.declared_elements = declared_elements

    def read(self, path, *, mesh_name=None, lazy=True):
        """Refuse: this format carries no topology.

        Args:
            path: The file.
            mesh_name: Unused.
            lazy: Unused.

        Raises:
            UnsupportedMeshFormatError: Always.
        """
        raise UnsupportedMeshFormatError("results-only; use attach_data")

    def read_data(self, path, dataset, *, lazy=True):
        """Return the variables read from the file, after checking the element count.

        Args:
            path: The results file.
            dataset: The mesh the values attach to.
            lazy: Defer reading values.

        Returns:
            The variables read, keyed by name. Only the driver's own — `attach_data` merges.

        Raises:
            ValueError: If the declared element count does not match the mesh.
        """
        declared = (
            dataset.n_face if self.declared_elements is None else self.declared_elements
        )
        if declared != dataset.n_face:
            raise ValueError(
                f"{path} declares {declared} elements; the mesh has {dataset.n_face}."
            )
        return {
            "velocity": MeshVariable(
                name="velocity",
                location="face",
                mesh_name=dataset.mesh_name,
                shape=(dataset.n_face,),
                _data=np.zeros(dataset.n_face),
            )
        }


class FrameOnlyDriver(FakeDriver):
    """A driver that writes a mesh frame but no variables."""

    name = "frame"
    long_name = "Frame Only"
    suffixes = (".frm",)
    capabilities = frozenset({"read_mesh", "write_mesh"})


class ReadOnlyDriver(FakeDriver):
    """A driver that cannot write."""

    name = "readonly"
    long_name = "Read Only"
    suffixes = (".ro",)
    capabilities = frozenset({"read_mesh"})


@pytest.fixture
def registered(monkeypatch):
    """Inject drivers directly into the registry cache.

    Returns:
        A callable taking driver instances and returning them, so a test can keep a handle on
        the instance the registry will hand to pyramids.
    """

    def _register(*drivers):
        monkeypatch.setattr(
            _drivers, "_CACHE", {driver.name: driver for driver in drivers}
        )
        return drivers[0] if len(drivers) == 1 else drivers

    return _register


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Monkeypatch `entry_points` so real discovery runs over in-memory entry points.

    Returns:
        A callable taking ``(name, factory)`` pairs. Each factory is published under a module
        attribute that the EntryPoint's ``value`` resolves to, so ``EntryPoint.load`` follows its
        normal path rather than being stubbed.
    """
    published: dict[str, object] = {}

    def _publish(*pairs):
        entries = []
        for index, (name, factory) in enumerate(pairs):
            attribute = f"_fake_driver_{index}"
            published[attribute] = factory
            setattr(_drivers, attribute, factory)
            entries.append(
                EntryPoint(
                    name=name,
                    value=f"pyramids.netcdf.ugrid._drivers:{attribute}",
                    group=ENTRY_POINT_GROUP,
                )
            )

        def _entry_points(*, group=None):
            return tuple(entries) if group == ENTRY_POINT_GROUP else ()

        monkeypatch.setattr(_drivers, "entry_points", _entry_points)

    yield _publish
    for attribute in published:
        monkeypatch.delattr(_drivers, attribute, raising=False)


class TestValidation:
    """Registration rejects a driver that does not meet the contract."""

    @pytest.mark.parametrize(
        ("attribute", "value", "expected"),
        [
            ("api_version", 99, "api_version"),
            ("suffixes", (), "no suffixes"),
            ("suffixes", ("slf",), "malformed suffix"),
            ("suffixes", (".SLF",), "malformed suffix"),
            (
                "capabilities",
                frozenset({"read_mesh", "teleport"}),
                "unknown capability",
            ),
            (
                "capabilities",
                frozenset({"read_mesh", "needs_mesh"}),
                "declares exactly one",
            ),
            ("capabilities", frozenset({"read_data"}), "declares exactly one"),
        ],
    )
    def test_invalid_driver_is_skipped_with_a_warning(
        self, fake_entry_points, attribute, value, expected
    ):
        """A driver breaking any contract rule is skipped, and the warning says which."""

        class Broken(FakeDriver):
            pass

        setattr(Broken, attribute, value)
        fake_entry_points(("broken", Broken))
        with pytest.warns(RuntimeWarning, match=expected):
            assert mesh_drivers() == {}

    def test_driver_missing_an_attribute_is_skipped(self, fake_entry_points):
        """A class lacking a contract attribute is reported by name, not an AttributeError."""

        class Nameless:
            suffixes = (".x",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"read_mesh"})

        fake_entry_points(("nameless", Nameless))
        with pytest.warns(RuntimeWarning, match="missing attribute"):
            assert mesh_drivers() == {}

    def test_every_capability_token_is_accepted(self, fake_entry_points):
        """The documented vocabulary is exactly what validation allows."""

        class Everything(FakeDriver):
            capabilities = CAPABILITIES - {"needs_mesh"}

        fake_entry_points(("everything", Everything))
        assert set(mesh_drivers()) == {"fake"}


class TestDiscovery:
    """Discovery is lazy, cached, and survives a broken plugin."""

    def test_driver_raising_on_load_is_skipped_with_a_warning(self, fake_entry_points):
        """A plugin whose import fails -- a missing optional dependency, say -- costs itself only."""

        def explode():
            raise ImportError("No module named 'mikeio'")

        fake_entry_points(("exploding", explode), ("fake", FakeDriver))
        with pytest.warns(RuntimeWarning, match="could not be loaded"):
            resolved = mesh_drivers()
        assert set(resolved) == {"fake"}

    def test_duplicate_name_keeps_the_first_and_warns(self, fake_entry_points):
        """Two plugins claiming one name is a conflict the user must be told about."""

        class Other(FakeDriver):
            long_name = "The Second One"

        fake_entry_points(("first", FakeDriver), ("second", Other))
        with pytest.warns(RuntimeWarning, match="registered twice"):
            resolved = mesh_drivers()
        assert resolved["fake"].long_name == "Fake Mesh Format"

    def test_discovery_runs_once_and_instantiates_once(self, fake_entry_points):
        """A driver may cache parsed state, so it must not be rebuilt per read."""
        built = []

        class Counted(FakeDriver):
            def __init__(self) -> None:
                super().__init__()
                built.append(self)

        fake_entry_points(("counted", Counted))
        first = mesh_drivers()
        second = mesh_drivers()
        assert len(built) == 1
        assert first is second

    def test_clear_cache_forces_rediscovery(self, fake_entry_points):
        """The test-only reset is what keeps the no-discovery guard honest."""
        fake_entry_points(("fake", FakeDriver))
        mesh_drivers()
        assert _drivers._CACHE is not None
        clear_driver_cache()
        assert _drivers._CACHE is None

    def test_drivers_for_suffix_normalises_and_preserves_order(self, registered):
        """Suffix lookup is case-insensitive, dot-optional, and in registration order."""
        first, second = registered(FakeDriver(), VetoDriver())
        assert drivers_for_suffix(".fake") == [first, second]
        assert drivers_for_suffix("FAKE") == [first, second]
        assert drivers_for_suffix(".missing") == []


class TestReadDispatch:
    """`read_file` resolution order."""

    def test_driver_kwarg_selects_that_driver(self, tmp_path, registered):
        """An explicit driver is used whatever the suffix says."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.whatever"
        target.write_bytes(b"")
        result = UgridDataset.read_file(target, driver="fake")
        assert result.n_face == 1
        assert driver.read_calls == [(str(target), None)]

    def test_driver_kwarg_does_not_sniff(self, tmp_path, registered):
        """Naming a driver skips `can_open`: the caller has already decided."""
        driver = registered(VetoDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target, driver="veto")
        assert driver.sniffed == []

    def test_unknown_driver_lists_what_is_installed(self, tmp_path, registered):
        """The error names the alternatives rather than only the failure."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(DriverNotExistError, match="fake"):
            UgridDataset.read_file(target, driver="nope")

    def test_can_open_veto_falls_through_to_gdal(self, tmp_path, registered):
        """A driver claiming the suffix but refusing the file does not capture the read."""
        driver = registered(VetoDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"not a mesh at all")
        with pytest.raises(UnsupportedMeshFormatError):
            UgridDataset.read_file(target)
        assert driver.sniffed == [str(target)]

    def test_unclaimed_suffix_reports_installed_drivers(self, tmp_path, registered):
        """GDAL's "not recognized" would point at a corrupt file, not a missing plugin."""
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01\x02")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert ".slf" in message
        assert "fake (.fake)" in message
        assert "pip install pyramids-mesh" in message

    def test_unsupported_format_error_is_catchable_as_the_old_one(
        self, tmp_path, registered
    ):
        """Existing `except FileFormatNotSupportedError` callers keep working."""
        registered(FakeDriver())
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00")
        with pytest.raises(FileFormatNotSupportedError):
            UgridDataset.read_file(target)

    def test_needs_mesh_driver_points_at_attach_data(self, tmp_path, registered):
        """A results-only file reached through `read_file` must say where to go."""
        registered(AttachDriver())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="attach_data"):
            UgridDataset.read_file(target)

    def test_driver_returning_network1d_is_reported(self, tmp_path, registered):
        """A 1D topology is a different class, so handing it back would break the caller."""
        driver = registered(FakeDriver())
        driver.returns = Network1d(
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 0.0]),
            edge_node_connectivity=Connectivity(
                data=np.array([[0, 1]]),
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )
        target = tmp_path / "network.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="Network1d"):
            UgridDataset.read_file(target)

    def test_driver_failure_names_the_driver(self, tmp_path, registered):
        """A parse failure is attributed, so the blame is unambiguous."""

        class Failing(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise ValueError("record length mismatch at byte 80")

        registered(Failing())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="'fake' driver failed"):
            UgridDataset.read_file(target)

    @pytest.mark.parametrize("suffix", [".nc", ".nc4", ".NC", ".cdf"])
    def test_netcdf_read_does_no_discovery(
        self, tmp_path, suffix, ugrid_convention_nc_path
    ):
        """The performance contract: a NetCDF read must not walk every installed distribution.

        Asserted by watching the cache rather than by timing, so it cannot flake.
        """
        target = tmp_path / f"mesh{suffix}"
        target.write_bytes(Path(ugrid_convention_nc_path).read_bytes())
        UgridDataset.read_file(target)
        assert _drivers._CACHE is None

    def test_vsi_path_does_no_discovery(self, ugrid_convention_nc_path):
        """A virtual-filesystem name has no suffix to dispatch on and must stay on GDAL."""
        UgridDataset.from_bytes(Path(ugrid_convention_nc_path).read_bytes())
        assert _drivers._CACHE is None

    def test_extensionless_file_does_not_crash(self, tmp_path, registered):
        """`Path.suffix` is empty here; dispatch must handle it, not raise."""
        registered(FakeDriver())
        target = tmp_path / "mesh"
        target.write_bytes(b"\x00")
        with pytest.raises(UnsupportedMeshFormatError, match="no suffix"):
            UgridDataset.read_file(target)

    def test_suffix_match_is_case_insensitive(self, tmp_path, registered):
        """`.FAKE` is a real spelling on Windows."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.FAKE"
        target.write_bytes(b"")
        UgridDataset.read_file(target)
        assert len(driver.read_calls) == 1

    def test_missing_file_still_raises_file_not_found(self, tmp_path, registered):
        """Dispatch must not mask the simplest error of all."""
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "absent.nc")


class TestAttachData:
    """`attach_data` for results-only formats."""

    def test_attach_returns_a_new_dataset_and_leaves_the_original(
        self, tmp_path, registered
    ):
        """The immutable-derivation contract every other pyramids operation keeps."""
        registered(AttachDriver())
        mesh = _square_mesh()
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        result = mesh.attach_data(target)
        assert "velocity" in result.data_variable_names
        assert "velocity" not in mesh.data_variable_names
        assert result is not mesh

    def test_attach_refuses_an_element_count_mismatch(self, tmp_path, registered):
        """The only safeguard against attaching results to the wrong mesh."""
        registered(AttachDriver(declared_elements=4096))
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        with pytest.raises(ValueError, match="4096"):
            _square_mesh().attach_data(target)

    def test_attach_refuses_a_driver_that_reads_its_own_mesh(
        self, tmp_path, registered
    ):
        """A `read_mesh` driver has nothing to attach."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="nothing to attach"):
            _square_mesh().attach_data(target, driver="fake")

    def test_attach_with_no_driver_for_the_suffix_reports_installed(
        self, tmp_path, registered
    ):
        """Detection failing says what is available, as the read path does."""
        registered(AttachDriver())
        target = tmp_path / "velocity.unknown"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="No installed driver"):
            _square_mesh().attach_data(target)

    def test_attach_rejects_a_variable_name_collision(self, tmp_path, registered):
        """Silently keeping either copy would lose data."""

        class Colliding(AttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return {
                    "depth": MeshVariable(
                        name="depth",
                        location="face",
                        mesh_name=dataset.mesh_name,
                        shape=(dataset.n_face,),
                        _data=np.ones(dataset.n_face),
                    )
                }

        registered(Colliding())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        with pytest.raises(ValueError, match="already defines"):
            _square_mesh().attach_data(target)

    def test_attach_unknown_driver_raises(self, tmp_path, registered):
        """Naming a driver that is not installed is an error, not a fallback."""
        registered(AttachDriver())
        target = tmp_path / "velocity.res"
        target.write_bytes(b"")
        with pytest.raises(DriverNotExistError):
            _square_mesh().attach_data(target, driver="nope")


class TestWriteDispatch:
    """`to_file` with a driver."""

    def test_default_write_is_unchanged(self, tmp_path):
        """No driver means the built-in UGRID writer, byte for byte as before."""
        target = tmp_path / "mesh.nc"
        _square_mesh().to_file(target)
        assert target.exists()
        assert UgridDataset.read_file(target).n_face == 1

    def test_driver_write_passes_options_through_verbatim(self, tmp_path, registered):
        """pyramids must not interpret or rename a format's own settings."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        _square_mesh().to_file(target, driver="fake", endian=">", precision="double")
        assert driver.write_calls == [
            (str(target), {"endian": ">", "precision": "double"})
        ]

    def test_write_driver_without_write_mesh_raises(self, tmp_path, registered):
        """A read-only driver says what it can do instead."""
        registered(ReadOnlyDriver())
        with pytest.raises(UnsupportedMeshFormatError, match="cannot write"):
            _square_mesh().to_file(tmp_path / "mesh.ro", driver="readonly")

    def test_frame_only_driver_warns_that_variables_are_dropped(
        self, tmp_path, registered
    ):
        """Losing data silently is the failure mode worth a warning."""
        registered(FrameOnlyDriver())
        with pytest.warns(RuntimeWarning, match="no data"):
            _square_mesh().to_file(tmp_path / "mesh.frm", driver="frame")

    def test_suffix_does_not_select_a_driver(self, tmp_path, registered):
        """Inferring Selafin from `.slf` would change behaviour for existing callers."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        _square_mesh().to_file(target)
        assert driver.write_calls == []
        assert UgridDataset.read_file(target, driver="fake").n_face == 1

    def test_options_without_a_driver_are_rejected(self, tmp_path):
        """Dropping them silently would hide a caller's mistake."""
        with pytest.raises(TypeError, match="driver="):
            _square_mesh().to_file(tmp_path / "mesh.nc", endian=">")

    def test_driver_write_failure_names_the_driver(self, tmp_path, registered):
        """Attribution, as on the read path."""

        class Failing(FakeDriver):
            def write(self, dataset, path, **options):
                raise OSError("disk full")

        registered(Failing())
        with pytest.raises(UnsupportedMeshFormatError, match="'fake' driver failed"):
            _square_mesh().to_file(tmp_path / "mesh.fake", driver="fake")

    def test_write_unknown_driver_raises(self, tmp_path, registered):
        """Consistent with the read and attach paths."""
        registered(FakeDriver())
        with pytest.raises(DriverNotExistError):
            _square_mesh().to_file(tmp_path / "mesh.fake", driver="nope")


class TestIntrospection:
    """`info(drivers=True)`, `mesh_names` and `read_all` routing."""

    def test_info_lists_the_builtin_reader_with_no_plugins(self, capsys):
        """The listing must read sensibly on a bare install, not render an empty section."""
        _square_mesh().info(drivers=True)
        captured = capsys.readouterr().out
        assert "mesh drivers (1)" in captured
        assert "ugrid (built-in)" in captured

    def test_info_lists_a_registered_driver(self, capsys, registered):
        """pyramids' answer to MDAL's driver introspection."""
        registered(FakeDriver())
        _square_mesh().info(drivers=True)
        captured = capsys.readouterr().out
        assert "fake  Fake Mesh Format  .fake" in captured

    def test_info_without_the_flag_does_no_discovery(self, capsys):
        """Summarising one dataset must not walk every installed distribution."""
        _square_mesh().info()
        capsys.readouterr()
        assert _drivers._CACHE is None

    def test_describe_drivers_reports_capabilities(self, registered):
        """Each line carries what the driver can actually do."""
        registered(FakeDriver())
        lines = describe_drivers()
        assert any("read_mesh" in line and "fake" in line for line in lines)

    def test_mesh_names_routes_to_the_driver(self, tmp_path, registered):
        """A multi-mesh third-party format is discoverable the same way UGRID is."""
        registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        assert UgridDataset.mesh_names(target) == ["first", "second"]

    def test_mesh_names_on_netcdf_does_no_discovery(self, ugrid_convention_nc_path):
        """The same short-circuit as `read_file`."""
        UgridDataset.mesh_names(ugrid_convention_nc_path)
        assert _drivers._CACHE is None

    def test_read_all_routes_through_the_driver(self, tmp_path, registered):
        """`read_all` is `mesh_names` plus a read per name, so both must dispatch."""
        driver = registered(FakeDriver())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        result = UgridDataset.read_all(target)
        assert sorted(result) == ["first", "second"]
        assert [call[1] for call in driver.read_calls] == ["first", "second"]


class TestBareInstall:
    """Behaviour with no plugin installed at all — the commonest real case.

    Worth its own class because every other test here registers something. The paths below run
    only when the registry is genuinely empty, which is what a fresh `pip install pyramids-gis`
    looks like.
    """

    def test_unsupported_message_says_none_are_installed(self, tmp_path):
        """The error reads sensibly with an empty registry rather than a dangling colon.

        Test scenario:
            With no driver registered, a file of an unknown format reports `none` for the
            installed list, still names the built-in reader, and still offers the install hint.
        """
        target = tmp_path / "model.slf"
        target.write_bytes(b"\x00\x01")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert "Installed drivers: none" in message, (
            f"An empty registry should report 'none', got: {message}"
        )
        assert "ugrid (.nc .nc4 .cdf)" in message, (
            f"The built-in reader should still be named, got: {message}"
        )
        assert "pip install pyramids-mesh" in message, (
            f"The install hint should still be offered, got: {message}"
        )

    def test_unknown_driver_reports_none_installed(self, tmp_path):
        """`driver=` against an empty registry names no alternatives, and says so.

        Test scenario:
            Asking for a driver on a bare install raises DriverNotExistError whose message
            reads `none` rather than listing an empty set.
        """
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(DriverNotExistError) as caught:
            UgridDataset.read_file(target, driver="selafin")
        assert "none" in str(caught.value), (
            f"Expected 'none' for an empty registry, got: {caught.value}"
        )

    def test_attach_data_on_a_bare_install_reports_none(self, tmp_path):
        """`attach_data` degrades the same way the read path does.

        Test scenario:
            With nothing registered, attaching a results file reports the empty catalogue
            instead of raising a KeyError or an IndexError from an empty candidate list.
        """
        target = tmp_path / "velocity.dat"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="Installed drivers: none"):
            _square_mesh().attach_data(target)

    def test_write_driver_on_a_bare_install_reports_none(self, tmp_path):
        """A write through a named driver fails the same way as a read.

        Test scenario:
            `to_file(driver=...)` on a bare install raises DriverNotExistError naming `none`.
        """
        with pytest.raises(DriverNotExistError, match="none"):
            _square_mesh().to_file(tmp_path / "mesh.slf", driver="selafin")


class TestErrorPassthrough:
    """A driver's own `UnsupportedMeshFormatError` must not be wrapped in another one.

    Dispatch attributes a driver's failure by name, which is right for an unexpected exception —
    a `struct.error` or a `KeyError` from a parser. But a driver that has already raised the
    package's own "I cannot read this" error has said everything useful, and re-wrapping turns a
    clear message into `UnsupportedMeshFormatError: The 'x' driver failed ...
    UnsupportedMeshFormatError(...)`.
    """

    def test_read_passes_an_unsupported_error_through_unwrapped(
        self, tmp_path, registered
    ):
        """A driver raising UnsupportedMeshFormatError from `read` keeps its own message.

        Test scenario:
            The driver raises UnsupportedMeshFormatError("this file is version 9"); the caller
            sees exactly that, with no "driver failed to read" wrapper around it.
        """

        class Picky(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise UnsupportedMeshFormatError(
                    "this file is version 9, which I cannot read"
                )

        registered(Picky())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            UgridDataset.read_file(target)
        message = str(caught.value)
        assert message == "this file is version 9, which I cannot read", (
            f"The driver's own message should survive unwrapped, got: {message}"
        )
        assert "driver failed to read" not in message, (
            f"An already-specific error must not be re-wrapped, got: {message}"
        )

    def test_write_passes_an_unsupported_error_through_unwrapped(
        self, tmp_path, registered
    ):
        """A driver raising UnsupportedMeshFormatError from `write` keeps its own message.

        Test scenario:
            The driver refuses to write a quad mesh and says so; the caller sees that reason,
            not a generic "driver failed to write" wrapper.
        """

        class RefusesQuads(FakeDriver):
            def write(self, dataset, path, **options):
                raise UnsupportedMeshFormatError("this format stores triangles only")

        registered(RefusesQuads())
        with pytest.raises(UnsupportedMeshFormatError) as caught:
            _square_mesh().to_file(tmp_path / "mesh.fake", driver="fake")
        message = str(caught.value)
        assert message == "this format stores triangles only", (
            f"The driver's own message should survive unwrapped, got: {message}"
        )

    def test_write_passes_a_driver_not_exist_error_through(self, tmp_path, registered):
        """A driver raising DriverNotExistError is not converted into a format error.

        Test scenario:
            A driver delegating to a sub-driver it cannot find raises DriverNotExistError;
            that type reaches the caller rather than being flattened into
            UnsupportedMeshFormatError, so `except DriverNotExistError` still works.
        """

        class Delegating(FakeDriver):
            def write(self, dataset, path, **options):
                raise DriverNotExistError("the sub-driver 'x' is not installed")

        registered(Delegating())
        with pytest.raises(DriverNotExistError, match="sub-driver"):
            _square_mesh().to_file(tmp_path / "mesh.fake", driver="fake")


class TestAttachThroughVirtualFilesystem:
    """`attach_data` with a GDAL virtual-filesystem name."""

    def test_vsi_path_has_no_suffix_to_dispatch_on(self, registered):
        """A `/vsi` name is not an OS path, so suffix detection must not be run on it.

        `Path("/vsimem/x").suffix` would read whatever follows the last dot in a virtual name,
        which is not a format hint. The resolver therefore treats a `/vsi` target as having no
        suffix, and reports that no driver claims it rather than guessing from the text.

        Test scenario:
            Attaching `/vsimem/results.res` finds no candidate — the registered driver claims
            `.res` but the virtual path contributes no suffix — and the error says so.
        """
        registered(AttachDriver())
        with pytest.raises(UnsupportedMeshFormatError, match="no suffix"):
            _square_mesh().attach_data("/vsimem/results.res")

    def test_attach_with_an_explicit_needs_mesh_driver_succeeds(
        self, tmp_path, registered
    ):
        """Naming a valid `needs_mesh` driver skips detection and attaches.

        The counterpart to `test_attach_refuses_a_driver_that_reads_its_own_mesh`: that one
        proves the capability gate rejects the wrong driver, this one proves it lets the right
        one through. Without it the gate's passing branch is never executed, so an inverted
        condition would still look tested.

        Test scenario:
            `attach_data(path, driver="attach")` on a file whose suffix the driver does not
            claim still attaches, because an explicit driver bypasses suffix detection.
        """
        registered(AttachDriver())
        target = tmp_path / "velocity.unclaimed"
        target.write_bytes(b"")
        result = _square_mesh().attach_data(target, driver="attach")
        assert "velocity" in result.data_variable_names, (
            f"Expected the attached variable, got: {result.data_variable_names}"
        )


class TestValidationIsContained:
    """Validation must never let a malformed plugin raise out of `mesh_drivers`.

    `_validate` reads attributes a plugin author controls, so it coerces values it has not
    type-checked — `suffix.startswith`, `set(capabilities)`. Running it outside the guarded
    block meant a plugin with `suffixes = (1,)` raised `AttributeError` straight out of
    `read_file`, breaking every non-NetCDF read and contradicting the module's documented
    "every failure is a warning, never an exception" guarantee.
    """

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("suffixes", (1,)),
            ("suffixes", 42),
            ("suffixes", None),
            ("capabilities", 7),
            ("capabilities", "read_mesh"),
            ("capabilities", None),
            ("name", 123),
            ("api_version", "one"),
        ],
    )
    def test_wrongly_typed_attribute_warns_and_does_not_raise(
        self, fake_entry_points, attribute, value
    ):
        """A plugin attribute of the wrong type is skipped with a warning, never raised.

        Args:
            attribute: The contract attribute to corrupt.
            value: A wrongly-typed value a plugin author could plausibly write.

        Test scenario:
            Discovery must survive each of these and return an empty registry, rather than
            propagating the TypeError or AttributeError that coercing the value produces.
        """

        class Malformed(FakeDriver):
            pass

        setattr(Malformed, attribute, value)
        fake_entry_points(("malformed", Malformed))
        with pytest.warns(RuntimeWarning):
            assert mesh_drivers() == {}, "A malformed driver must not register"

    def test_a_malformed_plugin_does_not_break_a_working_one(self, fake_entry_points):
        """One broken plugin costs itself, not the whole registry.

        Test scenario:
            A driver with integer suffixes registered alongside a valid driver leaves the
            valid one usable — which is the point of containing the failure.
        """

        class Malformed(FakeDriver):
            name = "malformed"
            suffixes = (1,)

        fake_entry_points(("malformed", Malformed), ("fake", FakeDriver))
        with pytest.warns(RuntimeWarning):
            resolved = mesh_drivers()
        assert set(resolved) == {"fake"}, (
            f"The valid driver should survive, got: {sorted(resolved)}"
        )

    def test_a_driver_missing_its_methods_is_rejected_at_registration(
        self, fake_entry_points
    ):
        """A driver without `can_open`/`read` is caught now, not at first use.

        Registering it and letting the AttributeError surface later reports a missing method
        as a *format* error, which sends the user looking at their file instead of the plugin.

        Test scenario:
            A class carrying every contract attribute but no methods is skipped, and the
            warning names the missing method.
        """

        class NoMethods:
            name = "nomethods"
            long_name = "No Methods"
            suffixes = (".nm",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"read_mesh"})

        fake_entry_points(("nomethods", NoMethods))
        with pytest.warns(RuntimeWarning, match="can_open"):
            assert mesh_drivers() == {}, "A method-less driver must not register"

    def test_a_needs_mesh_driver_must_implement_read_data(self, fake_entry_points):
        """The method a capability promises is the method validation requires.

        Test scenario:
            A `needs_mesh` driver without `read_data` is rejected, naming that method —
            `read` alone cannot serve a results-only format.
        """

        class NoReadData:
            name = "noreaddata"
            long_name = "No read_data"
            suffixes = (".nrd",)
            api_version = MESH_DRIVER_API_VERSION
            capabilities = frozenset({"needs_mesh"})

            def can_open(self, path):
                return True

        fake_entry_points(("noreaddata", NoReadData))
        with pytest.warns(RuntimeWarning, match="read_data"):
            assert mesh_drivers() == {}, "A needs_mesh driver needs read_data"

    def test_a_write_mesh_driver_must_implement_write(self, fake_entry_points):
        """Same rule for the write side.

        Test scenario:
            A driver declaring `write_mesh` without a `write` method is rejected at
            registration rather than failing mid-write with a half-created file.
        """

        class NoWrite(FakeDriver):
            capabilities = frozenset({"read_mesh", "write_mesh"})
            write = None

        fake_entry_points(("nowrite", NoWrite))
        with pytest.warns(RuntimeWarning, match="write"):
            assert mesh_drivers() == {}, "A write_mesh driver needs write"

    @pytest.mark.parametrize("name", ["ugrid", "UGRID", "", "   "])
    def test_a_reserved_or_blank_name_is_rejected(self, fake_entry_points, name):
        """A plugin must not be able to shadow the built-in reader or register nameless.

        Args:
            name: A name a plugin should not be allowed to claim.

        Test scenario:
            `ugrid` is the built-in reader's identity in `info()` listings and in `driver=`;
            a plugin taking it would make the listing ambiguous and `driver="ugrid"` resolve
            to the plugin. A blank name is unusable in `driver=` at all.
        """

        class Shadowing(FakeDriver):
            pass

        Shadowing.name = name
        fake_entry_points(("shadowing", Shadowing))
        with pytest.warns(RuntimeWarning):
            assert mesh_drivers() == {}, f"The name {name!r} must be rejected"


class TestMissingFileReporting:
    """A path that does not exist must say so, on every route.

    `read_file` documents `FileNotFoundError`, but the existence check lived only in the
    NetCDF reader, and the dispatch wrapper rewrote a driver's own `FileNotFoundError` into
    `UnsupportedMeshFormatError`. A filename typo therefore reported "no installed driver
    reads this format" — sending the user to install a plugin they already had.
    """

    def test_read_file_missing_driver_path_raises_file_not_found(
        self, tmp_path, registered
    ):
        """A missing file whose suffix a driver claims still raises FileNotFoundError.

        Test scenario:
            `mesh.fake` does not exist and the `fake` driver claims `.fake`. The error must
            name the missing file, not the format.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "typo.fake")

    def test_read_file_missing_with_explicit_driver_raises_file_not_found(
        self, tmp_path, registered
    ):
        """The explicit-driver route checks existence too.

        Test scenario:
            `driver="fake"` bypasses detection, so it needs its own existence check rather
            than inheriting the detection path's.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file(tmp_path / "typo.whatever", driver="fake")

    def test_attach_data_missing_file_raises_file_not_found(self, tmp_path, registered):
        """`attach_data` reports a missing results file as missing.

        Test scenario:
            A results file that is not there is a typo, not an unsupported format.
        """
        registered(AttachDriver())
        with pytest.raises(FileNotFoundError):
            _square_mesh().attach_data(tmp_path / "typo.res")

    def test_mesh_names_missing_driver_path_raises_file_not_found(
        self, tmp_path, registered
    ):
        """`mesh_names` matches `read_file` here, as its docstring claims.

        Test scenario:
            The same missing-file report on the topology-listing route.
        """
        registered(FakeDriver())
        with pytest.raises(FileNotFoundError):
            UgridDataset.mesh_names(tmp_path / "typo.fake")

    def test_a_driver_raising_file_not_found_is_not_rewritten(
        self, tmp_path, registered
    ):
        """A driver's own FileNotFoundError passes through unwrapped.

        A driver may open sibling files of its own — FLO-2D and ADCIRC are multi-file
        formats — and a missing sibling is a missing file, not an unsupported format.

        Test scenario:
            The primary file exists, the driver raises FileNotFoundError for a sibling, and
            the caller sees FileNotFoundError naming that sibling.
        """

        class NeedsSibling(FakeDriver):
            def read(self, path, *, mesh_name=None, lazy=True):
                raise FileNotFoundError(
                    "the companion file 'domain.sibling' is missing"
                )

        registered(NeedsSibling())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="companion file"):
            UgridDataset.read_file(target)

    def test_a_directory_is_not_reported_as_an_unsupported_format(
        self, tmp_path, registered
    ):
        """A directory gets an error about being a directory, not a plugin-install hint.

        A driver is still allowed to claim a directory — Esri TIN is a directory format — so
        this covers the *unclaimed* case, which is the one that used to misreport.

        Test scenario:
            With a driver that vetoes the path, the directory falls through to the built-in
            reader, which previously produced "no installed driver reads '.fake' — pip
            install pyramids-mesh" for what is really a wrong-path mistake.
        """
        registered(VetoDriver())
        folder = tmp_path / "a_directory.fake"
        folder.mkdir()
        with pytest.raises(
            (IsADirectoryError, FileNotFoundError, ValueError)
        ) as caught:
            UgridDataset.read_file(folder)
        assert "pip install" not in str(caught.value), (
            f"A directory must not be reported as a missing plugin: {caught.value}"
        )


class VariablesOnlyAttachDriver:
    """A `needs_mesh` driver under the contract where `read_data` returns only new variables."""

    name = "varattach"
    long_name = "Variables-Only Results"
    suffixes = (".vres",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"needs_mesh", "read_data"})

    def __init__(self, produces=("velocity",), declared_elements=None):
        """Set up the driver.

        Args:
            produces: Names the driver claims to have read from the file.
            declared_elements: The element count the file declares, for the mismatch case.
        """
        self.produces = produces
        self.declared_elements = declared_elements
        self.lazy_seen: list[bool] = []

    def can_open(self, path):
        """Accept any offered path."""
        return True

    def read_data(self, path, dataset, *, lazy=True):
        """Return one variable per name in `produces`, after the element-count check."""
        self.lazy_seen.append(lazy)
        declared = (
            dataset.n_face if self.declared_elements is None else self.declared_elements
        )
        if declared != dataset.n_face:
            raise ValueError(
                f"{path} declares {declared} elements; the mesh has {dataset.n_face}."
            )
        return {
            name: MeshVariable(
                name=name,
                location="face",
                mesh_name=dataset.mesh_name,
                shape=(dataset.n_face,),
                _data=np.zeros(dataset.n_face),
            )
            for name in self.produces
        }


class TestAttachMergeContract:
    """`read_data` returns its own variables; pyramids owns the merge.

    The first contract had the driver return a whole merged `UgridDataset`, and pyramids
    detected collisions with `attached[name] is not self._data_variables.get(name)`. That
    encoded an undocumented by-reference requirement: a driver rebuilding the dict — which
    is what `dataclasses.replace` produces, and how pyramids' own `rename_variable` builds
    its result — tripped a spurious `ValueError` naming variables that were not colliding,
    with a remedy ("rename one side") the user could not act on. It also could not tell a
    driver that silently dropped the mesh's own variables from a well-behaved one.

    Returning only the new variables removes the ambiguity: a name already on the dataset is
    unambiguously a collision, and the driver never touches what was already there.
    """

    def test_attach_merges_the_drivers_variables_onto_the_mesh(
        self, tmp_path, registered
    ):
        """The mesh keeps its own variables and gains the driver's.

        Test scenario:
            A mesh carrying `depth` attaches a file producing `velocity`; the result has
            both, and the original is untouched.
        """
        registered(VariablesOnlyAttachDriver())
        mesh = _square_mesh()
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        result = mesh.attach_data(target)
        assert sorted(result.data_variable_names) == ["depth", "velocity"], (
            f"Expected both variables, got: {sorted(result.data_variable_names)}"
        )
        assert mesh.data_variable_names == ["depth"], (
            f"The original must be unchanged, got: {mesh.data_variable_names}"
        )

    def test_a_rebuilt_variable_is_not_a_spurious_collision(self, tmp_path, registered):
        """A driver that rebuilds objects rather than reusing them attaches cleanly.

        This is the exact shape that produced M3: identity no longer participates, so a
        driver may construct its variables however it likes.

        Test scenario:
            The driver returns freshly-built `MeshVariable` objects; attaching succeeds.
        """
        registered(VariablesOnlyAttachDriver(produces=("velocity", "salinity")))
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        result = _square_mesh().attach_data(target)
        assert sorted(result.data_variable_names) == [
            "depth",
            "salinity",
            "velocity",
        ], f"All three expected, got: {sorted(result.data_variable_names)}"

    def test_a_genuine_name_collision_is_refused(self, tmp_path, registered):
        """A driver producing a name the mesh already defines is an unambiguous collision.

        Test scenario:
            The driver claims to have read `depth`, which the mesh already has; silently
            keeping either copy would lose data, so this raises and names `depth`.
        """
        registered(VariablesOnlyAttachDriver(produces=("depth",)))
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        with pytest.raises(ValueError, match="depth"):
            _square_mesh().attach_data(target)

    def test_attach_forwards_lazy_to_the_driver(self, tmp_path, registered):
        """`lazy` reaches `read_data`, so the capability is not inert.

        Test scenario:
            `attach_data(..., lazy=False)` is observable inside the driver.
        """
        driver = registered(VariablesOnlyAttachDriver())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        _square_mesh().attach_data(target, lazy=False)
        _square_mesh().attach_data(target, lazy=True)
        assert driver.lazy_seen == [False, True], (
            f"lazy must be forwarded verbatim, got: {driver.lazy_seen}"
        )

    def test_a_driver_returning_the_wrong_type_is_reported(self, tmp_path, registered):
        """A driver returning a dataset instead of a mapping is told so.

        The contract changed from "return the merged dataset" to "return your variables", so
        a driver written against the old shape must get a clear message rather than a
        confusing failure deep in the merge.

        Test scenario:
            `read_data` returns a `UgridDataset`; the error names the driver and the
            expected shape.
        """

        class OldStyle(VariablesOnlyAttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return dataset

        registered(OldStyle())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="mapping"):
            _square_mesh().attach_data(target)

    def test_a_driver_returning_a_non_mesh_variable_is_reported(
        self, tmp_path, registered
    ):
        """Values in the mapping must be MeshVariable objects.

        Test scenario:
            A driver returning a bare numpy array per name is caught at the boundary, not
            stored and left to fail at first use.
        """

        class RawArrays(VariablesOnlyAttachDriver):
            def read_data(self, path, dataset, *, lazy=True):
                return {"velocity": np.zeros(dataset.n_face)}

        registered(RawArrays())
        target = tmp_path / "v.vres"
        target.write_bytes(b"")
        with pytest.raises(UnsupportedMeshFormatError, match="MeshVariable"):
            _square_mesh().attach_data(target)


class TestLazyForwarding:
    """`lazy` must reach the driver, or the `lazy_data` capability is decoration.

    `read_file` had no `lazy` parameter and `_read_through_driver` called
    `driver.read(path, mesh_name=...)` without one, so a driver advertising `lazy_data` was
    always invoked with its own default. The docs told authors to honour `lazy=True`, which
    nothing ever set.
    """

    def test_read_file_forwards_lazy_to_the_driver(self, tmp_path, registered):
        """`lazy` is observable inside `driver.read`.

        Test scenario:
            Reading with `lazy=False` then `lazy=True` records both values in the driver.
        """

        class Recording(FakeDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(Recording())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target, lazy=False)
        UgridDataset.read_file(target, lazy=True)
        assert driver.lazy_seen == [False, True], (
            f"lazy must be forwarded verbatim, got: {driver.lazy_seen}"
        )

    def test_lazy_defaults_to_true(self, tmp_path, registered):
        """The default matches what the docs promise a driver will receive.

        Test scenario:
            Calling without `lazy` reaches the driver as True.
        """

        class Recording(FakeDriver):
            def __init__(self):
                super().__init__()
                self.lazy_seen: list[bool] = []

            def read(self, path, *, mesh_name=None, lazy=True):
                self.lazy_seen.append(lazy)
                return _square_mesh()

        driver = registered(Recording())
        target = tmp_path / "mesh.fake"
        target.write_bytes(b"")
        UgridDataset.read_file(target)
        assert driver.lazy_seen == [True], (
            f"The default must arrive as True, got: {driver.lazy_seen}"
        )

    def test_lazy_is_rejected_on_the_netcdf_path(self, ugrid_convention_nc_path):
        """The built-in reader has no lazy switch, so asking for one must not be silent.

        pyramids' NetCDF variables are already lazy by construction. Accepting `lazy=False`
        and ignoring it would promise an eager read the reader does not perform.

        Test scenario:
            `read_file(..., lazy=False)` on a `.nc` file raises rather than silently
            doing a lazy read.
        """
        with pytest.raises(TypeError, match="lazy"):
            UgridDataset.read_file(ugrid_convention_nc_path, lazy=False)
