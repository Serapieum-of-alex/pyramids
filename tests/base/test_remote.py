"""Unit tests for pyramids.base.remote."""

from __future__ import annotations

import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np  # noqa: E402
import pytest
from osgeo import gdal

from pyramids import _io
from pyramids.base.remote import (
    _BUCKET_URL_SCHEMES,
    _CLOUD_VSI_PREFIXES,
    _CREDENTIAL_FAMILIES,
    _NETWORK_VSI_PREFIXES,
    _OBJECT_STORE_VSI_PREFIXES,
    _PATH_SCOPED_KEYS,
    _VSI_PREFIXES,
    URL_SCHEMES,
    CloudConfig,
    _ambient_options,
    _ambient_stack,
    _bucket_prefix,
    _chain_archive_vsi,
    _drop_ambient,
    _family_of,
    _to_vsi,
    ambient_store_config,
    cloud_config_from_env,
    is_network_backed,
    is_remote,
    redact_credentials,
    signer_cloud_config,
)
from pyramids.netcdf import NetCDF

_DATA_RASTER = Path(__file__).parents[1] / "data" / "acc4000.tif"

pytestmark = pytest.mark.core

_STORE_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_S3_ENDPOINT",
    "AWS_ENDPOINT_URL",
    "AWS_VIRTUAL_HOSTING",
    "AWS_NO_SIGN_REQUEST",
    "AWS_REQUEST_PAYER",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_CONFIG_FILE",
    "GS_ACCESS_KEY_ID",
    "GS_SECRET_ACCESS_KEY",
    "GS_OAUTH2_REFRESH_TOKEN",
    "GS_NO_SIGN_REQUEST",
    "GS_USER_PROJECT",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_STORAGE_ACCESS_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_NO_SIGN_REQUEST",
    "GDAL_DISABLE_READDIR_ON_OPEN",
)


@pytest.fixture(autouse=True)
def _isolated_from_the_developer_environment(monkeypatch):
    """Clear every store variable this module asserts on, for the whole file.

    A path-specific miss falls through to the process config, so on a machine
    with credentials or an endpoint exported these tests assert against whatever
    that developer happens to have set: they pass on CI and fail locally, which
    is the worst way round. Scoped to one class it left its siblings exposed --
    `TestCloudConfigContextManager` asserts on `AWS_NO_SIGN_REQUEST` and
    `GDAL_DISABLE_READDIR_ON_OPEN` too.
    """
    for name in _STORE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _EnvSigner:
    """Minimal signer stand-in exposing only ``gdal_env()``."""

    def __init__(self, env):
        self._env = dict(env)

    def gdal_env(self):
        """Return the advertised GDAL config mapping."""
        return dict(self._env)


class TestSignerCloudConfig:
    """Tests for the shared ``signer_cloud_config`` helper (M4)."""

    def test_none_returns_nullcontext(self):
        """A ``None`` signer yields a no-op nullcontext.

        Test scenario:
            ``signer_cloud_config(None)`` installs no GDAL config.
        """
        assert isinstance(signer_cloud_config(None), nullcontext), (
            "expected a no-op context"
        )

    def test_signer_returns_seeded_cloudconfig(self):
        """A signer yields a CloudConfig carrying its ``gdal_env()``.

        Test scenario:
            The returned CloudConfig's config equals the signer's env.
        """
        ctx = signer_cloud_config(_EnvSigner({"AWS_REQUEST_PAYER": "requester"}))
        assert isinstance(ctx, CloudConfig), f"expected CloudConfig, got {type(ctx)}"
        assert ctx.as_gdal_config() == {"AWS_REQUEST_PAYER": "requester"}, (
            f"unexpected config: {ctx.as_gdal_config()}"
        )

    def test_config_active_within_block(self):
        """The signer's config is live inside the block and torn down after.

        Test scenario:
            ``GDAL_HTTP_TIMEOUT`` is set within the block and unset afterwards.
        """
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") is None, "precondition: unset"
        with signer_cloud_config(_EnvSigner({"GDAL_HTTP_TIMEOUT": "42"})):
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "42", (
                "config not active in block"
            )
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") is None, (
            "config not restored after block"
        )


class TestToVsi:
    def test_s3(self):
        assert _to_vsi("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_s3_nested_path(self):
        assert _to_vsi("s3://bucket/a/b/c/key.tif") == "/vsis3/bucket/a/b/c/key.tif"

    def test_gs(self):
        assert _to_vsi("gs://bucket/key.tif") == "/vsigs/bucket/key.tif"

    def test_az(self):
        assert _to_vsi("az://container/blob.tif") == "/vsiaz/container/blob.tif"

    def test_abfs_maps_to_vsiadls(self):
        """`abfs://` is the Gen2 scheme, so it routes to the Gen2 handler.

        It mapped to `/vsiaz/` (Blob) until #918. `abfs` is the Azure Blob
        *File System* driver, which is Data Lake Gen2 everywhere else in the
        Azure and Hadoop ecosystems; a flat Blob account is reached with
        `az://`.
        """
        assert _to_vsi("abfs://container/blob.tif") == "/vsiadls/container/blob.tif"

    def test_abfss_maps_to_vsiadls(self):
        """The TLS spelling of the Gen2 scheme resolves the same way."""
        assert _to_vsi("abfss://container/blob.tif") == "/vsiadls/container/blob.tif"

    def test_there_is_no_adls_scheme(self):
        """`adls://` is not a real scheme, so it is not accepted.

        The registered Azure Data Lake name is `adl://` (Gen1, which GDAL does
        not handle). Inventing `adls://` would collide with it visually and fail
        in the fsspec-backed readers, which resolve the scheme themselves.
        """
        assert "adls" not in URL_SCHEMES, "adls:// is not an ecosystem scheme"
        assert _to_vsi("adls://container/blob.tif") == "adls://container/blob.tif"

    @pytest.mark.parametrize("configured", [None, "prodlake", "otheracct"])
    def test_gen2_account_authority_rewrites_deterministically(self, configured):
        """The rewrite does not depend on ambient credentials.

        Args:
            configured: The `AZURE_STORAGE_ACCOUNT` in force, or `None`.

        Test scenario:
            `_to_vsi` runs on essentially every open, so the same URL must
            produce the same path whether or not credentials are configured and
            whether or not a `CloudConfig` block is active.
        """
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", configured):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = _to_vsi(
                    "abfss://raw@prodlake.dfs.core.windows.net/2026/x.tif"
                )
        assert resolved == "/vsiadls/raw/2026/x.tif", resolved

    def test_a_conflicting_account_warns_rather_than_raising(self):
        """A URL account that disagrees with the configured one is surfaced.

        Test scenario:
            GDAL takes the account from configuration, so the read would go
            somewhere other than the URL names. That is worth flagging — but not
            by raising from a path-rewriting helper on the open path.
        """
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", "otheracct"):
            with pytest.warns(UserWarning, match="otheracct"):
                _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")

    def test_a_matching_account_is_silent(self):
        """No warning when the URL and the configuration agree."""
        with gdal.config_option("AZURE_STORAGE_ACCOUNT", "prodlake"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")
                    == "/vsiadls/raw/x.tif"
                )

    def test_the_account_may_come_from_the_connection_string(self):
        """GDAL accepts the account embedded in the connection string.

        Test scenario:
            Reading only `AZURE_STORAGE_ACCOUNT` warned about a conflict on a
            setup that is perfectly valid for `/vsiadls/`.
        """
        connection = "DefaultEndpointsProtocol=https;AccountName=prodlake;AccountKey=k"
        with gdal.config_option("AZURE_STORAGE_CONNECTION_STRING", connection):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    _to_vsi("abfss://raw@prodlake.dfs.core.windows.net/x.tif")
                    == "/vsiadls/raw/x.tif"
                )

    def test_bare_gen2_authority_is_still_a_plain_container(self):
        """Without an `@`, the authority is the filesystem name as before."""
        assert _to_vsi("abfs://fs/2026/x.tif") == "/vsiadls/fs/2026/x.tif"

    def test_https_simple(self):
        assert _to_vsi("https://foo.com/x.tif") == "/vsicurl/https://foo.com/x.tif"

    def test_https_with_query(self):
        url = "https://foo.com/x.tif?sig=abc&exp=123"
        assert _to_vsi(url) == f"/vsicurl/{url}"

    def test_http_plain(self):
        assert _to_vsi("http://foo.com/x.tif") == "/vsicurl/http://foo.com/x.tif"

    def test_file_uri_posix(self):
        assert _to_vsi("file:///srv/data/x.tif") == "/srv/data/x.tif"

    def test_file_uri_windows(self):
        assert _to_vsi("file:///C:/data/x.tif") == "C:/data/x.tif"

    def test_already_vsi_s3_passthrough(self):
        assert _to_vsi("/vsis3/bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_already_vsi_curl_passthrough(self):
        assert _to_vsi("/vsicurl/https://x/y.tif") == "/vsicurl/https://x/y.tif"

    def test_already_vsi_mem_passthrough(self):
        assert _to_vsi("/vsimem/temp.tif") == "/vsimem/temp.tif"

    def test_already_vsi_zip_passthrough(self):
        assert _to_vsi("/vsizip/a.zip/b.tif") == "/vsizip/a.zip/b.tif"

    def test_local_posix_unchanged(self):
        assert _to_vsi("/home/user/data.tif") == "/home/user/data.tif"

    def test_local_windows_drive_unchanged(self):
        assert _to_vsi("C:/data/x.tif") == "C:/data/x.tif"

    def test_relative_path_unchanged(self):
        assert _to_vsi("data/x.tif") == "data/x.tif"

    def test_dods_maps_to_netcdf_dap(self):
        assert (
            _to_vsi("dods://test.opendap.org/opendap/data/nc/coads.nc")
            == 'NETCDF:"https://test.opendap.org/opendap/data/nc/coads.nc"'
        )

    def test_dods_preserves_query(self):
        assert _to_vsi("dods://h/path?a=b") == 'NETCDF:"https://h/path?a=b"'

    def test_dods_without_double_slash(self):
        # urlparse still classifies "dods:host/path" as scheme dods; must not crash.
        assert _to_vsi("dods:host/path") == 'NETCDF:"https://host/path"'

    def test_dods_uppercase_scheme(self):
        # scheme match is case-insensitive; the slice uses the lower-cased length.
        assert (
            _to_vsi("DODS://test.opendap.org/data.nc")
            == 'NETCDF:"https://test.opendap.org/data.nc"'
        )

    def test_dods_routed_through_parse_path(self):
        # The read path (read_file -> _parse_path -> _to_vsi) must yield the NETCDF: form.
        assert _io._parse_path("dods://h/x.nc") == 'NETCDF:"https://h/x.nc"'


class TestIsRemote:
    @pytest.mark.parametrize(
        "path",
        [
            "s3://bucket/key.tif",
            "gs://bucket/key.tif",
            "az://container/blob.tif",
            "abfs://container/blob.tif",
            "https://foo.com/x.tif",
            "http://foo.com/x.tif",
            "/vsis3/bucket/key.tif",
            "/vsicurl/https://foo/x.tif",
            "/vsimem/x.tif",
            "/vsizip/a.zip/b.tif",
            "dods://test.opendap.org/opendap/data/nc/coads.nc",
        ],
    )
    def test_true_cases(self, path):
        assert is_remote(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/data.tif",
            "C:/data/x.tif",
            "data/x.tif",
            "relative/path.tif",
            "./x.tif",
        ],
    )
    def test_false_cases(self, path):
        assert is_remote(path) is False


class TestIsNetworkBacked:
    """`is_network_backed` narrows `is_remote` to the sources that cross the network."""

    @pytest.mark.parametrize(
        "path, network",
        [
            ("/vsis3/bucket/key.tif", True),
            ("/vsigs/bucket/key.tif", True),
            ("/vsiaz/container/blob.tif", True),
            ("/vsicurl/https://foo/x.tif", True),
            ("/vsicurl?empty_dir=yes&url=https%3A%2F%2Fh%2Fa.tif", True),
            ("/vsicurl_streaming/https://foo/x.tif", True),
            ("/vsiswift/container/x.tif", True),
            ("s3://bucket/key.tif", True),
            ("abfs://container/blob.tif", True),
            ("dods://test.opendap.org/opendap/data/nc/coads.nc", True),
            ("/vsimem/x.tif", False),
            ("/vsizip/a.zip/b.tif", False),
            ("/vsigzip/a.gz", False),
            ("/vsitar/a.tar/b.tif", False),
        ],
    )
    def test_only_the_network_handlers_are_network_backed(self, path, network):
        """Every path here is remote; only the network-fetching ones are credentialed.

        Test scenario:
            The local virtual filesystems (`/vsimem/`, `/vsizip/`, `/vsigzip/`,
            `/vsitar/`) never authenticate, so a caller reasoning about credentials must
            not treat them like a cloud source — expected: `is_remote` is `True` for the
            whole table while `is_network_backed` splits it.
        """
        assert is_remote(path) is True, f"{path} should be remote"
        assert is_network_backed(path) is network, (
            f"is_network_backed({path!r}) should be {network}"
        )


class TestCloudConfigAsGdalConfig:
    def test_empty_default(self):
        assert CloudConfig().as_gdal_config() == {}

    def test_aws_full(self):
        cfg = CloudConfig(
            aws_access_key_id="AK",
            aws_secret_access_key="SEC",
            aws_session_token="TOK",
            aws_region="us-east-1",
        ).as_gdal_config()
        assert cfg == {
            "AWS_ACCESS_KEY_ID": "AK",
            "AWS_SECRET_ACCESS_KEY": "SEC",
            "AWS_SESSION_TOKEN": "TOK",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

    def test_skips_none_fields(self):
        cfg = CloudConfig(aws_region="eu-west-1").as_gdal_config()
        assert cfg == {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "eu-west-1"}

    def test_no_sign_request_true(self):
        assert CloudConfig(aws_no_sign_request=True).as_gdal_config() == {
            "AWS_NO_SIGN_REQUEST": "YES"
        }

    def test_no_sign_request_false_absent(self):
        assert (
            "AWS_NO_SIGN_REQUEST"
            not in CloudConfig(aws_no_sign_request=False).as_gdal_config()
        )

    def test_gs_fields(self):
        cfg = CloudConfig(
            gs_access_key_id="GA",
            gs_secret_access_key="GS",
        ).as_gdal_config()
        assert cfg == {"GS_ACCESS_KEY_ID": "GA", "GS_SECRET_ACCESS_KEY": "GS"}

    def test_azure_fields(self):
        cfg = CloudConfig(
            azure_storage_account="acct",
            azure_storage_access_key="key",
            azure_storage_sas_token="sas",
        ).as_gdal_config()
        assert cfg == {
            "AZURE_STORAGE_ACCOUNT": "acct",
            "AZURE_STORAGE_ACCESS_KEY": "key",
            "AZURE_STORAGE_SAS_TOKEN": "sas",
        }

    def test_extra_passthrough(self):
        cfg = CloudConfig(
            extra={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
        ).as_gdal_config()
        assert cfg == {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

    def test_extra_with_aws(self):
        cfg = CloudConfig(
            aws_region="us-east-1",
            extra={"VSI_CACHE": "TRUE"},
        ).as_gdal_config()
        assert cfg == {
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "VSI_CACHE": "TRUE",
        }

    def test_region_drives_both_keys_identically(self):
        """aws_region sets AWS_REGION and AWS_DEFAULT_REGION to the same value.

        Test scenario:
            A single aws_region field must emit both GDAL keys so an inherited
            AWS_DEFAULT_REGION env value cannot override an explicit region.
        """
        cfg = CloudConfig(aws_region="ap-south-1").as_gdal_config()
        assert cfg["AWS_REGION"] == "ap-south-1", f"AWS_REGION wrong: {cfg}"
        assert cfg["AWS_DEFAULT_REGION"] == "ap-south-1", (
            f"AWS_DEFAULT_REGION wrong: {cfg}"
        )

    def test_no_default_region_without_region(self):
        """AWS_DEFAULT_REGION is absent when aws_region is not provided.

        Test scenario:
            None-valued aws_region must drop both region keys (no empty leak).
        """
        cfg = CloudConfig(aws_no_sign_request=True).as_gdal_config()
        assert "AWS_REGION" not in cfg, f"unexpected AWS_REGION: {cfg}"
        assert "AWS_DEFAULT_REGION" not in cfg, f"unexpected AWS_DEFAULT_REGION: {cfg}"

    def test_extra_overrides_default_region(self):
        """An explicit extra AWS_DEFAULT_REGION wins over the aws_region-derived one.

        Test scenario:
            The extra escape hatch can decouple the two keys when a caller needs
            a different default region.
        """
        cfg = CloudConfig(
            aws_region="us-east-1",
            extra={"AWS_DEFAULT_REGION": "us-west-2"},
        ).as_gdal_config()
        assert cfg["AWS_REGION"] == "us-east-1", f"AWS_REGION wrong: {cfg}"
        assert cfg["AWS_DEFAULT_REGION"] == "us-west-2", f"extra did not win: {cfg}"


class TestCloudConfigContextManager:
    def test_enter_exit_no_options(self):
        with CloudConfig():
            pass  # no-op, should not raise

    def test_sets_options_inside_block(self):
        sentinel_key = "AWS_REGION"
        # Ensure it's not set ambiently
        gdal.SetConfigOption(sentinel_key, None)

        with CloudConfig(aws_region="us-east-2"):
            assert gdal.GetConfigOption(sentinel_key) == "us-east-2"

    def test_restores_previous_value_on_exit(self):
        key = "AWS_REGION"
        gdal.SetConfigOption(key, "us-west-2")
        try:
            with CloudConfig(aws_region="us-east-1"):
                assert gdal.GetConfigOption(key) == "us-east-1"
            assert gdal.GetConfigOption(key) == "us-west-2"
        finally:
            gdal.SetConfigOption(key, None)

    def test_restores_on_exception(self):
        key = "AWS_REGION"
        gdal.SetConfigOption(key, "before")
        cfg = CloudConfig(aws_region="during")
        # The in-context "during" value is covered by
        # test_restores_previous_value_on_exit; this test only asserts restoration
        # when the body raises, so the pytest.raises block holds a single raiser.
        try:
            with pytest.raises(RuntimeError):
                with cfg:
                    raise RuntimeError("boom")
            assert gdal.GetConfigOption(key) == "before"
        finally:
            gdal.SetConfigOption(key, None)

    def test_no_sign_request_applied(self):
        gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", None)
        with CloudConfig(aws_no_sign_request=True):
            assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") == "YES"
        assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") is None

    def test_extra_applied(self):
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", None)
        with CloudConfig(extra={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}):
            assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
        assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") is None


# ---------------------------------------------------------------------------
# End-to-end cloud I/O via a local HTTP server (Task 12)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHttpCogRead:
    """Read a COG served over HTTP through the /vsicurl/ rewrite."""

    def test_read_cog_over_http(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        assert ds.rows == 256
        assert ds.columns == 256
        assert ds.epsg == 4326

    def test_read_cog_over_http_is_cog_true(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        # is_cog works over VSI paths too
        assert ds.is_cog is True

    def test_read_cog_over_http_array_matches(self, http_server):
        from pyramids.dataset import Dataset

        url = f"{http_server}/valid.tif"
        ds = Dataset.read_file(url)
        arr = ds.read_array()
        expected = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        assert np.array_equal(arr, expected)

    def test_windowed_bbox_crop_over_http_matches_local(
        self, http_server, http_cog_dir, mocker
    ):
        """#957: a windowed bbox crop over /vsicurl reads only the AOI and equals the local crop.

        The 256x256 COG spans x in [0, 0.256], y in [-0.256, 0] at 0.001 deg cells.
        Cropping a small sub-bbox straight from the /vsicurl URL must (a) read the remote
        source with just the AOI pixel window (never a full-source read — that is what makes
        a range-read URL fetch only the AOI) and (b) return the same shape, geotransform and
        pixels as cropping the local file.
        """
        from pyramids.dataset import Dataset

        bbox = (
            0.1,
            -0.11,
            0.15,
            -0.06,
        )  # cols 100:150, rows 60:110 -> window [100,60,50,50]
        remote_ds = Dataset.read_file(f"{http_server}/valid.tif")
        spy = mocker.spy(Dataset, "read_array")
        remote = remote_ds.crop(bbox=bbox)
        source_windows = [
            kw.get("window", (a[1] if len(a) > 1 else None))
            for a, kw in spy.call_args_list
            if a and a[0] is remote_ds
        ]
        assert source_windows == [[100, 60, 50, 50]], (
            f"remote source must be read once, windowed to the AOI; saw {source_windows}"
        )

        local = Dataset.read_file(str(http_cog_dir / "valid.tif")).crop(bbox=bbox)
        assert remote.shape == local.shape, (
            f"remote crop shape {remote.shape} != local {local.shape}"
        )
        assert remote.geotransform == local.geotransform, (
            "remote crop geotransform differs from the local crop"
        )
        assert np.array_equal(remote.read_array(), local.read_array()), (
            "remote crop pixels differ from the local crop"
        )

    def test_read_plain_gtiff_over_http_may_require_range(self, http_server):
        """Plain (stripped) GTiff needs byte-range requests.

        Python's stdlib `http.server` does not support HTTP Range,
        so GDAL will fail for files that cannot be read sequentially.
        We only assert that *if* it raises, the URL was rewritten to
        /vsicurl/ first — i.e., the pipeline is correct even if the
        fixture HTTP server can't serve it.
        """
        from pyramids.dataset import Dataset

        url = f"{http_server}/plain.tif"
        try:
            ds = Dataset.read_file(url)
            assert ds.rows == 256
        except RuntimeError as exc:
            # Expected: stdlib HTTP server lacks Range support.
            assert "Range" in str(exc) or "range" in str(exc)


class TestS3UrlRewriteNoNetwork:
    """Verify string-level rewriting for s3:// without hitting the network."""

    def test_s3_rewrite_attempt_reaches_gdal(self):
        from pyramids.dataset import Dataset

        # We only care that pyramids doesn't raise ValueError about
        # unknown schemes, and that the rewrite reached /vsis3/.
        # GDAL raises RuntimeError or OSError for missing S3 resources.
        with pytest.raises((RuntimeError, OSError)):
            Dataset.read_file("s3://nonexistent-bucket-xyz-1234/nope.tif")

    def test_pipeline_uses_vsis3_prefix(self):
        from pyramids._io import _parse_path

        assert _parse_path("s3://b/k.tif") == "/vsis3/b/k.tif"


class TestToVsiArchiveChaining:
    """Tests for `_to_vsi`'s archive-chaining behavior.

    Named after the public entry point (`_to_vsi`) rather than the
    internal helper (`_chain_archive_vsi`) so the test class name
    remains stable if the helper is later renamed or inlined.

    Covers the pre-existing gap where
    `https://host/archive.tar/inner.tif` -> `/vsicurl/...` lost
    access to the inner file because GDAL needs `/vsitar//vsicurl/...`.
    """

    def test_tar_inside_https(self):
        """HTTPS URL pointing into .tar archive gets /vsitar/ prefix."""
        url = "https://example.com/archive.tar/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsitar//vsicurl/https://example.com/archive.tar/inner.tif", (
            f"Expected chained /vsitar/ + /vsicurl/, got: {result}"
        )

    def test_zip_inside_s3(self):
        """S3 URL pointing into .zip archive gets /vsizip/ prefix."""
        url = "s3://bucket/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsizip//vsis3/bucket/archive.zip/inner.tif", (
            f"Expected chained /vsizip/ + /vsis3/, got: {result}"
        )

    def test_gz_inside_https(self):
        """HTTPS URL pointing into .gz file gets /vsigzip/ prefix."""
        url = "https://example.com/data.gz/inner.asc"
        result = _to_vsi(url)
        assert result == "/vsigzip//vsicurl/https://example.com/data.gz/inner.asc", (
            f"Expected chained /vsigzip/ + /vsicurl/, got: {result}"
        )

    def test_tar_gz_inside_https(self):
        """HTTPS URL pointing into .tar.gz archive routes via /vsitar/."""
        url = "https://example.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert "/vsitar/" in result, (
            f".tar.gz must route through /vsitar/, got: {result}"
        )

    def test_tgz_inside_gs(self):
        """GCS URL pointing into .tgz archive routes via /vsitar/."""
        url = "gs://bucket/data.tgz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsigs/"), (
            f".tgz must chain /vsitar/ + /vsigs/, got: {result}"
        )

    def test_plain_tif_over_https_no_chain(self):
        """URL without an archive segment is not chained."""
        url = "https://example.com/scene.tif"
        result = _to_vsi(url)
        assert result == "/vsicurl/https://example.com/scene.tif", (
            f"Non-archive URL must not chain; got: {result}"
        )

    def test_archive_named_in_url_but_not_traversed(self):
        """URL ending at archive name (no trailing /) is not chained.

        Test scenario:
            If the user points at the archive file itself rather than
            a member inside it, GDAL can download and inspect the
            archive — no chained VSI needed.
        """
        url = "https://example.com/archive.tar"
        result = _to_vsi(url)
        assert result == "/vsicurl/https://example.com/archive.tar", (
            f"Archive-name-only URL must not chain; got: {result}"
        )

    def test_local_zip_path_unchanged_by_chain(self):
        """Local .zip/foo.tif is left for pyramids._io._parse_path to handle."""
        p = "/local/path/archive.zip/inner.tif"
        result = _to_vsi(p)
        assert result == p, (
            f"Local archive paths must be left to _parse_path, got: {result}"
        )


class TestCloudConfigEntryState:
    """CloudConfig's per-entry state is private, transient and not compared."""

    def test_no_entry_is_recorded_before_enter(self):
        """A fresh CloudConfig holds no active entry."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert cfg._entries == [], (
            f"a fresh CloudConfig must hold no entry, got: {cfg._entries!r}"
        )

    def test_the_entry_is_dropped_after_exit(self):
        """After `__exit__` no context reference lingers on the instance.

        Test scenario:
            The state is per *entry* rather than per instance, so the same config
            can be re-entered while open; the invariant the old `_ctx` field
            pinned still holds — nothing is retained once every block has exited.
        """
        cfg = CloudConfig(aws_region="us-east-1")
        with cfg:
            assert len(cfg._entries) == 1, "an entry must be recorded inside the block"
        assert cfg._entries == [], (
            f"entries must be dropped after exit, got: {cfg._entries!r}"
        )

    def test_entries_not_in_repr(self):
        """The entry stack is declared repr=False so it does not leak into repr()."""
        cfg = CloudConfig(aws_region="us-east-1")
        assert "_entries" not in repr(cfg), (
            f"_entries must be excluded from repr; got: {repr(cfg)}"
        )

    def test_entries_not_in_equality_comparison(self):
        """The entry stack is compare=False so __eq__ still works across blocks."""
        a = CloudConfig(aws_region="us-east-1")
        b = CloudConfig(aws_region="us-east-1")
        with a:
            assert a == b, (
                f"CloudConfigs with equal public fields must compare equal "
                f"regardless of _ctx state; got a={a!r}, b={b!r}"
            )


class TestToVsiArchiveChainingEdgeCases:
    """M1 (2nd review): boundary-anchored archive marker detection.

    These scenarios used to FALSE-POSITIVE under the substring-only
    implementation: query-string injection, hostname ending in an
    archive extension, and nested archives. The boundary-anchored
    regex + path-component extraction in `_extract_archive_search_region`
    must correctly handle them.
    """

    def test_query_string_with_dot_tar_not_chained(self):
        """Presigned URL containing `.tar/` in the query must not chain.

        Test scenario:
            A presigned URL may embed `archive.tar` in the query
            value (e.g. as part of a signed key). The file itself is
            a plain GeoTIFF; prepending /vsitar/ would break the read.
        """
        url = "https://foo.com/x.tif?key=archive.tar/inner&sig=abc"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .tar/ must not trigger archive chaining; got: {result}"
        )

    def test_query_string_with_dot_zip_not_chained(self):
        """Same protection for .zip inside a query string."""
        url = "https://foo.com/scene.tif?asset=pkg.zip/inner.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .zip/ must not trigger archive chaining; got: {result}"
        )

    def test_query_string_with_dot_gz_not_chained(self):
        """Same protection for .gz inside a query string."""
        url = "https://foo.com/scene.tif?backup=data.gz/inner"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Query-string .gz/ must not trigger archive chaining; got: {result}"
        )

    def test_hostname_ending_in_gz_not_chained(self):
        """Hostname whose label ends in .gz must not trigger archive chaining.

        Test scenario:
            A hostname like `data.gz.example.com` or `weird.gz` is
            legitimate and unrelated to gzip archives. The URL's path
            component is the only authoritative source for archive
            markers.
        """
        url = "https://weird.gz/file.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Hostname ending in .gz must not trigger archive chaining; got: {result}"
        )

    def test_hostname_with_dot_tar_not_chained(self):
        """Hostname containing .tar (e.g. tar.example.com) is not an archive."""
        url = "https://tar.example.com/file.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f"Hostname containing .tar must not trigger archive chaining; got: {result}"
        )

    def test_nested_outer_archive_wins(self):
        """Nested archive path only applies the outermost archive prefix.

        Test scenario:
            `outer.zip/inner.tar/file.tif` — only the OUTER.zip
            marker is honored; GDAL's chained-VSI syntax doesn't
            compose through arbitrary nesting, so silently applying
            /vsitar//vsizip/... would produce un-openable paths in
            most real cases. Documented single-layer limitation.
        """
        url = "https://foo.com/outer.zip/inner.tar/file.tif"
        result = _to_vsi(url)
        expected = "/vsizip//vsicurl/https://foo.com/outer.zip/inner.tar/file.tif"
        assert result == expected, (
            f"Nested archive: only outermost (.zip) should chain; got: {result}"
        )

    def test_path_with_dot_tar_in_directory_name_chained(self):
        """A legitimate `archive.tar/` segment in the path IS chained.

        Test scenario:
            Regression guard that the boundary-anchored regex still
            catches the common case: a path segment named
            `something.tar/` points INTO an archive and must be
            chained.
        """
        url = "https://foo.com/path/archive.tar/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsicurl/"), (
            f"Legitimate archive segment must chain; got: {result}"
        )

    def test_s3_key_with_dot_zip_segment_chained(self):
        """S3 key that traverses a .zip segment is correctly chained."""
        url = "s3://bucket/folder/archive.zip/inner.tif"
        result = _to_vsi(url)
        assert result == "/vsizip//vsis3/bucket/folder/archive.zip/inner.tif", (
            f"S3 archive segment must chain; got: {result}"
        )

    def test_tar_gz_prefers_vsitar_over_vsigzip(self):
        """.tar.gz/ must match before .gz/ in the regex alternation."""
        url = "https://foo.com/archive.tar.gz/inner.tif"
        result = _to_vsi(url)
        assert result.startswith("/vsitar//vsicurl/"), (
            f".tar.gz/ must route through /vsitar/, got: {result}"
        )
        assert not result.startswith("/vsigzip/"), (
            f".tar.gz/ must NOT route through /vsigzip/, got: {result}"
        )

    def test_uppercase_archive_extension_matched(self):
        """Case-insensitive matching — `.ZIP/` is treated like `.zip/`."""
        url = "https://foo.com/ARCHIVE.ZIP/INNER.TIF"
        result = _to_vsi(url)
        assert result.startswith("/vsizip//vsicurl/"), (
            f"Uppercase .ZIP/ must still chain; got: {result}"
        )

    def test_non_archive_extension_not_chained(self):
        """Non-archive extensions at path boundaries are not chained."""
        url = "https://foo.com/container.tif/inner.tif"
        result = _to_vsi(url)
        assert result == f"/vsicurl/{url}", (
            f".tif/ is not an archive; must not chain; got: {result}"
        )


@pytest.mark.slow
@pytest.mark.live
class TestLiveOpenDAP:
    """Read a real OPeNDAP/THREDDS dataset over dods:// via GDAL's netCDF DAP support."""

    URL = os.environ.get(
        "PYRAMIDS_OPENDAP_URL",
        "dods://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface/air.sig995.2012.nc",
    )
    VAR = os.environ.get("PYRAMIDS_OPENDAP_VAR", "air")

    def test_read_opendap_schema(self):
        """A dods:// URL opens as a NetCDF and its variable schema is read without a full download."""
        nc = NetCDF.read_file(self.URL)
        assert self.VAR in list(nc.variables), (
            f"{self.VAR!r} not in {list(nc.variables)[:8]}"
        )
        variable = nc.get_variable(self.VAR)
        assert variable.shape[0] >= 1


class TestRedactCredentials:
    """Credential values must not survive into a log line or message."""

    def test_embedded_header_value_is_blanked(self):
        """An embedded bearer header is replaced, the rest stays readable."""
        text = "Can't open /vsicurl?header.Authorization=Bearer%20SECRET&url=x"
        out = redact_credentials(text)
        assert "SECRET" not in out, f"token survived: {out}"
        assert "header.Authorization=<redacted>" in out, out

    def test_space_separated_scheme_and_token_is_blanked_whole(self):
        """An unencoded `Bearer <token>` loses the token, not just the scheme.

        Test scenario:
            The value pattern stops at whitespace, which is right for a query
            option but left the secret in an HTTP header value: `Bearer abc123`
            redacted only the word `Bearer` and printed the token. A leading
            auth scheme is now consumed together with what follows it.
        """
        text = "Can't open /vsicurl?header.Authorization=Bearer abc123SECRET"
        out = redact_credentials(text)
        assert "abc123SECRET" not in out, f"token survived: {out}"
        assert "header.Authorization=<redacted>" in out, out

    @pytest.mark.parametrize("scheme", ["Bearer", "Basic", "Digest", "Token"])
    def test_every_auth_scheme_is_consumed_with_its_token(self, scheme):
        """Each recognised scheme takes its credential with it.

        Args:
            scheme: An HTTP auth scheme that precedes the credential.
        """
        out = redact_credentials(
            f"/vsicurl?header.Authorization={scheme} SUPERSECRET&url=https://h/a.tif"
        )
        assert "SUPERSECRET" not in out, f"{scheme} token survived: {out}"
        assert "url=https://h/a.tif" in out, f"the rest was mangled: {out}"

    def test_a_scheme_word_does_not_swallow_following_prose(self):
        """Redaction stops at the next space, so a trailing sentence survives."""
        out = redact_credentials(
            "Can't open /vsicurl?header.Authorization=Bearer abc123. Skipping it"
        )
        assert "abc123" not in out, f"token survived: {out}"
        assert out.endswith(" Skipping it"), f"prose was swallowed: {out}"

    def test_sas_query_parameter_is_blanked(self):
        """A SAS-style `sig=` parameter is caught as well."""
        out = redact_credentials("https://h/a.tif?sv=2021&sig=SECRET")
        assert out == "https://h/a.tif?sv=2021&sig=<redacted>", out

    def test_ordinary_message_is_untouched(self):
        """A message with no credential is returned verbatim."""
        text = "Can't open /vsicurl/https://h/a.tif. Skipping it"
        assert redact_credentials(text) == text, "an innocent message was altered"

    def test_multiple_headers_all_blanked(self):
        """Every credential option in one string is redacted."""
        text = "/vsicurl?header.Authorization=Bearer%20A&header.X-Api-Key=B&url=x"
        out = redact_credentials(text)
        assert "Bearer%20A" not in out and "=B&" not in out, f"a value survived: {out}"

    def test_name_merely_ending_in_a_credential_word_is_kept(self):
        """A name that only ends with a credential word is not one.

        The previous `\b`-anchored pattern matched the tail of `my_token=` and
        `bucket-key=`, redacting values that were never secrets.
        """
        text = "https://h/a.tif?my_token=abc&bucket-key=def"
        assert redact_credentials(text) == text, (
            f"a non-credential was redacted: {text}"
        )

    def test_credential_after_an_ampersand_is_redacted(self):
        """A credential option later in the query is still caught."""
        out = redact_credentials("https://h/a.tif?sv=2021&sig=SECRET&sp=r")
        assert "SECRET" not in out, f"token survived: {out}"
        assert "sv=2021" in out and "sp=r" in out, f"neighbours mangled: {out}"

    def test_is_remote_accepts_the_query_form(self):
        """The `/vsicurl?` form is classified as remote like `/vsicurl/` is."""
        assert is_remote("/vsicurl?empty_dir=yes&url=https%3A%2F%2Fh%2Fa.tif"), (
            "the query form must be recognised as remote"
        )


class TestHandlerTableInvariant:
    """Tests for the relationship between the four handler tables (#918).

    The tables answer different questions and are not copies of one list, so
    the thing worth pinning is how they relate: `cloud` subset of `network`
    subset of `vsi`, and no purely local handler in the network set.
    """

    def test_network_is_a_subset_of_vsi(self):
        """Every network handler is recognised as a VSI path at all."""
        missing = set(_NETWORK_VSI_PREFIXES) - set(_VSI_PREFIXES)
        assert not missing, f"network handlers absent from _VSI_PREFIXES: {missing}"

    def test_cloud_is_a_subset_of_network(self):
        """Archive chaining is only offered for handlers that touch the network."""
        missing = set(_CLOUD_VSI_PREFIXES) - set(_NETWORK_VSI_PREFIXES)
        assert not missing, f"chaining handlers that are not network-backed: {missing}"

    @pytest.mark.parametrize(
        "prefix", ["/vsimem/", "/vsizip/", "/vsitar/", "/vsigzip/"]
    )
    def test_local_handlers_are_not_network_backed(self, prefix: str):
        """An in-memory or archive handler reads no network.

        Args:
            prefix: A purely local VSI prefix.
        """
        assert prefix in _VSI_PREFIXES, f"{prefix} should be a known VSI prefix"
        assert prefix not in _NETWORK_VSI_PREFIXES, f"{prefix} is not network-backed"

    def test_every_url_scheme_is_actually_rewritten(self):
        """No scheme in the map may fall through `_to_vsi` unchanged.

        The real risk the derivation guards against: a scheme added to
        `URL_SCHEMES` whose prefix no branch of `_to_vsi` handles would be
        returned as-is and handed to GDAL raw. Asserting on the rewrite is what
        catches that; asserting that the derived set matches its own derivation
        cannot fail.
        """
        for scheme in URL_SCHEMES:
            rewritten = _to_vsi(f"{scheme}://container/key.tif")
            assert rewritten != f"{scheme}://container/key.tif", (
                f"{scheme}:// is in URL_SCHEMES but _to_vsi leaves it unchanged"
            )

    @pytest.mark.parametrize(
        "prefix",
        [
            "/vsis3_streaming/",
            "/vsigs_streaming/",
            "/vsiaz_streaming/",
            "/vsioss_streaming/",
            "/vsiswift_streaming/",
        ],
    )
    def test_streaming_handlers_are_network_backed(self, prefix: str):
        """GDAL's `_streaming` twins read the same services over the network.

        Args:
            prefix: A streaming VSI prefix.

        Test scenario:
            Only `/vsicurl_streaming/` was listed, so the five object-store
            streaming handlers classified as local files.
        """
        path = f"{prefix}bucket/key.tif"
        assert is_remote(path), f"{prefix} must be remote"
        assert is_network_backed(path), f"{prefix} must be network-backed"

    def test_streaming_handlers_do_not_chain_archives(self):
        """Their remainder is an option list, so an archive marker is untrustworthy."""
        path = "/vsis3_streaming/b/a.zip/x.tif"
        assert _to_vsi(path) == path, "streaming handlers must not chain"


class TestAdlsHandler:
    """Tests for the ADLS Gen2 handler recognised in #918."""

    ADLS_PATH = "/vsiadls/container/x.tif"

    def test_is_remote(self):
        """A Gen2 path is remote, where it used to read as a local file."""
        assert is_remote(self.ADLS_PATH), "an ADLS path must be recognised as remote"

    def test_is_network_backed(self):
        """It is network-backed too, so credential reasoning applies."""
        assert is_network_backed(self.ADLS_PATH), "an ADLS path is network-backed"

    def test_chains_an_archive(self):
        """A zipped raster on Gen2 is rewritten the way the S3 equivalent is."""
        chained = _to_vsi("/vsiadls/c/a.zip/x.tif")
        assert chained == "/vsizip//vsiadls/c/a.zip/x.tif", chained


class TestNetworkHandlersChainArchives:
    """Tests that every network handler may chain an archive (#918).

    `_CLOUD_VSI_PREFIXES` used to be s3/gs/az/curl only, so a zipped raster on
    Alibaba OSS, OpenStack Swift or HDFS was never rewritten.

    These assert the path *rewrite* only. `/vsihdfs/` and `/vsiwebhdfs/` need a
    GDAL built with HDFS support and are not registered in every build (the
    project's own is one that lacks them), so nothing here opens a dataset.
    """

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/vsioss/c/a.zip/x.tif", "/vsizip//vsioss/c/a.zip/x.tif"),
            ("/vsiswift/c/a.zip/x.tif", "/vsizip//vsiswift/c/a.zip/x.tif"),
            (
                "/vsihdfs/hdfs://nn:8020/d/a.zip/x.tif",
                "/vsizip//vsihdfs/hdfs://nn:8020/d/a.zip/x.tif",
            ),
            (
                "/vsiwebhdfs/http://h:50070/webhdfs/v1/a.tar/x.tif",
                "/vsitar//vsiwebhdfs/http://h:50070/webhdfs/v1/a.tar/x.tif",
            ),
        ],
    )
    def test_chains(self, path: str, expected: str):
        """Each network handler gets the archive prefix it needs.

        Args:
            path: A VSI path pointing inside an archive.
            expected: The chained form GDAL needs to read the inner file.
        """
        assert _to_vsi(path) == expected, (
            "chaining must be reachable through the public rewrite"
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/vsihdfs/hdfs://data.gz/x.tif",
            "/vsiwebhdfs/http://data.zip/webhdfs/v1/x.tif",
        ],
    )
    def test_a_hostname_that_looks_like_an_archive_does_not_chain(self, path: str):
        """The URL-embedding handlers strip the hostname before searching.

        Args:
            path: A path whose *hostname* ends in an archive extension.

        Test scenario:
            `/vsihdfs/` and `/vsiwebhdfs/` embed a full URL after the prefix, so
            scanning the raw remainder would let a host named `data.gz` trigger
            chaining — the same trap `/vsicurl/` already guarded against.
        """
        assert _to_vsi(path) == path, "a hostname must not trigger chaining"


class TestCredentialFamilies:
    """Tests for `_family_of`, the rule the deferral's atomicity rests on.

    The mechanism that caused one round's regression (an explicit requester-pays
    block hiding an ambient identity) shipped with no test for what belongs in a
    family. These pin the rule rather than the instances.
    """

    @pytest.mark.parametrize(
        "key",
        [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_NO_SIGN_REQUEST",
        ],
    )
    def test_an_identity_key_belongs_to_the_aws_family(self, key):
        """Everything that decides who is reading defers together.

        Args:
            key: An AWS option that forms part of an identity.
        """
        family = _family_of(key)
        assert family is not None, f"{key} should belong to a family"
        assert "AWS_ACCESS_KEY_ID" in family

    @pytest.mark.parametrize(
        "key",
        [
            "AWS_REQUEST_PAYER",
            "GS_USER_PROJECT",
            "AWS_S3_ENDPOINT",
            "AWS_REGION",
            "AWS_VIRTUAL_HOSTING",
            "GDAL_HTTP_TIMEOUT",
        ],
    )
    def test_a_non_identity_key_stands_alone(self, key):
        """Billing, routing and transport defer key by key, not as a set.

        Args:
            key: An option that says where or how to read, not who is reading.

        Test scenario:
            Treating `AWS_REQUEST_PAYER` as part of the identity let an explicit
            requester-pays block claim the whole family and withhold the ambient
            credentials, so the read went out unsigned.
        """
        assert _family_of(key) is None, f"{key} should not claim an identity"

    def test_the_families_do_not_overlap(self):
        """A key belongs to at most one family, so a skip is unambiguous."""
        seen: dict[str, int] = {}
        for index, family in enumerate(_CREDENTIAL_FAMILIES):
            for key in family:
                assert key not in seen, f"{key} is in two families"
                seen[key] = index

    def test_every_family_key_is_path_scoped(self):
        """A family key that is not path-scoped could never be claimed.

        Test scenario:
            The claim set is built from the options actually pushed for a prefix,
            and only `_PATH_SCOPED_KEYS` are pushed. A family key outside that
            set would silently never participate.
        """
        orphans = sorted(
            key
            for family in _CREDENTIAL_FAMILIES
            for key in family
            if key not in _PATH_SCOPED_KEYS
        )
        assert orphans == [], f"family key(s) that are never path-scoped: {orphans}"


class TestCollectionCredentialGate:
    """Tests for the collection's per-timestep credential gate."""

    @staticmethod
    def _collection(files, env=None):
        """Build a collection over `files` with an optional explicit env."""
        from pyramids.dataset import Dataset
        from pyramids.dataset.collection import DatasetCollection

        template = Dataset(gdal.GetDriverByName("MEM").Create("m.tif", 4, 3, 1))
        return DatasetCollection(template, len(files), files=files, gdal_env=env)

    def test_a_captured_env_reaches_only_the_remote_timesteps(self):
        """A local timestep in a mixed collection gets no cloud secret."""
        local, remote = "local.tif", "/vsis3/eodata/remote.tif"
        with CloudConfig(extra={"AWS_SECRET_ACCESS_KEY": "SECRET"}):
            collection = self._collection([local, remote])
        assert collection._env_for(local) is None, "a local timestep got the secret"
        assert (collection._env_for(remote) or {}).get("AWS_SECRET_ACCESS_KEY") == (
            "SECRET"
        )

    def test_an_explicit_env_reaches_every_timestep(self):
        """What the caller passed is an instruction, not an inference.

        Test scenario:
            `from_files` supplies a behaviour-only env, and a local timestep
            still needs it — gating that was what broke the directory-scan
            default.
        """
        local = "local.tif"
        collection = self._collection(
            [local, "/vsis3/eodata/remote.tif"],
            env={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"},
        )
        assert (collection._env_for(local) or {}).get(
            "GDAL_DISABLE_READDIR_ON_OPEN"
        ) == "EMPTY_DIR"

    def test_a_bare_string_is_refused(self):
        """`files="a.tif"` is a mistake, not a one-file collection."""
        with pytest.raises(TypeError, match="sequence of paths"):
            self._collection("a.tif")

    def test_a_memory_handle_with_a_bucket_name_refuses_to_pickle(self):
        """A cosmetic bucket name must not put a captured secret in a recipe.

        Test scenario:
            The gate reads `GetDescription()`, which is a name rather than a
            source, so a MEM handle created with a `/vsis3/...` name captures.
            Its recipe could never be reopened anyway, and it carried the
            credentials into whatever the recipe was shipped to.
        """
        from pyramids.dataset import Dataset

        handle = gdal.GetDriverByName("MEM").Create("/vsis3/eodata/fake.tif", 4, 3, 1)
        with CloudConfig(extra={"AWS_SECRET_ACCESS_KEY": "SECRET"}):
            dataset = Dataset(handle)
        assert dataset.gdal_env.get("AWS_SECRET_ACCESS_KEY") == "SECRET"
        with pytest.raises(TypeError, match="no on-disk path"):
            dataset.__reduce__()

    def test_a_windows_path_is_recognised_as_remote(self):
        """A `Path` timestep must not read as local because of separators."""
        from pyramids.dataset.collection import _as_source

        assert _bucket_prefix(_as_source(PureWindowsPath("/vsis3/eodata/x.tif"))) == (
            "/vsis3/eodata/"
        )


class TestCloudConfigWorkerThreadVisibility:
    """Tests that a store's credentials reach GDAL's VSICurl worker threads (#983).

    `CloudConfig` applied its options thread-locally, and the worker threads that
    fetch the byte-ranges of a large read do not inherit thread-local config, so
    a custom `AWS_S3_ENDPOINT` never reached them and a non-AWS store 403'd. The
    fix applies the credential/endpoint subset per-bucket via path-specific
    options, which are worker-visible.
    """

    ENV = {
        "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
        "AWS_ACCESS_KEY_ID": "CDSEKEY",
        # A secret the pickle-recipe assertions can actually look for: without
        # one in the fixture those assertions were vacuously true.
        "AWS_SECRET_ACCESS_KEY": "CDSESECRET",
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "GDAL_HTTP_TIMEOUT": "60",
    }
    BUCKET = "/vsis3/eodata"
    OBJECT = "/vsis3/eodata/S2/B04_10m.jp2"

    @staticmethod
    def _worker_path_option(path: str, key: str):
        """Read a path-specific option from a separate (non-calling) thread."""
        result: dict[str, str | None] = {}
        thread = threading.Thread(
            target=lambda: result.__setitem__(
                "v", gdal.GetPathSpecificOption(path, key, None)
            )
        )
        thread.start()
        thread.join()
        return result["v"]

    @staticmethod
    def _worker_global_option(key: str):
        """Read a global config option from a separate thread."""
        result: dict[str, str | None] = {}
        thread = threading.Thread(
            target=lambda: result.__setitem__("v", gdal.GetConfigOption(key))
        )
        thread.start()
        thread.join()
        return result["v"]

    def test_a_worker_thread_sees_the_endpoint_and_key(self):
        """The credential/endpoint subset is visible off the calling thread.

        Test scenario:
            This is the bug: before the fix a worker read these back as `None`
            and fell through to real AWS.
        """
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            )
            assert (
                self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID") == "CDSEKEY"
            )

    def test_credentials_do_not_leak_to_another_bucket(self):
        """A different bucket does not inherit this bucket's credentials."""
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            assert (
                self._worker_path_option("/vsis3/other/x.tif", "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            )

    @pytest.mark.parametrize(
        "sibling",
        [
            "/vsis3/eodata2/b.jp2",
            "/vsis3/eodata-public/c.jp2",
            "/vsis3/eodataXYZ/d.jp2",
        ],
    )
    def test_credentials_do_not_leak_to_a_prefix_sharing_sibling(self, sibling: str):
        """A bucket whose name merely starts the same does not inherit the key.

        Args:
            sibling: A bucket sharing `eodata`'s name prefix but not the bucket.

        Test scenario:
            GDAL matches path-specific options by raw string prefix, so a key of
            `/vsis3/eodata` (no trailing slash) would leak `eodata`'s endpoint and
            secret to `/vsis3/eodata2/...`. The trailing slash prevents it.
        """
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            assert (
                self._worker_path_option(sibling, "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            )
            assert self._worker_path_option(sibling, "AWS_ACCESS_KEY_ID") != "CDSEKEY"

    def test_two_threads_reading_different_buckets_do_not_collide(self):
        """Concurrent reads of different buckets each see only their own config.

        Test scenario:
            Path-specific options are process-global, so the isolation must come
            from distinct bucket keys, not from thread-locality.
        """
        seen: dict[str, str | None] = {}
        barrier = threading.Barrier(2, timeout=10)

        def read(bucket: str, endpoint: str, tag: str):
            with CloudConfig(extra={"AWS_S3_ENDPOINT": endpoint}, path=bucket):
                barrier.wait()
                seen[tag] = self._worker_path_option(bucket, "AWS_S3_ENDPOINT")

        threads = [
            threading.Thread(target=read, args=("/vsis3/alpha/x", "alpha-ep", "a")),
            threading.Thread(target=read, args=("/vsis3/beta/x", "beta-ep", "b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == {"a": "alpha-ep", "b": "beta-ep"}

    def test_an_archive_chained_path_scopes_the_underlying_bucket(self):
        """A `/vsizip//vsis3/bucket/...` path scopes on the real store."""
        chained = "/vsizip//vsis3/eodata/a.zip/x.tif"
        with CloudConfig(extra=self.ENV, path=chained):
            assert (
                self._worker_path_option("/vsis3/eodata/a.zip/x.tif", "AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            )

    def test_a_raising_config_restore_does_not_strand_the_secret(self, monkeypatch):
        """`__exit__` pops the path options even if the config restore raises.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            The exit-path mirror of the `__enter__` rollback: a GDAL restore
            failure must not leave a per-bucket secret live in the process-global
            path state.
        """
        import pyramids.base.remote as remote

        class _BoomExit:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                raise RuntimeError("restore boom")

        monkeypatch.setattr(remote.gdal, "config_options", lambda cfg: _BoomExit())
        with pytest.raises(RuntimeError, match="restore boom"):
            with CloudConfig(extra={"AWS_SECRET_ACCESS_KEY": "S"}, path=self.OBJECT):
                pass
        monkeypatch.undo()
        assert (
            gdal.GetPathSpecificOption(self.OBJECT, "AWS_SECRET_ACCESS_KEY", "CLEAN")
            != "S"
        )
        assert not remote._PATH_OPTION_STACK

    def test_a_credential_provider_key_stays_thread_local(self):
        """`AWS_PROFILE` is not path-scoped — GDAL reads it via the global getter.

        Test scenario:
            Path-scoping a credential-provider key would hide it both ways, since
            a path-specific option is invisible to `gdal.GetConfigOption`.
        """
        with CloudConfig(extra={"AWS_PROFILE": "myprofile"}, path=self.OBJECT):
            assert gdal.GetConfigOption("AWS_PROFILE") == "myprofile"
            assert self._worker_path_option(self.OBJECT, "AWS_PROFILE") is None

    def test_options_are_unset_on_exit(self):
        """The path-specific options do not outlive the block."""
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            pass
        assert (
            gdal.GetPathSpecificOption(self.OBJECT, "AWS_S3_ENDPOINT", "GONE") == "GONE"
        )

    def test_a_nested_same_bucket_block_does_not_strip_the_outer(self):
        """An inner block's exit restores the outer block's value, not `None`.

        Test scenario:
            Wrapped engine methods can call each other on the same dataset, so a
            same-bucket `CloudConfig` can nest. A naive unset-on-exit would leave
            the outer read with no credentials.
        """
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            with CloudConfig(extra=self.ENV, path=self.OBJECT):
                pass
            assert (
                gdal.GetPathSpecificOption(self.OBJECT, "AWS_S3_ENDPOINT", "STRIPPED")
                == "eodata.dataspace.copernicus.eu"
            )
        assert (
            gdal.GetPathSpecificOption(self.OBJECT, "AWS_S3_ENDPOINT", "GONE") == "GONE"
        )

    def test_http_knobs_stay_thread_local(self):
        """A transport knob is not path-scoped — it has no bucket to attach to."""
        with CloudConfig(extra=self.ENV, path=self.OBJECT):
            assert self._worker_path_option(self.OBJECT, "GDAL_HTTP_TIMEOUT") is None, (
                "GDAL_HTTP_TIMEOUT must not become a per-bucket credential"
            )
            assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "60"

    def test_without_a_path_the_thread_local_copy_is_kept(self):
        """No path still leaves the options readable on the calling thread.

        Test scenario:
            `_configured_azure_account` and friends read credential keys back
            through `gdal.GetConfigOption`, which never sees a path-specific
            option, so a pathless block must keep its thread-local copy.
        """
        with CloudConfig(extra=self.ENV):
            assert self._worker_global_option("AWS_S3_ENDPOINT") is None
            assert (
                gdal.GetConfigOption("AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            )

    def test_a_pathless_block_publishes_nothing_process_wide(self):
        """A pathless block must not reach buckets the caller never named.

        Test scenario:
            Publishing to the whole `/vsis3/` handler would make the workers see
            the credentials, but that table is process-global with one slot per
            key, so an unrelated concurrent read of a third party's bucket would
            be signed with this caller's secret.
        """
        with CloudConfig(extra=self.ENV):
            for path in (
                "/vsis3/someone-elses-bucket/x.tif",
                "/vsis3_streaming/other/x.tif",
                self.OBJECT,
            ):
                # Asserting `is None` would only hold on a machine with no AWS
                # credentials exported: a path-specific miss falls through to the
                # process config. The block's own sentinel is what must be absent.
                assert (
                    self._worker_path_option(path, "AWS_ACCESS_KEY_ID") != "CDSEKEY"
                ), f"a pathless block leaked its key to {path}"
                assert (
                    self._worker_path_option(path, "AWS_S3_ENDPOINT")
                    != "eodata.dataspace.copernicus.eu"
                ), f"a pathless block leaked its endpoint to {path}"

    def test_the_ambient_config_rescopes_onto_the_read_s_own_bucket(self):
        """The reported defect: a bare block's endpoint must reach the workers.

        Test scenario:
            This is the issue's own reproduction — `with CloudConfig(extra=env):`
            around a read. The read knows its paths even though the block does
            not, so `ambient_store_config` re-scopes the credentials per bucket,
            which is worker-visible and isolated.
        """
        with CloudConfig(extra=self.ENV):
            with ambient_store_config(self.OBJECT):
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                    == "eodata.dataspace.copernicus.eu"
                ), "a worker must see the endpoint for the bucket being read"
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                    == "CDSEKEY"
                )
                assert (
                    self._worker_path_option("/vsis3/other/x.tif", "AWS_S3_ENDPOINT")
                    != "eodata.dataspace.copernicus.eu"
                ), "the re-scope must not reach a bucket this read never names"
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            ), "the re-scope outlived its block"

    def test_two_pathless_blocks_on_different_buckets_do_not_cross_wire(self):
        """Concurrent callers keep their own credentials.

        Test scenario:
            The regression this design exists to avoid: with the credentials
            published handler-wide instead, both threads observed whichever
            secret was pushed last, so one caller signed its whole read with
            another's key.
        """
        seen: dict[str, str | None] = {}
        start = threading.Barrier(2, timeout=10)
        inside = threading.Barrier(2, timeout=10)

        def read(tag: str, secret: str, bucket: str):
            env = {"AWS_S3_ENDPOINT": f"{tag}.example", "AWS_ACCESS_KEY_ID": secret}
            start.wait()
            with CloudConfig(extra=env):
                with ambient_store_config(f"/vsis3/{bucket}/x.tif"):
                    inside.wait()
                    seen[tag] = self._worker_path_option(
                        f"/vsis3/{bucket}/x.tif", "AWS_ACCESS_KEY_ID"
                    )

        threads = [
            threading.Thread(target=read, args=("a", "ASECRET", "abucket")),
            threading.Thread(target=read, args=("b", "BSECRET", "bbucket")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert not any(thread.is_alive() for thread in threads), (
            "a reader thread hung; the result below would be read from a partial run"
        )
        assert seen == {"a": "ASECRET", "b": "BSECRET"}, f"cross-wired: {seen}"

    def test_a_pathless_block_does_not_inject_into_a_nested_bucket_scope(self):
        """An outer pathless block must not add keys to an inner bucket's scope.

        Test scenario:
            GDAL merges path-specific scopes per key rather than shadowing them
            wholesale, so a handler-wide `AWS_NO_SIGN_REQUEST` would survive into
            an inner block that supplied real credentials and make an
            authenticated read go out anonymously.
        """
        with CloudConfig(aws_no_sign_request=True):
            with CloudConfig(extra={"AWS_ACCESS_KEY_ID": "REAL"}, path=self.OBJECT):
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_NO_SIGN_REQUEST")
                    != "YES"
                ), "a pathless outer block leaked AWS_NO_SIGN_REQUEST inward"

    def test_the_ambient_rescope_is_a_no_op_outside_any_block(self):
        """With no pathless block open, the re-scoper does nothing."""
        with ambient_store_config(self.OBJECT):
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            )

    def test_re_entering_one_instance_unwinds_only_the_inner_entry(self):
        """A `CloudConfig` entered while already open survives the inner exit.

        Test scenario:
            Engine methods wrap each other on the same dataset, so the same
            config object can be entered twice. Holding the context and the
            pushed-option list on the instance made the inner exit unwind the
            outer block's options and null the context, so the outer exit raised
            `AttributeError` and stranded its thread-local config.
        """
        config = CloudConfig(extra=self.ENV, path=self.OBJECT)
        with config:
            with config:
                pass
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            ), "the inner exit unwound the outer block's options"
        assert (
            self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
            != "eodata.dataspace.copernicus.eu"
        ), "the outer exit left its options live"

    def test_re_entering_one_pathless_instance_balances_the_ambient_stack(self):
        """A pathless config entered twice records and removes one entry each.

        Test scenario:
            A leaked ambient record would make a later unrelated read in the same
            thread silently pick up these credentials, long after the block that
            supplied them had exited.
        """
        config = CloudConfig(extra=self.ENV)
        depth_before = len(_ambient_stack())
        with config:
            with config:
                assert len(_ambient_stack()) == depth_before + 2
            assert len(_ambient_stack()) == depth_before + 1
        assert len(_ambient_stack()) == depth_before, "an ambient record leaked"

    def test_a_failed_entry_leaves_no_ambient_record(self):
        """A config whose entry raises does not leave its credentials recorded.

        Test scenario:
            `__enter__` raising means `__exit__` never runs, so the rollback has
            to drop the ambient record itself or the next read on this thread
            re-scopes credentials from a block that never opened.
        """
        depth_before = len(_ambient_stack())
        config = CloudConfig(extra=self.ENV)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                gdal,
                "config_options",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("config refused")),
            )
            with pytest.raises(RuntimeError, match="config refused"):
                config.__enter__()
        assert len(_ambient_stack()) == depth_before, "a failed entry leaked a record"

    def test_a_failed_entry_pops_the_secret_it_already_pushed(self):
        """A bucket-scoped entry that raises leaves nothing in the global table.

        Test scenario:
            The options are pushed before the thread-local context is entered,
            so a failure in between has already written a secret into GDAL's
            process-global path table. `__enter__` raising means `__exit__`
            never runs, so the rollback has to pop it or the credential outlives
            the block that supplied it, visible to every later read of that
            bucket in the process.
        """
        config = CloudConfig(extra=self.ENV, path=self.OBJECT)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                gdal,
                "config_options",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("config refused")),
            )
            with pytest.raises(RuntimeError, match="config refused"):
                config.__enter__()
        # Not `== "GONE"`: a path-specific miss falls through to the process
        # config, which on a machine with AWS credentials exported answers with
        # the ambient key. The block's own sentinels are what must be gone.
        assert (
            gdal.GetPathSpecificOption(self.OBJECT, "AWS_ACCESS_KEY_ID", "GONE")
            != "CDSEKEY"
        ), "a failed entry stranded its key in the process-global table"
        assert (
            gdal.GetPathSpecificOption(self.OBJECT, "AWS_S3_ENDPOINT", "GONE")
            != "eodata.dataspace.copernicus.eu"
        ), "a failed entry stranded its endpoint in the process-global table"
        assert config._entries == [], "a failed entry was recorded as active"

    def test_exiting_on_another_thread_removes_the_record_it_made(self):
        """A block closed off-thread must not leave its credentials recorded.

        Test scenario:
            A generator abandoned mid-block is finalized by GC, possibly on
            another thread. Popping the *current* thread's stack raised
            `IndexError` there — swallowed by the finalizer — and left the
            entering thread's record live forever, so a later read on that
            pooled thread re-scoped a dead block's credentials.
        """
        config = CloudConfig(extra=self.ENV)
        entered: dict[str, int] = {}

        def enter():
            config.__enter__()
            entered["depth"] = len(_ambient_stack())

        opener = threading.Thread(target=enter)
        opener.start()
        opener.join(timeout=10)
        assert entered["depth"] == 1, "the opening thread recorded nothing"

        closed: dict[str, str] = {}

        def close_elsewhere():
            try:
                config.__exit__(None, None, None)
                closed["result"] = "ok"
            except Exception as exc:
                closed["result"] = f"{type(exc).__name__}: {exc}"

        closer = threading.Thread(target=close_elsewhere)
        closer.start()
        closer.join(timeout=10)
        assert closed["result"] == "ok", f"off-thread exit failed: {closed['result']}"
        assert config._entries == [], "the entry survived an off-thread exit"

    def test_nested_pathless_blocks_merge_with_the_inner_winning(self):
        """An inner block overrides one key and inherits the rest.

        Test scenario:
            Taking only the innermost record dropped the outer block's keys,
            which then fell through to the process environment — so the calling
            thread could sign with one identity and the workers with another.
        """
        with CloudConfig(extra={"AWS_S3_ENDPOINT": "outer", "AWS_REGION": "eu-x"}):
            with CloudConfig(extra={"AWS_S3_ENDPOINT": "inner"}):
                with ambient_store_config(self.OBJECT):
                    assert (
                        self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                        == "inner"
                    ), "the inner block must win for the key it sets"
                    assert (
                        self._worker_path_option(self.OBJECT, "AWS_REGION") == "eu-x"
                    ), "the outer block's other keys must survive"

    def test_the_rescope_defers_to_an_explicit_bucket_scoped_block(self):
        """A broad pathless config must not outrank a narrow explicit one.

        Test scenario:
            Path options are last-push-wins, and the re-scope is entered inside
            the explicit block, so without deferral the deliberate, narrower
            credentials were silently replaced by the ambient ones for the whole
            operation — inverting every normal context-manager expectation.
        """
        with CloudConfig(extra={"AWS_ACCESS_KEY_ID": "OUTER_AMBIENT"}):
            with CloudConfig(
                extra={"AWS_ACCESS_KEY_ID": "INNER_EXPLICIT"}, path=self.OBJECT
            ):
                with ambient_store_config(self.OBJECT):
                    assert (
                        self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                        == "INNER_EXPLICIT"
                    ), "the ambient re-scope overrode an explicit path= block"
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                    == "INNER_EXPLICIT"
                ), "the re-scope's exit disturbed the explicit block"

    def test_one_instance_entered_on_two_threads_unwinds_cleanly(self):
        """Sharing a config across threads leaves no entry and no option behind.

        Test scenario:
            Per-instance state let one thread's exit restore the other thread's
            saved thread-local config. Each entry now records its own thread and
            only that thread closes its context.
        """
        shared = CloudConfig(extra=self.ENV, path=self.OBJECT)
        failures: list[str] = []

        def use(tag: str):
            try:
                with shared:
                    threading.Event().wait(0.02)
            except Exception as exc:
                failures.append(f"{tag}: {type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=use, args=(tag,)) for tag in ("first", "second")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert not any(thread.is_alive() for thread in threads), (
            "a thread hung inside the shared config"
        )
        assert failures == [], f"sharing one config across threads failed: {failures}"
        assert shared._entries == [], "entries leaked after both threads exited"
        assert (
            self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
            != "eodata.dataspace.copernicus.eu"
        ), "an option outlived both blocks"

    def test_a_non_bucket_path_records_no_duplicate(self):
        """Re-scoping onto a path with no bucket is a no-op, not a second record.

        Test scenario:
            The re-scope builds a `CloudConfig` carrying the same credentials.
            Entering it for a local destination would scope nothing and append a
            duplicate ambient record, which the next re-scope would then merge
            with itself.
        """
        with CloudConfig(extra=self.ENV):
            depth = len(_ambient_stack())
            with ambient_store_config("/tmp/local.tif"):
                assert len(_ambient_stack()) == depth, "a duplicate record was made"

    def test_merge_rasters_rescopes_its_sources(self):
        """`merge_rasters` hands its source paths to the re-scoper.

        Test scenario:
            The wiring is what makes a pathless `with CloudConfig(...)` around a
            merge reach the VSICurl workers. Spying on the call keeps this
            offline — the sources never have to be opened to prove the operation
            asks for the re-scope, and with which paths.
        """
        seen: list[Any] = []

        from pyramids.dataset import merge as merge_module

        def spy(path):
            seen.append(path)
            return nullcontext()

        def refuse(*args, **kwargs):
            raise RuntimeError("source not opened in this test")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(merge_module, "ambient_store_config", spy)
            # Stop before GDAL touches the network: the wiring is what is under
            # test, and the sources are deliberately unreachable.
            patch.setattr(merge_module, "open_network_dataset", refuse)
            with pytest.raises(RuntimeError, match="source not opened in this test"):
                merge_module.merge_rasters(
                    ["/vsis3/eodata/a.tif", "/vsis3/eodata/b.tif"], "out.tif"
                )
        assert seen, "merge_rasters never asked for an ambient re-scope"
        assert seen[0] == ["/vsis3/eodata/a.tif", "/vsis3/eodata/b.tif"], (
            f"the re-scope got the wrong paths: {seen[0]!r}"
        )

    def test_records_close_correctly_out_of_order(self):
        """An outer block closing before an inner one removes its own record.

        Test scenario:
            Context managers normally nest, but `__exit__` can be driven by hand
            or by a finalizer, so the records do not always come off the top.
            Removal is by identity for exactly this reason — popping the end
            would take the inner block's credentials and leave the outer block's
            live after it had gone.
        """
        outer = CloudConfig(extra={"AWS_S3_ENDPOINT": "outer.example"})
        inner = CloudConfig(extra={"AWS_S3_ENDPOINT": "inner.example"})
        depth = len(_ambient_stack())
        outer.__enter__()
        inner.__enter__()
        assert len(_ambient_stack()) == depth + 2

        outer.__exit__(None, None, None)
        assert len(_ambient_stack()) == depth + 1, (
            "the outer exit removed the wrong one"
        )
        assert _ambient_options().get("AWS_S3_ENDPOINT") == "inner.example", (
            "the surviving record is not the inner block's"
        )

        inner.__exit__(None, None, None)
        assert len(_ambient_stack()) == depth

    def test_dropping_a_record_twice_is_harmless(self):
        """A second removal of an already-removed record does nothing.

        Test scenario:
            A finalizer can run after an explicit close. Scanning for a record
            that is no longer there must leave the remaining records alone
            rather than removing whatever sits at the end.
        """
        survivor = CloudConfig(extra={"AWS_S3_ENDPOINT": "survivor.example"})
        with survivor:
            stale = CloudConfig(extra={"AWS_S3_ENDPOINT": "stale.example"})
            stale.__enter__()
            already_removed = _ambient_stack()[-1]
            stale.__exit__(None, None, None)
            depth = len(_ambient_stack())
            # The same object a second time, which is the finalizer-after-close
            # case. Passing a freshly built dict would only prove that a record
            # which was never added is ignored.
            _drop_ambient(_ambient_stack(), already_removed)
            assert len(_ambient_stack()) == depth, "an unrelated record was removed"
            assert _ambient_options().get("AWS_S3_ENDPOINT") == "survivor.example"

    @staticmethod
    def _dataset_named(source: str):
        """Wrap a MEM handle whose description is `source`.

        A real `/vsis3/` open needs the network, but the gate reads
        `src.GetDescription()` — and the MEM driver keeps whatever name it is
        created with, so a bucket-shaped description is reproducible offline.
        """
        from pyramids.dataset import Dataset

        handle = gdal.GetDriverByName("MEM").Create(source, 4, 3, 1, gdal.GDT_Byte)
        return Dataset(handle)

    @pytest.mark.parametrize(
        "source, captures",
        [
            ("/vsis3/eodata/S2/B04_10m.jp2", True),
            ('NETCDF:"/vsis3/eodata/x.nc":tas', True),
            ("/vsizip//vsis3/eodata/a.zip/x.tif", True),
            ("/vsicurl/https://host/a.tif", False),
            ("/vsimem/scratch.tif", False),
            ("plain-local.tif", False),
        ],
    )
    def test_the_capture_gate_follows_the_source_being_object_backed(
        self, source, captures
    ):
        """Only an object-backed source captures the ambient credentials.

        Args:
            source: The dataset's description, as GDAL reports it.
            captures: Whether the constructor should take the ambient config.

        Test scenario:
            Driven through `Dataset(...)` itself rather than re-deriving the
            rule, so deleting the gate fails this. `NETCDF:"…"` is the shape
            that made a remote container look local; `/vsicurl/` is excluded on
            purpose — it is not signed with SigV4.
        """
        with CloudConfig(extra=self.ENV):
            dataset = self._dataset_named(source)
        captured = bool(dataset.gdal_env)
        assert captured is captures, (
            f"{source!r} should {'capture' if captures else 'not capture'}; "
            f"got {sorted(dataset.gdal_env)}"
        )

    def test_an_explicit_gdal_env_still_wins_over_the_ambient_one(self):
        """A caller passing `gdal_env=` is not overridden by an open block.

        Test scenario:
            On a bucket-backed source, so the ambient side is actually in play —
            on a local one the gate suppresses it anyway and the precedence is
            never contested.
        """
        from pyramids.dataset import Dataset

        handle = gdal.GetDriverByName("MEM").Create(self.OBJECT, 4, 3, 1, gdal.GDT_Byte)
        with CloudConfig(extra=self.ENV):
            dataset = Dataset(handle, gdal_env={"AWS_S3_ENDPOINT": "explicit.example"})
        assert dataset.gdal_env == {"AWS_S3_ENDPOINT": "explicit.example"}, (
            f"the ambient config displaced an explicit one: {dataset.gdal_env}"
        )

    def test_the_rescope_from_a_capture_defers_to_an_explicit_block(self):
        """A captured env fills gaps around an explicit block, never outranks it.

        Test scenario:
            `_cloud_config` re-installs an implicitly captured config, so it has
            to defer the same way `ambient_store_config` does — otherwise the
            capture quietly beats a narrower `path=` block the caller opened
            around the read.
        """
        with CloudConfig(extra=self.ENV):
            dataset = self._dataset_named(self.OBJECT)
        with CloudConfig(extra={"AWS_ACCESS_KEY_ID": "EXPLICIT"}, path=self.OBJECT):
            with dataset._cloud_config():
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                    == "EXPLICIT"
                ), "the captured env outranked an explicit path= block"

    def test_a_later_read_scopes_the_captured_credentials_to_its_own_bucket(self):
        """Every read re-installs the captured config against the dataset's path.

        Test scenario:
            This is the half `merge_rasters` could not cover: the credentials
            must reach the workers at *read* time, scoped to the bucket this
            dataset came from and no other.
        """
        with CloudConfig(extra=self.ENV):
            dataset = self._dataset_named(self.OBJECT)
        with dataset._cloud_config():
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            ), "a later read left the workers blind"
            assert (
                self._worker_path_option("/vsis3/other/x.tif", "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            ), "the read widened onto a bucket it never names"
        assert (
            self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
            != "eodata.dataspace.copernicus.eu"
        ), "the scope outlived the read"

    def test_a_dataset_opened_outside_any_block_captures_nothing(self):
        """The common unsigned case still pays nothing per read."""
        from pyramids.dataset import Dataset

        dataset = Dataset.read_file(str(_DATA_RASTER))
        assert dataset.gdal_env == {}, f"captured unexpectedly: {dataset.gdal_env}"

    def test_a_local_dataset_captures_nothing_inside_a_pathless_block(self):
        """A local raster must not pick up credentials it can never use.

        Test scenario:
            The capture is for remote reads. Taking it unconditionally gave a
            local raster a cloud secret for its whole lifetime, re-pushed it on
            every read, and serialised it into the pickle recipe — so a local
            dataset sent to a worker carried a credential it had no use for.
        """
        from pyramids.dataset import Dataset

        with CloudConfig(extra=self.ENV):
            dataset = Dataset.read_file(str(_DATA_RASTER))
            depth = len(_ambient_stack())
            with dataset._cloud_config():
                assert len(_ambient_stack()) == depth, (
                    "a local read pushed a redundant ambient record"
                )
        assert dataset.gdal_env == {}, (
            f"a local raster captured cloud credentials: {sorted(dataset.gdal_env)}"
        )
        assert "AWS_SECRET_ACCESS_KEY" not in str(dataset.__reduce__()), (
            "the pickle recipe carries a credential the dataset never needs"
        )

    def test_a_path_vsi_rewriting_rejects_still_yields_a_decision(self):
        """`_bucket_prefix` answers `None` when `_to_vsi` refuses the path.

        Test scenario:
            `_to_vsi` raises `TypeError` on a non-string path (bytes, say). The
            gate must still answer rather than propagate, because it runs on
            every dataset construction — an exception there would break opening
            a dataset, not just the credential decision.
        """
        assert _bucket_prefix(b"/vsis3/eodata/x.tif") is None, (
            "a value _to_vsi rejects must not yield a bucket"
        )

    def test_two_deferring_readers_of_one_bucket_nest(self):
        """The first reader's exit must not strip the second reader's option.

        Test scenario:
            A deferred key that is skipped outright records nothing, so the
            stack does not match the number of blocks holding it: the first
            reader to exit popped the option while the second was still mid-read
            and its credentials fell through to the process config. Reachable
            from `read_windows`' own pool and from threaded dask reads, which is
            the path this branch exists to fix.
        """
        env = {"AWS_ACCESS_KEY_ID": "CDSEKEY"}
        first = cloud_config_from_env(env, path=self.OBJECT, defer=True)
        second = cloud_config_from_env(env, path=self.OBJECT, defer=True)
        first.__enter__()
        second.__enter__()
        try:
            assert (
                self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID") == "CDSEKEY"
            )
            first.__exit__(None, None, None)
            assert (
                self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID") == "CDSEKEY"
            ), "the first reader's exit stripped the second reader's credential"
        finally:
            second.__exit__(None, None, None)
        assert (
            self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID") != "CDSEKEY"
        ), "the option outlived both readers"

    def test_a_behaviour_only_block_does_not_hide_ambient_credentials(self):
        """Requester-pays says who is billed, not who is reading.

        Test scenario:
            Treating `AWS_REQUEST_PAYER` as part of the credential family made
            an explicit `CloudConfig(aws_request_payer=True, path=...)` claim the
            whole identity, so the ambient block's key and secret were withheld
            and the read went out unsigned.
        """
        with CloudConfig(
            extra={
                "AWS_ACCESS_KEY_ID": "CDSEKEY",
                "AWS_SECRET_ACCESS_KEY": "CDSESECRET",
            }
        ):
            with CloudConfig(aws_request_payer=True, path=self.OBJECT):
                with ambient_store_config(self.OBJECT):
                    assert (
                        self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                        == "CDSEKEY"
                    ), "a billing flag hid the ambient identity"
                    assert (
                        self._worker_path_option(self.OBJECT, "AWS_REQUEST_PAYER")
                        == "requester"
                    )

    def test_a_foreign_exit_leaves_a_live_owner_alone(self):
        """An exit from another thread must not unwind an entry still in use.

        Test scenario:
            Unwinding the oldest entry took a running thread's credentials out
            from under its read, and that thread's own exit then popped a
            stranger's entry or raised `IndexError` into user code. A finalizer
            may only clean up after an owner that has finished.
        """
        shared = CloudConfig(extra=self.ENV, path=self.OBJECT)
        inside = threading.Event()
        release = threading.Event()
        seen: dict[str, object] = {}

        def owner():
            try:
                with shared:
                    inside.set()
                    release.wait(10)
                    seen["during"] = self._worker_path_option(
                        self.OBJECT, "AWS_S3_ENDPOINT"
                    )
            except Exception as exc:  # noqa: BLE001 - recorded, then asserted on
                seen["owner_error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=owner)
        thread.start()
        assert inside.wait(10), "the owner never entered"
        shared.__exit__(None, None, None)
        release.set()
        thread.join(timeout=20)
        assert "owner_error" not in seen, f"the owner's own exit broke: {seen}"
        assert seen.get("during") == "eodata.dataspace.copernicus.eu", (
            "a foreign exit stripped the running owner's credentials"
        )
        assert shared._entries == [], "the owner's entry was not cleaned up"

    def test_a_behaviour_only_env_does_not_suppress_the_capture(self):
        """A non-credential `gdal_env=` must not disable #983's fix.

        Test scenario:
            Any explicit env used to suppress the capture, and `from_files`
            synthesises one carrying only `GDAL_DISABLE_READDIR_ON_OPEN` — so
            the fix was silently off for every collection built that way. A
            behaviour knob says nothing about who is reading.
        """
        from pyramids.dataset import Dataset

        handle = gdal.GetDriverByName("MEM").Create(self.OBJECT, 4, 3, 1)
        with CloudConfig(extra=self.ENV):
            dataset = Dataset(
                handle, gdal_env={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
            )
        assert dataset.gdal_env.get("AWS_ACCESS_KEY_ID") == "CDSEKEY", (
            f"a behaviour-only env suppressed the capture: {sorted(dataset.gdal_env)}"
        )
        assert dataset.gdal_env.get("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR", (
            "the caller's own key was dropped"
        )

    def test_an_explicit_credential_still_suppresses_the_capture(self):
        """Supplying credentials is a choice of identity, and it stands alone."""
        from pyramids.dataset import Dataset

        handle = gdal.GetDriverByName("MEM").Create(self.OBJECT, 4, 3, 1)
        with CloudConfig(extra=self.ENV):
            dataset = Dataset(handle, gdal_env={"AWS_ACCESS_KEY_ID": "EXPLICIT"})
        assert dataset.gdal_env == {"AWS_ACCESS_KEY_ID": "EXPLICIT"}, (
            f"the ambient config joined an explicit identity: {dataset.gdal_env}"
        )

    def test_a_signer_claims_the_identity_before_the_ambient_block(self):
        """A partial signer identity must not be completed by an ambient one.

        Test scenario:
            `merge_rasters` and `stack_bands` enter the signer's config first so
            it claims the credential family. Entered the other way round the
            ambient access key was signed alongside the signer's session token —
            the mixed identity the family rule exists to prevent.
        """

        class _TokenSigner:
            """A signer that supplies only a session token."""

            def gdal_env(self):
                return {"AWS_SESSION_TOKEN": "SIGNERTOKEN"}

            def sign_href(self, href):
                return href

        with CloudConfig(extra={"AWS_ACCESS_KEY_ID": "CDSEKEY"}):
            with (
                signer_cloud_config(_TokenSigner(), path=[self.OBJECT]),
                ambient_store_config([self.OBJECT]),
            ):
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_SESSION_TOKEN")
                    == "SIGNERTOKEN"
                )
                assert (
                    self._worker_path_option(self.OBJECT, "AWS_ACCESS_KEY_ID")
                    != "CDSEKEY"
                ), "an ambient key joined the signer's identity"

    def test_a_local_path_keeps_the_config_thread_local(self):
        """A local file names no bucket, so nothing is published."""
        with CloudConfig(extra=self.ENV, path="/tmp/local.tif"):
            assert self._worker_global_option("AWS_S3_ENDPOINT") is None
            assert (
                self._worker_path_option(self.OBJECT, "AWS_S3_ENDPOINT")
                != "eodata.dataspace.copernicus.eu"
            )

    def test_multiple_buckets_each_get_the_credentials(self):
        """A merge over sources in two buckets scopes both.

        Test scenario:
            `merge_rasters` wraps a list of sources, which may span buckets.
        """
        paths = ["/vsis3/eodata/a.jp2", "/vsis3/eodata2/b.jp2"]
        with CloudConfig(extra=self.ENV, path=paths):
            assert self._worker_path_option(paths[0], "AWS_S3_ENDPOINT") is not None
            assert self._worker_path_option(paths[1], "AWS_S3_ENDPOINT") is not None
