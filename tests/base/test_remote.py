"""Unit tests for pyramids.base.remote."""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids.base.remote import CloudConfig, _to_vsi, is_remote


class TestToVsi:
    def test_s3(self):
        assert _to_vsi("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_s3_nested_path(self):
        assert _to_vsi("s3://bucket/a/b/c/key.tif") == "/vsis3/bucket/a/b/c/key.tif"

    def test_gs(self):
        assert _to_vsi("gs://bucket/key.tif") == "/vsigs/bucket/key.tif"

    def test_az(self):
        assert _to_vsi("az://container/blob.tif") == "/vsiaz/container/blob.tif"

    def test_abfs_maps_to_vsiaz(self):
        assert _to_vsi("abfs://container/blob.tif") == "/vsiaz/container/blob.tif"

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
        }

    def test_skips_none_fields(self):
        cfg = CloudConfig(aws_region="eu-west-1").as_gdal_config()
        assert cfg == {"AWS_REGION": "eu-west-1"}

    def test_no_sign_request_true(self):
        assert CloudConfig(aws_no_sign_request=True).as_gdal_config() == {
            "AWS_NO_SIGN_REQUEST": "YES"
        }

    def test_no_sign_request_false_absent(self):
        assert "AWS_NO_SIGN_REQUEST" not in CloudConfig(aws_no_sign_request=False).as_gdal_config()

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
        cfg = CloudConfig(extra={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}).as_gdal_config()
        assert cfg == {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

    def test_extra_with_aws(self):
        cfg = CloudConfig(
            aws_region="us-east-1",
            extra={"VSI_CACHE": "TRUE"},
        ).as_gdal_config()
        assert cfg == {"AWS_REGION": "us-east-1", "VSI_CACHE": "TRUE"}


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
        try:
            with pytest.raises(RuntimeError):
                with CloudConfig(aws_region="during"):
                    assert gdal.GetConfigOption(key) == "during"
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
