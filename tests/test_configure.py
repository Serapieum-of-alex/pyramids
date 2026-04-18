"""Tests for :func:`pyramids.configure`.

DASK-5 entry point. Sets GDAL config options in the current process
and, when a dask client is given, broadcasts to every worker.
"""

from __future__ import annotations

import pytest
from osgeo import gdal

from pyramids import configure
from pyramids._configure import GDAL_CLOUD_DEFAULTS, _expand_credentials


class TestConfigureReturnValue:
    """`configure` returns the effective env dict that was applied."""

    def test_cloud_defaults_included(self):
        applied = configure(cloud_defaults=True)
        for key, value in GDAL_CLOUD_DEFAULTS.items():
            assert applied[key] == value

    def test_override_applied_after_cloud_defaults(self):
        applied = configure(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="3")
        assert applied["GDAL_HTTP_MAX_RETRY"] == "3"

    def test_extra_kwargs_stringified(self):
        applied = configure(GDAL_CACHEMAX=512)
        assert applied["GDAL_CACHEMAX"] == "512"

    def test_no_args_returns_empty(self):
        applied = configure()
        assert applied == {}


class TestConfigureAppliesToGdal:
    """Each key in the returned dict is visible via gdal.GetConfigOption."""

    def test_single_option_round_trip(self):
        configure(GDAL_CACHEMAX="64")
        assert gdal.GetConfigOption("GDAL_CACHEMAX") == "64"

    def test_cloud_defaults_applied_to_gdal(self):
        configure(cloud_defaults=True)
        assert gdal.GetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
        assert gdal.GetConfigOption("GDAL_HTTP_MULTIRANGE") == "YES"


class TestCredentialExpansion:
    """``aws=``, ``gs=``, ``azure=`` expand via ``_expand_credentials``."""

    def test_aws_unsigned_shortcut(self):
        out = _expand_credentials("AWS", {"aws_unsigned": True})
        assert out == {"AWS_NO_SIGN_REQUEST": "YES"}

    def test_aws_keys_upcased(self):
        out = _expand_credentials("AWS", {"access_key_id": "AKIA123"})
        assert out == {"AWS_ACCESS_KEY_ID": "AKIA123"}

    def test_none_input_empty(self):
        assert _expand_credentials("AWS", None) == {}

    def test_empty_dict_empty(self):
        assert _expand_credentials("GS", {}) == {}

    def test_gs_prefix(self):
        out = _expand_credentials("GS", {"project_id": "my-proj"})
        assert out == {"GS_PROJECT_ID": "my-proj"}

    def test_azure_prefix(self):
        out = _expand_credentials("AZURE", {"storage_account": "acct"})
        assert out == {"AZURE_STORAGE_ACCOUNT": "acct"}


class TestConfigureAwsEndToEnd:
    """`configure(aws=...)` applies expanded keys to GDAL."""

    def test_aws_unsigned_visible_in_gdal(self):
        configure(aws={"aws_unsigned": True})
        assert gdal.GetConfigOption("AWS_NO_SIGN_REQUEST") == "YES"


class TestImportSurface:
    """`pyramids.configure` is reachable from the package root."""

    def test_importable_from_pyramids_root(self):
        import pyramids

        assert callable(pyramids.configure)
