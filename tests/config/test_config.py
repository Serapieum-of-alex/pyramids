import os
import pytest
import unittest
from unittest.mock import patch
from pathlib import Path
from osgeo import gdal
from pyramids.config import Config
import io
from contextlib import redirect_stdout


class TestConfigEndToEnd(unittest.TestCase):
    """
    End-to-end tests for all methods in the Config class.

    These tests validate the actual functionality of the Config class without using mocks.
    The tests are executed across different operating systems and environments (e.g., Conda, non-Conda).
    """

    def setUp(self):
        """Set up a Config instance for tests."""
        self.config = Config()

    def test_set_env_conda(self):
        """End-to-end test for setting up GDAL in a Conda environment."""
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            # Conda environment
            gdal_plugins_path = Path(conda_prefix) / "Library/lib/gdalplugins"
            result = self.config.set_env_conda()
            if os.path.exists(gdal_plugins_path):
                self.assertEqual(result, gdal_plugins_path)
                self.assertEqual(
                    os.environ.get("GDAL_DRIVER_PATH"), str(gdal_plugins_path)
                )
            else:
                self.assertIsNone(result)
        else:
            # not a conda environment
            result = self.config.set_env_conda()
            self.assertIsNone(result)

    def test_dynamic_env_variables(self):
        """End-to-end test for setting dynamic environment variables for GDAL."""
        gdal_plugins_path = self.config.dynamic_env_variables()
        if gdal_plugins_path:
            self.assertTrue(gdal_plugins_path.exists())
            self.assertEqual(os.environ.get("GDAL_DRIVER_PATH"), str(gdal_plugins_path))
        else:
            self.assertIsNone(gdal_plugins_path)

    def test_set_error_handler(self):
        """End-to-end test for setting the GDAL error handler."""
        self.config.set_error_handler()


class TestConfigMock(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    @patch("os.environ", new_callable=dict)
    @patch("pyramids.config.Config.set_env_conda")
    @patch("pyramids.config.Config.dynamic_env_variables")
    @patch("osgeo.gdal.AllRegister")
    def test_initialize_gdal(self, mock_register, mock_dynamic, mock_conda, mock_env):
        mock_dynamic.return_value = Path("/usr/lib/gdalplugins")
        mock_conda.return_value = Path("/usr/lib/gdalplugins")
        self.config.initialize_gdal()

        # self.assertIn("GDAL_DRIVER_PATH", os.environ)
        # self.assertEqual(os.environ["GDAL_DRIVER_PATH"], "/usr/lib/gdalplugins")
        mock_register.assert_called_once()

    @patch("os.getenv", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    def test_set_env_conda_no_conda(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertIsNone(result)

    @patch("os.getenv", return_value="/fake/conda/prefix")
    @patch("pathlib.Path.exists", return_value=True)
    def test_set_env_conda_success(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertEqual(result, Path("/fake/conda/prefix/Library/lib/gdalplugins"))
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("/fake/conda/prefix/Library/lib/gdalplugins"))

    @patch("os.getenv", return_value="fake/conda/prefix")
    @patch("pathlib.Path.exists", return_value=False)
    def test_set_env_conda_plugins_path_not_exist(self, mock_exists, mock_getenv):
        result = self.config.set_env_conda()
        self.assertIsNone(result)

    @patch("site.getsitepackages", return_value=["C:/Python/site-packages"])
    @patch("pathlib.Path.exists", return_value=True)
    def test_dynamic_env_variables_windows(self, mock_exists, mock_site):
        with (
            patch("pyramids.config.Config.set_env_conda", return_value=None),
            patch("sys.platform", new="win32"),
        ):
            result = self.config.dynamic_env_variables()
        self.assertEqual(
            result, Path("C:/Python/site-packages/Library/Lib/gdalplugins")
        )
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("C:/Python/site-packages/Library/Lib/gdalplugins"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_dynamic_env_variables_linux(self, mock_exists):
        with (
            patch("pyramids.config.Config.set_env_conda", return_value=None),
            patch("sys.platform", new="linux"),
        ):
            result = self.config.dynamic_env_variables()
        self.assertEqual(result, Path("/usr/local/lib/gdalplugins"))
        path = Path(os.environ["GDAL_DRIVER_PATH"])
        self.assertEqual(path, Path("/usr/local/lib/gdalplugins"))

    @patch("osgeo.gdal.SetConfigOption")
    def test_set_error_handler(self, mock_set_config):
        self.config.set_error_handler()
        mock_set_config.assert_not_called()


@patch("osgeo.gdal.PushErrorHandler")
def test_set_error_handler_prints_for_low_error_class(mock_push):
    # Install the handler via Config and capture it from the patched GDAL entry point
    Config.set_error_handler()
    handler = mock_push.call_args[0][0]

    # Invoke the handler with an error class lower than CE_Warning to trigger printing
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler(0, 42, "oops")
    out = buf.getvalue().strip()

    assert out == "GDAL error (class 0, number 42): oops"
