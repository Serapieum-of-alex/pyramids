import os
import unittest
from pathlib import Path
from pyramids.config import Config


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
        # No explicit assertion needed; ensure no exceptions occur.
