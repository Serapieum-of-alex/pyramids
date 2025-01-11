"""Configuration module for the pyramids package."""

import os
import yaml
import logging
import sys
from typing import Union
from pathlib import Path
from osgeo import gdal, ogr
from pyramids import __path__ as root_path


class Config:
    """Configuration class for the pyramids package."""

    def __init__(self, config_file="config.yaml"):
        """Initialize the configuration."""
        self.setup_logging()
        self.config_file = config_file
        self.config = self.load_config()
        self.initialize_gdal()

    def load_config(self):
        """Load the configuration from the config file."""
        with open(f"{root_path[0]}/{self.config_file}", "r") as file:
            return yaml.safe_load(file)

    def initialize_gdal(self):
        """Initialize the GDAL and OGR configuration."""
        # By default, the GDAL and OGR Python bindings do not raise exceptions when errors occur. Instead, they return
        # an error value such as None and write an error message to sys.stdout, to report errors by raising
        # exceptions. You can enable this behavior in GDAL and OGR by calling the UseExceptions().
        gdal.UseExceptions()
        ogr.UseExceptions()
        # gdal.ErrorReset()
        for key, value in self.config.get("gdal", {}).items():
            gdal.SetConfigOption(key, value)
        for key, value in self.config.get("ogr", {}).items():
            gdal.SetConfigOption(key, value)

        gdal_plugins_path = self.dynamic_env_variables()

        if gdal_plugins_path:
            gdal.SetConfigOption("GDAL_DRIVER_PATH", str(gdal_plugins_path))

    def set_env_conda(self) -> Union[Path, None]:
        """Set the environment variables for Conda.

        Returns
        -------
        Path:
            The GDAL plugins path.
        """
        conda_prefix = Path(os.getenv("CONDA_PREFIX"))

        if not conda_prefix:
            self.logger.info("CONDA_PREFIX is not set. Ensure Conda is activated.")
            return None

        gdal_plugins_path = conda_prefix / "Library/lib/gdalplugins"

        if gdal_plugins_path.exists():
            os.environ["GDAL_DRIVER_PATH"] = str(gdal_plugins_path)
            self.logger.info(f"GDAL_DRIVER_PATH set to: {gdal_plugins_path}")
        else:
            self.logger.info(
                f"GDAL plugins path not found at: {gdal_plugins_path}. Please check your GDAL installation."
            )
            gdal_plugins_path = None

        return gdal_plugins_path

    def dynamic_env_variables(self) -> Path:
        """
        Dynamically locate the GDAL plugin path in a Conda or custom environment and set the GDAL_DRIVER_PATH environment variable.
        """
        # Check if we're in a Conda environment
        gdal_plugins_path = self.set_env_conda()

        if gdal_plugins_path is None:

            # For Windows, check Python site-packages
            if sys.platform == "win32":
                import site

                for site_path in site.getsitepackages():
                    gdal_plugins_path = Path(site_path) / "Library/Lib/gdalplugins"
                    if os.path.exists(gdal_plugins_path):
                        os.environ["GDAL_DRIVER_PATH"] = str(gdal_plugins_path)
                        self.logger.info(
                            f"GDAL_DRIVER_PATH set to: {gdal_plugins_path}"
                        )
            else:

                # Check typical system locations (Linux/MacOS)
                system_paths = [
                    "/usr/lib/gdalplugins",
                    "/usr/local/lib/gdalplugins",
                ]
                for path in system_paths:
                    path = Path(path)
                    if path.exists():
                        os.environ["GDAL_DRIVER_PATH"] = str(path)
                        gdal_plugins_path = path
                        self.logger.info(f"GDAL_DRIVER_PATH set to: {path}")

                # If the path is not found
                # print("GDAL plugins path could not be found. Please check your GDAL installation.")
        return gdal_plugins_path

    def setup_logging(self):
        """Set up the logging configuration."""
        log_config = {}  # self.config.get("logging", {})
        logging.basicConfig(
            level=log_config.get("level", "INFO"),
            format=log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            filename=log_config.get("file", None),
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging is configured.")

    @staticmethod
    def set_error_handler():
        """Set the error handler for GDAL."""

        def gdal_error_handler(err_class, err_num, err_msg):
            """Error handler for GDAL."""
            if err_class >= gdal.CE_Warning:
                pass  # Ignore warnings and higher level messages (errors, fatal errors)
            else:
                print(
                    "GDAL error (class {}, number {}): {}".format(
                        err_class, err_num, err_msg
                    )
                )

        gdal.PushErrorHandler(gdal_error_handler)
