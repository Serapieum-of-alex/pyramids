"""
Configuration module for the pyramids package.

This module provides a Config class to manage configuration settings, initialize GDAL and OGR options, and dynamically
adjust environment variables based on the system's setup.

Features:
- Load configurations from YAML files.
- Initialize GDAL and OGR settings.
- Dynamically set up environment variables for GDAL plugins.
- Handle error logging and error handlers.

Examples:
- Initialize and access the GDAL driver path:

  ```python
  >>> from pyramids.config import Config
  >>> config = Config()
  >>> config.initialize_gdal()
  >>> print(os.environ.get("GDAL_DRIVER_PATH"))  # doctest: +SKIP

  ```

- Load configuration from a specific file:

  ```python
  >>> config_file = "path/to/config.yaml"
  >>> config = Config(config_file=config_file) # doctest: +SKIP
  >>> config.load_config() # doctest: +SKIP

  ```

Classes:
- Config: The main configuration class for managing GDAL, OGR, and environment settings.

"""

import os
import yaml
import logging
import sys
from typing import Union
from pathlib import Path
from osgeo import gdal, ogr
from pyramids import __path__ as root_path


class Config:
    r"""
    Configuration class for the pyramids package.

    This class handles:
    - Loading configuration settings from YAML files.
    - Initializing GDAL and OGR configurations.
    - Dynamically setting environment variables for GDAL plugins based on the operating system and environment (e.g., Conda).

    Args:
        config_file (str, optional): Path to the configuration YAML file. Default is "config.yaml" in the module's directory.

    Attributes:
        config (dict): Loaded configuration settings.
        logger (logging.Logger): Logger for logging messages.

    Examples:
        - Initialize the configuration and load settings from the default config file:

          ```python
          >>> from pyramids.config import Config
          >>> config = Config() # doctest: +SKIP
          2025-01-11 23:13:48,889 - pyramids.config - INFO - Logging is configured.
          2025-01-11 23:13:48,891 - pyramids.config - INFO - GDAL_DRIVER_PATH set to: your\\conda\\env\\Library\\lib\\gdalplugins
          >>> config.initialize_gdal() # doctest: +SKIP
          2025-01-11 23:13:48,891 - pyramids.config - INFO - GDAL_DRIVER_PATH set to: your\\conda\\env\\Library\\lib\\gdalplugins
          >>> print(os.environ.get("GDAL_DRIVER_PATH")) # doctest: +SKIP
          C:\\Miniconda3\\envs\\pyramids\\Library\\lib\\gdalplugins

          ```

    Notes:
        - The GDAL and OGR Python bindings use exceptions for error reporting when `UseExceptions` is enabled.
        - Environment variable settings depend on the presence of Conda and platform-specific paths.

    See Also:
        - gdal.UseExceptions: Documentation on enabling GDAL exceptions.
        - ogr.UseExceptions: Documentation on enabling OGR exceptions.
    """

    def __init__(self, config_file="config.yaml"):
        """Initialize the configuration."""
        self.setup_logging()
        self.config_file = config_file
        self.config = self.load_config()
        self.initialize_gdal()

    def load_config(self):
        """
        Load the configuration from the specified YAML file.

        Returns:
            dict: A dictionary containing the configuration settings.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.

        Examples:
            - Load settings from a config file and print them:

              ```python
              >>> config = Config(config_file="config.yaml")
              >>> settings = config.load_config()
              >>> print(settings) # doctest: +NORMALIZE_WHITESPACE
              {'gdal': {'GDAL_CACHEMAX': '512',
               'GDAL_PAM_ENABLED': 'YES',
               'GDAL_VRT_ENABLE_PYTHON': 'YES',
               'GDAL_TIFF_INTERNAL_MASK': 'NO'},
               'ogr': {'OGR_SRS_PARSER': 'strict'},
               'logging': {'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'pyramids.log'}}

              ```
        """
        with open(f"{root_path[0]}/{self.config_file}", "r") as file:
            return yaml.safe_load(file)

    def initialize_gdal(self):
        """
        Initialize GDAL and OGR settings from the loaded configuration.

        Configures GDAL and OGR options and dynamically sets the GDAL_DRIVER_PATH environment variable based on the system setup.

        Notes:
            - Uses the `dynamic_env_variables` method to locate the GDAL plugins path.
            - By default, GDAL and OGR suppress exceptions unless explicitly enabled using `UseExceptions`.

        Examples:
            - Initialize GDAL and print the driver path:

              ```python
              >>> config = Config()
              >>> config.initialize_gdal()
              >>> print(os.environ.get("GDAL_DRIVER_PATH")) # doctest: +SKIP

              ```
        """
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
        gdal.AllRegister()

    def set_env_conda(self) -> Union[Path, None]:
        """
        Set the environment variables for GDAL in a Conda environment.

        Returns:
            Path | None: The GDAL plugins path if found, otherwise None.

        Notes:
            - Assumes the Conda environment variable `CONDA_PREFIX` is set.
            - The method verifies the existence of the `gdalplugins` directory under the Conda environment.

        Examples:
            - Set GDAL environment variables in a Conda environment and print the path:

              ```python
              >>> config = Config()
              >>> gdal_path = config.set_env_conda()
              >>> print(gdal_path) # doctest: +SKIP

              ```
        """
        conda_prefix = os.getenv("CONDA_PREFIX")

        if not conda_prefix:
            self.logger.info("CONDA_PREFIX is not set. Ensure Conda is activated.")
            return None

        gdal_plugins_path = Path(conda_prefix) / "Library/lib/gdalplugins"

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
        Dynamically locate the GDAL plugins path and set the GDAL_DRIVER_PATH environment variable.

        Returns:
            Path: The GDAL plugins path if found, otherwise None.

        Notes:
            - On Windows, it checks typical Python site-packages locations.
            - On Linux/macOS, it checks common system directories like `/usr/lib/gdalplugins`.

        Examples:
            - Locate the GDAL plugins path dynamically and print it:

              ```python
              >>> config = Config()
              >>> gdal_path = config.dynamic_env_variables()
              >>> print(gdal_path) # doctest: +SKIP

              ```
        """
        # Check if we're in a Conda environment
        gdal_plugins_path = self.set_env_conda()

        if gdal_plugins_path is None:

            # For Windows, check Python site-packages
            if sys.platform == "win32":
                import site

                for site_path in site.getsitepackages():
                    gdal_plugins_path = Path(site_path) / "Library/Lib/gdalplugins"
                    if gdal_plugins_path.exists():
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
        """
        Set up the logging configuration.

        Uses basic logging with configurable level and format.

        Examples:
            - Setup logging:

              ```python
              >>> config = Config()
              >>> config.setup_logging()

              ```
        """
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
        """
        Set the error handler for GDAL.

        Suppresses warnings and errors or redirects them to a custom handler.

        Examples:
            - Set a custom GDAL error handler:

              ```python
              >>> Config.set_error_handler()

              ```
        """

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
