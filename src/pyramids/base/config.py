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


class ColorFormatter(logging.Formatter):
    """Console formatter that colors the levelname based on log level."""
    RESET = "\x1b[0m"
    LEVEL_COLORS = {
        logging.DEBUG: "\x1b[36m",    # Cyan
        logging.INFO: "\x1b[32m",     # Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",    # Red
        logging.CRITICAL: "\x1b[35m", # Magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        import copy as _copy
        colored = _copy.copy(record)
        color = self.LEVEL_COLORS.get(record.levelno, "")
        colored.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(colored)


class LoggerManager:
    """Encapsulates logging setup and GDAL error handler installation for Pyramids.

    This class centralizes logging responsibilities to improve separation of concerns
    from the Config class. It provides static methods to configure logging and to
    integrate GDAL error output with the configured logger hierarchy.
    """
    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, level: Union[int, str] = logging.INFO, log_file: Union[str, Path, None] = None):
        """Initialize the logger manager."""
        self._setup_logging(level=level, log_file=log_file)
        self._set_error_handler()

    def _setup_logging(self, level: Union[int, str] = logging.INFO, log_file: Union[str, Path, None] = None) -> None:
        """
        Configure application-wide logging for Pyramids.

        This initializes a colored console handler and, optionally, a file handler using a consistent
        format. The configuration is idempotent: calling this method multiple times will not create
        duplicate handlers. It also reduces noise by elevating log levels for common third‑party
        libraries.
        """
        # Normalize level
        if isinstance(level, str):
            if level.upper() not in logging._nameToLevel:
                raise ValueError(f"Invalid log level: {level}")
            level = getattr(logging, level.upper(), logging.INFO)

        # Enable ANSI colors on Windows terminals when possible
        try:  # pragma: no cover - best effort without hard dependency
            import colorama

            colorama.just_fix_windows_console()
        except Exception:
            pass

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Determine if a console handler already exists
        console_handler = None
        file_handler_exists_for = set()
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                console_handler = h
            if isinstance(h, logging.FileHandler):
                try:
                    file_handler_exists_for.add(Path(h.baseFilename))
                except Exception:
                    pass

        # Create or update console handler
        if console_handler is None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(ColorFormatter(fmt=self.FMT, datefmt=self.DATE_FMT))
            root_logger.addHandler(console_handler)
        else:
            console_handler.setLevel(level)
            # Always ensure colored formatter on console
            if not isinstance(console_handler.formatter, ColorFormatter):
                console_handler.setFormatter(ColorFormatter(fmt=self.FMT, datefmt=self.DATE_FMT))

        # Create file handler if requested and not already present
        if log_file is not None:
            log_file_path = Path(log_file)
            if log_file_path not in file_handler_exists_for:
                fh = logging.FileHandler(log_file_path, encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(logging.Formatter(fmt=self.FMT, datefmt=self.DATE_FMT))
                root_logger.addHandler(fh)

        # Reduce noise from common third-party libraries
        for noisy in ("fiona", "rasterio", "shapely", "matplotlib", "urllib3", "osgeo"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Announce configuration via the module logger so tests can assert on it
        logging.getLogger(__name__).info("Logging is configured.")

    @staticmethod
    def _set_error_handler() -> None:
        """
        Link GDAL error output to the configured Pyramids logger and install the handler.

        Low-severity GDAL messages (below CE_Warning) are printed to stdout to preserve
        expected behavior in tests and certain GDAL workflows.
        """
        # Use a child of the module logger so it inherits the handlers/format configured in setup_logging()
        log = logging.getLogger(__name__).getChild("gdal")

        def gdal_error_handler(err_class, err_num, err_msg):
            """Error handler for GDAL mapped to logging levels."""
            try:
                # For error classes lower than CE_Warning, print to stdout (expected by tests)
                if err_class is not None and err_class < getattr(gdal, "CE_Warning", 2):
                    print(f"GDAL error (class {err_class}, number {err_num}): {err_msg}")
                    return

                # Map GDAL error classes to logging levels
                if err_class == gdal.CE_Debug:
                    log.debug(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Warning:
                    log.warning(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Failure:
                    log.error(f"GDAL[{err_num}] {err_msg}")
                elif err_class == gdal.CE_Fatal:
                    log.critical(f"GDAL[{err_num}] {err_msg}")
                else:
                    log.error(f"GDAL(class={err_class}, code={err_num}) {err_msg}")
            except Exception:
                # Fallback to error level if mapping fails for any reason
                log.error(f"GDAL(class={err_class}, code={err_num}) {err_msg}")

        gdal.PushErrorHandler(gdal_error_handler)


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

    def setup_logging(self, level: Union[int, str] = logging.INFO, log_file: Union[str, Path, None] = None):
        """
        Configure application-wide logging for Pyramids by delegating to LoggerManager.

        This method preserves the public API while separating responsibilities. It delegates
        the actual logging configuration to LoggerManager.setup_logging and then sets
        self.logger and self._logging_configured for convenience/compatibility.
        """
        LoggerManager(level=level, log_file=log_file)
        self._logging_configured = True
        self.logger = logging.getLogger(__name__)
