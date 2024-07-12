"""Configuration module for the pyramids package."""

import yaml
import logging
from osgeo import gdal, ogr
from . import __path__


class Config:
    """Configuration class for the pyramids package."""

    def __init__(self, config_file="config.yaml"):
        """Initialize the configuration."""
        self.config_file = config_file
        self.config = self.load_config()
        self.initialize_gdal()
        self.setup_logging()

    def load_config(self):
        """Load the configuration from the config file."""
        with open(f"{__path__[0]}/{self.config_file}", "r") as file:
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

    def setup_logging(self):
        """Set up the logging configuration."""
        log_config = self.config.get("logging", {})
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
