"""Custom Errors."""

import logging

logger = logging.getLogger(__name__)


class ReadOnlyError(Exception):
    """ReadOnlyError."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class DatasetNoFoundError(Exception):
    """DatasetNoFoundError."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class NoDataValueError(Exception):
    """NoDataValueError."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class AlignmentError(Exception):
    """Alignment Error."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class DriverNotExistError(Exception):
    """Driver-Not-exist Error."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class FileFormatNotSupported(Exception):
    """File Format Not Supported."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class OptionalPackageDoesNotExist(Exception):
    """Optional Package does not exist."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class FailedToSaveError(Exception):
    """Failed to save error."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)


class OutOfBoundsError(Exception):
    """Out-of-bounds error."""

    def __init__(self, error_message: str):
        """__init__."""
        logger.error(error_message)
