"""Custom Errors."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _PyramidsError(Exception):
    """Base class for all pyramids exceptions.

    Logs the error message at DEBUG level on construction for traceability,
    even when the exception is caught and handled. DEBUG is hidden by default
    and only appears when verbose logging is enabled.
    """

    def __init__(self, message: str):
        super().__init__(message)
        logger.debug(f"{type(self).__name__}: {message}")


class ReadOnlyError(_PyramidsError):
    """ReadOnlyError."""


class DatasetNotFoundError(_PyramidsError):
    """DatasetNotFoundError."""


class NoDataValueError(_PyramidsError):
    """NoDataValueError."""


class AlignmentError(_PyramidsError):
    """Alignment Error."""


class DriverNotExistError(_PyramidsError):
    """Driver-Not-exist Error."""


class FileFormatNotSupportedError(_PyramidsError):
    """File Format Not Supported."""


class OptionalPackageDoesNotExist(_PyramidsError):
    """Optional Package does not exist."""


class FailedToSaveError(_PyramidsError):
    """Failed to save error."""


class OutOfBoundsError(_PyramidsError):
    """Out-of-bounds error."""
