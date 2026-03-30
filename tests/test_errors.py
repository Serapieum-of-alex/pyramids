"""Tests for pyramids.base._errors custom exception classes.

Ensures every exception class can be instantiated and that its
constructor logs the error message via the module logger.
"""

import logging

import pytest

from pyramids.base._errors import (
    AlignmentError,
    DatasetNotFoundError,
    DriverNotExistError,
    FailedToSaveError,
    FileFormatNotSupportedError,
    NoDataValueError,
    OptionalPackageDoesNotExist,
    OutOfBoundsError,
    ReadOnlyError,
)


class TestReadOnlyError:
    """Tests for ReadOnlyError."""

    def test_instantiation_logs_message(self, caplog):
        """ReadOnlyError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(ReadOnlyError):
                raise ReadOnlyError("read-only test message")
        assert (
            "read-only test message" in caplog.text
        ), "ReadOnlyError should log its message"


class TestDatasetNotFoundError:
    """Tests for DatasetNotFoundError."""

    def test_instantiation_logs_message(self, caplog):
        """DatasetNotFoundError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(DatasetNotFoundError):
                raise DatasetNotFoundError("dataset not found")
        assert (
            "dataset not found" in caplog.text
        ), "DatasetNotFoundError should log its message"


class TestNoDataValueError:
    """Tests for NoDataValueError."""

    def test_instantiation_logs_message(self, caplog):
        """NoDataValueError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(NoDataValueError):
                raise NoDataValueError("no data value error")
        assert (
            "no data value error" in caplog.text
        ), "NoDataValueError should log its message"


class TestAlignmentError:
    """Tests for AlignmentError."""

    def test_instantiation_logs_message(self, caplog):
        """AlignmentError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(AlignmentError):
                raise AlignmentError("alignment error message")
        assert (
            "alignment error message" in caplog.text
        ), "AlignmentError should log its message"


class TestDriverNotExistError:
    """Tests for DriverNotExistError."""

    def test_instantiation_logs_message(self, caplog):
        """DriverNotExistError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(DriverNotExistError):
                raise DriverNotExistError("driver does not exist")
        assert (
            "driver does not exist" in caplog.text
        ), "DriverNotExistError should log its message"


class TestFileFormatNotSupportedError:
    """Tests for FileFormatNotSupportedError."""

    def test_instantiation_logs_message(self, caplog):
        """FileFormatNotSupportedError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(FileFormatNotSupportedError):
                raise FileFormatNotSupportedError("format not supported")
        assert (
            "format not supported" in caplog.text
        ), "FileFormatNotSupportedError should log its message"


class TestOptionalPackageDoesNotExist:
    """Tests for OptionalPackageDoesNotExist."""

    def test_instantiation_logs_message(self, caplog):
        """OptionalPackageDoesNotExist should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(OptionalPackageDoesNotExist):
                raise OptionalPackageDoesNotExist("optional package missing")
        assert (
            "optional package missing" in caplog.text
        ), "OptionalPackageDoesNotExist should log its message"


class TestFailedToSaveError:
    """Tests for FailedToSaveError."""

    def test_instantiation_logs_message(self, caplog):
        """FailedToSaveError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(FailedToSaveError):
                raise FailedToSaveError("failed to save")
        assert (
            "failed to save" in caplog.text
        ), "FailedToSaveError should log its message"


class TestOutOfBoundsError:
    """Tests for OutOfBoundsError."""

    def test_instantiation_logs_message(self, caplog):
        """OutOfBoundsError should log the provided message at ERROR level."""
        with caplog.at_level(logging.ERROR, logger="pyramids.base._errors"):
            with pytest.raises(OutOfBoundsError):
                raise OutOfBoundsError("out of bounds")
        assert "out of bounds" in caplog.text, "OutOfBoundsError should log its message"
