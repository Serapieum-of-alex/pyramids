import logging
from pathlib import Path
from contextlib import contextmanager

from unittest.mock import patch
import io
from contextlib import redirect_stdout
from osgeo import gdal
from pyramids.config import LoggerManager


@contextmanager
def isolated_root_logging():
    """
    Temporarily isolate root logger handlers and level so tests don't
    interfere with each other or the global test suite.
    """
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
        yield root
    finally:
        # Clear any handlers added by the test
        for h in list(root.handlers):
            root.removeHandler(h)
        # Restore originals
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)


def test_console_logging_colored_and_message(capsys):
    with isolated_root_logging():
        # Configure logging at INFO level using LoggerManager
        LoggerManager(level="INFO")

        # Emitted by setup_logging
        out = capsys.readouterr()
        stderr_text = out.err
        # Should have at least one ANSI escape (from ColorFormatter)
        assert "\x1b[" in stderr_text
        assert "Logging is configured." in stderr_text
        assert "pyramids.config" in stderr_text  # logger name

        # Also test that subsequent logs go to the console
        logging.getLogger(__name__).info("hello world")
        out2 = capsys.readouterr()
        assert "hello world" in out2.err
        assert "\x1b[" in out2.err  # colored level name


def test_file_logging_no_colors_and_writes(tmp_path: Path):
    log_file = tmp_path / "test.log"
    with isolated_root_logging():
        LoggerManager(level=logging.DEBUG, log_file=log_file)

        # setup_logging should log a message already
        # Now log something extra
        logging.getLogger("tests.config.test_logger").debug("file handler check")

    # Read file and assert contents
    text = log_file.read_text(encoding="utf-8")
    assert "Logging is configured." in text
    assert "file handler check" in text
    # Ensure no ANSI color sequences in the file
    assert "\x1b[" not in text


def test_idempotent_handlers(tmp_path: Path):
    log_file = tmp_path / "dup.log"
    with isolated_root_logging() as root:
        LoggerManager(level="INFO", log_file=log_file)
        # Call again with the same parameters
        LoggerManager(level="INFO", log_file=log_file)

        # Expect exactly one StreamHandler (console) and one FileHandler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
        assert len(file_handlers) == 1
        # And that file handler targets the same file
        assert Path(file_handlers[0].baseFilename) == log_file



@patch("osgeo.gdal.PushErrorHandler")
def test_set_error_handler_prints_for_low_error_class(mock_push):
    # Install the handler via LoggerManager and capture it from the patched GDAL entry point
    LoggerManager()
    handler = mock_push.call_args[0][0]

    # Invoke the handler with an error class lower than CE_Warning to trigger printing
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler(0, 42, "oops")
    out = buf.getvalue().strip()

    assert out == "GDAL error (class 0, number 42): oops"


def _collect_log_messages(records, logger_name: str):
    return [r for r in records if r.name == logger_name]


@patch("osgeo.gdal.PushErrorHandler")
def test_set_error_handler_logs_severities(mock_push, capsys):
    # Isolate root logging and configure
    with isolated_root_logging():
        LoggerManager(level="DEBUG")

        # Capture the handler installed during construction
        handler = mock_push.call_args[0][0]

        # Emit messages: warning and higher are routed to the configured logger -> console (stderr)
        handler(gdal.CE_Warning, 22, "warn msg")
        handler(gdal.CE_Failure, 33, "fail msg")
        handler(gdal.CE_Fatal, 44, "fatal msg")
        # Unknown class -> ERROR fallback
        handler(999, 55, "unknown class msg")

        out = capsys.readouterr()
        err_text = out.err
        # Assert substrings exist in stderr (avoid brittle timestamp/ANSI sequences)
        assert "pyramids.config.gdal | GDAL[22] warn msg" in err_text
        assert "pyramids.config.gdal | GDAL[33] fail msg" in err_text
        assert "pyramids.config.gdal | GDAL[44] fatal msg" in err_text
        assert "pyramids.config.gdal | GDAL(class=999, code=55) unknown class msg" in err_text
