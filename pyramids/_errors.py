from loguru import logger


class ReadOnlyError(Exception):
    """ReadOnlyError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class DatasetNoFoundError(Exception):
    """DatasetNoFoundError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class NoDataValueError(Exception):
    """NoDataValueError"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class AlignmentError(Exception):
    """Alignment Error"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class DriverNotExistError(Exception):
    """Driver Not exist Error"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class FileFormatNoSupported(Exception):
    """File Format Not Supported"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class OptionalPackageDoesNontExist(Exception):
    """Optional Package does not exist"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass


class FailedToSaveError(Exception):
    """Failed to save error"""

    def __init__(self, error_message: str):
        logger.error(error_message)

    pass
