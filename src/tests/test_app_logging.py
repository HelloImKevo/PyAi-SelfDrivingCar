import logging
from app_logging import get_logger


def test_logging():
    print("Running test...")
    logger: logging.Logger = get_logger('test')
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning("Warning message")
    logger.error("Error message")
    raise SystemExit(1)


def main():
    test_logging()
    print(r'All tests passed. Success! \o/')


main()
