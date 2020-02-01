"""
Logger object
=============

Different logging levels are available: debug, info, warning, error and critical.
"""

import logging


_level_to_tag_map = {
    logging.CRITICAL: 'E',
    logging.ERROR: 'E',
    logging.WARNING: 'W',
    logging.INFO: 'I',
    logging.DEBUG: 'D',
    logging.NOTSET: 'V',
}


class ConsoleFormatter(logging.Formatter):

    def __init__(self, message_format, timestamp_format):
        logging.Formatter.__init__(self, fmt=message_format, datefmt=timestamp_format)

    def format(self, record: logging.LogRecord):
        tag: str = _get_tag(record.levelno)
        record.levelname = tag
        return logging.Formatter.format(self, record)


def _get_tag(log_level) -> str:
    if log_level in _level_to_tag_map:
        return _level_to_tag_map.get(log_level)
    else:
        return _level_to_tag_map.get(logging.NOTSET)


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level=logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # 06-03 14:38:23.783/I: app.py:13 - Initializing app...
    formatter = ConsoleFormatter('%(asctime)s.%(msecs)d/%(levelname)s: %(filename)s:%(lineno)d - %(message)s',
                                 '%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(logging.NullHandler())
    logger.addHandler(console_handler)
    return logger
