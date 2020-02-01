import logging
from app_logging import get_logger
from map import CarApp


if __name__ == '__main__':
    # Configure app logging level here
    logger = get_logger('ai')
    logger.setLevel(level=logging.INFO)

    logger.info('Initializing app...')

    """Run the whole thing"""
    CarApp().run()
