import logging


def get_logger(name: str = "solar_storm") -> logging.Logger:
    """
    Returns a module-level logger with a simple INFO-level StreamHandler.
    Calling this multiple times with the same name is safe — handlers are
    only added once.

    Usage:
        from src.utils.logging_utils import get_logger
        log = get_logger(__name__)
        log.info("Loading data...")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger