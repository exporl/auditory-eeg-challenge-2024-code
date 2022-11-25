"""Logging tools."""
import logging


def enable_logging(**kwargs):
    """Enable logging.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to pass to logging.basicConfig
    """
    if "level" not in kwargs:
        kwargs["level"] = logging.INFO
    if "format" not in kwargs:
        kwargs["format"] = "%(asctime)s|%(levelname)s|%(pathname)s: " \
                           "%(message)s"

    logging.basicConfig(**kwargs)
