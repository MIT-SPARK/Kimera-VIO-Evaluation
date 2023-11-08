"""Configuration of python logging library."""
import logging
import click


DEFAULT_FORMAT = "[%(levelname)s] %(asctime)s @ %(filename)s:%(lineno)s: %(message)s"


class ClickHandler(logging.Handler):
    """Logging handler to color output using click."""

    def __init__(
        self,
        debug_color="green",
        warn_color="yellow",
        error_color="red",
        info_color=None,
    ):
        """Initialize the handler."""
        logging.Handler.__init__(self)
        self._debug_color = debug_color
        self._info_color = info_color
        self._warn_color = warn_color
        self._error_color = error_color

    def emit(self, record):
        """Send log record to console with appropriate coloring."""
        msg = self.format(record)

        if record.levelno <= logging.DEBUG:
            click.secho(msg, fg=self._debug_color)
            return

        if record.levelno <= logging.INFO:
            click.secho(msg, fg=self._info_color)
            return

        if record.levelno <= logging.WARNING:
            click.secho(msg, fg=self._warn_color, err=True)
            return

        click.secho(msg, fg=self._error_color, err=True)


def configure_logging(level="INFO", default_format=DEFAULT_FORMAT, **kwargs):
    """Set logging to use appropriate handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    level_map = {
        logging.getLevelName(x): x
        for x in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
    }

    handler = ClickHandler(**kwargs)
    handler.setLevel(level_map.get(level.upper(), logging.INFO))
    formatter = logging.Formatter(default_format)
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
