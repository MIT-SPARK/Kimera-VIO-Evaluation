"""Main entry point for running evaluations."""
import kimera_eval.core.logger
import kimera_eval.cli

from kimera_eval.core.trajectory_metrics import *
from kimera_eval.core.experiment_config import *

import click


@click.group()
@click.option("-l", "--log-level", default="INFO", help="log level")
def main(log_level):
    """Command-line tool to evaluate Kimera-VIO."""
    kimera_eval.core.logger.configure_logging(level=log_level)


main.add_command(kimera_eval.cli.run)
