"""Main entry point for running evaluations."""
import kimera_eval.logger
import kimera_eval.tools.cli

from kimera_eval.trajectory_metrics import *
from kimera_eval.experiment_config import *
from kimera_eval.dataset_runner import *

import click


@click.group()
@click.option("-l", "--log-level", default="INFO", help="log level")
def main(log_level):
    """Command-line tool to evaluate Kimera-VIO."""
    kimera_eval.logger.configure_logging(level=log_level)


main.add_command(kimera_eval.tools.cli.run)
