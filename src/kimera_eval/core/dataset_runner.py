"""Main library for evaluation."""
from kimera_eval.core.experiment_config import ExperimentConfig

import itertools
import logging
import pathlib
import subprocess
import shutil
import sys
import time
import yaml


def _run_vio(
    experiment,
    pipeline,
    sequence,
    output_path,
    minloglevel=0,
    spinner_width=10,
    spin_time=0.1,
):
    spinner = itertools.cycle(["-", "/", "|", "\\"])
    args = experiment.get_args(pipeline)
    args += sequence.args
    args += [
        f"--output_path={output_path}",
        f"--dataset_path={experiment.dataset_path / sequence.name}",
    ]

    args += [
        "--logtostderr=1",
        "--colorlogtostderr=1",
        "--log_prefix=1",
        "--log_output=true",
        f"--minloglevel={minloglevel}",
    ]

    pipe = subprocess.Popen(args)
    while pipe.poll() is None:
        if minloglevel > 0:
            # display spinner to show progress
            sys.stdout.write(next(spinner) * spinner_width)
            sys.stdout.flush()
            sys.stdout.write("\b" * spinner_width)

        time.sleep(spin_time)

    return pipe.wait() == 0


class DatasetRunner:
    """DatasetRunner is used to run the pipeline on datasets."""

    def __init__(self, experiment_config: ExperimentConfig, result_path: pathlib.Path):
        """
        Create a dataset runner.

        Args:
            experiment_config: Configuration for experiments
            result_path: Path to save results
        """
        self.config = experiment_config
        self.result_path = result_path

    def run_all(self, allow_removal=False, minloglevel=2, **kwargs):
        """
        Run all datasets and pipelines.

        Args:
            allow_removal: Allow removal of outputs
            minloglevel: Min log level setting for Kimera-VIO
            **kwargs: Arguments to be passed to _run_vio

        Returns:
            Dict[str, Dict[str, bool]]: Pipeline results
        """
        status = {}
        logging.info("Runing experiments...")
        for sequence in self.sequences:
            logging.info(f"Runing dataset '{sequence.name}'...")
            dataset_status = {}
            for pipeline in self.pipelines:
                output_path = self.result_path / sequence.name / pipeline.name
                if output_path.exists():
                    if allow_removal:
                        shutil.rmtree(output_path)
                    else:
                        status[sequence.name] = False
                        continue

                output_path.mkdir(parents=True, exist_ok=False)

                logging.info(f"Running pipeline '{pipeline}'...")
                dataset_status[pipeline.name] = _run_vio(
                    self.config,
                    pipeline,
                    sequence,
                    output_path,
                    minloglevel=minloglevel,
                )

            status[sequence.name] = dataset_status

        return status
