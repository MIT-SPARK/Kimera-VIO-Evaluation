"""Main library for evaluation."""
from kimera_eval.experiment_config import ExperimentConfig

import itertools
import logging
import pathlib
import subprocess
import shutil
import sys
import time


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
    args = experiment.args + pipeline.args + sequence.args
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

    logging.debug(f"  - starting experiment with args: {args}")
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
        logging.info("runing experiments...")
        for sequence in self.config.sequences:
            logging.info(f"running dataset '{sequence.name}' ...")
            dataset_status = {}
            for pipeline in self.config.pipelines:
                output_path = self.result_path / sequence.name / pipeline.name
                if output_path.exists():
                    if allow_removal:
                        shutil.rmtree(output_path)
                    else:
                        logging.warning(f"  - skipping pipeline '{pipeline.name}' ...")
                        dataset_status[pipeline.name] = (False, "Results exist")
                        continue

                output_path.mkdir(parents=True, exist_ok=False)

                logging.info(f"  - running pipeline '{pipeline.name}' ...")
                try:
                    start = time.perf_counter()
                    valid = _run_vio(
                        self.config,
                        pipeline,
                        sequence,
                        output_path,
                        minloglevel=minloglevel,
                    )
                    stop = time.perf_counter()
                    dataset_status[pipeline.name] = (
                        valid,
                        f"elapsed: {stop - start} [s]",
                    )
                except Exception as e:
                    dataset_status[pipeline.name] = (False, f"{e}")

            status[sequence.name] = dataset_status

        return status
