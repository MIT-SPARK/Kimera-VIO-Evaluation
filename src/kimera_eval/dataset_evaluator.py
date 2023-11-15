"""Main library for evaluation."""
import kimera_eval.plotting
import kimera_eval.trajectory_metrics

import matplotlib.pyplot as plt
import itertools
import logging
import pathlib


class DatasetEvaluator:
    """DatasetEvaluator is used to evaluate performance of the pipeline on datasets."""

    def __init__(self, config, csv_name="traj_vio.csv"):
        """Make an evaluation class to handle evaluting kimera results."""
        self.config = config
        self.traj_vio_csv_name = csv_name
        self.traj_gt_csv_name = "traj_gt.csv"
        self.traj_pgo_csv_name = "traj_pgo.csv"

    def evaluate(self, all_results_path: pathlib.Path):
        """Run datasets if necessary, evaluate all."""
        combos = itertools.product(self.config.sequences, self.config.pipelines)
        failed = []
        for sequence, pipeline in combos:
            result_path = all_results_path / sequence.name / pipeline.name

            logging.info(f"starting analysis of {pipeline.name} for {sequence.name}...")
            logging.debug(f"{sequence.name}:{pipeline.name} located @ '{result_path}'")
            try:
                self._run_analysis(sequence, result_path)
            except Exception as e:
                logging.critical(f"analysis failed for '{result_path}': {e}")
                failed.append(f"{sequence.name}:{pipeline.name}")
                continue

            logging.info(f"finished analysis of {pipeline.name} for {sequence.name}")

        return failed

    def _run_analysis(self, sequence, result_path):
        logging.debug(f"loading trajectories from '{result_path}'...")
        traj = kimera_eval.trajectory_metrics.TrajectoryGroup.load(
            result_path,
            ref_name=self.traj_gt_csv_name,
            vio_name=self.traj_vio_csv_name,
            pgo_name=self.traj_pgo_csv_name,
        )
        traj = traj.align(sequence.analysis).reduce(sequence.analysis)

        vio_results = kimera_eval.trajectory_metrics.TrajectoryResults.analyze(
            sequence.analysis, traj.vio
        )
        vio_results.save(result_path / "results_vio.pickle")
        pgo_results = None
        if traj.pgo is not None:
            pgo_results = kimera_eval.trajectory_metrics.TrajectoryResults.analyze(
                sequence.analysis, traj.pgo
            )
            pgo_results.save(result_path / "results_pgo.pickle")

        plots = kimera_eval.plotting.add_results_to_collection(
            traj.vio, vio_results, "VIO"
        )
        if traj.pgo is not None:
            plots = kimera_eval.plotting.add_results_to_collection(
                traj.vio,
                pgo_results,
                "PGO + VIO",
                plots=plots,
                extra_trajectory=traj.pgo.est,
            )

        pdf_path = result_path / "plots.pdf"
        logging.debug(f"Saving plots to: {pdf_path}")
        plots.export(str(pdf_path))
        plt.close("all")
