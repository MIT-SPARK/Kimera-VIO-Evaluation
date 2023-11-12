"""Main library for evaluation."""
import kimera_eval.plotting
import kimera_eval.trajectory_metrics

import yaml
import logging
import pathlib


class DatasetEvaluator:
    """DatasetEvaluator is used to evaluate performance of the pipeline on datasets."""

    def __init__(self, config, result_path, csv_name="traj_vio.csv"):
        """Make an evaluation class to handle evaluting kimera results."""
        self.result_path = result_path
        self.traj_vio_csv_name = csv_name
        self.traj_gt_csv_name = "traj_gt.csv"
        self.traj_pgo_csv_name = "traj_pgo.csv"

    def evaluate(self):
        """Run datasets if necessary, evaluate all."""
        for dataset in self.datasets_to_eval:
            logging.info("evaluating dataset {dataset['name']}")
            for pipeline in dataset["pipelines"]:
                if not self._evaluate_run(pipeline, dataset):
                    continue

    def _evaluate_run(self, pipeline, dataset):
        dataset_name = dataset["name"]
        results_path = self.results_path / dataset_name / pipeline

        logging.info(f"starting analysis of pipeline '{pipeline}'...")
        logging.debug(f"{dataset_name}:{pipeline} results located @ '{results_path}'")
        try:
            self._run_analysis(dataset, results_path)
        except Exception as e:
            logging.critical(f"analysis failed '{results_path}': {e}")
            return False

        logging.info(f"finished analysis of pipeline '{pipeline}'")
        return True

    def _run_analysis(self, dataset, results_path):
        dataset_name = dataset["name"]
        segments = dataset["segments"]

        logging.debug(f"loading trajectories from '{results_path}'...")
        traj = kimera_eval.trajectory_metrics.TrajectoryGroup.load(
            results_path,
            ref_name=self.traj_gt_csv_name,
            vio_name=self.traj_vio_csv_name,
            pgo_name=self.traj_pgo_csv_name,
        )

        logging.info("registering and aligning trajectories")
        traj_args = {
            "discard_n_start_poses": dataset["discard_n_start_poses"],
            "discard_n_end_poses": dataset["discard_n_end_poses"],
        }
        reduced = traj.aligned(**traj_args).reduced(**traj_args)

        vio_results = reduced.vio.analyze(segments)
        pgo_results = None if reduced.pgo is None else reduced.pgo.analyze(segments)
        self.save_results_to_file(vio_results, "results_vio", results_path)
        if pgo_results is not None:
            self.save_results_to_file(pgo_results, "results_pgo", results_path)

        logging.debug(f"Plotting: {dataset_name}")
        plots = kimera_eval.plotting.add_results_to_collection(
            reduced.vio, vio_results, "VIO"
        )
        if reduced.pgo is not None:
            plots = kimera_eval.plotting.add_results_to_collection(
                reduced.vio,
                pgo_results,
                "PGO + VIO",
                plots=plots,
                extra_trajectory=reduced.pgo.est,
            )

        self.save_plots_to_file(plots, results_path)

    def save_results_to_file(self, results, title, dataset_pipeline_result_dir):
        """
        Write a result dictionary to file as a yaml file.

        Args:
            results: a dictionary trajectory statistics
            title: filename without extension
            dataset_pipeline_result_dir: directory to save file to
        """
        dataset_path = pathlib.Path(dataset_pipeline_result_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)
        results_path = dataset_path / f"{title}.yaml"
        logging.debug(f"Saving analysis results to: {results_path}")

        with results_path.open("w") as fout:
            fout.write(yaml.dump(results, default_flow_style=False))

    def save_plots_to_file(self, plot_collection, dataset_pipeline_result_dir):
        """Wrie plot collection to disk as both eps and pdf."""
        dataset_path = pathlib.Path(dataset_pipeline_result_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)
        pdf_path = dataset_path / "plots.pdf"
        logging.debug(f"Saving plots to: {pdf_path}")
        plot_collection.export(pdf_path, False)
