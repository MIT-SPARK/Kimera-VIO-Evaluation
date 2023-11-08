"""Main library for evaluation."""
import yaml
import pandas as pd
import logging
import pathlib


class DatasetEvaluator:
    """DatasetEvaluator is used to evaluate performance of the pipeline on datasets."""

    def __init__(self, config, result_path, csv_name="traj_vio.csv"):
        """Make an evaluation class to handle evaluting kimera results."""
        self.result_path = result_path
        self.datasets_to_eval = experiment_params["datasets_to_run"]

        self.display_plots = args.plot
        self.save_results = args.save_results
        self.save_plots = args.save_plots
        self.write_website = args.write_website
        self.save_boxplots = args.save_boxplots
        self.analyze_vio = args.analyze_vio

        self.traj_vio_csv_name = csv_name
        self.traj_gt_csv_name = "traj_gt.csv"
        self.traj_pgo_csv_name = "traj_pgo.csv"

        # Class to write the results to the Jenkins website
        self.website_builder = evt.WebsiteBuilder(
            self.results_dir, self.traj_vio_csv_name
        )

    def evaluate(self):
        """Run datasets if necessary, evaluate all."""
        for dataset in self.datasets_to_eval:
            # Evaluate each dataset if needed:
            if self.analyze_vio:
                self.evaluate_dataset(dataset)

        if self.write_website:
            logging.info("Writing full website...")
            stats = aggregate_ape_results(self.results_dir)

            if len(stats) > 0:
                logging.info("Drawing APE boxplots.")
                draw_ape_boxplots(stats, self.results_dir)

            self.website_builder.write_boxplot_website(stats)
            self.website_builder.write_datasets_website()
            logging.info("Finished writing website.")

        return True

    def evaluate_dataset(self, dataset):
        """Evaluate VIO performance on given dataset."""
        logging.info("Evaluating dataset {dataset['name']}")
        for pipeline in dataset["pipelines"]:
            if not self._evaluate_run(pipeline, dataset):
                logging.error(
                    f"Failed to evaluate `{dataset['name']}` for pipeline `{pipeline}`"
                )
                raise Exception("Failed evaluation.")

        if self.save_boxplots:
            self.save_boxplots_to_file(dataset["pipelines"], dataset)

    def _evaluate_run(self, pipeline_type, dataset):
        """Evaluate performance of one set of Kimera-VIO results.

        Assumes that the files traj_gt.csv traj_vio.csv and traj_pgo.csv are present.

        Args:
            dataset: Dataset to evaluate on
            pipeline_type: Pipeline that was evaluated

        Returns: True if there are no exceptions during evaluation, False otherwise.
        """
        dataset_name = dataset["name"]
        curr_results_path = self.results_path / dataset_name / pipeline_type

        traj_gt_path = curr_results_path / self.traj_gt_csv_name
        traj_vio_path = curr_results_path / self.traj_vio_csv_name
        traj_pgo_path = curr_results_path / self.traj_pgo_csv_name

        logging.debug(f"Analysing results @ `{curr_results_path}`")
        logging.info(f"Starting analysis of pipeline: {pipeline_type}")

        discard_n_start_poses = dataset["discard_n_start_poses"]
        discard_n_end_poses = dataset["discard_n_end_poses"]
        segments = dataset["segments"]

        [plot_collection, results_vio, results_pgo] = self.run_analysis(
            traj_gt_path,
            traj_vio_path,
            traj_pgo_path,
            segments,
            dataset_name,
            discard_n_start_poses,
            discard_n_end_poses,
        )

        if self.save_results:
            if results_vio is not None:
                self.save_results_to_file(results_vio, "results_vio", curr_results_path)
            if results_pgo is not None:
                self.save_results_to_file(results_pgo, "results_pgo", curr_results_path)

        if self.display_plots and plot_collection is not None:
            logging.debug("Displaying plots.")
            plot_collection.show()

        if self.save_plots and plot_collection is not None:
            self.save_plots_to_file(plot_collection, curr_results_path)

        if self.write_website:
            logging.info(f"Writing performance website for dataset: {dataset_name}")
            self.website_builder.add_dataset_to_website(
                dataset_name, pipeline_type, curr_results_path
            )
            self.website_builder.write_datasets_website()

        return True

    def run_analysis(
        self,
        traj_ref_path,
        traj_vio_path,
        traj_pgo_path,
        segments,
        dataset_name="",
        discard_n_start_poses=0,
        discard_n_end_poses=0,
    ):
        """Analyze data from a set of trajectory csv files.

        Args:
            traj_ref_path: string representing filepath of the reference (ground-truth) trajectory.
            traj_vio_path: string representing filepath of the vio estimated trajectory.
            traj_pgo_path: string representing filepath of the pgo estimated trajectory.
            segments: list of segments for RPE calculation, defined in the experiments yaml file.
            dataset_name: string representing the dataset's name
            discard_n_start_poses: int representing number of poses to discard from start of analysis.
            discard_n_end_poses: int representing the number of poses to discard from end of analysis.
        """
        import copy

        # Mind that traj_est_pgo might be None
        traj_ref, traj_est_vio, traj_est_pgo = self.read_traj_files(
            traj_ref_path, traj_vio_path, traj_pgo_path
        )

        # We copy to distinguish from the pgo version that may be created
        traj_ref_vio = copy.deepcopy(traj_ref)

        # Register and align trajectories:
        logging.info("Registering and aligning trajectories")
        traj_ref_vio, traj_est_vio = sync.associate_trajectories(
            traj_ref_vio, traj_est_vio
        )
        traj_est_vio = trajectory.align_trajectory(
            traj_est_vio,
            traj_ref_vio,
            correct_scale=False,
            discard_n_start_poses=int(discard_n_start_poses),
            discard_n_end_poses=int(discard_n_end_poses),
        )

        # We do the same for the PGO trajectory if needed:
        traj_ref_pgo = None
        if traj_est_pgo is not None:
            traj_ref_pgo = copy.deepcopy(traj_ref)
            traj_ref_pgo, traj_est_pgo = sync.associate_trajectories(
                traj_ref_pgo, traj_est_pgo
            )
            traj_est_pgo = trajectory.align_trajectory(
                traj_est_pgo,
                traj_ref_pgo,
                correct_scale=False,
                discard_n_start_poses=int(discard_n_start_poses),
                discard_n_end_poses=int(discard_n_end_poses),
            )

        # We need to pick the lowest num_poses before doing any computation:
        num_of_poses = traj_est_vio.num_poses
        if traj_est_pgo is not None:
            num_of_poses = min(num_of_poses, traj_est_pgo.num_poses)
            traj_est_pgo.reduce_to_ids(
                range(
                    int(discard_n_start_poses),
                    int(num_of_poses - discard_n_end_poses),
                    1,
                )
            )
            traj_ref_pgo.reduce_to_ids(
                range(
                    int(discard_n_start_poses),
                    int(num_of_poses - discard_n_end_poses),
                    1,
                )
            )

        traj_est_vio.reduce_to_ids(
            range(
                int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1
            )
        )
        traj_ref_vio.reduce_to_ids(
            range(
                int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1
            )
        )

        # Calculate all metrics:
        (
            ape_metric_vio,
            rpe_metric_trans_vio,
            rpe_metric_rot_vio,
            results_vio,
        ) = self.process_trajectory_data(traj_ref_vio, traj_est_vio, segments, True)

        # We do the same for the pgo trajectory if needed:
        ape_metric_pgo = None
        rpe_metric_trans_pgo = None
        rpe_metric_rot_pgo = None
        results_pgo = None
        if traj_est_pgo is not None:
            (
                ape_metric_pgo,
                rpe_metric_trans_pgo,
                rpe_metric_rot_pgo,
                results_pgo,
            ) = self.process_trajectory_data(
                traj_ref_pgo, traj_est_pgo, segments, False
            )

        # Generate plots for return:
        plot_collection = None
        if self.display_plots or self.save_plots:
            logging.debug(f"Plotting: {dataset_name}")
            plot_collection = plot.PlotCollection("Example")

            if traj_est_pgo is not None:
                # APE Metric Plot:
                plot_collection.add_figure(
                    "PGO_APE_translation",
                    plot_metric(ape_metric_pgo, "PGO + VIO APE Translation"),
                )

                # Trajectory Colormapped with ATE Plot:
                plot_collection.add_figure(
                    "PGO_APE_translation_trajectory_error",
                    plot_traj_colormap_ape(
                        ape_metric_pgo,
                        traj_ref_pgo,
                        traj_est_vio,
                        traj_est_pgo,
                        "PGO + VIO ATE Mapped Onto Trajectory",
                    ),
                )

                # RPE Translation Metric Plot:
                plot_collection.add_figure(
                    "PGO_RPE_translation",
                    plot_metric(rpe_metric_trans_pgo, "PGO + VIO RPE Translation"),
                )

                # Trajectory Colormapped with RTE Plot:
                plot_collection.add_figure(
                    "PGO_RPE_translation_trajectory_error",
                    plot_traj_colormap_rpe(
                        rpe_metric_trans_pgo,
                        traj_ref_pgo,
                        traj_est_vio,
                        traj_est_pgo,
                        "PGO + VIO RPE Translation Error Mapped Onto Trajectory",
                    ),
                )

                # RPE Rotation Metric Plot:
                plot_collection.add_figure(
                    "PGO_RPE_Rotation",
                    plot_metric(rpe_metric_rot_pgo, "PGO + VIO RPE Rotation"),
                )

                # Trajectory Colormapped with RTE Plot:
                plot_collection.add_figure(
                    "PGO_RPE_rotation_trajectory_error",
                    plot_traj_colormap_rpe(
                        rpe_metric_rot_pgo,
                        traj_ref_pgo,
                        traj_est_vio,
                        traj_est_pgo,
                        "PGO + VIO RPE Rotation Error Mapped Onto Trajectory",
                    ),
                )

            # Plot VIO results
            plot_collection.add_figure(
                "VIO_APE_translation",
                plot_metric(ape_metric_vio, "VIO APE Translation"),
            )

            plot_collection.add_figure(
                "VIO_APE_translation_trajectory_error",
                plot_traj_colormap_ape(
                    ape_metric_vio,
                    traj_ref_vio,
                    traj_est_vio,
                    None,
                    "VIO ATE Mapped Onto Trajectory",
                ),
            )

            plot_collection.add_figure(
                "VIO_RPE_translation",
                plot_metric(rpe_metric_trans_vio, "VIO RPE Translation"),
            )

            plot_collection.add_figure(
                "VIO_RPE_translation_trajectory_error",
                plot_traj_colormap_rpe(
                    rpe_metric_trans_vio,
                    traj_ref_vio,
                    traj_est_vio,
                    None,
                    "VIO RPE Translation Error Mapped Onto Trajectory",
                ),
            )

            plot_collection.add_figure(
                "VIO_RPE_Rotation", plot_metric(rpe_metric_rot_vio, "VIO RPE Rotation")
            )

            plot_collection.add_figure(
                "VIO_RPE_rotation_trajectory_error",
                plot_traj_colormap_rpe(
                    rpe_metric_rot_vio,
                    traj_ref_vio,
                    traj_est_vio,
                    None,
                    "VIO RPE Rotation Error Mapped Onto Trajectory",
                ),
            )

        return [plot_collection, results_vio, results_pgo]

    def process_trajectory_data(self, traj_ref, traj_est, segments, is_vio_traj=True):
        """Process trajectory data."""
        suffix = "VIO" if is_vio_traj else "PGO"
        data = (traj_ref, traj_est)

        logging.info(f"Calculating APE translation part for {suffix}")
        ape_metric = get_ape_trans(data)
        ape_result = ape_metric.get_result()
        logging.debug(f"APE translation: {ape_result.stats['mean']}")

        logging.info(f"Calculating RPE translation part for {suffix}")
        rpe_metric_trans = get_rpe_trans(data)

        logging.info(f"Calculating RPE rotation angle for {suffix}")
        rpe_metric_rot = get_rpe_rot(data)

        # Collect results:
        results = dict()
        results["absolute_errors"] = ape_result

        results["relative_errors"] = self.calc_rpe_results(
            rpe_metric_trans, rpe_metric_rot, data, segments
        )

        # Add as well how long hte trajectory was.
        results["trajectory_length_m"] = traj_est.path_length()

        return (ape_metric, rpe_metric_trans, rpe_metric_rot, results)

    def read_traj_files(self, traj_ref_path, traj_vio_path, traj_pgo_path):
        """Outputs PoseTrajectory3D objects for csv trajectory files.

        Args:
            traj_ref_path: string representing filepath of the reference (ground-truth) trajectory.
            traj_vio_path: string representing filepath of the vio estimated trajectory.
            traj_pgo_path: string representing filepath of the pgo estimated trajectory.

        Returns: A 3-tuple with the PoseTrajectory3D objects representing the reference trajectory,
            vio trajectory, and pgo trajectory in that order.
            NOTE: traj_est_pgo is optional and might be None
        """
        # Read reference trajectory file:
        traj_ref = None
        try:
            traj_ref = pandas_bridge.df_to_trajectory(
                pd.read_csv(traj_ref_path, sep=",", index_col=0)
            )
        except IOError as e:
            raise Exception(
                "\033[91mMissing ground-truth output csv! \033[93m {}.".format(e)
            )

        # Read estimated vio trajectory file:
        traj_est_vio = None
        try:
            traj_est_vio = pandas_bridge.df_to_trajectory(
                pd.read_csv(traj_vio_path, sep=",", index_col=0)
            )
        except IOError as e:
            raise Exception(
                "\033[91mMissing vio estimated output csv! \033[93m {}.".format(e)
            )

        # Read estimated pgo trajectory file:
        traj_est_pgo = None
        try:
            traj_est_pgo = pandas_bridge.df_to_trajectory(
                pd.read_csv(traj_pgo_path, sep=",", index_col=0)
            )
        except IOError as e:
            log.warning("Missing pgo estimated output csv: {}.".format(e))
            log.warning("Not plotting pgo results.")

        return (traj_ref, traj_est_vio, traj_est_pgo)

    def calc_rpe_results(self, rpe_metric_trans, rpe_metric_rot, data, segments):
        """Create and return a dictionary containing stats and results RRE and RTE for a datset.

        Args:
            rpe_metric_trans: an evo.core.metric object representing the RTE.
            rpe_metric_rot: an evo.core.metric object representing the RRE.
            data: a 2-tuple with reference and estimated trajectories as PoseTrajectory3D objects
                in that order.
            segments: a list of segments for RPE.

        Returns: a dictionary containing all relevant RPE results.
        """
        # Calculate RPE results of segments and save
        rpe_results = dict()
        for segment in segments:
            rpe_results[segment] = dict()
            print_purple("RPE analysis of segment: %d" % segment)
            print_lightpurple("Calculating RPE segment translation part")
            rpe_segment_metric_trans = metrics.RPE(
                metrics.PoseRelation.translation_part,
                float(segment),
                metrics.Unit.meters,
                0.01,
                True,
            )
            rpe_segment_metric_trans.process_data(data)
            # TODO(Toni): Save RPE computation results rather than the statistics
            # you can compute statistics later... Like done for ape!
            rpe_segment_stats_trans = rpe_segment_metric_trans.get_all_statistics()
            rpe_results[segment]["rpe_trans"] = rpe_segment_stats_trans

            print_lightpurple("Calculating RPE segment rotation angle")
            rpe_segment_metric_rot = metrics.RPE(
                metrics.PoseRelation.rotation_angle_deg,
                float(segment),
                metrics.Unit.meters,
                0.01,
                True,
            )
            rpe_segment_metric_rot.process_data(data)
            rpe_segment_stats_rot = rpe_segment_metric_rot.get_all_statistics()
            rpe_results[segment]["rpe_rot"] = rpe_segment_stats_rot

        return rpe_results

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
        print(f"Saving analysis results to: {results_path}")

        with results_path.open("w") as fout:
            fout.write(yaml.dump(results, default_flow_style=False))

    def save_plots_to_file(
        self, plot_collection, dataset_pipeline_result_dir, save_pdf=True
    ):
        """Wrie plot collection to disk as both eps and pdf.

        Args:
            - plot_collection: a PlotCollection containing all the plots to save to file.
            - dataset_pipeline_result_dir: a string representing the filepath for the location to
                which the plot files are saved.
            - save_pdf: whether to save figures to pdf or eps format
        """
        # Config output format (pdf, eps, ...) using evo_config...
        if save_pdf:
            pdf_output_file_path = os.path.join(
                dataset_pipeline_result_dir, "plots.pdf"
            )
            evt.print_green("Saving plots to: %s" % pdf_output_file_path)
            plot_collection.export(pdf_output_file_path, False)
        else:
            eps_output_file_path = os.path.join(
                dataset_pipeline_result_dir, "plots.eps"
            )
            evt.print_green("Saving plots to: %s" % eps_output_file_path)
            plot_collection.export(eps_output_file_path, False)

    def save_boxplots_to_file(self, pipelines_to_run_list, dataset):
        """Writes boxplots for all pipelines of a given dataset to disk.

        Args:
            pipelines_to_run_list: a list containing all pipelines to run for a dataset.
            dataset: a single dataset, as taken from the experiments yaml file.
        """
        dataset_name = dataset["name"]
        dataset_segments = dataset["segments"]

        # TODO(Toni) is this really saving the boxplots?
        stats = dict()
        for pipeline_type in pipelines_to_run_list:
            results_dataset_dir = os.path.join(self.results_dir, dataset_name)
            results_vio = os.path.join(
                results_dataset_dir, pipeline_type, "results_vio.yaml"
            )
            if not os.path.exists(results_vio):
                raise Exception(
                    "\033[91mCannot plot boxplots: missing results for %s pipeline \
                                and dataset: %s"
                    % (pipeline_type, dataset_name)
                    + "\033[99m \n \
                                Expected results here: %s"
                    % results_vio
                    + "\033[99m \n \
                                Ensure that `--save_results` is passed at commandline."
                )

            try:
                stats[pipeline_type] = yaml.load(
                    open(results_vio, "r"), Loader=yaml.Loader
                )
            except yaml.YAMLError as e:
                raise Exception("Error in results_vio file: ", e)

            log.info("Check stats %s in %s" % (pipeline_type, results_vio))
            try:
                evt.check_stats(stats[pipeline_type])
            except Exception as e:
                log.warning(e)

        if "relative_errors" in stats:
            log.info("Drawing RPE boxplots.")
            evt.draw_rpe_boxplots(results_dataset_dir, stats, len(dataset_segments))
        else:
            log.info("Missing RPE results, not drawing RPE boxplots.")
