#!/usr/bin/env python

from __future__ import print_function
import copy
import os
from ruamel import yaml
import math
import subprocess
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from shutil import copyfile, move, rmtree, copytree, copy2

import pandas as pd

from evo.core import trajectory, sync, metrics
from evo.tools import plot, pandas_bridge
from evo.tools import file_interface

import evaluation.tools as evt


def aggregate_all_results(results_dir, use_pgo=False):
    """ Aggregate APE results and draw APE boxplot as well as write latex table
    with results:
        Args:
            - result_dir: path to the directory with results ordered as follows:
               \* dataset_name:
               |___\* pipeline_type:
               |   |___results.yaml
               |___\* pipeline_type:
               |   |___results.yaml
               \* dataset_name:
               |___\* pipeline_type:
               |   |___results.yaml
               Basically all subfolders with a results.yaml will be examined.
            - use_pgo: whether to aggregate all results for VIO or for PGO trajectory.
                set to True for PGO and False (default) for VIO
        Returns:
            - stats: a nested dictionary with the statistics and results of all pipelines:
                * First level ordered with dataset_name as keys:
                * Second level ordered with pipeline_type as keys:
                * Each stats[dataset_name][pipeline_type] value has:
                    * absolute_errors: an evo Result type with trajectory and APE stats.
                    * relative_errors: RPE stats.
    """
    import fnmatch
    # Load results.
    log.info("Aggregate dataset results.")
    # Aggregate all stats for each pipeline and dataset
    yaml_filename = 'results_vio.yaml'
    if use_pgo:
        yaml_filename = 'results_pgo.yaml'
    stats = dict()
    for root, dirnames, filenames in os.walk(results_dir):
        for results_filename in fnmatch.filter(filenames, yaml_filename):
            results_filepath = os.path.join(root, results_filename)
            # Get pipeline name
            pipeline_name = os.path.basename(root)
            # Get dataset name
            dataset_name = os.path.basename(os.path.split(root)[0])
            # Collect stats
            if stats.get(dataset_name) is None:
                stats[dataset_name] = dict()

            try:
                stats[dataset_name][pipeline_name] = yaml.load(open(results_filepath, 'r'), Loader=yaml.Loader)
            except yaml.YAMLError as e:
                raise Exception("Error in results file: ", e)
            except:
                log.fatal("\033[1mFailed opening file: \033[0m\n %s" % results_filepath)

            log.debug("Check stats from: " + results_filepath)
            try:
                evt.check_stats(stats[dataset_name][pipeline_name])
            except Exception as e:
                log.warning(e)

    return stats

def aggregate_ape_results(results_dir):
    """ Aggregate APE results and draw APE boxplot as well as write latex table
    with results:
        Args:
            - result_dir: path to the directory with results ordered as follows:
               \* dataset_name:
               |___\* pipeline_type:
               |   |___results.yaml
               |___\* pipeline_type:
               |   |___results.yaml
               \* dataset_name:
               |___\* pipeline_type:
               |   |___results.yaml
               Basically all subfolders with a results.yaml will be examined.

        Returns:
            - stats: a nested dictionary with the statistics and results of all pipelines:
                * First level ordered with dataset_name as keys:
                * Second level ordered with pipeline_type as keys:
                * Each stats[dataset_name][pipeline_type] value has:
                    * absolute_errors: an evo Result type with trajectory and APE stats.
                    * relative_errors: RPE stats.
    """
    stats = aggregate_all_results(results_dir)
    # Draw APE boxplot
    if (len(list(stats.values())) > 0):
        log.info("Drawing APE boxplots.")
        evt.draw_ape_boxplots(stats, results_dir)
        # Write APE table
        log.info("Writing APE latex table.")
        evt.write_latex_table(stats, results_dir)
    return stats


from tqdm import tqdm
""" DatasetRunner is used to run the pipeline on datasets """
class DatasetRunner:
    def __init__(self, experiment_params, args, extra_flagfile_path = ''):
        self.vocabulary_path = os.path.expandvars(experiment_params['vocabulary_path'])
        self.results_dir     = os.path.expandvars(experiment_params['results_dir'])
        self.params_dir      = os.path.expandvars(experiment_params['params_dir'])
        self.dataset_dir     = os.path.expandvars(experiment_params['dataset_dir'])
        self.executable_path = os.path.expandvars(experiment_params['executable_path'])
        self.datasets_to_run = experiment_params['datasets_to_run']
        self.verbose_vio     = args.verbose_sparkvio

        self.extra_flagfile_path = extra_flagfile_path

        self.pipeline_output_dir = os.path.join(self.results_dir, "tmp_output/output/")
        evt.create_full_path_if_not_exists(self.pipeline_output_dir)

    def run_all(self):
        """ Runs all datasets in experiments file. """
        # Run experiments.
        log.info("Run experiments")
        successful_run = True
        for dataset in tqdm(self.datasets_to_run):
            log.info("Run dataset: %s" % dataset['name'])
            if not self.run_dataset(dataset):
                log.info("\033[91m Dataset: %s failed!! \033[00m" %
                         dataset['name'])
                successful_run = False
        return successful_run

    def run_dataset(self, dataset):
        """ Run a single dataset from an experiments file and save all output. This is done
            for every pipeline requested for the dataset.

            Args:
                dataset: a dataset to run as defined in the experiments yaml file.

            Returns: True if all pipelines for the dataset succeed, False otherwise.
        """
        dataset_name = dataset['name']

        has_a_pipeline_failed = False
        pipelines_to_run_list = dataset['pipelines']
        if len(pipelines_to_run_list) == 0:
            log.warning("Not running pipeline...")
        for pipeline_type in pipelines_to_run_list:
            # TODO shouldn't this break when a pipeline has failed? Not necessarily
            # if we want to plot all pipelines except the failing ones.
            evt.print_green("Run pipeline: %s" % pipeline_type)
            pipeline_success = self.__run_vio(dataset, pipeline_type)
            if pipeline_success:
                evt.print_green("Successful pipeline run.")
            else:
                log.error("Failed pipeline run!")
                has_a_pipeline_failed = True

        if not has_a_pipeline_failed:
            evt.print_green("All pipeline runs were successful.")
        evt.print_green("Finished evaluation for dataset: " + dataset_name)
        return not has_a_pipeline_failed

    def __run_vio(self, dataset, pipeline_type):
        """ Performs subprocess call for a specific pipeline on a specific dataset.

            Args:
                dataset: a dataset to run as defined in the experiments yaml file.
                pipeline_type: a pipeline representing a set of parameters to use, as
                    defined in the experiments yaml file for the dataset in question.

            Returns: True if the thread exits successfully, False otherwise.
        """
        def kimera_vio_thread(thread_return, minloglevel=0):
            """ Function to run Kimera-VIO in another thread """
            # Subprocess returns 0 if Ok, any number bigger than 1 if not ok.
            command = "{} \
                    --logtostderr=1 --colorlogtostderr=1 --log_prefix=1 \
                    --minloglevel={} \
                    --dataset_path={}/{} --output_path={} \
                    --params_folder_path={}/{} \
                    --vocabulary_path={} \
                    --flagfile={}/{}/{} --flagfile={}/{}/{} \
                    --flagfile={}/{}/{} --flagfile={}/{}/{} \
                    --flagfile={}/{}/{} --flagfile={}/{} \
                    --visualize=false \
                    --visualize_lmk_type=false \
                    --visualize_mesh=false \
                    --visualize_mesh_with_colored_polygon_clusters=false \
                    --visualize_point_cloud=false \
                    --visualize_convex_hull=false \
                    --visualize_plane_constraints=false \
                    --visualize_planes=false \
                    --visualize_plane_label=false \
                    --visualize_semantic_mesh=false \
                    --visualize_mesh_in_frustum=false \
                    --viz_type=2 \
                    --initial_k={} --final_k={} --use_lcd={} \
                    --log_euroc_gt_data=true --log_output=true".format(
                        self.executable_path,
                        minloglevel,
                        self.dataset_dir, dataset["name"], self.pipeline_output_dir,
                        self.params_dir, pipeline_type,
                        self.vocabulary_path,
                        self.params_dir, pipeline_type, "flags/stereoVIOEuroc.flags",
                        self.params_dir, pipeline_type, "flags/Mesher.flags",
                        self.params_dir, pipeline_type, "flags/VioBackEnd.flags",
                        self.params_dir, pipeline_type, "flags/RegularVioBackEnd.flags",
                        self.params_dir, pipeline_type, "flags/Visualizer3D.flags",
                        self.params_dir, self.extra_flagfile_path,
                        dataset["initial_frame"], dataset["final_frame"], dataset["use_lcd"])
            # print("Starting Kimera-VIO with command:\n")
            # print(command)
            return_code = subprocess.call(command, shell=True)
            if return_code is 0:
                thread_return['success'] = True
            else:
                thread_return['success'] = False


        import threading
        import time
        import itertools, sys # just for spinner
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        thread_return={'success': False}
        minloglevel = 2 # Set Kimera-VIO verbosity level to ERROR
        if self.verbose_vio:
            minloglevel = 0 # Set Kimera-VIO verbosity level to INFO
        thread = threading.Thread(target=kimera_vio_thread, args=(thread_return, minloglevel,))
        thread.start()
        while thread.is_alive():
            if not self.verbose_vio:
                # If Kimera-VIO is not in verbose mode, the user might think the python script is hanging.
                # So, instead, display a spinner of 80 characters.
                sys.stdout.write(next(spinner) * 10)  # write the next character
                sys.stdout.flush()                     # flush stdout buffer (actual character display)
                sys.stdout.write('\b' * 10)            # erase the last written char
            time.sleep(0.100) # Sleep 100ms while Kimera-VIO is running
        thread.join()

        # Move output files for future evaluation:
        self.move_output_files(pipeline_type, dataset)

        return thread_return['success']

    def move_output_files(self, pipeline_type, dataset):
        """ Moves all output files for a particular pipeline and dataset
            from their temporary logging location during runtime to the evaluation location.

            Args:
                pipeline_type: a pipeline representing a set of parameters to use, as
                    defined in the experiments yaml file for the dataset in question.
                dataset: a dataset to run as defined in the experiments yaml file.
        """
        dataset_name = dataset["name"]
        dataset_results_dir = os.path.join(self.results_dir, dataset_name)
        dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)

        log.debug("\033[1mMoving output dir:\033[0m \n %s \n \033[1m to destination:\033[0m \n %s" %
            (self.pipeline_output_dir, dataset_pipeline_result_dir))

        try:
            evt.move_output_from_to(self.pipeline_output_dir, dataset_pipeline_result_dir)
        except:
            log.fatal("\033[1mFailed copying output dir: \033[0m\n %s \n \033[1m to destination: %s \033[0m\n" %
                (self.pipeline_output_dir, dataset_pipeline_result_dir))


""" DatasetEvaluator is used to evaluate performance of the pipeline on datasets """
class DatasetEvaluator:
    def __init__(self, experiment_params, args, extra_flagfile_path):
        self.results_dir      = os.path.expandvars(experiment_params['results_dir'])
        self.datasets_to_eval = experiment_params['datasets_to_run']

        self.display_plots = args.plot
        self.save_results  = args.save_results
        self.save_plots    = args.save_plots
        self.write_website = args.write_website
        self.save_boxplots = args.save_boxplots
        self.run_vio       = args.run_pipeline
        self.analyze_vio   = args.analyze_vio

        self.runner = DatasetRunner(experiment_params, args, extra_flagfile_path)

        self.traj_vio_csv_name = "traj_vio.csv"
        self.traj_gt_csv_name = "traj_gt.csv"
        self.traj_pgo_csv_name = "traj_pgo.csv"

        # Class to write the results to the Jenkins website
        self.website_builder = evt.WebsiteBuilder(self.results_dir)

    def evaluate(self):
        """ Run datasets if necessary, evaluate all. """
        for dataset in tqdm(self.datasets_to_eval):
            # Run the dataset if needed:
            if self.run_vio:
                log.info("Run dataset: %s" % dataset['name'])
                if not self.runner.run_dataset(dataset):
                    log.info("\033[91m Dataset: %s failed!! \033[00m" %
                            dataset['name'])
                    raise Exception("Failed to run dataset %s." % dataset['name'])

            # Evaluate each dataset if needed:
            if self.analyze_vio:
                self.evaluate_dataset(dataset)

        if self.write_website:
            log.info("Writing full website.")
            stats = aggregate_ape_results(self.results_dir)
            if (len(list(stats.values())) > 0):
                self.website_builder.write_boxplot_website(stats)
            self.website_builder.write_datasets_website()

        return True

    def evaluate_dataset(self, dataset):
        """ Evaluates VIO performance on given dataset """
        log.info("Evaluate dataset: %s" % dataset['name'])
        pipelines_to_evaluate_list = dataset['pipelines']
        for pipeline_type in pipelines_to_evaluate_list:
            if not self.__evaluate_run(pipeline_type, dataset):
                log.error("Failed to evaluate dataset %s for pipeline %s."
                        % dataset['name'] % pipeline_type)
                raise Exception("Failed evaluation.")

        if self.save_boxplots:
            self.save_boxplots_to_file(pipelines_to_evaluate_list, dataset)

    def __evaluate_run(self, pipeline_type, dataset):
        """ Evaluate performance of one pipeline of one dataset, as defined in the experiments
            yaml file.

            Assumes that the files traj_gt.csv traj_vio.csv and traj_pgo.csv are present.

            Args:
                dataset: a dataset to evaluate as defined in the experiments yaml file.
                pipeline_type: a pipeline representing a set of parameters to use, as
                    defined in the experiments yaml file for the dataset in question.

            Returns: True if there are no exceptions during evaluation, False otherwise.
        """
        dataset_name = dataset["name"]
        dataset_results_dir = os.path.join(self.results_dir, dataset_name)
        dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)

        traj_gt_path = os.path.join(dataset_pipeline_result_dir, self.traj_gt_csv_name)
        traj_vio_path = os.path.join(dataset_pipeline_result_dir, self.traj_vio_csv_name)
        traj_pgo_path = os.path.join(dataset_pipeline_result_dir, self.traj_pgo_csv_name)

        # Analyze dataset:
        log.debug("\033[1mAnalysing dataset:\033[0m \n %s \n \033[1m for pipeline \033[0m %s."
                % (dataset_results_dir, pipeline_type))
        evt.print_green("Starting analysis of pipeline: %s" % pipeline_type)

        discard_n_start_poses = dataset["discard_n_start_poses"]
        discard_n_end_poses = dataset["discard_n_end_poses"]
        segments = dataset["segments"]

        [plot_collection, results_vio, results_pgo] = self.run_analysis(
            traj_gt_path, traj_vio_path, traj_pgo_path, segments,
            dataset_name, discard_n_start_poses, discard_n_end_poses)

        if self.save_results:
            if results_vio is not None:
                self.save_results_to_file(results_vio, "results_vio", dataset_pipeline_result_dir)
            if results_pgo is not None:
                self.save_results_to_file(results_pgo, "results_pgo", dataset_pipeline_result_dir)

        if self.display_plots and plot_collection is not None:
            evt.print_green("Displaying plots.")
            plot_collection.show()

        if self.save_plots and plot_collection is not None:
            self.save_plots_to_file(plot_collection, dataset_pipeline_result_dir)

        if self.write_website:
            log.info("Writing performance website for dataset: %s" % dataset_name)
            self.website_builder.add_dataset_to_website(dataset_name, pipeline_type, dataset_pipeline_result_dir)
            self.website_builder.write_datasets_website()

        return True

    def run_analysis(self, traj_ref_path, traj_vio_path, traj_pgo_path, segments,
                     dataset_name="", discard_n_start_poses=0, discard_n_end_poses=0):
        """ Analyze data from a set of trajectory csv files.

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
        traj_ref, traj_est_vio, traj_est_pgo = self.read_traj_files(traj_ref_path, traj_vio_path, traj_pgo_path)

        # We copy to distinguish from the pgo version that may be created
        traj_ref_vio = copy.deepcopy(traj_ref)

        # Register and align trajectories:
        evt.print_purple("Registering and aligning trajectories")
        traj_ref_vio, traj_est_vio = sync.associate_trajectories(traj_ref_vio, traj_est_vio)
        traj_est_vio = trajectory.align_trajectory(traj_est_vio, traj_ref_vio, correct_scale = False,
                                                   discard_n_start_poses = int(discard_n_start_poses),
                                                   discard_n_end_poses = int(discard_n_end_poses))

        # We do the same for the PGO trajectory if needed:
        traj_ref_pgo = None
        if traj_est_pgo is not None:
            traj_ref_pgo = copy.deepcopy(traj_ref)
            traj_ref_pgo, traj_est_pgo = sync.associate_trajectories(traj_ref_pgo, traj_est_pgo)
            traj_est_pgo = trajectory.align_trajectory(traj_est_pgo, traj_ref_pgo, correct_scale = False,
                                                       discard_n_start_poses = int(discard_n_start_poses),
                                                       discard_n_end_poses = int(discard_n_end_poses))

        # We need to pick the lowest num_poses before doing any computation:
        num_of_poses = traj_est_vio.num_poses
        if traj_est_pgo is not None:
            num_of_poses = min(num_of_poses, traj_est_pgo.num_poses)
            traj_est_pgo.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
            traj_ref_pgo.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

        traj_est_vio.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
        traj_ref_vio.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

        # Calculate all metrics:
        (ape_metric_vio, rpe_metric_trans_vio, rpe_metric_rot_vio, results_vio) = self.process_trajectory_data(traj_ref_vio, traj_est_vio, segments, True)

        # We do the same for the pgo trajectory if needed:
        ape_metric_pgo = None
        rpe_metric_trans_pgo = None
        rpe_metric_rot_pgo = None
        results_pgo = None
        if traj_est_pgo is not None:
            (ape_metric_pgo, rpe_metric_trans_pgo, rpe_metric_rot_pgo, results_pgo) = self.process_trajectory_data(traj_ref_pgo, traj_est_pgo, segments, False)

        # Generate plots for return:
        plot_collection = None
        if self.display_plots or self.save_plots:
            evt.print_green("Plotting:")
            log.info(dataset_name)
            plot_collection = plot.PlotCollection("Example")

            if traj_est_pgo is not None:
                # APE Metric Plot:
                plot_collection.add_figure(
                    "PGO_APE_translation",
                    plot_metric(ape_metric_pgo, "PGO + VIO APE Translation")
                )

                # Trajectory Colormapped with ATE Plot:
                plot_collection.add_figure(
                    "PGO_APE_translation_trajectory_error",
                    plot_traj_colormap_ape(ape_metric_pgo, traj_ref_pgo,
                                               traj_est_vio, traj_est_pgo,
                                               "PGO + VIO ATE Mapped Onto Trajectory")
                )

                # RPE Translation Metric Plot:
                plot_collection.add_figure(
                    "PGO_RPE_translation",
                    plot_metric(rpe_metric_trans_pgo, "PGO + VIO RPE Translation")
                )

                # Trajectory Colormapped with RTE Plot:
                plot_collection.add_figure(
                    "PGO_RPE_translation_trajectory_error",
                    plot_traj_colormap_rpe(rpe_metric_trans_pgo, traj_ref_pgo,
                                               traj_est_vio, traj_est_pgo,
                                               "PGO + VIO RPE Translation Error Mapped Onto Trajectory")
                )

                # RPE Rotation Metric Plot:
                plot_collection.add_figure(
                    "PGO_RPE_Rotation",
                    plot_metric(rpe_metric_rot_pgo, "PGO + VIO RPE Rotation")
                )

                # Trajectory Colormapped with RTE Plot:
                plot_collection.add_figure(
                    "PGO_RPE_rotation_trajectory_error",
                    plot_traj_colormap_rpe(rpe_metric_rot_pgo, traj_ref_pgo,
                                               traj_est_vio, traj_est_pgo,
                                               "PGO + VIO RPE Rotation Error Mapped Onto Trajectory")
                )

            # Plot VIO results
            plot_collection.add_figure(
                "VIO_APE_translation",
                plot_metric(ape_metric_vio, "VIO APE Translation")
            )

            plot_collection.add_figure(
                "VIO_APE_translation_trajectory_error",
                plot_traj_colormap_ape(ape_metric_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO ATE Mapped Onto Trajectory")
            )

            plot_collection.add_figure(
                "VIO_RPE_translation",
                plot_metric(rpe_metric_trans_vio, "VIO RPE Translation")
            )

            plot_collection.add_figure(
                "VIO_RPE_translation_trajectory_error",
                plot_traj_colormap_rpe(rpe_metric_trans_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO RPE Translation Error Mapped Onto Trajectory")
            )

            plot_collection.add_figure(
                "VIO_RPE_Rotation",
                plot_metric(rpe_metric_rot_vio, "VIO RPE Rotation")
            )

            plot_collection.add_figure(
                "VIO_RPE_rotation_trajectory_error",
                plot_traj_colormap_rpe(rpe_metric_rot_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO RPE Rotation Error Mapped Onto Trajectory")
            )

        return [plot_collection, results_vio, results_pgo]

    def process_trajectory_data(self, traj_ref, traj_est, segments, is_vio_traj=True):
        """
        """
        suffix = "VIO" if is_vio_traj else "PGO"
        data = (traj_ref, traj_est)

        evt.print_purple("Calculating APE translation part for " + suffix)
        ape_metric = get_ape_trans(data)
        ape_result = ape_metric.get_result()

        evt.print_purple("Calculating RPE translation part for " + suffix)
        rpe_metric_trans = get_rpe_trans(data)

        evt.print_purple("Calculating RPE rotation angle for " + suffix)
        rpe_metric_rot = get_rpe_rot(data)

        # Collect results:
        results = dict()
        results["absolute_errors"] = ape_result

        results["relative_errors"] = self.calc_rpe_results(rpe_metric_trans, rpe_metric_rot, data, segments)

        # Add as well how long hte trajectory was.
        results["trajectory_length_m"] = traj_est.path_length()

        return (ape_metric, rpe_metric_trans, rpe_metric_rot, results)

    def read_traj_files(self, traj_ref_path, traj_vio_path, traj_pgo_path):
        """ Outputs PoseTrajectory3D objects for csv trajectory files.

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
            traj_ref = pandas_bridge.df_to_trajectory(pd.read_csv(traj_ref_path, sep=',', index_col=0))
        except IOError as e:
            raise Exception("\033[91mMissing ground-truth output csv! \033[93m {}.".format(e))

        # Read estimated vio trajectory file:
        traj_est_vio = None
        try:
            traj_est_vio = pandas_bridge.df_to_trajectory(pd.read_csv(traj_vio_path, sep=',', index_col=0))
        except IOError as e:
            raise Exception("\033[91mMissing vio estimated output csv! \033[93m {}.".format(e))

        # Read estimated pgo trajectory file:
        traj_est_pgo = None
        try:
            traj_est_pgo = pandas_bridge.df_to_trajectory(pd.read_csv(traj_pgo_path, sep=',', index_col=0))
        except IOError as e:
            log.warning("Missing pgo estimated output csv: {}.".format(e))
            log.warning("Not plotting pgo results.")

        return (traj_ref, traj_est_vio, traj_est_pgo)

    def calc_rpe_results(self, rpe_metric_trans, rpe_metric_rot, data, segments):
        """ Create and return a dictionary containing stats and results RRE and RTE for a datset.

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
            evt.print_purple("RPE analysis of segment: %d"%segment)
            evt.print_lightpurple("Calculating RPE segment translation part")
            rpe_segment_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                                   float(segment), metrics.Unit.meters, 0.01, True)
            rpe_segment_metric_trans.process_data(data)
            # TODO(Toni): Save RPE computation results rather than the statistics
            # you can compute statistics later... Like done for ape!
            rpe_segment_stats_trans = rpe_segment_metric_trans.get_all_statistics()
            rpe_results[segment]["rpe_trans"] = rpe_segment_stats_trans

            evt.print_lightpurple("Calculating RPE segment rotation angle")
            rpe_segment_metric_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                                 float(segment), metrics.Unit.meters, 0.01, True)
            rpe_segment_metric_rot.process_data(data)
            rpe_segment_stats_rot = rpe_segment_metric_rot.get_all_statistics()
            rpe_results[segment]["rpe_rot"] = rpe_segment_stats_rot

        return rpe_results

    def save_results_to_file(self, results, title, dataset_pipeline_result_dir):
        """ Writes a result dictionary to file as a yaml file.

            Args:
                results: a dictionary containing ape, rpe rotation and rpe translation results and
                    statistics.
                title: a string representing the filename without the '.yaml' extension.
                dataset_pipeline_result_dir: a string representing the filepath for the location to
                    save the results file.
        """
        results_file = os.path.join(dataset_pipeline_result_dir, title + '.yaml')
        evt.print_green("Saving analysis results to: %s" % results_file)
        evt.create_full_path_if_not_exists(results_file)
        with open(results_file,'w') as outfile:
            outfile.write(yaml.dump(results, default_flow_style=False))

    def save_plots_to_file(self, plot_collection, dataset_pipeline_result_dir,
                          save_pdf=True):
        """ Wrie plot collection to disk as both eps and pdf.

            Args:
                - plot_collection: a PlotCollection containing all the plots to save to file.
                - dataset_pipeline_result_dir: a string representing the filepath for the location to
                    which the plot files are saved.
                - save_pdf: whether to save figures to pdf or eps format
        """
        # Config output format (pdf, eps, ...) using evo_config...
        if save_pdf:
            pdf_output_file_path = os.path.join(dataset_pipeline_result_dir, "plots.pdf")
            evt.print_green("Saving plots to: %s" % pdf_output_file_path)
            plot_collection.export(pdf_output_file_path, False)
        else:
            eps_output_file_path = os.path.join(dataset_pipeline_result_dir, "plots.eps")
            evt.print_green("Saving plots to: %s" % eps_output_file_path)
            plot_collection.export(eps_output_file_path, False)

    def save_boxplots_to_file(self, pipelines_to_run_list, dataset):
        """ Writes boxplots for all pipelines of a given dataset to disk.

            Args:
                pipelines_to_run_list: a list containing all pipelines to run for a dataset.
                dataset: a single dataset, as taken from the experiments yaml file.
        """
        dataset_name = dataset['name']
        dataset_segments = dataset['segments']

        # TODO(Toni) is this really saving the boxplots?
        stats = dict()
        for pipeline_type in pipelines_to_run_list:
            results_dataset_dir = os.path.join(self.results_dir, dataset_name)
            results_vio = os.path.join(results_dataset_dir, pipeline_type, "results_vio.yaml")
            if not os.path.exists(results_vio):
                raise Exception("\033[91mCannot plot boxplots: missing results for %s pipeline \
                                and dataset: %s" % (pipeline_type, dataset_name) + "\033[99m \n \
                                Expected results here: %s" % results_vio + "\033[99m \n \
                                Ensure that `--save_results` is passed at commandline.")

            try:
                stats[pipeline_type]  = yaml.load(open(results_vio,'r'), Loader=yaml.Loader)
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


# Miscellaneous methods

def get_ape_rot(data):
    """ Return APE rotation metric for input data.

        Args:
            data: A 2-tuple containing the reference trajectory and the
                estimated trajectory as PoseTrajectory3D objects.

        Returns:
            A metrics object containing the desired results.
    """
    ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_rot.process_data(data)

    return ape_rot


def get_ape_trans(data):
    """ Return APE translation metric for input data.

        Args:
            data: A 2-tuple containing the reference trajectory and the
                estimated trajectory as PoseTrajectory3D objects.

        Returns:
            A metrics object containing the desired results.
    """
    ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_trans.process_data(data)

    return ape_trans


def get_rpe_rot(data):
    """ Return RPE rotation metric for input data.

        Args:
            data: A 2-tuple containing the reference trajectory and the
                estimated trajectory as PoseTrajectory3D objects.

        Returns:
            A metrics object containing the desired results.
    """
    rpe_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                          1.0, metrics.Unit.frames, 1.0, False)
    rpe_rot.process_data(data)

    return rpe_rot

def get_rpe_trans(data):
    """ Return RPE translation metric for input data.

        Args:
            data: A 2-tuple containing the reference trajectory and the
                estimated trajectory as PoseTrajectory3D objects.

        Returns:
            A metrics object containing the desired results.
    """
    rpe_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                            1.0, metrics.Unit.frames, 0.0, False)
    rpe_trans.process_data(data)

    return rpe_trans


def plot_metric(metric, plot_title="", figsize=(8,8)):
    """ Adds a metric plot to a plot collection.

        Args:
            plot_collection: a PlotCollection containing plots.
            metric: an evo.core.metric object with statistics and information.
            plot_title: a string representing the title of the plot.
            figsize: a 2-tuple representing the figure size.

        Returns:
            A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    stats = metric.get_all_statistics()

    plot.error_array(fig, metric.error, statistics=stats,
                        title=plot_title,
                        xlabel="Keyframe index [-]",
                        ylabel=plot_title + " " + metric.unit.value)

    return fig


def plot_traj_colormap_ape(ape_metric, traj_ref, traj_est1, traj_est2=None,
                           plot_title="", figsize=(8,8)):
    """ Adds a trajectory colormap of ATE metrics to a plot collection.

        Args:
            ape_metric: an evo.core.metric object with statistics and information for APE.
            traj_ref: a PoseTrajectory3D object representing the reference trajectory.
            traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
            traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory. Optional.
            plot_title: a string representing the title of the plot.
            figsize: a 2-tuple representing the figure size.

        Returns:
            A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)

    ape_stats = ape_metric.get_all_statistics()

    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')

    colormap_traj = traj_est1
    if traj_est2 is not None:
        plot.traj(ax, plot_mode, traj_est1, '.', 'gray', 'reference without pgo')
        colormap_traj = traj_est2

    plot.traj_colormap(ax, colormap_traj, ape_metric.error, plot_mode,
                        min_map=0.0, max_map=math.ceil(ape_stats['max']*10)/10,
                        title=plot_title)

    return fig


def plot_traj_colormap_rpe(rpe_metric, traj_ref, traj_est1, traj_est2=None,
                           plot_title="", figsize=(8,8)):
    """ Adds a trajectory colormap of RPE metrics to a plot collection.

        Args:
            ape_metric: an evo.core.metric object with statistics and information for RPE.
            traj_ref: a PoseTrajectory3D object representing the reference trajectory.
            traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
            traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory. Optional.
            plot_title: a string representing the title of the plot.
            figsize: a 2-tuple representing the figure size.

        Returns:
            A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)

    # We have to make deep copies to avoid altering the original data: TODO(marcus): figure out why
    traj_ref = copy.deepcopy(traj_ref)
    traj_est1 = copy.deepcopy(traj_est1)
    traj_est2 = copy.deepcopy(traj_est2)

    rpe_stats = rpe_metric.get_all_statistics()
    traj_ref.reduce_to_ids(rpe_metric.delta_ids)
    traj_est1.reduce_to_ids(rpe_metric.delta_ids)

    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')

    colormap_traj = traj_est1
    if traj_est2 is not None:
        traj_est2.reduce_to_ids(rpe_metric.delta_ids)
        plot.traj(ax, plot_mode, traj_est1, '.', 'gray', 'reference without pgo')
        colormap_traj = traj_est2

    plot.traj_colormap(ax, colormap_traj, rpe_metric.error, plot_mode,
                        min_map=0.0, max_map=math.ceil(rpe_stats['max']*10)/10,
                        title=plot_title)
    
    return fig


def convert_abs_traj_to_rel_traj(traj, up_to_scale=False):
    """ Converts an absolute-pose trajectory to a relative-pose trajectory.
    
        The incoming trajectory is processed element-wise. At each timestamp
        starting from the second (index 1), the relative pose 
        from the previous timestamp to the current one is calculated (in the previous-
        timestamp's coordinate frame). This relative pose is then appended to the 
        resulting trajectory.
        The resulting trajectory has timestamp indices corresponding to poses that represent
        the relative transformation between that timestamp and the **next** one.
        
        Args:
            traj: A PoseTrajectory3D object with timestamps as indices containing, at a minimum,
                columns representing the xyz position and wxyz quaternion-rotation at each
                timestamp, corresponding to the absolute pose at that time.
            up_to_scale: A boolean. If set to True, relative poses will have their translation
                part normalized.
        
        Returns:
            A PoseTrajectory3D object with xyz position and wxyz quaternion fields for the 
            relative pose trajectory corresponding to the absolute one given in `traj`.
    """
    from evo.core import transformations
    from evo.core import lie_algebra as lie

    new_poses = []
    
    for i in range(1, len(traj.timestamps)):
        rel_pose = lie.relative_se3(traj.poses_se3[i-1], traj.poses_se3[i])

        if up_to_scale:
            bim1_t_bi = rel_pose[:3, 3]
            norm = np.linalg.norm(bim1_t_bi)
            if norm > 1e-6:
                bim1_t_bi = bim1_t_bi / norm
                rel_pose[:3, 3] = bim1_t_bi
    
        new_poses.append(rel_pose)

    return trajectory.PoseTrajectory3D(timestamps=traj.timestamps[1:], poses_se3=new_poses)

def convert_rel_traj_from_body_to_cam(rel_traj, body_T_cam):
    """Converts a relative pose trajectory from body frame to camera frame
    
    Args: 
        rel_traj: Relative trajectory, a PoseTrajectory3D object containing timestamps
            and relative poses at each timestamp. It has to have the poses_se3 field.
            
        body_T_cam: The SE(3) transformation from camera from to body frame. Also known
            as camera extrinsics matrix.
        
    Returns: 
        A PoseTrajectory3D object in camera frame
    """
    def assert_so3(R):
        assert(np.isclose(np.linalg.det(R), 1, atol=1e-06))
        assert(np.allclose(np.matmul(R, R.transpose()), np.eye(3), atol=1e-06)) 

    assert_so3(body_T_cam[0:3, 0:3])
 
    new_poses = []
    for i in range(len(rel_traj.timestamps)):
        im1_body_T_body_i = rel_traj.poses_se3[i]
        assert_so3(im1_body_T_body_i[0:3,0:3])
 
        im1_cam_T_cam_i = np.matmul(np.matmul(np.linalg.inv(body_T_cam), im1_body_T_body_i), body_T_cam)

        assert_so3(np.linalg.inv(body_T_cam)[0:3,0:3])
        assert_so3(im1_cam_T_cam_i[0:3,0:3])
 
        new_poses.append(im1_cam_T_cam_i)
 
    return trajectory.PoseTrajectory3D(timestamps=rel_traj.timestamps, poses_se3=new_poses)
    
