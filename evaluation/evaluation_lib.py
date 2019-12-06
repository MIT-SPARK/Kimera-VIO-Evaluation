#!/usr/bin/env python

from __future__ import print_function
import copy
import os
# import yaml
from ruamel import yaml
import math
import subprocess
import numpy as np
import glog as log
from evo.tools import plot
import matplotlib.pyplot as plt
from shutil import copyfile, move, rmtree, copytree, copy2

from evo.core import trajectory, sync, metrics
import evaluation.tools as evt


def aggregate_all_results(results_dir):
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
    import fnmatch
    # Load results.
    log.info("Aggregate dataset results.")
    # Aggregate all stats for each pipeline and dataset
    stats = dict()
    for root, dirnames, filenames in os.walk(results_dir):
        for results_filename in fnmatch.filter(filenames, 'results_vio.yaml'):
            results_filepath = os.path.join(root, results_filename)
            # Get pipeline name
            pipeline_name = os.path.basename(root)
            # Get dataset name
            dataset_name = os.path.basename(os.path.split(root)[0])
            # Collect stats
            if stats.get(dataset_name) is None:
                stats[dataset_name] = dict()
            stats[dataset_name][pipeline_name] = yaml.load(open(results_filepath, 'r'), Loader=yaml.Loader)
            log.debug("Check stats from: " + results_filepath)
            check_stats(stats[dataset_name][pipeline_name])
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
    """
    stats = aggregate_all_results(results_dir)
    # Draw APE boxplot
    log.info("Drawing APE boxplots.")
    evt.draw_ape_boxplots(stats, results_dir)
    # Write APE table
    log.info("Writting APE latex table.")
    evt.write_latex_table(stats, results_dir)

def check_stats(stats):
    if not "relative_errors" in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception("\033[91mWrong stats format: no relative_errors... \n"
                        "Are you sure you runned the pipeline and "
                        "saved the results? (--save_results).\033[99m")
    else:
        if len(stats["relative_errors"]) == 0:
            raise Exception("\033[91mNo relative errors available... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")

        if not "rpe_rot" in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception("\033[91mWrong stats format: no rpe_rot... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")
        if not "rpe_trans" in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception("\033[91mWrong stats format: no rpe_trans... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")
    if not "absolute_errors" in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception("\033[91mWrong stats format: no absolute_errors... \n"
                        "Are you sure you runned the pipeline and "
                        "saved the results? (--save_results).\033[99m")


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
        self.output_file_vio = os.path.join(self.pipeline_output_dir, "output_posesVIO.csv")
        self.output_file_pgo = os.path.join(self.pipeline_output_dir, "output_lcd_optimized_traj.csv")

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
        dataset_segments = dataset['segments']

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
            return_code = subprocess.call("{} \
                                --logtostderr=1 --colorlogtostderr=1 --log_prefix=1 \
                                --dataset_path={}/{} --output_path={} \
                                --left_cam_params_path={}/{}/{} \
                                --right_cam_params_path={}/{}/{} \
                                --imu_params_path={}/{}/{} \
                                --backend_params_path={}/{}/{} \
                                --frontend_params_path={}/{}/{} \
                                --lcd_params_path={}/{}/{} \
                                --vocabulary_path={} \
                                --flagfile={}/{}/{} --flagfile={}/{}/{} \
                                --flagfile={}/{}/{} --flagfile={}/{}/{} \
                                --flagfile={}/{}/{} --flagfile={}/{} \
                                --initial_k={} --final_k={} --use_lcd={} \
                                --log_output=True --minloglevel={} \
                                --parallel_run={}".format(
                self.executable_path, self.dataset_dir, dataset["name"], self.pipeline_output_dir,
                self.params_dir, pipeline_type, "LeftCameraParams.yaml",
                self.params_dir, pipeline_type, "RightCameraParams.yaml",
                self.params_dir, pipeline_type, "ImuParams.yaml",
                self.params_dir, pipeline_type, "regularVioParameters.yaml",
                self.params_dir, pipeline_type, "trackerParameters.yaml",
                self.params_dir, pipeline_type, "LCDParameters.yaml",
                self.vocabulary_path,
                self.params_dir, pipeline_type, "flags/stereoVIOEuroc.flags",
                self.params_dir, pipeline_type, "flags/Mesher.flags",
                self.params_dir, pipeline_type, "flags/VioBackEnd.flags",
                self.params_dir, pipeline_type, "flags/RegularVioBackEnd.flags",
                self.params_dir, pipeline_type, "flags/Visualizer3D.flags",
                self.params_dir, self.extra_flagfile_path,
                dataset["initial_frame"], dataset["final_frame"], dataset["use_lcd"], minloglevel,
                dataset["parallel_run"]),
                shell=True)
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
        """ Copies trajectory csvs and moves all output files for a particular pipeline and dataset
            from their temporary logging location during runtime to the evaluation location.

            Args:
                dataset: a dataset to run as defined in the experiments yaml file.
                pipeline_type: a pipeline representing a set of parameters to use, as
                    defined in the experiments yaml file for the dataset in question.
        """
        dataset_name = dataset["name"]
        dataset_results_dir = os.path.join(self.results_dir, dataset_name)
        dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)

        traj_ref_path = os.path.join(self.dataset_dir, dataset_name, "mav0/state_groundtruth_estimate0/data.csv") # TODO make it not specific to EUROC

        traj_vio = os.path.join(dataset_results_dir, pipeline_type, "traj_vio.csv")
        traj_pgo = os.path.join(dataset_results_dir, pipeline_type, "traj_pgo.csv")
        evt.create_full_path_if_not_exists(traj_vio)
        evt.create_full_path_if_not_exists(traj_pgo)

        log.debug("\033[1mCopying output file: \033[0m \n %s \n \033[1m to results file:\033[0m\n %s" %
            (self.output_file_vio, traj_vio))
        copyfile(self.output_file_vio, traj_vio)

        if dataset["use_lcd"]:
            log.debug("\033[1mCopying output file: \033[0m \n %s \n \033[1m to results file:\033[0m\n %s" %
                (self.output_file_pgo, traj_pgo))
            copyfile(self.output_file_pgo, traj_pgo)

        output_destination_dir = os.path.join(dataset_pipeline_result_dir, "output")
        log.debug("\033[1mMoving output dir:\033[0m \n %s \n \033[1m to destination:\033[0m \n %s" %
            (self.pipeline_output_dir, output_destination_dir))

        try:
            evt.move_output_from_to(self.pipeline_output_dir, output_destination_dir)
        except:
            log.fatal("\033[1mFailed copying output dir: \033[0m\n %s \n \033[1m to destination: %s \033[0m\n" %
                (self.pipeline_output_dir, output_destination_dir))


""" DatasetEvaluator is used to evaluate performance of the pipeline on datasets """
class DatasetEvaluator:
    def __init__(self, experiment_params, args, extra_flagfile_path):
        self.results_dir      = os.path.expandvars(experiment_params['results_dir'])
        self.datasets_to_eval = experiment_params['datasets_to_run']
        self.dataset_dir      = os.path.expandvars(experiment_params['dataset_dir'])

        self.display_plots = args.plot
        self.save_results  = args.save_results
        self.save_plots    = args.save_plots
        self.save_boxplots = args.save_boxplots
        self.run_vio       = args.run_pipeline
        self.analyze_vio   = args.analyze_vio

        self.runner = DatasetRunner(experiment_params, args, extra_flagfile_path)

    def evaluate(self):
        """ Run datasets if necessary, evaluate all. """
        if self.run_vio and not self.analyze_vio:
            return self.runner.run_all()
        
        elif not self.run_vio and self.analyze_vio:
            return self.evaluate_all()

        for dataset in tqdm(self.datasets_to_eval):
            # Run the dataset if needed:
            if self.run_vio:
                successful_run = True
                log.info("Run dataset: %s" % dataset['name'])
                if not self.runner.run_dataset(dataset):
                    log.info("\033[91m Dataset: %s failed!! \033[00m" %
                            dataset['name'])
                    raise Exception("Failed to run dataset %s." % dataset['name'])
                
            # Evaluate each dataset if needed:
            if self.analyze_vio:
                log.info("Evaluate dataset: %s" % dataset['name'])
                pipelines_to_run_list = dataset['pipelines']
                for pipeline_type in pipelines_to_run_list:
                    if not self.__evaluate_run(pipeline_type, dataset):
                        log.error("Failed to evaluate dataset %s for pipeline %s. Exiting."
                                  % dataset['name'] % pipeline_type)
                        raise Exception("Failed evaluation.")

                if self.save_boxplots:
                    self.save_boxplots_to_file(pipelines_to_run_list, dataset)

        return True

    def evaluate_all(self):
        """ Evaluate performance on every pipeline of every dataset defined in the experiments
            yaml file.
        """
        # Run analysis.
        log.info("Run analysis for all experiments")
        for dataset in tqdm(self.datasets_to_eval):
            log.info("Evaluate dataset: %s" % dataset['name'])
            pipelines_to_run_list = dataset['pipelines']
            for pipeline_type in pipelines_to_run_list:
                if not self.__evaluate_run(pipeline_type, dataset):
                    log.error("Failed to evaluate dataset %s for pipeline %s. Exiting.")
                    raise Exception("Failed evaluation.")

            if self.save_boxplots:
                self.save_boxplots_to_file(pipelines_to_run_list, dataset)

        return True

    def __evaluate_run(self, pipeline_type, dataset):
        """ Evaluate performance of one pipeline of one dataset, as defined in the experiments
            yaml file.

            Args:
                dataset: a dataset to run as defined in the experiments yaml file.
                pipeline_type: a pipeline representing a set of parameters to use, as
                    defined in the experiments yaml file for the dataset in question.

            Returns: True if the there are no exceptions during evaluation, False otherwise.
        """
        dataset_name = dataset["name"]
        dataset_results_dir = os.path.join(self.results_dir, dataset_name)
        dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)
        use_lcd = dataset["use_lcd"]
        plot_vio_and_pgo = dataset["plot_vio_and_pgo"]

        if not use_lcd and plot_vio_and_pgo:
            log.error("\033[1mCannot plot PGO results if 'use_lcd' is set to False:\033[0m")
            plot_vio_and_pgo = False

        traj_ref_path = os.path.join(
            self.dataset_dir, dataset_name, "mav0/state_groundtruth_estimate0/data.csv") # TODO make it not specific to EUROC

        traj_vio_path = os.path.join(dataset_results_dir, pipeline_type, "traj_vio.csv")
        traj_pgo_path = os.path.join(dataset_results_dir, pipeline_type, "traj_pgo.csv")


        # Analyze dataset:
        log.debug("\033[1mAnalysing dataset:\033[0m \n %s \n \033[1m for pipeline \033[0m %s."
                % (dataset_results_dir, pipeline_type))
        evt.print_green("Starting analysis of pipeline: %s" % pipeline_type)

        discard_n_start_poses = dataset["discard_n_start_poses"]
        discard_n_end_poses = dataset["discard_n_end_poses"]
        segments = dataset["segments"]

        [plot_collection, results_vio, results_pgo] = self.run_analysis(
            traj_ref_path, traj_vio_path, traj_pgo_path, segments,
            use_lcd, plot_vio_and_pgo, dataset_name, discard_n_start_poses,
            discard_n_end_poses)

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

        return True

    def run_analysis(self, traj_ref_path, traj_vio_path, traj_pgo_path, segments, generate_pgo=False,
                     plot_vio_and_pgo=False, dataset_name="",
                     discard_n_start_poses=0, discard_n_end_poses=0):
        """ Analyze data from a set of trajectory csv files.

            Args:
                traj_ref_path: string representing filepath of the reference (ground-truth) trajectory.
                traj_vio_path: string representing filepath of the vio estimated trajectory.
                traj_pgo_path: string representing filepath of the pgo estimated trajectory.
                segments: list of segments for RPE calculation, defined in the experiments yaml file.
                generate_pgo: boolean; if True, analysis will generate results and plots for pgo trajectories.
                plot_vio_and_pgo: if True, the plots will include both pgo and vio-only trajectories.
                dataset_name: string representing the dataset's name
                discard_n_start_poses: int representing number of poses to discard from start of analysis.
                discard_n_end_poses: int representing the number of poses to discard from end of analysis.
        """
        import copy

        traj_ref, traj_est_vio, traj_est_pgo = self.read_traj_files(traj_ref_path, traj_vio_path,
                                                                    traj_pgo_path, generate_pgo)

        # We copy to distinguish from the pgo version that may be created
        traj_ref_vio = copy.deepcopy(traj_ref)

        # Register and align trajectories:
        evt.print_purple("Registering and aligning trajectories")
        traj_ref_vio, traj_est_vio = sync.associate_trajectories(traj_ref_vio, traj_est_vio)
        traj_est_vio = trajectory.align_trajectory(traj_est_vio, traj_ref_vio, correct_scale = False,
                                                   discard_n_start_poses = int(discard_n_start_poses),
                                                   discard_n_end_poses = int(discard_n_end_poses))

        num_of_poses = traj_est_vio.num_poses
        # We need to pick the lowest num_poses before doing any computation:
        if traj_est_pgo is not None:
            num_of_poses = min(num_of_poses, traj_est_pgo.num_poses)
        traj_est_vio.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
        traj_ref_vio.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

        # Calculate all metrics:
        data = (traj_ref_vio, traj_est_vio)
        evt.print_purple("Calculating APE translation part for VIO")
        ape_metric_vio = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric_vio.process_data(data)
        evt.print_purple("Calculating RPE translation part for VIO")
        rpe_metric_trans_vio = metrics.RPE(metrics.PoseRelation.translation_part,
                                           1.0, metrics.Unit.frames, 0.0, False)
        rpe_metric_trans_vio.process_data(data)
        evt.print_purple("Calculating RPE rotation angle for VIO")
        rpe_metric_rot_vio = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                         1.0, metrics.Unit.frames, 1.0, False)
        rpe_metric_rot_vio.process_data(data)

        # Calculate results dictionary for vio and pgo trajectories if needed:
        results_vio = self.calc_results(ape_metric_vio, rpe_metric_trans_vio,
                                        rpe_metric_rot_vio, data, segments)


        # We do the same for the pgo trajectory if needed:
        traj_ref_pgo = None
        ape_metric_pgo = None
        rpe_metric_trans_pgo = None
        rpe_metric_rot_pgo = None
        results_pgo = None
        if traj_est_pgo is not None:
            traj_ref_pgo = copy.deepcopy(traj_ref)
            traj_ref_pgo, traj_est_pgo = sync.associate_trajectories(traj_ref_pgo, traj_est_pgo)
            traj_est_pgo = trajectory.align_trajectory(traj_est_pgo, traj_ref_pgo, correct_scale = False,
                                                       discard_n_start_poses = int(discard_n_start_poses),
                                                       discard_n_end_poses = int(discard_n_end_poses))

            traj_est_pgo.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
            traj_ref_pgo.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

            data = (traj_ref_pgo, traj_est_pgo)
            evt.print_purple("Calculating APE translation part for PGO")
            ape_metric_pgo = metrics.APE(metrics.PoseRelation.translation_part)
            ape_metric_pgo.process_data(data)
            evt.print_purple("Calculating RPE translation part for PGO")
            rpe_metric_trans_pgo = metrics.RPE(metrics.PoseRelation.translation_part,
                                               1.0, metrics.Unit.frames, 0.0, False)
            rpe_metric_trans_pgo.process_data(data)
            evt.print_purple("Calculating RPE rotation angle for PGO")
            rpe_metric_rot_pgo = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                             1.0, metrics.Unit.frames, 1.0, False)
            rpe_metric_rot_pgo.process_data(data)

            results_pgo = self.calc_results(ape_metric_pgo, rpe_metric_trans_pgo,
                                            rpe_metric_rot_pgo, data, segments)

        # Generate plots for return:
        plot_collection = None
        if self.display_plots or self.save_plots:
            evt.print_green("Plotting:")
            log.info(dataset_name)
            plot_collection = plot.PlotCollection("Example")

            if traj_est_pgo is not None:
                # APE Metric Plot:
                self.add_metric_plot(plot_collection, dataset_name, ape_metric_pgo,
                                     "PGO_APE_translation", "PGO + VIO APE Translation", "[m]")

                # Trajectory Colormapped with ATE Plot:
                self.add_traj_colormap_ape(plot_collection, ape_metric_pgo, traj_ref_pgo,
                                           traj_est_vio, traj_est_pgo,
                                           "PGO_APE_translation_trajectory_error",
                                           "PGO + VIO ATE Mapped Onto Trajectory")

                # RPE Translation Metric Plot:
                self.add_metric_plot(plot_collection, dataset_name, rpe_metric_trans_pgo,
                                     "PGO_RPE_translation", "PGO + VIO RPE Translation", "[m]")

                # Trajectory Colormapped with RTE Plot:
                self.add_traj_colormap_rpe(plot_collection, rpe_metric_trans_pgo, traj_ref_pgo,
                                           traj_est_vio, traj_est_pgo,
                                           "PGO_RPE_translation_trajectory_error",
                                           "PGO + VIO RPE Translation Error Mapped Onto Trajectory")

                # RPE Rotation Metric Plot:
                self.add_metric_plot(plot_collection, dataset_name, rpe_metric_rot_pgo,
                                     "PGO_RPE_Rotation", "PGO + VIO RPE Rotation", "[m]")

                # Trajectory Colormapped with RTE Plot:
                self.add_traj_colormap_rpe(plot_collection, rpe_metric_rot_pgo, traj_ref_pgo,
                                           traj_est_vio, traj_est_pgo,
                                           "PGO_RPE_rotation_trajectory_error",
                                           "PGO + VIO RPE Rotation Error Mapped Onto Trajectory")

            if traj_est_pgo is None or plot_vio_and_pgo:
                self.add_metric_plot(plot_collection, dataset_name, ape_metric_vio,
                                     "VIO_APE_translation", "VIO APE Translation", "[m]")
                self.add_traj_colormap_ape(plot_collection, ape_metric_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO_APE_translation_trajectory_error",
                                           "VIO ATE Mapped Onto Trajectory")
                self.add_metric_plot(plot_collection, dataset_name, rpe_metric_trans_vio,
                                     "VIO_RPE_translation", "VIO RPE Translation", "[m]")
                self.add_traj_colormap_rpe(plot_collection, rpe_metric_trans_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO_RPE_translation_trajectory_error",
                                           "VIO RPE Translation Error Mapped Onto Trajectory")
                self.add_metric_plot(plot_collection, dataset_name, rpe_metric_rot_vio,
                                     "VIO_RPE_Rotation", "VIO RPE Rotation", "[m]")
                self.add_traj_colormap_rpe(plot_collection, rpe_metric_rot_vio, traj_ref_vio,
                                           traj_est_vio, None,
                                           "VIO_RPE_rotation_trajectory_error",
                                           "VIO RPE Rotation Error Mapped Onto Trajectory")

        return [plot_collection, results_vio, results_pgo]

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

    def save_plots_to_file(self, plot_collection, dataset_pipeline_result_dir):
        """ Wrie plot collection to disk as both eps and pdf.

            Args:
                plot_collection: a PlotCollection containing all the plots to save to file.
                dataset_pipeline_result_dir: a string representing the filepath for the location to
                    which the plot files are saved.
        """
        # Config output format (pdf, eps, ...) using evo_config...
        eps_output_file_path = os.path.join(dataset_pipeline_result_dir, "plots.eps")
        pdf_output_file_path = os.path.join(dataset_pipeline_result_dir, "plots.pdf")
        evt.print_green("Saving plots to: %s" % eps_output_file_path)
        evt.print_green("Saving plots to: %s" % pdf_output_file_path)
        plot_collection.export(eps_output_file_path, False)
        plot_collection.export(pdf_output_file_path, False)

    def read_traj_files(self, traj_ref_path, traj_vio_path, traj_pgo_path, generate_pgo=False):
        """ Outputs PoseTrajectory3D objects for csv trajectory files.

            Args:
                traj_ref_path: string representing filepath of the reference (ground-truth) trajectory.
                traj_vio_path: string representing filepath of the vio estimated trajectory.
                traj_pgo_path: string representing filepath of the pgo estimated trajectory.
                generate_pgo: boolean; if True, analysis will generate results and plots for pgo trajectories.

            Returns: A 3-tuple with the PoseTrajectory3D objects representing the reference trajectory,
                vio trajectory, and pgo trajectory in that order.
        """
        from evo.tools import file_interface

        # Read reference trajectory file:
        traj_ref = None
        try:
            traj_ref = file_interface.read_euroc_csv_trajectory(traj_ref_path) # TODO make it non-euroc specific.
        except file_interface.FileInterfaceException as e:
            raise Exception("\033[91mMissing ground truth csv! \033[93m {}.".format(e))

        # Read estimated vio trajectory file:
        traj_est_vio = None
        try:
            traj_est_vio = file_interface.read_euroc_csv_trajectory(traj_vio_path)
        except file_interface.FileInterfaceException as e:
            raise Exception("\033[91mMissing vio estimated output csv! \033[93m {}.".format(e))

        # Read estimated pgo trajectory file:
        traj_est_pgo = None
        if generate_pgo:
            try:
                traj_est_pgo = file_interface.read_pose_csv_trajectory(traj_pgo_path)
            except file_interface.FileInterfaceException as e:
                raise Exception("\033[91mMissing pgo estimated output csv! \033[93m {}.".format(e))

        return (traj_ref, traj_est_vio, traj_est_pgo)

    def add_metric_plot(self, plot_collection, dataset_name, metric, fig_title="",
                        plot_title="", metric_units=""):
        """ Adds a metric plot to a plot collection.

            Args:
                plot_collection: a PlotCollection containing plots.
                dataset_name: a string representing the name of the dataset being evaluated.
                metric: an evo.core.metric object with statistics and information.
                fig_title: a string representing the title of the figure. Must be unique in the plot_collection.
                plot_title: a string representing the title of the plot.
                metric_units: a string representing the units of the metric being plotted.
        """
        fig = plt.figure(figsize=(8, 8))
        stats = metric.get_all_statistics()

        plot.error_array(fig, metric.error, statistics=stats,
                         name=plot_title, title=plot_title,
                         xlabel="Keyframe index [-]",
                         ylabel=plot_title + " " + metric_units)
        plot_collection.add_figure(fig_title, fig)

    def add_traj_colormap_ape(self, plot_collection, ape_metric, traj_ref, traj_est1, traj_est2=None,
                              fig_title="", plot_title=""):
        """ Adds a trajectory colormap of ATE metrics to a plot collection.

            Args:
                plot_collection: a PlotCollection containing plots.
                ape_metric: an evo.core.metric object with statistics and information for APE.
                traj_ref: a PoseTrajectory3D object representing the reference trajectory.
                traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
                traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory. Optional.
                fig_title: a string representing the title of the figure. Must be unique in the plot_collection.
                plot_title: a string representing the title of the plot.
        """
        fig = plt.figure(figsize=(8, 8))
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
        plot_collection.add_figure(fig_title, fig)

    def add_traj_colormap_rpe(self, plot_collection, rpe_metric, traj_ref, traj_est1, traj_est2=None,
                              fig_title="", plot_title=""):
        """ Adds a trajectory colormap of RPE metrics to a plot collection.

            Args:
                plot_collection: a PlotCollection containing plots.
                ape_metric: an evo.core.metric object with statistics and information for RPE.
                traj_ref: a PoseTrajectory3D object representing the reference trajectory.
                traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
                traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory. Optional.
                fig_title: a string representing the title of the figure. Must be unique in the plot_collection.
                plot_title: a string representing the title of the plot.
        """
        fig = plt.figure(figsize=(8, 8))
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
        plot_collection.add_figure(fig_title, fig)

    def calc_results(self, ape_metric, rpe_metric_trans, rpe_metric_rot, data, segments):
        """ Create and return a dictionary containing stats and results for ATE, RRE and RTE for a datset.

            Args:
                ape_metric: an evo.core.metric object representing the ATE.
                rpe_metric_trans: an evo.core.metric object representing the RTE.
                rpe_metric_rot: an evo.core.metric object representing the RRE.
                data: a 2-tuple with reference and estimated trajectories as PoseTrajectory3D objects
                    in that order.
                segments: a list of segments for RPE.

            Returns: a dictionary containing all relevant results.
        """
        # Calculate APE results:
        results = dict()
        ape_result = ape_metric.get_result()
        results["absolute_errors"] = ape_result

        # Calculate RPE results:
        # TODO(Toni): Save RPE computation results rather than the statistics
        # you can compute statistics later...
        rpe_stats_trans = rpe_metric_trans.get_all_statistics()
        rpe_stats_rot = rpe_metric_rot.get_all_statistics()

        # Calculate RPE results of segments and save
        results["relative_errors"] = dict()
        for segment in segments:
            results["relative_errors"][segment] = dict()
            evt.print_purple("RPE analysis of segment: %d"%segment)
            evt.print_lightpurple("Calculating RPE segment translation part")
            rpe_segment_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                                   float(segment), metrics.Unit.meters, 0.01, True)
            rpe_segment_metric_trans.process_data(data)
            rpe_segment_stats_trans = rpe_segment_metric_trans.get_all_statistics()
            results["relative_errors"][segment]["rpe_trans"] = rpe_segment_stats_trans

            evt.print_lightpurple("Calculating RPE segment rotation angle")
            rpe_segment_metric_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                                 float(segment), metrics.Unit.meters, 0.01, True)
            rpe_segment_metric_rot.process_data(data)
            rpe_segment_stats_rot = rpe_segment_metric_rot.get_all_statistics()
            results["relative_errors"][segment]["rpe_rot"] = rpe_segment_stats_rot

        return results

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
            check_stats(stats[pipeline_type])

        log.info("Drawing boxplots.")
        evt.draw_rpe_boxplots(results_dataset_dir, stats, len(dataset_segments))
