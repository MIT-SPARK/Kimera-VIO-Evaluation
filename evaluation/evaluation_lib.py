#!/usr/bin/env python

from __future__ import print_function
import copy
import os
import yaml
import math
import subprocess
import numpy as np
import glog as log
from evo.tools import plot
import matplotlib.pyplot as plt
from shutil import copyfile, move, rmtree, copytree, copy2

from evo.core import trajectory, sync, metrics
import evaluation.tools as evt

FIX_MAX_Y = True
Y_MAX_APE_TRANS = {
    "MH_01_easy": 0.3,
    "MH_02_easy": 0.25,
    "MH_03_medium": 0.35,
    "MH_04_difficult": 0.5,
    "MH_05_difficult": 0.36,
    "V1_01_easy": 0.125,
    "V1_02_medium": 0.16,
    "V1_03_difficult": 0.4,
    "V2_01_easy": 0.175,
    "V2_02_medium": 0.24,
    "V2_03_difficult": 0.7
}
Y_MAX_RPE_TRANS = {
    "MH_01_easy": 0.028,
    "MH_02_easy": 0.025,
    "MH_03_medium": 0.091,
    "MH_04_difficult": 0.21,
    "MH_05_difficult": 0.07,
    "V1_01_easy": 0.03,
    "V1_02_medium": 0.04,
    "V1_03_difficult": 0.15,
    "V2_01_easy": 0.04,
    "V2_02_medium": 0.06,
    "V2_03_difficult": 0.17
}
Y_MAX_RPE_ROT = {
    "MH_01_easy": 0.4,
    "MH_02_easy": 0.6,
    "MH_03_medium": 0.35,
    "MH_04_difficult": 1.0,
    "MH_05_difficult": 0.3,
    "V1_01_easy": 0.6,
    "V1_02_medium": 1.5,
    "V1_03_difficult": 1.25,
    "V2_01_easy": 0.6,
    "V2_02_medium": 1.0,
    "V2_03_difficult": 2.6
}

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
        for results_filename in fnmatch.filter(filenames, 'results.yaml'):
            results_filepath = os.path.join(root, results_filename)
            # Get pipeline name
            pipeline_name = os.path.basename(root)
            # Get dataset name
            dataset_name = os.path.basename(os.path.split(root)[0])
            # Collect stats
            if stats.get(dataset_name) is None:
                stats[dataset_name] = dict()
            stats[dataset_name][pipeline_name] = yaml.load(open(results_filepath, 'r'))
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

def run_analysis(traj_ref_path, traj_est_path, segments, save_results, display_plot, save_plots,
                 save_folder, confirm_overwrite = False, dataset_name = "", discard_n_start_poses=0,
                discard_n_end_poses=0):
    """ Run analysis on given trajectories, saves plots on given path:
    :param traj_ref_path: path to the reference (ground truth) trajectory.
    :param traj_est_path: path to the estimated trajectory.
    :param save_results: saves APE, and RPE per segment results.
    :param save_plots: whether to save the plots.
    :param save_folder: where to save the plots.
    :param confirm_overwrite: whether to confirm overwriting plots or not.
    :param dataset_name: optional param, to allow setting the same scale on different plots.
    """
    # Load trajectories.
    from evo.tools import file_interface
    traj_ref = None
    try:
        traj_ref = file_interface.read_euroc_csv_trajectory(traj_ref_path) # TODO make it non-euroc specific.
    except file_interface.FileInterfaceException as e:
        raise Exception("\033[91mMissing ground truth csv! \033[93m {}.".format(e))

    traj_est = None
    try:
        traj_est = file_interface.read_swe_csv_trajectory(traj_est_path)
    except file_interface.FileInterfaceException as e:
        log.info(e)
        raise Exception("\033[91mMissing vio output csv.\033[99m")

    evt.print_purple("Registering trajectories")
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    evt.print_purple("Aligning trajectories")
    traj_est = trajectory.align_trajectory(traj_est, traj_ref, correct_scale = False,
                                           discard_n_start_poses = int(discard_n_start_poses),
                                           discard_n_end_poses = int(discard_n_end_poses))

    num_of_poses = traj_est.num_poses
    traj_est.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
    traj_ref.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

    results = dict()

    evt.print_purple("Calculating APE translation part")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_result = ape_metric.get_result()
    results["absolute_errors"] = ape_result

    log.info(ape_result.pretty_str(info=True))

    # TODO(Toni): Save RPE computation results rather than the statistics
    # you can compute statistics later...
    evt.print_purple("Calculating RPE translation part for plotting")
    rpe_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                   1.0, metrics.Unit.frames, 0.0, False)
    rpe_metric_trans.process_data(data)
    rpe_stats_trans = rpe_metric_trans.get_all_statistics()
    log.info("mean: %f" % rpe_stats_trans["mean"])

    evt.print_purple("Calculating RPE rotation angle for plotting")
    rpe_metric_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                 1.0, metrics.Unit.frames, 1.0, False)
    rpe_metric_rot.process_data(data)
    rpe_stats_rot = rpe_metric_rot.get_all_statistics()
    log.info("mean: %f" % rpe_stats_rot["mean"])

    results["relative_errors"] = dict()
    # Read segments file
    for segment in segments:
        results["relative_errors"][segment] = dict()
        evt.print_purple("RPE analysis of segment: %d"%segment)
        evt.print_lightpurple("Calculating RPE segment translation part")
        rpe_segment_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                       float(segment), metrics.Unit.meters, 0.01, True)
        rpe_segment_metric_trans.process_data(data)
        rpe_segment_stats_trans = rpe_segment_metric_trans.get_all_statistics()
        results["relative_errors"][segment]["rpe_trans"] = rpe_segment_stats_trans
        # print(rpe_segment_stats_trans)
        # print("mean:", rpe_segment_stats_trans["mean"])

        evt.print_lightpurple("Calculating RPE segment rotation angle")
        rpe_segment_metric_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                     float(segment), metrics.Unit.meters, 0.01, True)
        rpe_segment_metric_rot.process_data(data)
        rpe_segment_stats_rot = rpe_segment_metric_rot.get_all_statistics()
        results["relative_errors"][segment]["rpe_rot"] = rpe_segment_stats_rot
        # print(rpe_segment_stats_rot)
        # print("mean:", rpe_segment_stats_rot["mean"])

    if save_results:
        # Save results file
        results_file = os.path.join(save_folder, 'results.yaml')
        evt.print_green("Saving analysis results to: %s" % results_file)
        with open(results_file,'w') as outfile:
            if confirm_overwrite:
                if evt.user.check_and_confirm_overwrite(results_file):
                        outfile.write(yaml.dump(results, default_flow_style=False))
                else:
                    log.info("Not overwritting results.")
            else:
                outfile.write(yaml.dump(results, default_flow_style=False))

    # For each segment in segments file
    # Calculate rpe with delta = segment in meters with all-pairs set to True
    # Calculate max, min, rmse, mean, median etc

    # Plot boxplot, or those cumulative figures you see in evo (like demographic plots)
    if display_plot or save_plots:
        evt.print_green("Plotting:")
        log.info(dataset_name)
        plot_collection = plot.PlotCollection("Example")
        # metric values
        fig_1 = plt.figure(figsize=(8, 8))
        ymax = -1
        if dataset_name is not "" and FIX_MAX_Y:
            ymax = Y_MAX_APE_TRANS[dataset_name]

        ape_statistics = ape_metric.get_all_statistics()
        plot.error_array(fig_1, ape_metric.error, statistics=ape_statistics,
                         name="APE translation", title=""#str(ape_metric)
                         , xlabel="Keyframe index [-]",
                         ylabel="APE translation [m]", y_min= 0.0, y_max=ymax)
        plot_collection.add_figure("APE_translation", fig_1)

        # trajectory colormapped with error
        fig_2 = plt.figure(figsize=(8, 8))
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig_2, plot_mode)
        plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
        plot.traj_colormap(ax, traj_est, ape_metric.error, plot_mode,
                           min_map=0.0, max_map=math.ceil(ape_statistics['max']*10)/10,
                           title="ATE mapped onto trajectory [m]")
        plot_collection.add_figure("APE_translation_trajectory_error", fig_2)

        # RPE
        ## Trans
        ### metric values
        fig_3 = plt.figure(figsize=(8, 8))
        if dataset_name is not "" and FIX_MAX_Y:
            ymax = Y_MAX_RPE_TRANS[dataset_name]
        plot.error_array(fig_3, rpe_metric_trans.error, statistics=rpe_stats_trans,
                         name="RPE translation", title=""#str(rpe_metric_trans)
                         , xlabel="Keyframe index [-]", ylabel="RPE translation [m]", y_max=ymax)
        plot_collection.add_figure("RPE_translation", fig_3)

        ### trajectory colormapped with error
        fig_4 = plt.figure(figsize=(8, 8))
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig_4, plot_mode)
        traj_ref_trans = copy.deepcopy(traj_ref)
        traj_ref_trans.reduce_to_ids(rpe_metric_trans.delta_ids)
        traj_est_trans = copy.deepcopy(traj_est)
        traj_est_trans.reduce_to_ids(rpe_metric_trans.delta_ids)
        plot.traj(ax, plot_mode, traj_ref_trans, '--', 'gray', 'Reference')
        plot.traj_colormap(ax, traj_est_trans, rpe_metric_trans.error, plot_mode,
                           min_map=0.0, max_map=math.ceil(rpe_stats_trans['max']*10)/10,
                           title="RPE translation error mapped onto trajectory [m]"
                          )
        plot_collection.add_figure("RPE_translation_trajectory_error", fig_4)

        ## Rot
        ### metric values
        fig_5 = plt.figure(figsize=(8, 8))
        if dataset_name is not "" and FIX_MAX_Y:
            ymax = Y_MAX_RPE_ROT[dataset_name]
        plot.error_array(fig_5, rpe_metric_rot.error, statistics=rpe_stats_rot,
                         name="RPE rotation error", title=""#str(rpe_metric_rot)
                         , xlabel="Keyframe index [-]", ylabel="RPE rotation [deg]", y_max=ymax)
        plot_collection.add_figure("RPE_rotation", fig_5)

        ### trajectory colormapped with error
        fig_6 = plt.figure(figsize=(8, 8))
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig_6, plot_mode)
        traj_ref_rot = copy.deepcopy(traj_ref)
        traj_ref_rot.reduce_to_ids(rpe_metric_rot.delta_ids)
        traj_est_rot = copy.deepcopy(traj_est)
        traj_est_rot.reduce_to_ids(rpe_metric_rot.delta_ids)
        plot.traj(ax, plot_mode, traj_ref_rot, '--', 'gray', 'Reference')
        plot.traj_colormap(ax, traj_est_rot, rpe_metric_rot.error, plot_mode,
                           min_map=0.0, max_map=math.ceil(rpe_stats_rot['max']*10)/10,
                           title="RPE rotation error mapped onto trajectory [deg]")
        plot_collection.add_figure("RPE_rotation_trajectory_error", fig_6)

        if display_plot:
            evt.print_green("Displaying plots.")
            plot_collection.show()

        if save_plots:
            evt.print_green("Saving plots to: ")
            log.info(save_folder)
            # Config output format (pdf, eps, ...) using evo_config...
            plot_collection.export(os.path.join(save_folder, "plots.eps"), False)
            plot_collection.export(os.path.join(save_folder, "plots.pdf"), False)

# Run pipeline as a subprocess.
def run_vio(executable_path, dataset_dir, dataset_name, params_dir,
            pipeline_output_dir, pipeline_type, initial_k, final_k,
            extra_flagfile_path="", verbose_sparkvio=False):
    """ Runs pipeline depending on the pipeline_type using a subprocess.
    Args:
        - executable_path: where the SparkVIO executable is.
        - dataset_dir: where the Euroc dataset is.
        - dataset_name: Euroc dataset to be run.
        - params_dir: directory where the SparkVIO parameters are stored. Needs to follow
            a convention: flagfiles must have the names below, same for yaml files.
        - pipeline_output_dir: directory where to store output information from SparkVIO.
        - pipeline_type: must be one of ['S', 'SP', 'SPR']
        - initial_k: k_th frame where to start running SparkVIO
        - final_k: k_th frame where to stop SparkVIO
        - extra_flagfile_path: to be used in order to override other flags or add new ones.
            Useful for regression tests when the param to be regressed is a gflag.
        - verbose_sparkvio: whether to print the SparkVIO messages or not.
            This is useful for debugging, but too verbose when you want to see APE/RPE results.
    """

    def spark_vio_thread(thread_return, minloglevel=0):
        """ Function to run SparkVIO in another thread """
        thread_return['success'] = subprocess.call("{} \
                            --logtostderr=1 --colorlogtostderr=1 --log_prefix=1 \
                            --dataset_path={}/{} --output_path={} \
                            --vio_params_path={}/{}/{} \
                            --tracker_params_path={}/{}/{} \
                            --flagfile={}/{}/{} --flagfile={}/{}/{} \
                            --flagfile={}/{}/{} --flagfile={}/{}/{} \
                            --flagfile={}/{}/{} --flagfile={}/{} \
                            --initial_k={} --final_k={} \
                            --log_output=True --minloglevel={}".format(
            executable_path, dataset_dir, dataset_name, pipeline_output_dir,
            params_dir, pipeline_type, "regularVioParameters.yaml",
            params_dir, pipeline_type, "trackerParameters.yaml",
            params_dir, pipeline_type, "flags/stereoVIOEuroc.flags",
            params_dir, pipeline_type, "flags/Mesher.flags",
            params_dir, pipeline_type, "flags/VioBackEnd.flags",
            params_dir, pipeline_type, "flags/RegularVioBackEnd.flags",
            params_dir, pipeline_type, "flags/Visualizer3D.flags",
            params_dir, extra_flagfile_path,
            initial_k, final_k, minloglevel),
            shell=True)

    import threading
    import time
    import itertools, sys # just for spinner
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    thread_return={'success': False}
    minloglevel = 2 # Set SparkVIO verbosity level to ERROR
    if verbose_sparkvio:
        minloglevel = 0 # Set SparkVIO verbosity level to INFO
    thread = threading.Thread(target=spark_vio_thread, args=(thread_return, minloglevel,))
    thread.start()
    while thread.is_alive():
        if not verbose_sparkvio:
            # If SparkVIO is not in verbose mode, the user might think the python script is hanging.
            # So, instead, display a spinner of 80 characters.
            sys.stdout.write(next(spinner) * 80)  # write the next character
            sys.stdout.flush()                     # flush stdout buffer (actual character display)
            sys.stdout.write('\b' * 80)            # erase the last written char
        time.sleep(0.100) # Sleep 100ms while SparkVIO is running
    thread.join()
    return thread_return['success']

def process_vio(executable_path, dataset_dir, dataset_name, results_dir, params_dir, pipeline_output_dir,
                pipeline_type, SEGMENTS, save_results, plot, save_plots, output_file, run_pipeline,
                analyse_vio, discard_n_start_poses, discard_n_end_poses, initial_k, final_k, extra_flagfile_path='',
                verbose_sparkvio=False):
    """ 
    Args:
        - executable_path: path to the pipeline executable (i.e. `./build/spark_vio`).
        - dataset_dir: directory of the dataset, must contain traj_gt.csv (the ground truth trajectory for analysis to work).
        - dataset_name: specific dataset to run.
        - results_dir: directory where the results of the run will reside:
        -   used as results_dir/dataset_name/S, results_dir/dataset_name/SP, results_dir/dataset_name/SPR
        -   where each directory have traj_est.csv (the estimated trajectory), and plots if requested.
        - params_dir: directory where the parameters for each pipeline reside:
        -   used as params_dir/S, params_dir/SP, params_dir/SPR.
        - pipeline_output_dir: where to store all output_* files produced by the pipeline.
        - pipeline_type: type of pipeline to process (1: S, 2: SP, 3: SPR).
        - SEGMENTS: segments for RPE boxplots.
        - save_results: saves APE, and RPE per segment results of the run.
        - plot: whether to plot the APE/RPE results or not.
        - save_plots: saves plots of APE/RPE.
        - output_file: the name of the trajectory estimate output of the vio which will then be copied as traj_est.csv.
        - run_pipeline: whether to run the VIO to generate a new traj_est.csv.
        - analyse_vio: whether to analyse traj_est.csv or not.
        - extra_flagfile_path: to be used in order to override other flags or add new ones.
            Useful for regression tests when the param to be regressed is a gflag.
        - verbose_sparkvio: whether to print the SparkVIO messages or not.
            This is useful for debugging, but too verbose when you want to see APE/RPE results.
    """
    dataset_results_dir = os.path.join(results_dir, dataset_name)
    dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)
    traj_ref_path = os.path.join(dataset_dir, dataset_name, "mav0/state_groundtruth_estimate0/data.csv") # TODO make it not specific to EUROC
    traj_es = os.path.join(dataset_results_dir, pipeline_type, "traj_es.csv")
    evt.create_full_path_if_not_exists(traj_es)
    if run_pipeline:
        evt.print_green("Run pipeline: %s" % pipeline_type)
        # The override flags are used by the regression tests.
        if run_vio(executable_path, dataset_dir, dataset_name, params_dir,
                   pipeline_output_dir, pipeline_type, initial_k, final_k,
                   extra_flagfile_path, verbose_sparkvio) == 0:
            evt.print_green("Successful pipeline run.")
            log.debug("\033[1mCopying output file: \033[0m \n %s \n \033[1m to results file:\033[0m\n %s" % 
                (output_file, traj_es))
            copyfile(output_file, traj_es)
            output_destination_dir = os.path.join(dataset_pipeline_result_dir, "output")
            log.debug("\033[1mMoving output dir:\033[0m \n %s \n \033[1m to destination:\033[0m \n %s" % 
                (pipeline_output_dir, output_destination_dir))
            try:
                evt.move_output_from_to(pipeline_output_dir, output_destination_dir)
            except:
                log.fatal("\033[1mFailed copying output dir: \033[0m\n %s \n \033[1m to destination: %s \033[0m\n" % 
                    (pipeline_output_dir, output_destination_dir))
        else:
            log.error("Pipeline failed on dataset: " + dataset_name)
            # Avoid writting results.yaml with analysis if the pipeline failed.
            log.info("Not writting results.yaml")
            return False

    if analyse_vio:
        log.debug("\033[1mAnalysing dataset:\033[0m \n %s \n \033[1m for pipeline \033[0m %s."
                 % (dataset_results_dir, pipeline_type))
        evt.print_green("Starting analysis of pipeline: %s" % pipeline_type)
        run_analysis(traj_ref_path, traj_es, SEGMENTS,
                     save_results, plot, save_plots, dataset_pipeline_result_dir, False,
                     dataset_name,
                     discard_n_start_poses,
                     discard_n_end_poses)
    return True

# TODO(Toni): we are passing all params all the time.... Make a class!!
def run_dataset(results_dir, params_dir, dataset_dir, dataset_properties, executable_path,
                run_pipeline, analyse_vio,
                plot, save_results, save_plots, save_boxplots, pipelines_to_run_list,
                initial_k, final_k, discard_n_start_poses = 0, discard_n_end_poses = 0, extra_flagfile_path = '',
                verbose_sparkvio = False):
    """ Evaluates pipeline using Structureless(S), Structureless(S) + Projection(P), \
            and Structureless(S) + Projection(P) + Regular(R) factors \
            and then compiles a list of results """
    dataset_name = dataset_properties['name']
    dataset_segments = dataset_properties['segments']

    ################### RUN PIPELINE ################################
    pipeline_output_dir = os.path.join(results_dir, "tmp_output/output")
    evt.create_full_path_if_not_exists(pipeline_output_dir)
    output_file = os.path.join(pipeline_output_dir, "output_posesVIO.csv")
    has_a_pipeline_failed = False
    if len(pipelines_to_run_list) == 0:
        log.warning("Not running pipeline...")
    for pipeline_type in pipelines_to_run_list:
        has_a_pipeline_failed = not process_vio(
            executable_path, dataset_dir, dataset_name, results_dir, params_dir,
            pipeline_output_dir, pipeline_type, dataset_segments, save_results,
            plot, save_plots, output_file, run_pipeline, analyse_vio,
            discard_n_start_poses, discard_n_end_poses,
            initial_k, final_k, extra_flagfile_path, verbose_sparkvio)

    # Save boxplots
    if save_boxplots:
        # TODO(Toni) is this really saving the boxplots?
        if not has_a_pipeline_failed:
            stats = dict()
            for pipeline_type in pipelines_to_run_list:
                results_dataset_dir = os.path.join(results_dir, dataset_name)
                results = os.path.join(results_dataset_dir, pipeline_type, "results.yaml")
                if not os.path.exists(results):
                    raise Exception("\033[91mCannot plot boxplots: missing results for %s pipeline \
                                    and dataset: %s" % (pipeline_type, dataset_name) + "\033[99m \n \
                                    Expected results here: %s" % results)

                try:
                    stats[pipeline_type]  = yaml.load(open(results,'r'), Loader=yaml.Loader)
                except yaml.YAMLError as e:
                    raise Exception("Error in results file: ", e)

                log.info("Check stats %s in %s" % (pipeline_type, results))
                check_stats(stats[pipeline_type])

            log.info("Drawing boxplots.")
            evt.draw_rpe_boxplots(results_dataset_dir, stats, len(dataset_segments))
        else:
            log.warning("A pipeline run has failed... skipping boxplot drawing.")

    if not has_a_pipeline_failed:
        evt.print_green("All pipeline runs were successful.")
    else:
        log.error("A pipeline has failed!")
    evt.print_green("Finished evaluation for dataset: " + dataset_name)
    return not has_a_pipeline_failed
