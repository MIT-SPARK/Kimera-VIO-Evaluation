#!/usr/bin/env python

from __future__ import print_function
import copy
import os
import yaml
import numpy as np
from evo.tools import plot
import matplotlib.pyplot as plt
from shutil import copyfile, move, rmtree, copytree, copy2

from evo.core import trajectory, sync, metrics
import evaluation.tools as evt

Y_MAX_APE_TRANS = {
    "MH_01_easy": 0.3,
    "MH_02_easy": 0.25,
    "MH_03_medium": 0.35,
    "MH_04_difficult": 0.5,
    "MH_05_difficult": 0.36,
    "V1_01_easy": 0.170,
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

def aggregate_ape_results(list_of_datasets, list_of_pipelines, results_dir):
    """ Aggregate APE results and draw APE boxplot as well as write latex table
    with results """
    # Load results.
    print("Loading dataset results")

    # Aggregate all stats for each pipeline and dataset
    stats = dict()
    for dataset_name in list_of_datasets:
        dataset_dir = os.path.join(results_dir, dataset_name)
        stats[dataset_name] = dict()
        for pipeline_name in list_of_pipelines:
            pipeline_dir = os.path.join(dataset_dir, pipeline_name)
            # Get results.
            results_file = os.path.join(pipeline_dir, 'results.yaml')
            stats[dataset_name][pipeline_name] = yaml.load(open(results_file, 'r'))
            print("Check stats from " + results_file)
            check_stats(stats[dataset_name][pipeline_name])

    print("Drawing APE boxplots.")
    evt.draw_ape_boxplots(stats, results_dir)
    # Write APE table
    evt.write_latex_table(stats, results_dir)
    # Write APE table without S pipeline

def check_stats(stats):
    if not "relative_errors" in stats:
        print("Stats: ")
        print(stats)
        raise Exception("\033[91mWrong stats format: no relative_errors... \n"
                        "Are you sure you runned the pipeline and "
                        "saved the results? (--save_results).\033[99m")
    else:
        if len(stats["relative_errors"]) == 0:
            raise Exception("\033[91mNo relative errors available... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")

        if not "rpe_rot" in list(stats["relative_errors"].values())[0]:
            print("Stats: ")
            print(stats)
            raise Exception("\033[91mWrong stats format: no rpe_rot... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")
        if not "rpe_trans" in list(stats["relative_errors"].values())[0]:
            print("Stats: ")
            print(stats)
            raise Exception("\033[91mWrong stats format: no rpe_trans... \n"
                            "Are you sure you runned the pipeline and "
                            "saved the results? (--save_results).\033[99m")
    if not "absolute_errors" in stats:
        print("Stats: ")
        print(stats)
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
        print(e)
        raise Exception("\033[91mMissing ground truth csv.\033[99m")

    print("Registering trajectories")
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    print("Aligning trajectories")
    traj_est = trajectory.align_trajectory(traj_est, traj_ref, correct_scale = False,
                                           discard_n_start_poses = int(discard_n_start_poses),
                                           discard_n_end_poses = int(discard_n_end_poses))

    num_of_poses = traj_est.num_poses
    traj_est.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))
    traj_ref.reduce_to_ids(range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1))

    results = dict()

    results["absolute_errors"] = dict()
    print("Calculating APE translation part")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()
    results["absolute_errors"] = ape_statistics
    print("mean:", ape_statistics["mean"])

    print("Calculating RPE translation part for plotting")
    rpe_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                   1.0, metrics.Unit.frames, 0.0, False)
    rpe_metric_trans.process_data(data)
    rpe_stats_trans = rpe_metric_trans.get_all_statistics()
    print("mean:", rpe_stats_trans["mean"])

    print("Calculating RPE rotation angle for plotting")
    rpe_metric_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                                 1.0, metrics.Unit.frames, 1.0, False)
    rpe_metric_rot.process_data(data)
    rpe_stats_rot = rpe_metric_rot.get_all_statistics()
    print("mean:", rpe_stats_rot["mean"])

    results["relative_errors"] = dict()
    # Read segments file
    for segment in segments:
        results["relative_errors"][segment] = dict()
        print("RPE analysis of segment: %d"%segment)
        print("Calculating RPE segment translation part")
        rpe_segment_metric_trans = metrics.RPE(metrics.PoseRelation.translation_part,
                                       float(segment), metrics.Unit.meters, 0.01, True)
        rpe_segment_metric_trans.process_data(data)
        rpe_segment_stats_trans = rpe_segment_metric_trans.get_all_statistics()
        results["relative_errors"][segment]["rpe_trans"] = rpe_segment_stats_trans
        # print(rpe_segment_stats_trans)
        # print("mean:", rpe_segment_stats_trans["mean"])

        print("Calculating RPE segment rotation angle")
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
        print("Saving analysis results to: " + results_file)
        if confirm_overwrite:
            if not evt.user.check_and_confirm_overwrite(results_file):
                return
        with open(results_file,'w') as outfile:
            outfile.write(yaml.dump(results, default_flow_style=False))

    # For each segment in segments file
    # Calculate rpe with delta = segment in meters with all-pairs set to True
    # Calculate max, min, rmse, mean, median etc
    # Plot boxplot, or those cumulative figures you see in evo (like demographic plots)

    if display_plot or save_plots:
        print("plotting")
        plot_collection = plot.PlotCollection("Example")
        # metric values
        fig_1 = plt.figure(figsize=(8, 8))
        ymax = -1
        print(dataset_name)
        if dataset_name is not "":
            ymax = Y_MAX_APE_TRANS[dataset_name]
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
                           min_map=0.0, max_map=ymax,
                           title="APE translation error mapped onto trajectory [m]")
        plot_collection.add_figure("APE_translation_trajectory_error", fig_2)

        # RPE
        ## Trans
        ### metric values
        fig_3 = plt.figure(figsize=(8, 8))
        if dataset_name is not "":
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
                           min_map=0.0, max_map=ymax,
                           title="RPE translation error mapped onto trajectory [m]"
                          )
        plot_collection.add_figure("RPE_translation_trajectory_error", fig_4)

        ## Rot
        ### metric values
        fig_5 = plt.figure(figsize=(8, 8))
        if dataset_name is not "":
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
                           min_map=0.0, max_map=ymax,
                           title="RPE rotation error mapped onto trajectory [deg]")
        plot_collection.add_figure("RPE_rotation_trajectory_error", fig_6)

        if display_plot:
            print("Displaying plots.")
            plot_collection.show()

        if save_plots:
            print("Saving plots to: " + save_folder)
            plot_collection.export(save_folder + "/plots.pdf", False)

# Run pipeline as a subprocess.
def run_vio(build_dir, dataset_dir, dataset_name, params_dir,
            pipeline_output_dir, pipeline_type, initial_k, final_k,
            extra_flagfile_path=""):
    """ Runs pipeline depending on the pipeline_type using a subprocess."""
    import subprocess
    return subprocess.call("{}/spark_vio \
                           --logtostderr=1 --colorlogtostderr=1 --log_prefix=0 \
                           --dataset_path={}/{} --output_path={} \
                           --vio_params_path={}/{}/{} \
                           --tracker_params_path={}/{}/{} \
                           --flagfile={}/{}/{} --flagfile={}/{}/{} \
                           --flagfile={}/{}/{} --flagfile={}/{}/{} \
                           --flagfile={}/{}/{} --flagfile={}/{}/{} \
                           --initial_k={} --final_k={} \
                           --log_output=True".format(
                               build_dir, dataset_dir, dataset_name, pipeline_output_dir,
                               params_dir, pipeline_type, "regularVioParameters.yaml",
                               params_dir, pipeline_type, "trackerParameters.yaml",
                               params_dir, pipeline_type, "flags/stereoVIOEuroc.flags",
                               params_dir, pipeline_type, "flags/Mesher.flags",
                               params_dir, pipeline_type, "flags/VioBackEnd.flags",
                               params_dir, pipeline_type, "flags/RegularVioBackEnd.flags",
                               params_dir, pipeline_type, "flags/Visualizer3D.flags",
                               params_dir, pipeline_type, extra_flagfile_path,
                               initial_k, final_k), \
                           shell=True)

def process_vio(build_dir, dataset_dir, dataset_name, results_dir, params_dir, pipeline_output_dir,
                pipeline_type, SEGMENTS, save_results, plot, save_plots, output_file, run_pipeline,
                analyse_vio, discard_n_start_poses, discard_n_end_poses, initial_k, final_k):
    """ 
    * build_dir: directory where the pipeline executable resides.
    * dataset_dir: directory of the dataset, must contain traj_gt.csv (the ground truth trajectory for analysis to work).
    * dataset_name: specific dataset to run.
    * results_dir: directory where the results of the run will reside:
        used as results_dir/dataset_name/S, results_dir/dataset_name/SP, results_dir/dataset_name/SPR
        where each directory have traj_est.csv (the estimated trajectory), and plots if requested.
    * params_dir: directory where the parameters for each pipeline reside:
        used as params_dir/S, params_dir/SP, params_dir/SPR.
    * pipeline_output_dir: where to store all output_* files produced by the pipeline.
    * pipeline_type: type of pipeline to process (1: S, 2: SP, 3: SPR).
    * SEGMENTS: segments for RPE boxplots.
    * save_results: saves APE, and RPE per segment results of the run.
    * plot: whether to plot the APE/RPE results or not.
    * save_plots: saves plots of APE/RPE.
    * output_file: the name of the trajectory estimate output of the vio which will then be copied as traj_est.csv.
    * run_pipeline: whether to run the VIO to generate a new traj_est.csv.
    * analyse_vio: whether to analyse traj_est.csv or not.
    """
    dataset_results_dir = results_dir + "/" + dataset_name + "/"
    dataset_pipeline_result_dir = dataset_results_dir + "/" + pipeline_type + "/"
    traj_ref_path = dataset_dir + "/" + dataset_name + "/mav0/state_groundtruth_estimate0/data.csv" # TODO make it not specific to EUROC
    traj_es = dataset_results_dir + "/" + pipeline_type + "/" + "traj_es.csv"
    evt.create_full_path_if_not_exists(traj_es)
    if run_pipeline:
        if run_vio(build_dir, dataset_dir, dataset_name, params_dir,
                   pipeline_output_dir, pipeline_type, initial_k, final_k) == 0:
            print("Successful pipeline run.")
            print("\033[1mCopying output file: " + output_file + "\n to results file:\n" + \
                  traj_es + "\033[0m")
            copyfile(output_file, traj_es)
            output_destination_dir = dataset_pipeline_result_dir + "/output/"
            print("\033[1mMoving output dir: " + pipeline_output_dir
                  + "\n to destination:\n" + output_destination_dir + "\033[0m")
            try:
                evt.move_output_from_to(pipeline_output_dir, output_destination_dir)
            except:
                print("\033[1mFailed copying output dir: " + pipeline_output_dir
                      + "\n to destination:\n" + output_destination_dir + "\033[0m")
        else:
            print("Pipeline failed on dataset: " + dataset_name)
            return False

    if analyse_vio:
        print("\033[1mAnalysing dataset: " + dataset_results_dir + " for pipeline "
              + pipeline_type + ".\033[0m")
        run_analysis(traj_ref_path, traj_es, SEGMENTS,
                     save_results, plot, save_plots, dataset_pipeline_result_dir, False,
                     dataset_name,
                     discard_n_start_poses,
                     discard_n_end_poses)
    return True

def run_dataset(results_dir, params_dir, dataset_dir, dataset_properties, build_dir,
                run_pipeline, analyse_vio,
                plot, save_results, save_plots, save_boxplots, pipelines_to_run_list,
                initial_k, final_k, discard_n_start_poses = 0, discard_n_end_poses = 0):
    """ Evaluates pipeline using Structureless(S), Structureless(S) + Projection(P), \
            and Structureless(S) + Projection(P) + Regular(R) factors \
            and then compiles a list of results """
    dataset_name = dataset_properties['name']
    dataset_segments = dataset_properties['segments']

    ################### RUN PIPELINE ################################
    pipeline_output_dir = results_dir + "/tmp_output/output/"
    evt.create_full_path_if_not_exists(pipeline_output_dir)
    output_file = pipeline_output_dir + "/output_posesVIO.csv"
    has_a_pipeline_failed = False
    if len(pipelines_to_run_list) == 0:
        print("Not running pipeline...")
    for pipeline_type in pipelines_to_run_list:
        has_a_pipeline_failed = not process_vio(
            build_dir, dataset_dir, dataset_name, results_dir, params_dir,
            pipeline_output_dir, pipeline_type, dataset_segments, save_results,
            plot, save_plots, output_file, run_pipeline, analyse_vio,
            discard_n_start_poses, discard_n_end_poses,
            initial_k, final_k)

    # Save boxplots
    if save_boxplots:
        if has_a_pipeline_failed == False:
            print("Saving boxplots.")

            stats = dict()
            for pipeline_type in pipelines_to_run_list:
                results = results_dir + "/" + dataset_name + "/" + pipeline_type + "/results.yaml"
                if not os.path.exists(results):
                    raise Exception("\033[91mCannot plot boxplots: missing results for %s pipeline \
                                    and dataset: %s"%(pipeline_type, dataset_name) + "\033[99m")

                stats[pipeline_type]  = yaml.load(open(results,'r'))
                print("Check stats %s "%(pipeline_type) + results)
                check_stats(stats[pipeline_type])

            print("Drawing boxplots.")
            evt.draw_rpe_boxplots(results_dir + "/" + dataset_name, stats, len(dataset_segments))
        else:
            print("A pipeline run has failed... skipping boxplot drawing.")

    if not has_a_pipeline_failed:
        print("All pipeline runs were successful.")
    else:
        print("A pipeline has failed!")
    print("Finished evaluation for dataset: " + dataset_name)
    return not has_a_pipeline_failed
