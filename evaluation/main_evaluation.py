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

Y_MAX_APE_TRANS={
    "MH_01_easy": 0.3, "MH_02_easy": 0.25, "MH_03_medium": 0.35,
    "mh_04_difficult": 0.5, "MH_05_difficult": 0.36, "V1_01_easy": 0.170,
    "V1_02_medium": 0.16, "V1_03_difficult": 0.4,"V2_01_easy": 0.175,
    "V2_02_medium": 0.24,"v2_03_difficult": 0.7
    }
Y_MAX_RPE_TRANS={
    "MH_01_easy": 0.028, "MH_02_easy": 0.025, "MH_03_medium": 0.091,
    "mh_04_difficult": 0.21, "MH_05_difficult": 0.07, "V1_01_easy": 0.03,
    "V1_02_medium": 0.04, "V1_03_difficult": 0.15,"V2_01_easy": 0.04,
    "V2_02_medium": 0.06,"v2_03_difficult": 0.17
    }
Y_MAX_RPE_ROT={
"MH_01_easy":0.4,
"MH_02_easy":0.6,
"MH_03_medium":0.35,
"mh_04_difficult":1.0,
"MH_05_difficult":0.3,
"V1_01_easy":0.6,
"V1_02_medium":1.5,
"V1_03_difficult":1.25,
"V2_01_easy":0.6,
"V2_02_medium":1.0,
"v2_03_difficult":2.6
}

def create_full_path_if_not_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def move_output_from_to(pipeline_output_dir, output_destination_dir):
    try:
        if (os.path.exists(output_destination_dir)):
            rmtree(output_destination_dir)
    except:
        print("Directory:" + output_destination_dir + " does not exist, we can safely move output.")
    try:
        if (os.path.isdir(pipeline_output_dir)):
            move(pipeline_output_dir, output_destination_dir)
        else:
            print("There is no output directory...")
    except:
        print("Could not move output from: " + pipeline_output_dir + " to: "
              + output_destination_dir)
        raise
    try:
        os.makedirs(pipeline_output_dir)
    except:
        print("Could not mkdir: " + pipeline_output_dir)
        raise

def ensure_dir(dir_path):
    """ Check if the path directory exists: if it does, returns true,
    if not creates the directory dir_path and returns if it was successful"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True

def aggregate_ape_results(list_of_datasets, list_of_pipelines):
    RESULTS_DIR = '/home/tonirv/code/evo-1/results'
    # Load results.
    print("Loading dataset results")

    # Aggregate all stats for each pipeline and dataset
    stats = dict()
    for dataset_name in list_of_datasets:
        dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        stats[dataset_name] = dict()
        for pipeline_name in list_of_pipelines:
            pipeline_dir = os.path.join(dataset_dir, pipeline_name)
            # Get results.
            results_file = os.path.join(pipeline_dir, 'results.yaml')
            stats[dataset_name][pipeline_name] = yaml.load(open(results_file, 'r'))
            print("Check stats from " + results_file)
            checkStats(stats[dataset_name][pipeline_name])

    print("Drawing APE boxplots.")
    evt.draw_ape_boxplots(stats, RESULTS_DIR)
    # Write APE table
    write_latex_table(stats, RESULTS_DIR)
    # Write APE table without S pipeline

def checkStats(stats):
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

def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:,0:3],axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances,distances),1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                      if smallest == element]

def write_latex_table(stats, results_dir):
    """ Write latex table with median, mean and rmse from stats:
            which is a list that contains:
                - dataset name (string) (like V1_01_easy, MH_01_easy etc):
                    - pipeline type (string) (like S, SP or SPR):
                        - "absolute_errors":
                            - "max"
                            - "min"
                            - "mean"
                            - "median"
                            - "q1"
                            - "q3"
                            - "rmse"
            This function iterates over the pipeline types, and for each pipeline type, it plots
            the metrics achieved, as a boxplot. So the boxplot has in x-axis the dataset name,
            and in y-axis one boxplot per pipeline."""
    # Tex stuff.
    # start_line = """\\begin{table}[H]
    # \\centering
    # \\resizebox{\\textwidth}{!}{
    # \\begin{tabular}{l p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm} p{1.4cm}}
    # \\hline
    # Sequence             & \\multicolumn{2}{c}{\\textbf{S}} & \\multicolumn{2}{c}{\\textbf{S + P}}  & \\multicolumn{2}{c}{\\textbf{S + P + R} (Proposed)}          \\\\ \\hline
                         # & Median APE Translation (m)  & Mean APE Translation (m) & RMSE APE Translation (m) &
                         # Median APE Translation (m)  & Mean APE Translation (m) & RMSE APE Translation (m) & Median
                         # APE Translation (m) & Mean APE Translation (m)  & RMSE APE translation (m) \\\\
    # """
    start_line = """\\begin{table}[H]
  \\centering
  \\caption{Accuracy of the state estimation when using Structureless and Projection factors (S + P), and our proposed approach using Structureless, Projection and Regularity factors (S + P + R)}
  \\label{tab:accuracy_comparison}
  \\begin{tabularx}{\\textwidth}{l *6{Y}}
    \\toprule
    & \\multicolumn{6}{c}{APE Translation} \\\\
    \\cmidrule{2-7}
    & \\multicolumn{3}{c}{\\textbf{S + P}}  & \\multicolumn{3}{c}{\\textbf{S + P + R} (Proposed)} \\\\
    \\cmidrule(r){2-4} \\cmidrule(l){5-7}
    Sequence & Median [cm] & Mean [cm] & RMSE [cm] & Median [cm] & Mean [cm] & RMSE [cm] \\\\
    \\midrule
    """

    end_line = """
    \\bottomrule
  \\end{tabularx}%
\\end{table}
"""
    bold_in = '& \\textbf{{'
    bold_out = '}} '
    end = '\\\\\n'

    all_lines = start_line

    winners = dict()
    for dataset_name, pipeline_types in sorted(stats.items()):
        median_error_pos = []
        # mean_error_pos = []
        rmse_error_pos = []
        for _, pipeline_stats in sorted(pipeline_types.items()):
            # if pipeline_type is not "S": # Ignore S pipeline
            median_error_pos.append(pipeline_stats["absolute_errors"]["median"])
            # mean_error_pos.append(pipeline_stats["absolute_errors"]["mean"])
            rmse_error_pos.append(pipeline_stats["absolute_errors"]["rmse"])

        # Find winning pipeline
        _, median_idx_min = locate_min(median_error_pos)
        # _, mean_idx_min = locate_min(mean_error_pos)
        _, rmse_idx_min = locate_min(rmse_error_pos)

        # Store winning pipeline
        winners[dataset_name] = [median_idx_min,
                                 # mean_idx_min,
                                 rmse_idx_min]

    for dataset_name, pipeline_types in sorted(stats.items()):
        start = '{:>25} '.format(dataset_name.replace('_', '\\_'))
        one_line = start
        pipeline_idx = 0
        for _, pipeline_stats in sorted(pipeline_types.items()):
            # if pipeline_type is not "S": # Ignore S pipeline
            median_error_pos = pipeline_stats["absolute_errors"]["median"] * 100 # as we report in cm
            # mean_error_pos = pipeline_stats["absolute_errors"]["mean"] * 100 # as we report in cm
            rmse_error_pos = pipeline_stats["absolute_errors"]["rmse"] * 100 # as we report in cm

            # Bold for min median error
            if len(winners[dataset_name][0]) == 1 and pipeline_idx == winners[dataset_name][0][0]:
                one_line += bold_in + '{:.1f}'.format(median_error_pos) + bold_out
            else:
                one_line += '& {:.1f} '.format(median_error_pos)

            # Bold for min mean error
            # if len(winners[dataset_name][1]) == 1 and winners[dataset_name][1][0] == pipeline_idx:
                # one_line += bold_in + '{:.1f}'.format(mean_error_pos) + bold_out
            # else:
                # one_line += '& {:.1f} '.format(mean_error_pos)

            # Bold for min rmse error
            # Do not bold, if multiple max
            if len(winners[dataset_name][1]) == 1 and winners[dataset_name][1][0] == pipeline_idx:
                one_line += bold_in + '{:.1f}'.format(rmse_error_pos) + bold_out
            else:
                one_line += '& {:.1f} '.format(rmse_error_pos)

            pipeline_idx += 1

        one_line += end
        all_lines += one_line
    all_lines += end_line

    # Save table
    results_file = os.path.join(results_dir, 'APE_table.tex')
    print("Saving table of APE results to: " + results_file)
    with open(results_file,'w') as outfile:
        outfile.write(all_lines)

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
        traj_ref = file_interface.read_euroc_csv_trajectory(traj_ref_path)
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
            plot_collection.export(save_folder + "/plots", False)

    ## Plot results
    #if args.plot or args.save_plot or args.serialize_plot:
        #    common.plot(
        #        args, result,
        #        result.trajectories[ref_name],
        #        result.trajectories[est_name])

    ## Save results
    #if args.save_results:
        #    logger.debug(SEP)
        #    if not SETTINGS.save_traj_in_zip:
            #        del result.trajectories[ref_name]
            #        del result.trajectories[est_name]
            #    file_interface.save_res_file(
            #        args.save_results, result, confirm_overwrite=not args.no_warnings)

# Run pipeline as a subprocess.
def run_vio(build_dir, dataset_dir, dataset_name, results_dir, pipeline_output_dir, pipeline_type,
           extra_flagfile_path = ""):
    """ Runs pipeline depending on the pipeline_type"""
    import subprocess
    return subprocess.call("{}/stereoVIOEuroc \
                           --logtostderr=1 --colorlogtostderr=1 --log_prefix=0 \
                           --dataset_path={}/{} --output_path={}\
                           --vio_params_path={}/params/{}/{} \
                           --tracker_params_path={}/params/{}/{} \
                           --flagfile={}/params/{}/{} --flagfile={}/params/{}/{} \
                           --flagfile={}/params/{}/{} --flagfile={}/params/{}/{} \
                           --flagfile={}/params/{}/{} --flagfile={}/params/{}/{} \
                           --log_output=True".format(
                               build_dir, dataset_dir, dataset_name, pipeline_output_dir,
                               results_dir, pipeline_type, "regularVioParameters.yaml",
                               results_dir, pipeline_type, "trackerParameters.yaml",
                               results_dir, pipeline_type, "flags/stereoVIOEuroc.flags",
                               results_dir, pipeline_type, "flags/Mesher.flags",
                               results_dir, pipeline_type, "flags/VioBackEnd.flags",
                               results_dir, pipeline_type, "flags/RegularVioBackEnd.flags",
                               results_dir, pipeline_type, "flags/Visualizer3D.flags",
                               results_dir, pipeline_type, extra_flagfile_path), \
                           shell=True)

def process_vio(build_dir, dataset_dir, dataset_name, results_dir, pipeline_output_dir,
                pipeline_type, SEGMENTS, save_results, plot, save_plots, output_file, run_pipeline, analyse_vio, discard_n_start_poses, discard_n_end_poses):
    """ build_dir: directory where the pipeline executable resides,
    dataset_dir: directory of the dataset,
    dataset_name: specific dataset to run,
    results_dir: directory where the results of the run will reside:
        used as results_dir/dataset_name/S, results_dir/dataset_name/SP, results_dir/dataset_name/SPR
        where each directory have traj_est.csv (the estimated trajectory), and plots if requested.
        results_dir/dataset_name/ must contain traj_gt.csv (the ground truth trajectory for analysis to work),
    pipeline_output_dir: where to store all output_* files produced by the pipeline,
    pipeline_type: type of pipeline to process (1: S, 2: SP, 3: SPR)
    SEGMENTS: segments for RPE boxplots,
    save_results: saves APE, and RPE per segment results of the run,
    plot: whether to plot the APE/RPE results or not,
    save_plots: saves plots of APE/RPE,
    output_file: the name of the trajectory estimate output of the vio which will then be copied as traj_est.csv,
    run_pipeline: whether to run the VIO to generate a new traj_est.csv,
    analyse_vio: whether to analyse traj_est.csv or not"""
    dataset_result_dir = results_dir + "/" + dataset_name + "/"
    dataset_pipeline_result_dir = dataset_result_dir + "/" + pipeline_type + "/"
    traj_ref_path = dataset_result_dir + "/traj_gt.csv"
    traj_es = dataset_result_dir + "/" + pipeline_type + "/" + "traj_es.csv"
    create_full_path_if_not_exists(traj_es)
    if run_pipeline:
        if run_vio(build_dir, dataset_dir, dataset_name, results_dir,
                   pipeline_output_dir, pipeline_type) == 0:
            print("Successful pipeline run.")
            print("\033[1mCopying output file: " + output_file + "\n to results file:\n" + \
                  traj_es + "\033[0m")
            copyfile(output_file, traj_es)
            try:
                output_destination_dir = dataset_pipeline_result_dir + "/output/"
                print("\033[1mCopying output dir: " + pipeline_output_dir
                      + "\n to destination:\n" + output_destination_dir + "\033[0m")
                move_output_from_to(pipeline_output_dir, output_destination_dir)
            except:
                print("\033[1mFailed copying output dir: " + pipeline_output_dir
                      + "\n to destination:\n" + output_destination_dir + "\033[0m")
        else:
            print("Pipeline failed on dataset: " + dataset_name)
            return False

    if analyse_vio:
        print("\033[1mAnalysing dataset: " + dataset_result_dir + " for pipeline "
              + pipeline_type + ".\033[0m")
        run_analysis(traj_ref_path, traj_es, SEGMENTS,
                     save_results, plot, save_plots, dataset_pipeline_result_dir, False,
                     dataset_name,
                     discard_n_start_poses,
                     discard_n_end_poses)
    return True

def run_dataset(results_dir, dataset_dir, dataset_properties, build_dir,
                run_pipeline, analyse_vio,
                plot, save_results, save_plots, save_boxplots, pipelines_to_run_list,
                discard_n_start_poses = 0, discard_n_end_poses = 0):
    """ Evaluates pipeline using Structureless(S), Structureless(S) + Projection(P), \
            and Structureless(S) + Projection(P) + Regular(R) factors \
            and then compiles a list of results """
    dataset_name = dataset_properties['name']
    dataset_segments = dataset_properties['segments']

    ################### RUN PIPELINE ################################
    pipeline_output_dir = results_dir + "/tmp_output/output/"
    create_full_path_if_not_exists(pipeline_output_dir)
    output_file = pipeline_output_dir + "/output_posesVIO.csv"
    has_a_pipeline_failed = False
    if len(pipelines_to_run_list) == 0:
        print("Not running pipeline...")
    for pipeline_type in pipelines_to_run_list:
        if process_vio(build_dir, dataset_dir, dataset_name, results_dir, pipeline_output_dir,
                       pipeline_type, dataset_segments, save_results, plot, save_plots,
                       output_file, run_pipeline, analyse_vio,
                       discard_n_start_poses, discard_n_end_poses) == False:
            has_a_pipeline_failed = True

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
                checkStats(stats[pipeline_type])

            print("Drawing boxplots.")
            evt.draw_rpe_boxplots(results_dir + "/" + dataset_name, stats, len(dataset_segments))
        else:
            print("A pipeline run has failed... skipping boxplot drawing.")

    if not has_a_pipeline_failed:
        print ("All pipeline runs were successful.")
    else:
        print("A pipeline has failed!")
    print("Finished evaluation for dataset: " + dataset_name)
    return has_a_pipeline_failed

def write_flags_parameters(param_name, param_new_value, params_path):
    directory = os.path.dirname(params_path)
    if not os.path.exists(directory):
        raise Exception("\033[91mCould not find directory: " + directory + "\033[99m")
    params_flagfile = open(params_path, "a+")
    params_flagfile.write("--" + param_name + "=" + param_new_value)
    params_flagfile.close()

def check_and_create_regression_test_structure(regression_tests_path, param_names, param_values,
                                               dataset_names, pipeline_types, extra_params_to_modify):
    """ Makes/Checks that the file structure is the correct one, and updates the parameters with the given values"""
    # Make or check regression_test directory
    assert(ensure_dir(regression_tests_path))
    # Make or check param_name directory
    # Use as param_name the concatenated elements of param_names
    param_names_dir = ""
    for i in param_names:
        param_names_dir += str(i) + "-"
    param_names_dir = param_names_dir[:-1]
    assert(ensure_dir("{}/{}".format(regression_tests_path, param_names_dir)))
    for param_value in param_values:
        # Create/Check param_value folder
        param_value_dir = ""
        if isinstance(param_value, list):
            for i in param_value:
                param_value_dir += str(i) + "-"
            param_value_dir = param_value_dir[:-1]
        else:
            param_value_dir = param_value
        ensure_dir("{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir))
        # Create params folder by copying from current official one.
        param_dir = "{}/{}/{}/params".format(regression_tests_path, param_names_dir, param_value_dir)
        if (os.path.exists(param_dir)):
            rmtree(param_dir)
        copytree("/home/tonirv/code/evo/results/params", param_dir)

        # Modify param with param value
        for pipeline_type in pipeline_types:
            param_pipeline_dir = "{}/{}".format(param_dir, pipeline_type)
            ensure_dir(param_pipeline_dir)
            written_extra_param_names = []
            is_param_name_written = [False] * len(param_names)
            # VIO params
            vio_file = param_pipeline_dir + "/vioParameters.yaml"
            vio_params = []
            with open(vio_file, 'r') as infile:
                # Skip first yaml line: it contains %YAML:... which can't be read...
                _ = infile.readline()
                vio_params = yaml.load(infile)
                for idx, param_name in enumerate(param_names):
                    if param_name in vio_params:
                        # Modify param_name with param_value
                        if isinstance(param_value, list):
                            vio_params[param_name] = param_value[idx]
                        else:
                            vio_params[param_name] = param_value
                        is_param_name_written[idx] = True
                for extra_param_name, extra_param_value in extra_params_to_modify.items():
                    if extra_param_name in vio_params:
                        vio_params[extra_param_name] = extra_param_value
                        written_extra_param_names.append(extra_param_name)
                # Store param_names with param_value
                with open(vio_file,'w') as outfile:
                    outfile.write("%YAML:1.0\n")
                with open(vio_file,'a') as outfile:
                    outfile.write(yaml.dump(vio_params))

            # Tracker params
            tracker_file = param_pipeline_dir + "/trackerParameters.yaml"
            tracker_params = []
            with open(tracker_file, 'r') as infile:
                # Skip first yaml line: it contains %YAML:... which can't be read...
                _ = infile.readline()
                tracker_params = yaml.load(infile)
                for idx, param_name in enumerate(param_names):
                    if param_name in tracker_params:
                        # Modify param_name with param_value
                        if isinstance(param_value, list):
                            tracker_params[param_name] = param_value[idx]
                        else:
                            tracker_params[param_name] = param_value
                        is_param_name_written[idx] = True
                for extra_param_name, extra_param_value in extra_params_to_modify.items():
                    if extra_param_name in tracker_params:
                        tracker_params[extra_param_name] = extra_param_value
                        written_extra_param_names.append(extra_param_name)
                with open(tracker_file,'w') as outfile:
                    outfile.write("%YAML:1.0\n")
                with open(tracker_file,'a') as outfile:
                    outfile.write(yaml.dump(tracker_params, default_flow_style=False))

            # Gflags
            for idx, param_name in enumerate(param_names):
                if not is_param_name_written[idx]:
                    # Could not find param_name in vio_params nor tracker_params
                    # it must be a gflag:
                    if isinstance(param_value, list):
                        write_flags_parameters(param_name, param_value[idx],
                                               param_pipeline_dir + "/flags/override.flags")
                    else:
                        write_flags_parameters(param_name, param_value,
                                               param_pipeline_dir + "/flags/override.flags")
            for extra_param_name, extra_param_value in extra_params_to_modify.items():
                if extra_param_name not in written_extra_param_names:
                    write_flags_parameters(extra_param_name,
                                           extra_param_value,
                                           param_pipeline_dir + "/flags/override.flags")

        # Create/Check tmp_output folder
        ensure_dir("{}/{}/{}/tmp_output/output".format(regression_tests_path, param_names_dir, param_value_dir))

        for dataset_name in dataset_names:
            ensure_dir("{}/{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir, dataset_name))
            # Create ground truth trajectory by copying from current official one.
            copy2("/home/tonirv/code/evo/results/{}/traj_gt.csv".format(dataset_name),
                 "{}/{}/{}/{}/traj_gt.csv".format(regression_tests_path, param_names_dir,
                                                  param_value_dir, dataset_name))
            # Create segments by copying from current official one.
            copy2("/home/tonirv/code/evo/results/{}/segments.txt".format(dataset_name),
                 "{}/{}/{}/{}/segments.txt".format(regression_tests_path, param_names_dir,
                                                  param_value_dir, dataset_name))
            for pipeline_type in pipeline_types:
                ensure_dir("{}/{}/{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir,
                                                   dataset_name, pipeline_type))

    # Make/Check results dir for current param_names
    ensure_dir("{}/{}/results".format(regression_tests_path, param_names_dir))
    for dataset_name in dataset_names:
        # Make/Check dataset dir for current param_names_dir, as the performance given the param depends on the dataset.
        ensure_dir("{}/{}/results/{}".format(regression_tests_path, param_names_dir, dataset_name))

def build_list_of_pipelines_to_run(pipelines_to_run):
    pipelines_to_run_list = []
    if pipelines_to_run == 0:
        pipelines_to_run_list = ['S', 'SP', 'SPR']
    if pipelines_to_run == 1:
        pipelines_to_run_list = ['S']
    if pipelines_to_run == 2:
        pipelines_to_run_list = ['SP']
    if pipelines_to_run == 3:
        pipelines_to_run_list = ['SPR']
    if pipelines_to_run == 4:
        pipelines_to_run_list = ['S', 'SP']
    if pipelines_to_run == 5:
        pipelines_to_run_list = ['S', 'SPR']
    if pipelines_to_run == 6:
        pipelines_to_run_list = ['SP', 'SPR']
    return pipelines_to_run_list

def regression_test_simple(test_name, param_names, param_values, only_compile_regression_test_results,
                           run_pipelines, pipelines_to_run, extra_params_to_modify):
    """ Runs the vio pipeline with different values for the given param
    and draws graphs to decide best value for the param:
        - param_names: names of the parameters to fine-tune: e.g ["monoNoiseSigma", "stereoNoiseSigma"]
        - param_values: values that the parameter should take: e.g [[1.0, 1.3], [1.0, 1.2]]
        - only_compile_regression_test_results: just draw boxplots for regression test,
            skip all per pipeline analysis and runs, assumes we have results.yaml for
            each param value, dataset and pipeline.
        - run_pipelines: run pipelines, if set to false, it won't run pipelines and will assume we have a traj_est.csv already.
        - pipelines_to_run: which pipeline to run, useful when a parameter only affects a single pipeline."""
    # Ensure input is correct.
    if isinstance(param_names, list):
        if len(param_names) > 1:
            assert(len(param_names) == len(param_values[0]))
            for i in range(2, len(param_names)):
                # Ensure all rows have the same number of parameter changes
                assert(len(param_values[i-2]) == len(param_values[i-1]))

    # Check and create file structure
    dataset_names = ["V1_01_easy"]
    pipelines_to_run_list = build_list_of_pipelines_to_run(pipelines_to_run)
    REGRESSION_TESTS_DIR = "/home/tonirv/code/evo-1/regression_tests/" + test_name
    check_and_create_regression_test_structure(REGRESSION_TESTS_DIR, param_names, param_values,
                                               dataset_names, pipelines_to_run_list, extra_params_to_modify)

    param_names_dir = ""
    for i in param_names:
        param_names_dir += str(i) + "-"
    param_names_dir = param_names_dir[:-1]
    DATASET_DIR = '/home/tonirv/datasets/EuRoC'
    BUILD_DIR = '/home/tonirv/code/spark_vio/build'
    if not only_compile_regression_test_results:
        for param_value in param_values:
            param_value_dir = ""
            if isinstance(param_value, list):
                for i in param_value:
                    param_value_dir += str(i) + "-"
                param_value_dir = param_value_dir[:-1]
            else:
                param_value_dir = param_value
            results_dir = "{}/{}/{}".format(REGRESSION_TESTS_DIR, param_names_dir, param_value_dir)
            for dataset_name in dataset_names:
                run_dataset(results_dir, DATASET_DIR, dataset_name, BUILD_DIR,
                               run_pipelines, # Should we re-run pipelines?
                               True, # Should we run the analysis of per pipeline errors?
                               False, # Should we display plots?
                               True, # Should we save results?
                               True, # Should we save plots?
                               False, # Should we save boxplots?
                               pipelines_to_run_list) # Should we run 0: all pipelines, 1: S, 2:SP 3:SPR

                print("Finished analysis of pipelines for param_value: {} for parameter: {}".format(param_value_dir, param_names_dir))
                print("Finished pipeline runs/analysis for regression test of param_name: {}".format(param_names_dir))

    # Compile results for current param_name
    print("Drawing boxplot APE for regression test of param_name: {}".format(param_names_dir))
    for dataset_name in dataset_names:
        stats = dict()
        for param_value in param_values:
            param_value_dir = ""
            if isinstance(param_value, list):
                for i in param_value:
                    param_value_dir += str(i) + "-"
                param_value_dir = param_value_dir[:-1]
            else:
                param_value_dir = param_value
            stats[param_value_dir] = dict()
            for pipeline in pipelines_to_run_list:
                results_file = "{}/{}/{}/{}/{}/results.yaml".format(REGRESSION_TESTS_DIR, param_names_dir,
                                                                    param_value_dir, dataset_name, pipeline)
                if os.path.isfile(results_file):
                    stats[param_value_dir][pipeline] = yaml.load(open(results_file,'r'))
                else:
                    print("Could not find results file: {}".format(results_file) + ". Adding cross to boxplot...")
                    stats[param_value_dir][pipeline] = False

        print("Drawing regression simple APE boxplots for dataset: " + dataset_name)
        plot_dir = "{}/{}/results/{}".format(REGRESSION_TESTS_DIR, param_names_dir, dataset_name)
        max_y = -1
        if dataset_name == "V2_02_medium":
            max_y = 0.40
        if dataset_name == "V1_01_easy":
            max_y = 0.20
        evt.draw_regression_simple_boxplot_APE(param_names, stats, plot_dir, max_y)
    print("Finished regression test for param_name: {}".format(param_names_dir))

def run(args):
    from evo.tools import log
    from evo.tools.settings import SETTINGS

    # Get experiment information from yaml file.
    experiment_params = yaml.load(args.experiments_path)

    results_dir = experiment_params['results_dir']
    dataset_dir = experiment_params['dataset_dir']
    build_dir = experiment_params['build_dir']
    datasets_to_run = experiment_params['datasets_to_run']

    # Run experiments.
    print("Run experiments")
    for dataset in datasets_to_run:
        print("Run dataset:", dataset['name'])
        pipelines_to_run = dataset['pipelines']
        run_dataset(results_dir, dataset_dir, dataset, build_dir,
                    args.run_pipeline, args.analyse_vio,
                    args.plot, args.save_results,
                    args.save_plots, args.save_boxplots,
                    pipelines_to_run,
                    dataset['discard_n_start_poses'],
                    dataset['discard_n_end_poses'])

def parser():
    import argparse
    basic_desc = "Full evaluation of SPARK VIO pipeline (APE trans + RPE trans + RPE rot) metric app"

    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))
    input_opts = shared_parser.add_argument_group("input options")
    algo_opts = shared_parser.add_argument_group("algorithm options")
    output_opts = shared_parser.add_argument_group("output options")
    usability_opts = shared_parser.add_argument_group("usability options")

    input_opts.add_argument("experiments_path", type=argparse.FileType('r'),
                           help="Path to the yaml file with experiments settings.",
                            default = "./experiments.yaml")

    algo_opts.add_argument("-r", "--run_pipeline", action="store_true",
                           help="Run vio?")
    algo_opts.add_argument("-a", "--analyse_vio", action="store_true",
                           help="Analyse vio, compute APE and RPE")

    output_opts.add_argument("--plot", action="store_true", help="show plot window",)
    output_opts.add_argument("--plot_colormap_max", type=float,
                             help="The upper bound used for the color map plot "
                             "(default: maximum error value)")
    output_opts.add_argument("--plot_colormap_min", type=float,
                             help="The lower bound used for the color map plot "
                             "(default: minimum error value)")
    output_opts.add_argument("--plot_colormap_max_percentile", type=float,
                             help="Percentile of the error distribution to be used "
                             "as the upper bound of the color map plot "
                             "(in %%, overrides --plot_colormap_min)")
    output_opts.add_argument("--save_plots", action="store_true",
                             help="Save plots?")
    output_opts.add_argument("--save_boxplots", action="store_true",
                             help="Save boxplots?")
    output_opts.add_argument("--save_results", action="store_true",
                             help="Save results?")

    main_parser = argparse.ArgumentParser(
        description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True

    return shared_parser

import argcomplete
if __name__ == '__main__':
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    run(args)

