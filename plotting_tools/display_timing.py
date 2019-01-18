#!/usr/bin/env python
import os
import math

import matplotlib.pyplot as plt
import numpy as np

import evaluation.tools as evt
import plotting_tools as pt

def draw_timing_plot(filename,
                     keyframe_ids,
                     pipelines_times,
                     ylabel = 'Optimization time [s]',
                     display_plot=False,
                     display_x_label=True,
                     latexify=True,
                     fig_width=6,
                     fig_height=3):
    """ Plots timing information for each pipeline contained in the list of dicts
    pipelines_times:
    - filename: where to save the figure.
    - pipelines_times: list of dicts of the form:
    [{
        'pipeline_name': pipeline_name,
        'line_color': np.random.rand(3),
        'line_style': '-',
        'times': update_times
    }, ... ]
    - keyframe_ids: corresponds to the x ticks, so update_times and keyframe_ids must
    be the same length.
    - ylabel: Y label used for the plot.
    - display_plot: whether to display the plot or not.
    - display_x_label: whether to display the x label of the plot or not.
    - latexify: whether to use latex for the generation of the plot.
    """
    if latexify:
        pt.latexify(fig_width, fig_height)
    fig = plt.figure(figsize=[fig_width, fig_height], dpi=1000)
    i = 0
    for pipeline_time in pipelines_times:
        assert len(keyframe_ids) == len(pipeline_time['times'])
        plt.plot(
            keyframe_ids,
            pipeline_time['times'],
            linestyle=pipeline_time['line_style'],
            color=pipeline_time['line_color'],
            linewidth=0.5,
            label="$t_{" + pipeline_time['pipeline_name'] + "}^{opt}$")
        i = i + 1
    plt.ylabel(ylabel)
    if display_x_label:
        plt.xlabel('Keyframe Index [-]')
    plt.xlim(min(keyframe_ids), max(keyframe_ids))
    plt.ylim(bottom=0)
    plt.grid(axis='both', linestyle='--')
    plt.legend()
    # Create path to filename if it does not exist.
    evt.create_full_path_if_not_exists(filename)
    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=1000)
    if display_plot:
        plt.show()

def get_pipeline_times(results_folder, pipeline_names):
    """ Returns a list of keyframe_ids, together with pipelines_times, a list of dict for information
    on the pipeline timing and how to plot it. See draw_timing_plot for the 
    actual structure of pipelines_times.
        - 'results_folder': should point to the following filesystem structure:
        results_folder/pipeline_names[1]/output/output_timingVIO.txt
        results_folder/pipeline_names[2]/output/output_timingVIO.txt etc
        results_folder/pipeline_names[etc]/output/output_timingVIO.txt etc
        - pipeline_names: are the names of the pipelines for which results are available.
        This function will warn if there are no results available for the given pipeline.
    """
    # Set random seed for returning always the same colors
    np.random.seed(0)
    keyframe_ids = []
    pipeline_times=[]
    # Only add times for those pipelines which have results.
    prev_pipeline_name=''
    prev_keyframe_ids=[] # Used to check that all keyframe_ids are the same.
    for pipeline_name in pipeline_names:
        if os.path.isdir(results_folder):
            timing_results_dir = os.path.join(results_folder, pipeline_name, "output/output_timingVIO.txt")
            if os.path.exists(timing_results_dir):
                print 'Parsing results for pipeline: %s in file: %s' \
                % (pipeline_name, timing_results_dir)
                filename = open(timing_results_dir, 'r')
                keyframe_ids, update_times = \
                np.loadtxt(filename, delimiter=' ', usecols=(0,3), unpack=True) # 4th column are results.
                pipeline_times.append({
                    'pipeline_name': pipeline_name,
                    'line_color': np.random.rand(3),
                    'line_style': '-',
                    'times': update_times
                })
                if len(prev_keyframe_ids) is not 0:
                    assert len(keyframe_ids) == len(prev_keyframe_ids), \
                     'You are comparing timing information with'\
                    ' different keyframe ids for pipeline %s and %s,'\
                    ' this is not normal!' % (pipeline_name, prev_pipeline_name)
                prev_pipeline_name = pipeline_name
                prev_keyframe_ids = keyframe_ids
            else:
                print 'WARNING: pipeline with name: %s has missing results... ' \
                 'No file found at: %s' % (pipeline_name, timing_results_dir)
        else:
            raise Exception ('ERROR: invalid results folder: %s' % results_folder)
    return keyframe_ids, pipeline_times

def parser():
    import argparse
    basic_desc = "Plot timing results for VIO pipeline."
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    path_to_vio_output = main_parser.add_argument_group("input options")
    path_to_vio_output.add_argument(
        "path_to_vio_output",
        help="Path to the directory containing the VIO timing output files."
        "The file structure below this path should be {S, SP, SPR, *}/output/output_timingVIO.txt.",
        default="/home/tonirv/code/evo/results/V1_01_easy/")
    return main_parser

def main(results_folder, pipeline_names):
    """
        Displays timing of VIO stored in 'results_folder' parameter as a path.
        In particular 'results_folder' should point to the following filesystem structure:
        results_folder/pipeline_names[1]/output/output_timingVIO.txt
        results_folder/pipeline_names[2]/output/output_timingVIO.txt etc
        results_folder/pipeline_names[etc]/output/output_timingVIO.txt etc
        Where pipeline_names are the names of the pipelines for which results are available.
        This function will warn if there are not results available for the given pipeline.
    """
    keyframe_ids, pipelines_times = get_pipeline_times(results_folder, pipeline_names)
    assert len(keyframe_ids) > 0, 'There does not seem to be keyframe_ids, these are the x axis of the plot'
    assert len(pipelines_times) > 0, 'Missing pipeline timing information.'
    draw_timing_plot(os.path.join(results_folder, "timing/all_timing_for_paper.pdf"),
                     keyframe_ids, pipelines_times)

if __name__ == "__main__":
    import argcomplete
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # HARDCODED pipeline_names for convenience.
    PIPELINE_NAMES=['S', 'SP', 'SPR']
    main(args.path_to_vio_output, PIPELINE_NAMES)