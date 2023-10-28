"""Utils for making latex tables."""
import os
from evo.core import result
import glog as log


def locate_min(a):
    """Locate min in column."""
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) if smallest == element]


def write_latex_table_header(cols_names_list, sub_cols_names_list):
    """
    Write latex table header.

    If you don't want sub_cols in the table just set it to 1.

    Args:
        cols_names_list: List of names of the columns,
            typically pipeline names (S, SP, SPR, ...).
        sub_cols_names_list: List of names of the sub-columns,
            typically metrics names (Median, RMSE, Drift, ...).
    """
    assert type(cols_names_list) == list
    assert type(sub_cols_names_list) == list
    cols = len(cols_names_list)
    sub_cols = len(sub_cols_names_list)

    start_line = """\\begin{table*}[h]
  \\centering
  \\caption{Accuracy of the State Estimation}
  \\label{tab:accuracy_comparison}
  \\begin{tabularx}{\\textwidth}{l *%s{Y}}
    \\toprule
    & \\multicolumn{%s}{c}{APE Translation} \\\\
    \\cmidrule{2-%s}
    """ % (
        cols * sub_cols,
        cols * sub_cols,
        cols * sub_cols + 1,
    )

    cols_header_line = ""
    if sub_cols <= 1:
        cols_header_line = """Sequence """

    mid_rule_line = ""
    col_counter = 0
    for col_name in cols_names_list:
        if sub_cols > 1:
            cols_header_line = (
                cols_header_line
                + """& \\multicolumn{%s}{c}{\\textbf{%s}} """ % (sub_cols, col_name)
            )
            mid_rule_line = mid_rule_line + """\\cmidrule(r){%s-%s}""" % (
                col_counter,
                col_counter + sub_cols,
            )
        else:
            cols_header_line = cols_header_line + """& \\textbf{%s} """ % (col_name)
        col_counter = col_counter + sub_cols

    break_row = """ \\\\"""
    sub_cols_header_line = ""
    if sub_cols > 1:
        sub_cols_header_line = """Sequence """
        for col_name in cols_names_list:
            for sub_col_name in sub_cols_names_list:
                sub_cols_header_line = sub_cols_header_line + """& %s """ % (
                    sub_col_name
                )

    start_line = (
        start_line
        + cols_header_line
        + break_row
        + "\n"
        + sub_cols_header_line
        + break_row
        + "\n \\midrule \n"
    )
    return start_line


def write_latex_table(stats, results_dir):
    """
    Write latex table with median, mean and rmse from stats.

    This is a list that contains:
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
                - "trajectory_length_m"
    """
    # Assumes an equal number of cols/keys per row
    cols_names_list = list(sorted(stats[list(stats.keys())[0]].keys()))
    sub_cols_names_list = ["Median [cm]", "RMSE [cm]", "Drift [\\%]"]
    start_line = write_latex_table_header(cols_names_list, sub_cols_names_list)
    end_line = """
    \\bottomrule
  \\end{tabularx}%
\\end{table*}{}
"""
    bold_in = "& \\textbf{"
    bold_out = "} "
    end = "\\\\\n"

    all_lines = start_line

    winners = dict()
    for dataset_name, pipeline_types in sorted(stats.items()):
        median_error_pos = []
        # mean_error_pos = []
        rmse_error_pos = []
        drift = []
        i = 0
        for pipeline_type, pipeline_stats in sorted(pipeline_types.items()):
            assert (
                cols_names_list[i] == pipeline_type
            )  # Ensure col names and results are consistent!
            i += 1
            assert isinstance(pipeline_stats["absolute_errors"], result.Result)
            # if pipeline_type is not "S": # Ignore S pipeline
            median_error_pos.append(pipeline_stats["absolute_errors"].stats["median"])
            # mean_error_pos.append(pipeline_stats["absolute_errors"]["mean"])
            rmse = pipeline_stats["absolute_errors"].stats["rmse"]
            rmse_error_pos.append(rmse)
            assert pipeline_stats["trajectory_length_m"] > 0
            # THIS IS NOT ACTUALLY DRIFT: bcs the trajectory_length_m is the length of
            # the estimated traj, not the ground-truth one...
            drift.append(rmse / pipeline_stats["trajectory_length_m"])
            log.error("DRIFT IS: %f" % (rmse / pipeline_stats["trajectory_length_m"]))

        # Find winning pipeline
        _, median_idx_min = locate_min(median_error_pos)
        # _, mean_idx_min = locate_min(mean_error_pos)
        _, rmse_idx_min = locate_min(rmse_error_pos)
        _, drift_idx_min = locate_min(drift)

        # Store winning pipeline
        winners[dataset_name] = [
            median_idx_min,
            # mean_idx_min,
            rmse_idx_min,
            drift_idx_min,
        ]

    for dataset_name, pipeline_types in sorted(stats.items()):
        start = "{:>25} ".format(dataset_name.replace("_", "\\_"))
        one_line = start
        pipeline_idx = 0
        for pipeline_type, pipeline_stats in sorted(pipeline_types.items()):
            assert isinstance(pipeline_stats["absolute_errors"], result.Result)
            log.info("Pipeline type: %s" % pipeline_type)
            # if pipeline_type is not "S": # Ignore S pipeline
            median_error_pos = (
                pipeline_stats["absolute_errors"].stats["median"] * 100
            )  # as we report in cm
            rmse = pipeline_stats["absolute_errors"].stats["rmse"]
            rmse_error_pos = rmse * 100  # as we report in cm
            assert pipeline_stats["trajectory_length_m"] > 0
            drift = (
                rmse / pipeline_stats["trajectory_length_m"] * 100
            )  # as we report in %

            # Bold for min median error
            if (
                len(winners[dataset_name][0]) == 1
                and pipeline_idx == winners[dataset_name][0][0]
            ):
                one_line += bold_in + "{:.1f}".format(median_error_pos) + bold_out
            else:
                one_line += "& {:.1f} ".format(median_error_pos)

            # Bold for min rmse error
            # Do not bold, if multiple max
            if (
                len(winners[dataset_name][1]) == 1
                and winners[dataset_name][1][0] == pipeline_idx
            ):
                one_line += bold_in + "{:.1f}".format(rmse_error_pos) + bold_out
            else:
                one_line += "& {:.1f} ".format(rmse_error_pos)

            # Bold for min drift error
            # Do not bold, if multiple max
            if (
                len(winners[dataset_name][2]) == 1
                and winners[dataset_name][2][0] == pipeline_idx
            ):
                one_line += bold_in + "{:.1f}".format(drift) + bold_out
            else:
                one_line += "& {:.1f} ".format(drift)

            pipeline_idx += 1

        one_line += end
        all_lines += one_line
    all_lines += end_line

    # Save table
    results_file = os.path.join(results_dir, "APE_table.tex")
    print("Saving table of APE results to: " + results_file)
    with open(results_file, "w") as outfile:
        outfile.write(all_lines)
