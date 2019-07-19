import os
from evaluation.tools.math_utils import locate_min

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