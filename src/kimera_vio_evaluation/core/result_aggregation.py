"""Result aggregation functions."""
import os
import yaml
import glog as log
from kimera_vio_evaluation.tools.utils import check_stats
from kimera_vio_evaluation.tools.latex_utils import write_latex_table
from kimera_vio_evaluation.tools.plotly_plotter import draw_ape_boxplots


def aggregate_all_results(results_dir, use_pgo=False):
    r"""
    Aggregate APE results and draw APE boxplot as well as write latex table.

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
    yaml_filename = "results_vio.yaml"
    if use_pgo:
        yaml_filename = "results_pgo.yaml"
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
                stats[dataset_name][pipeline_name] = yaml.load(
                    open(results_filepath, "r"), Loader=yaml.Loader
                )
            except yaml.YAMLError as e:
                raise Exception("Error in results file: ", e)
            except Exception:
                log.fatal("\033[1mFailed opening file: \033[0m\n %s" % results_filepath)

            log.debug("Check stats from: " + results_filepath)
            try:
                check_stats(stats[dataset_name][pipeline_name])
            except Exception as e:
                log.warning(e)

    return stats


def aggregate_ape_results(results_dir):
    r"""
    Aggregate APE results and draw APE boxplot as well as write latex table.

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
    if len(list(stats.values())) > 0:
        log.info("Drawing APE boxplots.")
        draw_ape_boxplots(stats, results_dir)
        # Write APE table
        log.info("Writing APE latex table.")
        write_latex_table(stats, results_dir)
    return stats
