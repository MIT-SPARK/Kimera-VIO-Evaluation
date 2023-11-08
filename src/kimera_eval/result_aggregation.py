"""Result aggregation functions."""
import yaml
import logging
import pathlib
import numpy as np


def check_stats(stats):
    """Check stat contents."""
    if "relative_errors" not in stats:
        logging.error(f"Stats are missing required metrics: {stats}")

    if len(stats["relative_errors"]) == 0:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "rpe_rot" not in list(stats["relative_errors"].values())[0]:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "rpe_trans" not in list(stats["relative_errors"].values())[0]:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "absolute_errors" not in stats:
        logging.error(f"Stats are missing required metrics: {stats}")
        return False

    return True


def aggregate_ape_results(results_dir, use_pgo=False):
    """
    Aggregate APE results and draw APE boxplot as well as write latex table.

    Args:
      - result_dir: path to directory containing yaml result files
      - use_pgo: whether to aggregate all results for VIO or for PGO trajectory.

    Returns:
        Dict[str, Dict[str, Any]]: results keyed by dataset then pipeline
    """
    logging.debug(f"Aggregating dataset results @ '{results_dir}'")

    yaml_filename = "results_vio.yaml"
    if use_pgo:
        yaml_filename = "results_pgo.yaml"

    stats = {}
    results_path = pathlib.Path(results_dir)
    filepaths = sorted(list(results_path.glob(f"**/{yaml_filename}")))
    for filepath in filepaths:
        pipeline_name = filepath.parent.stem
        dataset_name = filepath.parent.parent.stem
        if dataset_name not in stats:
            stats[dataset_name] = {}

        with filepath.open("r") as fin:
            stats[dataset_name][pipeline_name] = yaml.safe_load(fin.read())

        logging.debug(f"Checking stats from `{filepath}`")
        if not check_stats(stats[dataset_name][pipeline_name]):
            logging.warning(f"invalid stats for {dataset_name}:{pipeline_name}")

    return stats


def get_pipeline_times(results_path, pipeline_names, column):
    """
    Load pipeline times.

    Args:
        results_path: path to VIO pipeline results
        pipeline_names: are the names of the pipelines for which results are available
        column: Specifies the column in the CSV file containing timing info

    Returns: keyframe indices and list of timing info
    """
    times = []
    keyframes = None
    prev_name = None
    for color_idx, name in enumerate(sorted(pipeline_names)):
        timing_path = results_path / name / "output" / "output_timingVIO.txt"
        if not timing_path.exists():
            print(f"WARNING: skipping missing {name} at '{timing_path}'... ")
            continue

        print("Parsing results for pipeline: {name} in file: '{timing_path}'")
        with timing_path.open("r") as fin:
            ret = np.loadtxt(fin, delimiter=" ", usecols=(0, column), unpack=True)

        times.append(
            {
                "pipeline_name": name,
                "line_color": color_idx,
                "line_style": "-",
                "times": ret[1],
            }
        )

        if keyframes is None:
            keyframes = ret[0]
        elif len(keyframes) == len(ret[0]):
            raise Exception(f"keyframe ids do not match between {name} and {prev_name}")

        prev_name = name

    return keyframes, times
