"""Result aggregation functions."""
import numpy as np


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
