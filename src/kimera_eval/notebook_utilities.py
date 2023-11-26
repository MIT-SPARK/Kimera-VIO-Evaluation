"""Helper functions for debugging notebooks."""
import pandas as pd
import pathlib


def _get_df_summary(df):
    def _get_mean(attrib):
        ls = df[attrib].tolist()
        return float(sum(ls)) / len(ls)

    def _get_min(attrib):
        return min(df[attrib])

    def _get_max(attrib):
        return max(df[attrib])

    return [
        ("Average number of detected features", _get_mean("nrDetectedFeatures")),
        ("Minimum number of detected features", _get_min("nrDetectedFeatures")),
        ("Average number of tracked features", _get_mean("nrTrackerFeatures")),
        ("Minimum number of tracked features", _get_min("nrTrackerFeatures")),
        ("Average number of mono ransac inliers", _get_mean("nrMonoInliers")),
        ("Minimum number of mono ransac inliers", _get_min("nrMonoInliers")),
        ("Average number of stereo ransac inliers", _get_mean("nrStereoInliers")),
        ("Minimum number of stereo ransac inliers", _get_min("nrStereoInliers")),
        ("Average number of mono ransac putatives", _get_mean("nrMonoPutatives")),
        ("Minimum number of mono ransac putatives", _get_min("nrMonoPutatives")),
        ("Average number of stereo ransac putatives", _get_mean("nrStereoPutatives")),
        ("Minimum number of stereo ransac putatives", _get_min("nrStereoPutatives")),
    ]


def _print_summary(summary_stats):
    attrib_len = [len(attrib[0]) for attrib in summary_stats]
    max_attrib_len = max(attrib_len)

    print("\nStatistic summary:\n")
    for entry in summary_stats:
        attrib = entry[0]
        value = entry[1]
        spacing = max_attrib_len - len(attrib)
        print(attrib + " " * spacing + ": " + str(value))


def load_frontend_statistics(vio_output_path, print_summary=True):
    """Load frontend statistics."""
    vio_output_path = pathlib.Path(vio_output_path).expanduser().absolute()
    stats_path = vio_output_path / "output_frontend_stats.csv"

    # Convert to tidy pandas DataFrame object.
    df = pd.read_csv(stats_path, sep=",", index_col=False)
    if print_summary:
        df.head()

        summary = _get_df_summary(df)
        _print_summary(summary)

    return df
