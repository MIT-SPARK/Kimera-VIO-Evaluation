"""Helper functions for debugging notebooks."""
import pandas as pd
import numpy as np
import logging
import pathlib
import copy
import evo

from evo.core import transformations
from evo.core import lie_algebra as lie


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


def rename_pim_df(df):
    """
    Rename a DataFrame built from PIM measurements to be converted to a trajectory.

    This is an 'inplace' argument and returns nothing.

    Args:
        df: A pandas.DataFrame object.
    """
    df.index.names = ["timestamp"]
    df.rename(columns={"tx": "x", "ty": "y", "tz": "z"}, inplace=True)


def draw_trajectory_poses(ax, traj, downsample=20, **kwargs):
    """Draw actual poses via coordinate axes for a trajectory."""
    if downsample <= 1:
        traj_to_draw = traj
    else:
        traj_to_draw = copy.deepcopy(traj)
        traj_to_draw.reduce_to_ids(np.arange(traj.num_poses)[::downsample])

    evo.tools.plot.draw_coordinate_axes(ax, traj_to_draw, **kwargs)


def draw_start_and_end(ax, traj, plot_mode):
    """Draw start and end points of the trajectory."""
    start_pose = traj.poses_se3[0]
    end_pose = traj.poses_se3[-1]
    if plot_mode == evo.tools.plot.PlotMode.xy:
        ax.plot(start_pose[0, 3], start_pose[1, 3], "bo")
        ax.plot(end_pose[0, 3], end_pose[1, 3], "rx")


def setup_logging(name):
    """Configure logging for notebook."""
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        log.addHandler(ch)


def downsize_lc_df(df):
    """
    Remove all entries from a pandas DataFrame object that have '0' for the timestamp.

    This includes all entries that do not have loop closures.

    Args:
        df: A pandas.DataFrame object of loop-closure detections, indexed by timestamp.

    Returns:
        A pandas.DataFrame object with only loop closure entries.
    """
    df = df[~df.index.duplicated()]
    ts = np.array(df.index.tolist())
    good_ts = ts[np.where(ts > 0)]
    res = df.reindex(index=good_ts)
    return res


def downsize_lc_result_df(df):
    """
    Remove all entries from a pandas DataFrame object that have '0' for the timestamp.

    Same as downsize_lc_df but checks the `isloop` field of the DataFrame for the LCD
    result DataFrame, instead of the timestamp.
    """
    df = df[~df.index.duplicated()]
    ts = np.array(df.index.tolist())
    isloop = np.array(df.isLoop.tolist())
    good_ts = ts[np.where(isloop == 1)]
    res = df.reindex(index=good_ts)
    return res


def closest_num(ls, query):
    """Get the closest index to the query index."""
    ls_len = len(ls)
    return ls[min(range(ls_len), key=lambda i: abs(ls[i] - query))]


def get_gt_rel_pose(gt_df, match_ts, query_ts, to_scale=True):
    """Return the relative pose from match to query for given timestamps.

    Args:
        gt_df: A pandas.DataFrame object with timestamps as indices containing, at a minimum,
            columns representing the xyz position and wxyz quaternion-rotation at each
            timestamp, corresponding to the absolute pose at that time.
        match_ts: An integer representing the match frame timestamp.
        query_ts: An integer representing the query frame timestamp.
        to_scale: A boolean. If set to False, relative poses will have their translation
            part normalized.
    Returns:
        A 4x4 numpy array representing the relative pose from match to query frame.
    """
    w_T_bmatch = None
    w_T_bquery = None

    try:
        closest_ts = closest_num(gt_df.index, match_ts)
        if closest_ts != match_ts:
            # print("using closest match for timestamps")
            pass

        w_t_bmatch = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
        w_q_bmatch = np.array(
            [gt_df.at[closest_ts, idx] for idx in ["qw", "qx", "qy", "qz"]]
        )
        w_T_bmatch = transformations.quaternion_matrix(w_q_bmatch)
        w_T_bmatch[:3, 3] = w_t_bmatch
    except Exception:
        print(
            "Failed to convert an abs pose to a rel pose. Timestamp ",
            match_ts,
            " is not available in ground truth df.",
        )
        return None

    try:
        closest_ts = closest_num(gt_df.index, query_ts)
        if closest_ts != query_ts:
            # print("using closest match for timestamps")
            pass

        w_t_bquery = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
        w_q_bquery = np.array(
            [gt_df.at[closest_ts, idx] for idx in ["qw", "qx", "qy", "qz"]]
        )
        w_T_bquery = transformations.quaternion_matrix(w_q_bquery)
        w_T_bquery[:3, 3] = w_t_bquery
    except Exception:
        print(
            "Failed to convert an abs pose to a rel pose. Timestamp ",
            query_ts,
            " is not available in ground truth df.",
        )
        return None

    bmatch_T_bquery = lie.relative_se3(w_T_bmatch, w_T_bquery)
    bmatch_t_bquery = bmatch_T_bquery[:3, 3]

    if not to_scale:
        norm = np.linalg.norm(bmatch_t_bquery)
        if norm > 1e-6:
            bmatch_t_bquery = bmatch_t_bquery / np.linalg.norm(bmatch_t_bquery)

    bmatch_T_bquery[:3, 3] = bmatch_t_bquery

    return bmatch_T_bquery


def convert_abs_traj_to_rel_traj_lcd(df, lcd_df, to_scale=True):
    """
    Convert an absolute-pose trajectory to a relative-pose trajectory.

    The incoming DataFrame df is processed element-wise. At each kf timestamp (which is
    the index of the DataFrame row) starting from the second (index 1), the relative
    pose from the match timestamp to the query stamp is calculated (in the match-
    timestamp's coordinate frame). This relative pose is then appended to the
    resulting DataFrame.
    The resulting DataFrame has timestamp indices corresponding to poses that represent
    the relative transformation between the match timestamp and the query one.

    Args:
        df: A pandas.DataFrame object with timestamps as indices containing, at a
            minimum, columns representing the xyz position and wxyz quaternion-rotation
            at each timestamp, corresponding to the absolute pose at that time.
        lcd_df: A pandas.DataFrame object with timestamps as indices containing, at a
            minimum, columns representing the timestamp of query frames and the
            timestamps of the match frames.
        to_scale: A boolean. If set to False, relative poses will have their translation
            part normalized.

    Returns:
        A pandas.DataFrame object with xyz position and wxyz quaternion fields for the
        relative pose trajectory corresponding to the absolute one given in 'df', and
        relative by the given match and query timestamps.
    """
    rows_list = []
    index_list = []

    for i in range(len(lcd_df.index)):
        match_ts = lcd_df.timestamp_match[lcd_df.index[i]]
        query_ts = lcd_df.timestamp_query[lcd_df.index[i]]

        if match_ts == 0 and query_ts == 0:
            continue

        bi_T_bidelta = get_gt_rel_pose(df, match_ts, query_ts, to_scale)

        if bi_T_bidelta is not None:
            bi_R_bidelta = copy.deepcopy(bi_T_bidelta)
            bi_R_bidelta[:, 3] = np.array([0, 0, 0, 1])
            bi_q_bidelta = transformations.quaternion_from_matrix(bi_R_bidelta)
            bi_t_bidelta = bi_T_bidelta[:3, 3]

            new_row = {
                "x": bi_t_bidelta[0],
                "y": bi_t_bidelta[1],
                "z": bi_t_bidelta[2],
                "qw": bi_q_bidelta[0],
                "qx": bi_q_bidelta[1],
                "qy": bi_q_bidelta[2],
                "qz": bi_q_bidelta[3],
            }
            rows_list.append(new_row)
            index_list.append(lcd_df.index[i])

    return pd.DataFrame(data=rows_list, index=index_list)


def rename_euroc_gt_df(df):
    """Rename a DataFrame built from a EuRoC ground-truth file to be easier to read.

    Column labels are changed to be more readable and to be identical to the generic
    pose trajectory format used with other csv files. Note that '#timestamp' will not
    actually be renamed if it is the index of the DataFrame (which it should be). It
    will be appropriately renamed if it is the index name.

    This operation is 'inplace': It does not return a new DataFrame but simply changes
    the existing one.

    Args:
        df: A pandas.DataFrame object.
    """
    df.index.names = ["timestamp"]
    df.rename(
        columns={
            " p_RS_R_x [m]": "x",
            " p_RS_R_y [m]": "y",
            " p_RS_R_z [m]": "z",
            " q_RS_w []": "qw",
            " q_RS_x []": "qx",
            " q_RS_y []": "qy",
            " q_RS_z []": "qz",
            " v_RS_R_x [m s^-1]": "vx",
            " v_RS_R_y [m s^-1]": "vy",
            " v_RS_R_z [m s^-1]": "vz",
            " b_w_RS_S_x [rad s^-1]": "bgx",
            " b_w_RS_S_y [rad s^-1]": "bgy",
            " b_w_RS_S_z [rad s^-1]": "bgz",
            " b_a_RS_S_x [m s^-2]": "bax",
            " b_a_RS_S_y [m s^-2]": "bay",
            " b_a_RS_S_z [m s^-2]": "baz",
        },
        inplace=True,
    )


def rename_lcd_result_df(df):
    """Rename a DataFrame built from LCD results measurements to a trajectory.

    This is an 'inplace' argument and returns nothing.

    Args:
        df: A pandas.DataFrame object.
    """
    df.index.names = ["timestamp"]
    df.rename(
        columns={
            "#timestamp_match": "timestamp_match",
            "px": "x",
            "py": "y",
            "pz": "z",
        },
        inplace=True,
    )
