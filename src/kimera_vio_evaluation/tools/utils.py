"""Logging utilities."""
import errno
import glog
import os


def check_stats(stats):
    """Check stat contents."""
    exception_msg = "Wrong stats format: no relative_errors... \n"
    "Are you sure you ran the pipeline and saved the results? (--save_results)."
    if "relative_errors" not in stats:
        glog.error("Stats: ")
        glog.error(stats)
        raise Exception(exception_msg)
    else:
        if len(stats["relative_errors"]) == 0:
            raise Exception(exception_msg)

        if "rpe_rot" not in list(stats["relative_errors"].values())[0]:
            glog.error("Stats: ")
            glog.error(stats)
            raise Exception(exception_msg)
        if "rpe_trans" not in list(stats["relative_errors"].values())[0]:
            glog.error("Stats: ")
            glog.error(stats)
            raise Exception(exception_msg)
    if "absolute_errors" not in stats:
        glog.error("Stats: ")
        glog.error(stats)
        raise Exception(exception_msg)


def create_full_path_if_not_exists(filename):
    """Create parent of file if it doesn't exist."""
    if not os.path.exists(os.path.dirname(filename)):
        try:
            glog.debug("Creating non-existent path: %s" % filename)
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                glog.fatal("Could not create inexistent filename: " + filename)


def ensure_dir(dir_path):
    """
    Check if the path directory exists: if it does, returns true.

    If not creates the directory dir_path and returns if it was successful
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True
