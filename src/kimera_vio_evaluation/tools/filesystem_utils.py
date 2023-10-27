"""Utilities for handling directories."""
import os
import glog
import errno


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
