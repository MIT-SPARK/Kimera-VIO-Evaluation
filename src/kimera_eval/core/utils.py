"""Logging utilities."""
import glog


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
