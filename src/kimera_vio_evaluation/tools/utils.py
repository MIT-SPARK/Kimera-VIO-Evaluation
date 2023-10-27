"""Logging utilities."""
import glog as log


# print in colors
def print_red(skk):
    """Print red string."""
    print("\033[91m {}\033[00m".format(skk))


def print_green(skk):
    """Print green string."""
    print("\033[92m {}\033[00m".format(skk))


def print_lightpurple(skk):
    """Print light purple string."""
    print("\033[94m {}\033[00m".format(skk))


def print_purple(skk):
    """Print purple string."""
    print("\033[95m {}\033[00m".format(skk))


def check_stats(stats):
    """Check stat contents."""
    exception_msg = "Wrong stats format: no relative_errors... \n"
    "Are you sure you ran the pipeline and saved the results? (--save_results)."
    if "relative_errors" not in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception(exception_msg)
    else:
        if len(stats["relative_errors"]) == 0:
            raise Exception(exception_msg)

        if "rpe_rot" not in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception(exception_msg)
        if "rpe_trans" not in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception(exception_msg)
    if "absolute_errors" not in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception(exception_msg)
