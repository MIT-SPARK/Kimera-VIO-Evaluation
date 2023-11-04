"""Logging utilities."""
import logging


def check_stats(stats):
    """Check stat contents."""
    if "relative_errors" not in stats:
        logging.error("Stats: ")
        logging.error(stats)
        raise Exception("missing required metrics")

    if len(stats["relative_errors"]) == 0:
        raise Exception("missing required metrics")

    if "rpe_rot" not in list(stats["relative_errors"].values())[0]:
        logging.error("Stats: ")
        logging.error(stats)
        raise Exception("missing required metrics")

    if "rpe_trans" not in list(stats["relative_errors"].values())[0]:
        logging.error("Stats: ")
        logging.error(stats)
        raise Exception("missing required metrics")

    if "absolute_errors" not in stats:
        logging.error("Stats: ")
        logging.error(stats)
        raise Exception("missing required metrics")
