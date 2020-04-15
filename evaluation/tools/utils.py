#!/usr/bin/env python

import glog as log

# print in colors
def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
def print_green(skk): print("\033[92m {}\033[00m" .format(skk))
def print_yellow(skk): print("\033[93m {}\033[00m" .format(skk))
def print_lightpurple(skk): print("\033[94m {}\033[00m" .format(skk))
def print_purple(skk): print("\033[95m {}\033[00m" .format(skk))
def print_cyan(skk): print("\033[96m {}\033[00m" .format(skk))
def print_lightgray(skk): print("\033[97m {}\033[00m" .format(skk))
def print_black(skk): print("\033[98m {}\033[00m" .format(skk))

# Get items from a dictionary
def get_items(dict_object):
    for key in dict_object:
        yield key, dict_object[key]

def check_stats(stats):
    exception_msg = "Wrong stats format: no relative_errors... \n"
    "Are you sure you ran the pipeline and saved the results? (--save_results)."
    if not "relative_errors" in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception(exception_msg)
    else:
        if len(stats["relative_errors"]) == 0:
            raise Exception(exception_msg)

        if not "rpe_rot" in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception(exception_msg)
        if not "rpe_trans" in list(stats["relative_errors"].values())[0]:
            log.error("Stats: ")
            log.error(stats)
            raise Exception(exception_msg)
    if not "absolute_errors" in stats:
        log.error("Stats: ")
        log.error(stats)
        raise Exception(exception_msg)
