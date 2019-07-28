#!/usr/bin/env python

import os
from shutil import rmtree, move
import glog as log
import errno

def create_full_path_if_not_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            log.info('Creating non-existent path: %s' % filename)
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                log.fatal("Could not create inexistent filename: " + filename)

def move_output_from_to(from_dir, to_dir):
    try:
        if (os.path.exists(to_dir)):
            rmtree(to_dir)
    except:
        log.info("Directory:" + to_dir + " does not exist, we can safely move output.")
    try:
        if (os.path.isdir(from_dir)):
            move(from_dir, to_dir)
        else:
            log.info("There is no output directory...")
    except:
        print("Could not move output from: " + from_dir + " to: " + to_dir)
        raise
    try:
        os.makedirs(from_dir)
    except:
        log.fatal("Could not mkdir: " + from_dir)

def ensure_dir(dir_path):
    """ Check if the path directory exists: if it does, returns true,
    if not creates the directory dir_path and returns if it was successful"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True
