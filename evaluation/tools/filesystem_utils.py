#!/usr/bin/env python
import os
import errno

def create_full_path_if_not_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print("Could not create inexistent filename: " + filename)
                raise
