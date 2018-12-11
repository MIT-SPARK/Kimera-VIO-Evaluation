#!/usr/bin/env python
import os
import logging

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

def prompt_val(msg="Enter a value:"):
    return input(msg + "\n")

def confirm(msg="Enter 'y' to confirm or any other key to cancel.", key='y'):
    return input(msg + "\n") == key

def check_and_confirm_overwrite(file_path):
    if os.path.isfile(file_path):
        print(file_path + " exists, do you want to overwrite?")
        return confirm()
    else:
        return True
