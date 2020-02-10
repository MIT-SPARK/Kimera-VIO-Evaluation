#!/usr/bin/env python

from __future__ import print_function
import glog as log
import os
from ruamel import yaml
from tqdm import tqdm

# Ignore warnings from matplotlib...
import warnings
warnings.filterwarnings("ignore")

from evaluation.evaluation_lib import DatasetEvaluator, DatasetRunner, aggregate_ape_results

def run(args):
    # Get experiment information from yaml file.
    experiment_params = yaml.load(args.experiments_path, Loader=yaml.Loader)
    # Create dataset evaluator: runs vio depending on given params and analyzes output.
    extra_flagfile_path = ""  # TODO(marcus): parse from experiments
    # TODO(marcus): choose which of the following based on -r -a flags
    dataset_evaluator = DatasetEvaluator(experiment_params, args, extra_flagfile_path)
    dataset_evaluator.evaluate()
    # Aggregate results in results directory
    aggregate_ape_results(os.path.expandvars(experiment_params['results_dir']))
    return True

def parser():
    import argparse
    basic_desc = "Full evaluation of SPARK VIO pipeline (APE trans + RPE trans + RPE rot) metric app"

    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))

    input_opts = shared_parser.add_argument_group("input options")
    evaluation_opts = shared_parser.add_argument_group("algorithm options")
    output_opts = shared_parser.add_argument_group("output options")

    input_opts.add_argument("experiments_path", type=argparse.FileType('r'),
                            help="Path to the yaml file with experiments settings.",
                            default="./experiments.yaml")

    evaluation_opts.add_argument("-r", "--run_pipeline", action="store_true",
                                 help="Run vio?")
    evaluation_opts.add_argument("-a", "--analyze_vio", action="store_true",
                                 help="Analyze vio, compute APE and RPE")

    output_opts.add_argument(
        "--plot", action="store_true", help="show plot window",)
    output_opts.add_argument("--save_plots", action="store_true",
                             help="Save plots?")
    output_opts.add_argument("--write_website", action="store_true",
                             help="Write website with results?")
    output_opts.add_argument("--save_boxplots", action="store_true",
                             help="Save boxplots?")
    output_opts.add_argument("--save_results", action="store_true",
                             help="Save results?")
    output_opts.add_argument("-v", "--verbose_sparkvio", action="store_true",
                             help="Make SparkVIO log all verbosity to console. Useful for debugging if a run failed.")

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser

import argcomplete
import sys
if __name__ == '__main__':
    log.setLevel("INFO")
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # try:
    if run(args):
        sys.exit(os.EX_OK)
    # except Exception as e:
    #     print("error: ", e)
    #     raise Exception("Main evaluation run failed.")
