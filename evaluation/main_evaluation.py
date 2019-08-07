#!/usr/bin/env python

from __future__ import print_function
import glog as log
import os
import yaml

from evaluation_lib import run_dataset, aggregate_ape_results

def run(args):
    # Get experiment information from yaml file.
    experiment_params = yaml.load(args.experiments_path)

    results_dir = os.path.expandvars(experiment_params['results_dir'])
    params_dir = os.path.expandvars(experiment_params['params_dir'])
    dataset_dir = os.path.expandvars(experiment_params['dataset_dir'])
    executable_path = os.path.expandvars(experiment_params['executable_path'])
    datasets_to_run = experiment_params['datasets_to_run']

    # Run experiments.
    log.info("Run experiments")
    successful_run = True
    for dataset in datasets_to_run:
        log.info("Run dataset: %s" % dataset['name'])
        pipelines_to_run = dataset['pipelines']
        if not run_dataset(results_dir, params_dir, dataset_dir, dataset, executable_path,
                           args.run_pipeline, args.analyse_vio,
                           args.plot, args.save_results,
                           args.save_plots, args.save_boxplots,
                           pipelines_to_run,
                           dataset['initial_frame'],
                           dataset['final_frame'],
                           dataset['discard_n_start_poses'],
                           dataset['discard_n_end_poses']):
            log.info("\033[91m Dataset: %s failed!! \033[00m" % dataset['name'])
            successful_run = False

    aggregate_ape_results(results_dir)
    return successful_run

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
    evaluation_opts.add_argument("-a", "--analyse_vio", action="store_true",
                                 help="Analyse vio, compute APE and RPE")

    output_opts.add_argument(
        "--plot", action="store_true", help="show plot window",)
    output_opts.add_argument("--save_plots", action="store_true",
                             help="Save plots?")
    output_opts.add_argument("--save_boxplots", action="store_true",
                             help="Save boxplots?")
    output_opts.add_argument("--save_results", action="store_true",
                             help="Save results?")

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
    if run(args):
        sys.exit(os.EX_OK)
    else:
        raise Exception("Main evaluation run failed.")
