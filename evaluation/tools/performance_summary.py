#!/usr/bin/env python

import os
import sys
import argparse
import argcomplete
import yaml
import csv

import evaluation.tools as evt

def parser():
    basic_desc = "Plot timing results for VIO pipeline."
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    input_options = main_parser.add_argument_group("input options")
    input_options.add_argument(
        "vio_results_path", help="Path to the **YAML** file containing the VIO results.",
        default="./results/V1_01_easy/S/results.yaml")
    input_options.add_argument(
        "vio_results_summary_path", help="Path to the **CSV** file containing the VIO summary results.",
        default="./results/V1_01_easy/S/results_summary.csv")
    return main_parser

def write_vio_results_summary(results, vio_results_summary_path):
    # Get APE.
    ATE_mean = results['absolute_errors']['mean']
    ATE_rmse = results['absolute_errors']['rmse']
    # Get RPE for smallest segments.
    #assert(len(results['relative_errors']) > 0)
    #RPE_mean = results['relative_errors'][0]['mean']
    #RPE_rmse = results['relative_errors'][0]['rmse']
    # Generate path to summary if it does not exist.
    evt.create_full_path_if_not_exists(vio_results_summary_path)
    # Write to CSV file.
    with open(vio_results_summary_path, 'w') as vio_results_summary_file:
        print('Writing VIO summary results to: %s' % vio_results_summary_path)
        performance_metrics = ['ATE_mean', 'ATE_rmse', 'Timing']
        writer = csv.DictWriter(vio_results_summary_file, fieldnames=performance_metrics)    
        writer.writeheader()
        writer.writerow({'ATE_mean': ATE_mean, 'ATE_rmse': ATE_rmse, 'Timing': 0})

def main(vio_results_path, vio_results_summary_path):
    # Read vio results yaml file.
    print("Reading VIO results from: %s" % vio_results_path)
    if os.path.exists(vio_results_path):
        with open(vio_results_path,'r') as input:
            results = yaml.load(input)
            write_vio_results_summary(results, vio_results_summary_path)

if __name__ == "__main__":
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(args.vio_results_path, args.vio_results_summary_path)
    sys.exit(os.EX_OK)