#!/usr/bin/env python

from __future__ import print_function
import os
import yaml
import glog as log
from shutil import rmtree, copytree, copy2

from evaluation.tools.filesystem_utils import ensure_dir
from evaluation.tools.plotter import draw_regression_simple_boxplot_APE
from evaluation.tools.utils import get_items
from evaluation_lib import run_dataset

def write_flags_params(param_filepath, param_name, param_value):
    """ Write params to gflags file.

    Args:
        param_filepath: path to the gflag file.
        param_name: name of the parameter to write.
        param_value: value of the parameter to write.

    Returns:
    """
    directory = os.path.dirname(param_filepath)
    if not os.path.exists(directory):
        raise Exception("\033[91mCould not find directory: \033[99m \n %s" % directory )
    params_flagfile = open(param_filepath, "a+")
    params_flagfile.write("--" + param_name + "=" + param_value)
    params_flagfile.close()

def write_yaml_params(param_filepath, param_name, param_value, default_flow_style):
    """ Writes a line in a YAML file with a param_name and a param_value by searching
    for the param_name and overwritting its param_value

    Args:
        param_filepath: path to the file containing the params.
        param_name: name of the parameter to write.
        param_value: value of the parameter to write.

    Returns:
        is_param_name_written: bool specifying if the parameter was written or not.
    """
    directory = os.path.dirname(param_filepath)
    if not os.path.exists(directory):
        raise Exception(
            "\033[91mCould not find directory: \033[99m \n %s" % directory ) # Check param_file exists: assert(os.path.isfile(param_filepath))
    is_param_name_written = False
    # New params store old params and changes just the param_name with the new param_value
    # Then we dump/write these new params to file again.
    new_params = []
    with open(param_filepath, 'r') as infile:
        # Skip first yaml line: it contains %YAML:... which can't be read...
        _ = infile.readline()
        try:
            new_params = yaml.load(infile)
        except:
             raise Exception("Error loading YAML file: {}" % param_filepath)
        if param_name in new_params:
            # Modify param_name with param_value
            new_params[param_name] = param_value
            is_param_name_written = True
        else:
            # Param_name was not written to file because it is not present in file.
            is_param_name_written = False
    # Store param_names with param_value
    with open(param_filepath, 'w') as outfile:
        # First line must contain %YAML:1.0
        outfile.write("%YAML:1.0\n")
    with open(param_filepath, 'a') as outfile:
        # Write the actual new parameters.
        if not default_flow_style:
            outfile.write(yaml.dump(new_params, default_flow_style=False))
        else:
            outfile.write(yaml.dump(new_params)) # Do not use default_flow_style=True, adds braces...
    return is_param_name_written

def check_and_create_regression_test_structure(regression_tests_path, baseline_params_dir, param_name_to_values,
                                               ground_truth_path):
    """ Makes/Checks that the file structure is the correct one, and updates the parameters with the given values

    Args:
        regression_tests_path: path to the root folder where all regression results will be stored.
        baseline_params_dir: path to the directory containing the baseline parameters used for the VIO, which we modify with
            the regressed parameters.
        param_name_to_value: dict mapping a parameter name to its value.
        ground_truth_path: path to the ground-truth of euroc's dataset containing a traj_gt.csv file per dataset.
        (NOT NEEDED) dataset_names: name of the datasets for which we will be running regression tests.
        (NOT NEEDED) pipeline_types: list of the pipeline types to run [S, SP and/or SPR].
        (NOT NEEDED FOR NOW) extra_params_to_modify: extra parameters to modify besides the baseline regression (disabled for now.).

    Returns:

    """
    # Make or check regression_test root directory
    assert(ensure_dir(regression_tests_path))
    for param_name, param_values in get_items(param_name_to_values):
        # Create or check param_name directory
        param_name_dir = os.path.join(regression_tests_path, param_name)
        assert(ensure_dir(param_name_dir))

        # Create/Check tmp_output folder
        #tmp_output_dir = os.path.join(param_name_dir, "tmp_output/output")
        #ensure_dir(tmp_output_dir)

        for param_value in param_values:
            # Create or check param_value folder
            param_name_value_dir = os.path.join(param_name_dir, str(param_value))
            assert(ensure_dir(param_name_value_dir))

            # Create params folder by copying from current baseline one.
            modified_baseline_params_dir = os.path.join(param_name_value_dir, "params")
            if (os.path.exists(modified_baseline_params_dir)):
                rmtree(modified_baseline_params_dir)
            # TODO(Toni): check that baseline_params_dir is good
            copytree(baseline_params_dir, modified_baseline_params_dir)

            ######## MODIFY REGRESSED PARAMETER ##############################
            # Modify param with param value by searching in all yaml files
            is_param_name_written_in_yaml_file = False
            for root, _, files in os.walk(modified_baseline_params_dir):
                for file in files:
                    if file.endswith(".yaml"):
                        if "tracker" in file:
                            is_param_name_written_in_yaml_file = write_yaml_params(
                                os.path.join(root, file), param_name, param_value, False)
                        else:
                            is_param_name_written_in_yaml_file = write_yaml_params(
                                os.path.join(root, file), param_name, param_value, True)

            # Modify gflags parameters
            if not is_param_name_written_in_yaml_file:
                # Could not find param_name in vio_params nor tracker_params it must be a gflag:
                write_flags_params(os.path.join(modified_baseline_params_dir, "flags/override.flags"),
                                   param_name, param_value)
            #for extra_param_name, extra_param_value in extra_params_to_modify.items():
            #    if extra_param_name not in written_extra_param_names:
            #        write_flags_params(extra_param_name,
            #                               extra_param_value,
            #                               param_pipeline_dir + "/flags/override.flags")
            ###################################################################

            # Create/Check dataset name directory
            #for dataset in dataset_names:
            #    # Create/Check dataset_name folder
            #    param_name_value_dataset_dir = os.path.join(param_name_value_dir, dataset)
            #    assert(ensure_dir(param_name_value_dataset_dir))

            #    # Create/Check pipeline name directory
            #    for pipeline_type in pipeline_types:
            #        # Create/Check pipeline folder
            #        param_name_value_dataset_pipeline_dir = os.path.join(param_name_value_dataset_dir, pipeline_type)
            #        assert(ensure_dir(param_name_value_dataset_pipeline_dir))

    # Make/Check results dir for current param_names
    param_name_results_dir = os.path.join(param_name_dir, "results")
    assert(ensure_dir(param_name_results_dir))
    #for dataset_name in dataset_names:
    #    # Make/Check dataset dir for current param_names_dir, as the performance given the param depends on the dataset.
    #    assert(ensure_dir(os.path.join(param_name_results_dir, dataset_name)))

def run(args):
    # Get experiment information from yaml file.
    experiment_params = yaml.load(args.experiments_path)

    regression_tests_dir = os.path.expandvars(experiment_params['regression_tests_dir'])
    params_dir = os.path.expandvars(experiment_params['params_dir'])
    dataset_dir = os.path.expandvars(experiment_params['dataset_dir'])
    executable_path = os.path.expandvars(experiment_params['executable_path'])

    datasets_to_run = experiment_params['datasets_to_run']
    regression_params = experiment_params['regression_parameters']

    # Build dictionary from parameter name to list of parameter values
    param_name_to_values = dict()
    for regression_param in regression_params:
        param_name_to_values[regression_param['name']] = regression_param['values']

    log.info("Setup regression tests.")
    check_and_create_regression_test_structure(regression_tests_dir,
                                               params_dir,
                                               param_name_to_values,
                                               dataset_dir)

    # Run experiments.
    log.info("Run regression tests.")
    for regression_param in regression_params:
        stats = dict()
        # Redirect to param_name_value dir param_name = regression_param['name']
        param_name = regression_param['name']
        log.info("Run for regression parameter: %s" % param_name)
        for param_value in regression_param['values']:
            results_dir = os.path.join(regression_tests_dir, param_name, str(param_value))
            # Redirect to modified params_dir
            params_dir = os.path.join(results_dir, 'params')
            for dataset in datasets_to_run:
                dataset_name = dataset['name']
                log.info("\033[93mRun: %s = %f in dataset %s\033[00m" % (param_name, param_value, dataset_name))
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
                    raise Exception("\033[91m Dataset: ", dataset_name, " failed!! \033[00m")
                stats[param_value] = dict()
                for pipeline in pipelines_to_run:
                    results_file = os.path.join(results_dir, dataset_name, pipeline, "results.yaml")
                    log.error(results_file)
                    if os.path.isfile(results_file):
                        stats[param_value][pipeline] = yaml.load(open(results_file,'r'))
                    else:
                        log.warning("Could not find results file: {}. Adding cross to boxplot...".format(results_file))
                        stats[param_value][pipeline] = False



        # Compile results for current param_name
        log.info("Drawing boxplot APE for regression test of param_name: %s" % param_name)
        for dataset in datasets_to_run:
            dataset_name = dataset['name']

            log.info("Drawing regression simple APE boxplots for dataset: %s" % dataset_name)
            plot_dir = os.path.join(regression_tests_dir, param_name, "results", dataset_name)
            max_y = -1
            draw_regression_simple_boxplot_APE(param_name, stats, plot_dir, max_y)
        log.info("Finished regression test for param_name: %s" % param_name)

    return True

def parser():
    import argparse
    basic_desc = "Regression tests of SPARK VIO pipeline."

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
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if run(args):
        sys.exit(os.EX_OK)
    else:
        raise Exception("Regression tests failed.")
