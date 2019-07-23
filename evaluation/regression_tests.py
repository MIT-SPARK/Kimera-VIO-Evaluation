#!/usr/bin/env python

from __future__ import print_function
import os
import yaml
from shutil import rmtree, copytree, copy2

import evaluation.tools as evt

def write_flags_params(param_filepath, param_name_to_value):
    """ Write params to gflags file 

    Args:
        param_filepath: path to the gflag file
        param_name_to_value: dict from param name to value for those parameters that we want to write

    Returns:
    """
    directory = os.path.dirname(param_filepath)
    if not os.path.exists(directory):
        raise Exception("\033[91mCould not find directory: " + directory + "\033[99m")
    params_flagfile = open(param_filepath, "a+")
    for param_name, param_value in param_name_to_value:
        params_flagfile.write("--" + param_name + "=" + param_value)
    params_flagfile.close()

def write_yaml_params(param_filepath, param_name_to_value):
    """ Writes a line in a YAML file with a param_name and a param_value by searching
    for the param_name and overwritting its param_value

    Args:
        param_filepath: path to the file containing the params.
        param_name_to_value: dict mapping a parameter name to its value.

    Returns:
        is_param_name_written: dict specifying if the parameter was written or not.

    """
    directory = os.path.dirname(param_filepath)
    if not os.path.exists(directory):
        raise Exception(
            "\033[91mCould not find directory: " + directory + "\033[99m")
    # Check param_file exists:
    assert(os.path.isfile(param_filepath))
    # Check param_name_to_value is a dict()
    assert(isinstance(param_name_to_value, dict))
    is_param_name_written = {}
    new_params = []
    with open(param_filepath, 'r') as infile:
        # Skip first yaml line: it contains %YAML:... which can't be read...
        _ = infile.readline()
        try:
            new_params = yaml.load(infile)
        except:
             raise Exception("Error loading YAML file: {}" % param_filepath)
        for param_name, param_value in enumerate(param_name_to_value):
            if param_name in new_params:
                # Modify param_name with param_value
                new_params[param_name] = param_value
                is_param_name_written[param_name] = True
            else:
                # Param_name was not written to file because it is not present in file.
                is_param_name_written[param_name] = False
    # Store param_names with param_value
    with open(param_filepath, 'w') as outfile:
        # First line must contain %YAML:1.0
        outfile.write("%YAML:1.0\n")
    with open(param_filepath, 'a') as outfile:
        # Write the actual new parameters.
        outfile.write(yaml.dump(new_params, default_flow_style=False))
    return is_param_name_written

def check_and_create_regression_test_structure(regression_tests_path, param_name_to_values,
                                               dataset_names, pipeline_types, extra_params_to_modify):
    """ Makes/Checks that the file structure is the correct one, and updates the parameters with the given values

    Args:
        regression_tests_path:
        param_name_to_value: dict mapping a parameter name to its value.
        dataset_names:
        pipeline_types:
        extra_params_to_modify:

    Returns:

    """
    # Make or check regression_test directory
    assert(evt.ensure_dir(regression_tests_path))

    # Make or check param_name directory
    # Use as param_name the concatenated elements of param_names
    param_names_dir = ""
    for i in param_name_to_values:
        param_names_dir += str(i) + "-"
    param_names_dir = param_names_dir[:-1]
    assert(evt.ensure_dir("{}/{}".format(regression_tests_path, param_names_dir)))

    for param_name, param_value in param_name_to_values:
        # Create/Check param_value folder

        ##########################
        param_value_dir = ""
        if isinstance(param_value, list):
            for i in param_value:
                param_value_dir += str(i) + "-"
            param_value_dir = param_value_dir[:-1]
        else:
            param_value_dir = param_value
        ###########################

        evt.ensure_dir("{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir))

        # Create params folder by copying from current official one.
        param_dir = "{}/{}/{}/params".format(regression_tests_path, param_names_dir, param_value_dir)
        if (os.path.exists(param_dir)):
            rmtree(param_dir)
        copytree("/home/tonirv/code/evo/results/params", param_dir)

        # Modify param with param value
        for pipeline_type in pipeline_types:
            param_pipeline_dir = "{}/{}".format(param_dir, pipeline_type)
            evt.ensure_dir(param_pipeline_dir)
            written_extra_param_names = []

            # TODO(Toni) Remove hardcoded...
            ###########################################################
            # VIO params
            vio_file = param_pipeline_dir + "/vioParameters.yaml"
            is_param_name_written_in_vio_file = write_yaml_params(vio_file, param_name_to_values)
            ###################################################################
            # Tracker params
            tracker_file = param_pipeline_dir + "/trackerParameters.yaml"
            is_param_name_written_in_tracker_file = write_yaml_params(tracker_file, param_name_to_values)
            ###################################################################

            # Join both dictionaries, the non-written params to yaml files must be gflags.
            is_param_name_written = dict()
            for key in param_name_to_values:
                is_param_name_written[key] = is_param_name_written_in_vio_file[key] or is_param_name_written_in_tracker_file[key]

            ###################################################################
            # Gflags
            for param_name, param_value in param_name_to_values:
                if not is_param_name_written[param_name]:
                    # Could not find param_name in vio_params nor tracker_params
                    # it must be a gflag:
                    write_flags_params(param_pipeline_dir + "/flags/override.flags", param_name_to_values)
            #for extra_param_name, extra_param_value in extra_params_to_modify.items():
            #    if extra_param_name not in written_extra_param_names:
            #        write_flags_params(extra_param_name,
            #                               extra_param_value,
            #                               param_pipeline_dir + "/flags/override.flags")
            ###################################################################

        # Create/Check tmp_output folder
        evt.ensure_dir("{}/{}/{}/tmp_output/output".format(regression_tests_path, param_names_dir, param_value_dir))

        ###################################################################
        # TODO(TONI): remove hardcoded
        for dataset_name in dataset_names:
            evt.ensure_dir("{}/{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir, dataset_name))
            # Create ground truth trajectory by copying from current official one.
            copy2("/home/tonirv/code/evo/results/{}/traj_gt.csv".format(dataset_name),
                 "{}/{}/{}/{}/traj_gt.csv".format(regression_tests_path, param_names_dir,
                                                  param_value_dir, dataset_name))
            # Create segments by copying from current official one.
            copy2("/home/tonirv/code/evo/results/{}/segments.txt".format(dataset_name),
                 "{}/{}/{}/{}/segments.txt".format(regression_tests_path, param_names_dir,
                                                  param_value_dir, dataset_name))
            for pipeline_type in pipeline_types:
                evt.ensure_dir("{}/{}/{}/{}/{}".format(regression_tests_path, param_names_dir, param_value_dir,
                                                   dataset_name, pipeline_type))

    # Make/Check results dir for current param_names
    evt.ensure_dir("{}/{}/results".format(regression_tests_path, param_names_dir))
    for dataset_name in dataset_names:
        # Make/Check dataset dir for current param_names_dir, as the performance given the param depends on the dataset.
        evt.ensure_dir("{}/{}/results/{}".format(regression_tests_path, param_names_dir, dataset_name))

# TODO(TONI): remove this, as it is parsed in YAML
def build_list_of_pipelines_to_run(pipelines_to_run):
    pipelines_to_run_list = []
    if pipelines_to_run == 0:
        pipelines_to_run_list = ['S', 'SP', 'SPR']
    if pipelines_to_run == 1:
        pipelines_to_run_list = ['S']
    if pipelines_to_run == 2:
        pipelines_to_run_list = ['SP']
    if pipelines_to_run == 3:
        pipelines_to_run_list = ['SPR']
    if pipelines_to_run == 4:
        pipelines_to_run_list = ['S', 'SP']
    if pipelines_to_run == 5:
        pipelines_to_run_list = ['S', 'SPR']
    if pipelines_to_run == 6:
        pipelines_to_run_list = ['SP', 'SPR']
    return pipelines_to_run_list

def regression_test_simple(test_name, param_names, param_values, only_compile_regression_test_results,
                           run_pipelines, pipelines_to_run, extra_params_to_modify):
    """ Runs the vio pipeline with different values for the given param
    and draws graphs to decide best value for the param:
        - param_names: names of the parameters to fine-tune: e.g ["monoNoiseSigma", "stereoNoiseSigma"]
        - param_values: values that the parameter should take: e.g [[1.0, 1.3], [1.0, 1.2]]
        - only_compile_regression_test_results: just draw boxplots for regression test,
            skip all per pipeline analysis and runs, assumes we have results.yaml for
            each param value, dataset and pipeline.
        - run_pipelines: run pipelines, if set to false, it won't run pipelines and will assume we have a traj_est.csv already.
        - pipelines_to_run: which pipeline to run, useful when a parameter only affects a single pipeline."""
    # Ensure input is correct.
    if isinstance(param_names, list):
        if len(param_names) > 1:
            assert(len(param_names) == len(param_values[0]))
            for i in range(2, len(param_names)):
                # Ensure all rows have the same number of parameter changes
                assert(len(param_values[i-2]) == len(param_values[i-1]))

    # Check and create file structure
    dataset_names = ["V1_01_easy"]
    pipelines_to_run_list = build_list_of_pipelines_to_run(pipelines_to_run)
    REGRESSION_TESTS_DIR = "/home/tonirv/code/evo-1/regression_tests/" + test_name
    check_and_create_regression_test_structure(REGRESSION_TESTS_DIR, param_names, param_values,
                                               dataset_names, pipelines_to_run_list, extra_params_to_modify)

    param_names_dir = ""
    for i in param_names:
        param_names_dir += str(i) + "-"
    param_names_dir = param_names_dir[:-1]
    DATASET_DIR = '/home/tonirv/datasets/EuRoC'
    BUILD_DIR = '/home/tonirv/code/spark_vio/build'
    if not only_compile_regression_test_results:
        for param_value in param_values:
            param_value_dir = ""
            if isinstance(param_value, list):
                for i in param_value:
                    param_value_dir += str(i) + "-"
                param_value_dir = param_value_dir[:-1]
            else:
                param_value_dir = param_value
            results_dir = "{}/{}/{}".format(REGRESSION_TESTS_DIR, param_names_dir, param_value_dir)
            PARAMS_DIR = results_dir
            for dataset_name in dataset_names:
                run_dataset(results_dir, PARAMS_DIR, DATASET_DIR, dataset_name, BUILD_DIR,
                               run_pipelines, # Should we re-run pipelines?
                               True, # Should we run the analysis of per pipeline errors?
                               False, # Should we display plots?
                               True, # Should we save results?
                               True, # Should we save plots?
                               False, # Should we save boxplots?
                               pipelines_to_run_list) # Should we run 0: all pipelines, 1: S, 2:SP 3:SPR

                print("Finished analysis of pipelines for param_value: {} for parameter: {}".format(param_value_dir, param_names_dir))
            print("Finished pipeline runs/analysis for regression test of param_name: {}".format(param_names_dir))

    # Compile results for current param_name
    print("Drawing boxplot APE for regression test of param_name: {}".format(param_names_dir))
    for dataset_name in dataset_names:
        stats = dict()
        for param_value in param_values:
            param_value_dir = ""
            if isinstance(param_value, list):
                for i in param_value:
                    param_value_dir += str(i) + "-"
                param_value_dir = param_value_dir[:-1]
            else:
                param_value_dir = param_value
            stats[param_value_dir] = dict()
            for pipeline in pipelines_to_run_list:
                results_file = "{}/{}/{}/{}/{}/results.yaml".format(REGRESSION_TESTS_DIR, param_names_dir,
                                                                    param_value_dir, dataset_name, pipeline)
                if os.path.isfile(results_file):
                    stats[param_value_dir][pipeline] = yaml.load(open(results_file,'r'))
                else:
                    print("Could not find results file: {}".format(results_file) + ". Adding cross to boxplot...")
                    stats[param_value_dir][pipeline] = False

        print("Drawing regression simple APE boxplots for dataset: " + dataset_name)
        plot_dir = "{}/{}/results/{}".format(REGRESSION_TESTS_DIR, param_names_dir, dataset_name)
        max_y = -1
        if dataset_name == "V2_02_medium":
            max_y = 0.40
        if dataset_name == "V1_01_easy":
            max_y = 0.20
        evt.draw_regression_simple_boxplot_APE(param_names, stats, plot_dir, max_y)
    print("Finished regression test for param_name: {}".format(param_names_dir))
