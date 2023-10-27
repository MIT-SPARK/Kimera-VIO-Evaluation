"""Run evaluation on goseek."""
import glog as log
import os
from ruamel import yaml

# Ignore warnings from matplotlib...
import warnings

warnings.filterwarnings("ignore")

from evaluation.evaluation_lib import (
    DatasetEvaluator,
    DatasetRunner,
    aggregate_ape_results,
)


def find_submissions(results_dir):
    """Finds all folders having a traj_vio.csv file. We assume these folders have the following
    filesystem:
            - logs:
            \* <submission_1>
            |___\* kimera_vio_logs
            |   |___\* Tesse
            |   |   |___ traj_vio.csv
            |   |___\* Tesse_1
            |       |___ traj_vio.csv
            |___\* other_logs...
            \* <submission_2>
            |___\* kimera_vio_logs
            |   |___\* Tesse
            |       |___ traj_vio.csv
            |___\* other_logs...
            |   |

        Args: results_dir path to the `logs` directory (see filesystem above)

        Return: List of submission ids where a traj_vio.csv was found.
    """
    import fnmatch

    # Load results.
    # Aggregate all stats for each pipeline and dataset
    submissions = dict()
    for root, _, filenames in os.walk(results_dir):
        for _ in fnmatch.filter(filenames, "traj_vio.csv"):
            # results_filepath = os.path.join(root, results_filename)
            # Get pipeline name
            pipeline_name = os.path.basename(root)
            # Only check first five chars, others correspond to log id
            assert pipeline_name[:5] == "Tesse"
            # Get kimera_vio_ros name
            mid_folder_path = os.path.split(root)[0]
            mid_folder_name = os.path.basename(mid_folder_path)
            if mid_folder_name != "kimera_vio_logs":
                raise Exception(
                    "Wrong mid folder name: \n \
                                - expected: kimera_vio_logs \n \
                                - got: %s"
                    % mid_folder_name
                )

            # Get submission id name
            submission_id = os.path.basename(os.path.split(mid_folder_path)[0])
            pipeline = mid_folder_name + "/" + pipeline_name
            # Collect stats
            submissions.setdefault(submission_id, []).append(pipeline)

    return submissions


def run(args):
    # Get experiment information from yaml file.
    experiment_params = yaml.load(args.experiments_path, Loader=yaml.Loader)
    results_dir = os.path.expandvars(experiment_params["results_dir"])

    submissions = find_submissions(results_dir)

    if len(submissions) == 0:
        log.warning("No submissions found!")
        return False

    experiment_params["vocabulary_path"] = ""
    experiment_params["params_dir"] = ""
    experiment_params["dataset_dir"] = ""
    experiment_params["executable_path"] = ""
    experiment_params["datasets_to_run"] = []
    for submission_id, pipelines in submissions.items():
        dataset_to_evaluate = dict()
        dataset_to_evaluate["name"] = submission_id
        dataset_to_evaluate["use_lcd"] = False  # This is only for running VIO, not eval
        dataset_to_evaluate["segments"] = experiment_params["segments"]
        dataset_to_evaluate["pipelines"] = pipelines
        dataset_to_evaluate["discard_n_start_poses"] = experiment_params[
            "discard_n_start_poses"
        ]
        dataset_to_evaluate["discard_n_end_poses"] = experiment_params[
            "discard_n_end_poses"
        ]
        experiment_params["datasets_to_run"].append(dataset_to_evaluate)

    # Create dataset evaluator: evaluates vio output.
    dataset_evaluator = DatasetEvaluator(experiment_params, args, "")
    dataset_evaluator.evaluate()

    # Aggregate results in results directory
    # TODO(Toni): do a go-seek version
    # aggregate_ape_results(os.path.expandvars(experiment_params['results_dir']))
    return True


def parser():
    import argparse

    basic_desc = "Full evaluation of SPARK VIO pipeline (APE trans + RPE trans + RPE rot) metric app"

    shared_parser = argparse.ArgumentParser(
        add_help=True, description="{}".format(basic_desc)
    )

    input_opts = shared_parser.add_argument_group("input options")
    evaluation_opts = shared_parser.add_argument_group("algorithm options")
    output_opts = shared_parser.add_argument_group("output options")

    input_opts.add_argument(
        "experiments_path",
        type=argparse.FileType("r"),
        help="Path to the yaml file with experiments settings.",
        default="./experiments.yaml",
    )

    evaluation_opts.add_argument(
        "-r", "--run_pipeline", action="store_true", help="Run vio?"
    )
    evaluation_opts.add_argument(
        "-a",
        "--analyze_vio",
        action="store_true",
        help="Analyze vio, compute APE and RPE",
    )

    output_opts.add_argument(
        "--plot",
        action="store_true",
        help="show plot window",
    )
    output_opts.add_argument("--save_plots", action="store_true", help="Save plots?")
    output_opts.add_argument(
        "--write_website", action="store_true", help="Write website with results?"
    )
    output_opts.add_argument(
        "--save_boxplots", action="store_true", help="Save boxplots?"
    )
    output_opts.add_argument(
        "--save_results", action="store_true", help="Save results?"
    )
    output_opts.add_argument(
        "-v",
        "--verbose_sparkvio",
        action="store_true",
        help="Make SparkVIO log all verbosity to console. Useful for debugging if a run failed.",
    )

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser


import argcomplete
import sys

if __name__ == "__main__":
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
