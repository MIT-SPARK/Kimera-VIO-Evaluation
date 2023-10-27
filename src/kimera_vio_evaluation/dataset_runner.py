"""Main library for evaluation."""
import os
import subprocess
import glog as log
import shutil
import tqdm
import pathlib


def print_green(skk):
    """Print green string."""
    print("\033[92m {}\033[00m".format(skk))


class DatasetRunner:
    """DatasetRunner is used to run the pipeline on datasets."""

    def __init__(self, params, extra_flags_path="", verbose=False):
        """Create a dataset runner."""
        self.datasets_to_run = params["datasets_to_run"]
        self.extra_flagfile_path = extra_flags_path
        self.verbose_vio = verbose

        self.results_dir = pathlib.Path(params["results_dir"]).absolute()
        self.vocabulary_path = os.path.expandvars(params["vocabulary_path"])
        self.params_dir = os.path.expandvars(params["params_dir"])
        self.dataset_dir = os.path.expandvars(params["dataset_dir"])
        self.executable_path = os.path.expandvars(params["executable_path"])

        self.pipeline_output_dir = self.results_dir / "tmp_output" / "output"
        self.pipeline_output_dir.mkdir(parents=True, exists=True)

    def run_all(self):
        """Run all datasets in experiments file."""
        # Run experiments.
        log.info("Run experiments")
        successful_run = True
        for dataset in tqdm.tqdm(self.datasets_to_run):
            log.info("Run dataset: %s" % dataset["name"])
            if not self.run_dataset(dataset):
                log.info("\033[91m Dataset: %s failed!! \033[00m" % dataset["name"])
                successful_run = False

        return successful_run

    def run_dataset(self, dataset):
        """
        Run a single dataset from an experiments file and save all output.

        This is done for every pipeline requested for the dataset.

        Args:
            dataset: a dataset to run as defined in the experiments yaml file.

        Returns: True if all pipelines for the dataset succeed, False otherwise.
        """
        dataset_name = dataset["name"]

        has_a_pipeline_failed = False
        pipelines_to_run_list = dataset["pipelines"]
        if len(pipelines_to_run_list) == 0:
            log.warning("Not running pipeline...")
        for pipeline_type in pipelines_to_run_list:
            # TODO shouldn't this break when a pipeline has failed? Not necessarily
            # if we want to plot all pipelines except the failing ones.
            print_green("Run pipeline: %s" % pipeline_type)
            pipeline_success = self._run_vio(dataset, pipeline_type)
            if pipeline_success:
                print_green("Successful pipeline run.")
            else:
                log.error("Failed pipeline run!")
                has_a_pipeline_failed = True

        if not has_a_pipeline_failed:
            print_green("All pipeline runs were successful.")

        print_green("Finished evaluation for dataset: " + dataset_name)
        return not has_a_pipeline_failed

    def _run_vio(self, dataset, pipeline_type):
        def _kimera_vio_thread(thread_return, minloglevel=0):
            # Subprocess returns 0 if Ok, any number bigger than 1 if not ok.
            command = "{} \
                    --logtostderr=1 --colorlogtostderr=1 --log_prefix=1 \
                    --minloglevel={} \
                    --dataset_path={}/{} --output_path={} \
                    --params_folder_path={}/{} \
                    --vocabulary_path={} \
                    --flagfile={}/{}/{} --flagfile={}/{}/{} \
                    --flagfile={}/{}/{} --flagfile={}/{}/{} \
                    --flagfile={}/{}/{} --flagfile={}/{} \
                    --visualize=false \
                    --visualize_lmk_type=false \
                    --visualize_mesh=false \
                    --visualize_mesh_with_colored_polygon_clusters=false \
                    --visualize_point_cloud=false \
                    --visualize_convex_hull=false \
                    --visualize_plane_constraints=false \
                    --visualize_planes=false \
                    --visualize_plane_label=false \
                    --visualize_semantic_mesh=false \
                    --visualize_mesh_in_frustum=false \
                    --viz_type=2 \
                    --initial_k={} --final_k={} --use_lcd={} \
                    --log_euroc_gt_data=true --log_output=true".format(
                self.executable_path,
                minloglevel,
                self.dataset_dir,
                dataset["name"],
                self.pipeline_output_dir,
                self.params_dir,
                pipeline_type,
                self.vocabulary_path,
                self.params_dir,
                pipeline_type,
                "flags/stereoVIOEuroc.flags",
                self.params_dir,
                pipeline_type,
                "flags/Mesher.flags",
                self.params_dir,
                pipeline_type,
                "flags/VioBackend.flags",
                self.params_dir,
                pipeline_type,
                "flags/RegularVioBackend.flags",
                self.params_dir,
                pipeline_type,
                "flags/Visualizer3D.flags",
                self.params_dir,
                self.extra_flagfile_path,
                dataset["initial_frame"],
                dataset["final_frame"],
                dataset["use_lcd"],
            )
            # print("Starting Kimera-VIO with command:\n")
            # print(command)
            return_code = subprocess.call(command, shell=True)
            if return_code == 0:
                thread_return["success"] = True
            else:
                thread_return["success"] = False

        import threading
        import time
        import itertools
        import sys

        spinner = itertools.cycle(["-", "/", "|", "\\"])
        thread_return = {"success": False}
        minloglevel = 2  # Set Kimera-VIO verbosity level to ERROR
        if self.verbose_vio:
            minloglevel = 0  # Set Kimera-VIO verbosity level to INFO
        thread = threading.Thread(
            target=_kimera_vio_thread,
            args=(
                thread_return,
                minloglevel,
            ),
        )
        thread.start()
        while thread.is_alive():
            if not self.verbose_vio:
                # If Kimera-VIO is not in verbose mode, the user might think the python
                # script is hanging.
                # So, instead, display a spinner of 80 characters.
                sys.stdout.write(next(spinner) * 10)  # write the next character
                sys.stdout.flush()  # flush stdout buffer (actual character display)
                sys.stdout.write("\b" * 10)  # erase the last written char
            time.sleep(0.100)  # Sleep 100ms while Kimera-VIO is running
        thread.join()

        # Move output files for future evaluation:
        self.move_output_files(pipeline_type, dataset)

        return thread_return["success"]

    def move_output_files(self, pipeline_type, dataset):
        """
        Move output files to proper location.

        Moves output files for a particular pipeline and dataset
        from their temporary logging location during runtime to the evaluation location.

        Args:
            pipeline_type: a pipeline representing a set of parameters to use, as
                defined in the experiments yaml file for the dataset in question.
            dataset: a dataset to run as defined in the experiments yaml file.
        """
        dataset_name = dataset["name"]
        dataset_results_dir = os.path.join(self.results_dir, dataset_name)
        dataset_pipeline_result_dir = os.path.join(dataset_results_dir, pipeline_type)
        log.debug(f"Moving {self.pipeline_output_dir} to {dataset_pipeline_result_dir}")

        if os.path.exists(dataset_pipeline_result_dir):
            shutil.rmtree(dataset_pipeline_result_dir)

        if not os.path.isdir(self.pipeline_output_dir):
            log.info("There is no output directory...")

        shutil.move(self.pipeline_output_dir, dataset_pipeline_result_dir)
        try:
            os.makedirs(self.pipeline_output_dir)
        except Exception:
            log.fatal("Could not mkdir: " + dataset_pipeline_result_dir)
