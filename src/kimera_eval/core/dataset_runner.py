"""Main library for evaluation."""
from dataclasses import dataclass, field
from typing import Optional, List

import dataclasses
import itertools
import logging
import pathlib
import subprocess
import shutil
import sys
import time


DEFAULT_FLAG_NAMES = [
    "stereoVIOEuroc",
    "Mesher",
    "VioBackend",
    "RegularVioBackend",
    "Visualizer3D",
]

VISUALIZER_FLAGS = [
    "--visualize=false",
    "--visualize_lmk_type=false",
    "--visualize_mesh=false",
    "--visualize_mesh_with_colored_polygon_clusters=false",
    "--visualize_point_cloud=false",
    "--visualize_convex_hull=false",
    "--visualize_plane_constraints=false",
    "--visualize_planes=false",
    "--visualize_plane_label=false",
    "--visualize_semantic_mesh=false",
    "--visualize_mesh_in_frustum=false",
    "--viz_type=2",
]


def _normalize_path(path):
    return path.expanduser().absolute()


@dataclass
class PipelineConfig:
    """Configuration for pipeline."""

    executable_path: pathlib.Path
    vocabulary_path: pathlib.Path
    params_path: pathlib.Path
    verbose: bool = False
    extra_flags_path: Optional[pathlib.Path] = None

    def __post_init__(self):
        """Normalize paths."""
        self.executable_path = _normalize_path(self.executable_path)
        self.vocabulary_path = _normalize_path(self.vocabulary_path)
        self.params_path = _normalize_path(self.params_path)
        self.extra_flags_path = _normalize_path(self.extra_flags_path)

    @property
    def flag_files(self):
        """Get list of full paths to flag files."""
        flag_path = self.params_path / "flags"
        full_paths = [flag_path / f"{name}.flags" for name in DEFAULT_FLAG_NAMES]
        if self.extra_flags_path:
            full_paths.append(self.extra_flags_path)

        return full_paths

    @property
    def base_args(self):
        """Get base args for config."""
        return [
            str(self.executable_path),
            f"--params_folder_path={self.params_path}",
            f"--vocabulary_path={self.vocabulary_path}",
        ] + [f"--flagfile={flag_path}" for flag_path in self.flag_files]


# TODO(nathan): ["--log_euroc_gt_data=true"]
@dataclass
class DatasetConfig:
    """Configuration for dataset."""

    dataset_root: pathlib.Path
    name: str
    initial_frame: int = 0
    final_frame: Optional[int] = None
    use_lcd: bool = False
    extra_args: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, dataset_root, **kwargs):
        """Parse a config dictionary."""
        if not kwargs["name"]:
            raise ValueError("name required for dataset")

        valid_args = [x.name for x in dataclasses.fields(cls)]
        return cls(dataset_root, {k: v for k, v in kwargs.items() if k in valid_args})

    @property
    def args(self):
        """Get dataset arguments."""
        return [
            f"--dataset_path={self.dataset_root / self.dataset_name}"
            f"--initial_k={self.initial_frame}",
            f"--final_k={self.final_frame}",
            f"--use_lcd={self.use_lcd}",
        ]


def _run_vio(
    config, sequence, output_path, minloglevel=0, spinner_width=10, spin_time=0.1
):
    spinner = itertools.cycle(["-", "/", "|", "\\"])
    args = config.base_args + sequence.args + [f"--output_path={output_path}"]

    args += [
        "--logtostderr=1",
        "--colorlogtostderr=1",
        "--log_prefix=1",
        "--log_output=true",
        f"--minloglevel={minloglevel}",
    ]

    pipe = subprocess.Popen(args)
    while pipe.poll() is None:
        if minloglevel > 0:
            # display spinner to show progress
            sys.stdout.write(next(spinner) * spinner_width)
            sys.stdout.flush()
            sys.stdout.write("\b" * spinner_width)

        time.sleep(spin_time)

    return pipe.wait() == 0


class DatasetRunner:
    """DatasetRunner is used to run the pipeline on datasets."""

    def __init__(self, dataset_path: pathlib.Path, result_path: pathlib.Path, params):
        """
        Create a dataset runner.

        Args:
            dataset_path: Path to dataset
            result_path: Path to save results
            params: Experiment parameters
        """
        self.results_path = result_path
        self.sequences = [
            DatasetConfig.from_config(dataset_path, **x) for x in params["datasets"]
        ]
        self.pipelines = [PipelineConfig(**x) for x in params["pipelines"]]

    def run_all(self, allow_removal=False, minloglevel=2, **kwargs):
        """
        Run all datasets and pipelines.

        Args:
            allow_removal: Allow removal of outputs
            minloglevel: Min log level setting for Kimera-VIO
            **kwargs: Arguments to be passed to _run_vio

        Returns:
            Dict[str, Dict[str, bool]]: Pipeline results
        """
        status = {}
        logging.info("Runing experiments...")
        for sequence in self.sequences:
            logging.info(f"Runing dataset '{sequence.name}'...")
            dataset_status = {}
            for pipeline in self.pipelines:
                output_path = self.result_path / sequence.name / pipeline.name
                if output_path.exists():
                    if allow_removal:
                        shutil.rmtree(output_path)
                    else:
                        status[sequence.name] = False
                        continue

                output_path.mkdir(parents=True, exist_ok=False)

                logging.info(f"Running pipeline '{pipeline}'...")
                dataset_status[pipeline.name] = _run_vio(
                    sequence, pipeline, output_path, minloglevel=minloglevel
                )

            status[sequence.name] = dataset_status

        return status
