"""Main library for evaluation."""
from dataclasses import dataclass, field
from typing import Optional, List

import itertools
import logging
import pathlib
import subprocess
import shutil
import sys
import time
import yaml


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

    @classmethod
    def from_yaml(cls, config_str):
        """Create a configuration from a yaml string."""
        return cls(**yaml.safe_load(config_str))

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
    dataset_name: str
    initial_frame: int = 0
    final_frame: Optional[int] = None
    use_lcd: bool = False
    extra_args: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, dataset_root, config_str):
        """Create a configuration from a yaml string."""
        return cls(**yaml.safe_load(config_str))

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
    config, dataset, output_path, minloglevel=0, spinner_width=10, spin_time=0.1
):
    spinner = itertools.cycle(["-", "/", "|", "\\"])
    args = config.base_args + dataset.args + [f"--output_path={output_path}"]

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

    def __init__(self, result_path: pathlib.Path, params):
        """Create a dataset runner."""
        self.results_path = pathlib.Path(params["results_dir"]).absolute()

    def run_all(
        self,
        datasets: List[DatasetConfig],
        pipelines: List[PipelineConfig],
        allow_removal=False,
        minloglevel=2,
        **kwargs,
    ):
        """
        Run all datasets and pipelines.

        Args:
            datasets: datasets to run
            pipelines: pipelines to run

        Returns:
            Dict[str, Dict[str, bool]]: Pipeline results
        """
        status = {}
        logging.info("Runing experiments...")
        for dataset in datasets:
            logging.info(f"Runing dataset '{dataset.name}'...")
            dataset_status = {}
            for pipeline in pipelines:
                output_path = self.result_path / dataset.name / pipeline.name
                if output_path.exists():
                    if allow_removal:
                        shutil.rmtree(output_path)
                    else:
                        status[dataset.name] = False
                        continue

                output_path.mkdir(parents=True, exist_ok=False)

                logging.info(f"Running pipeline '{pipeline}'...")
                dataset_status[pipeline.name] = _run_vio(
                    dataset, pipeline, output_path, minloglevel=minloglevel
                )

            status[dataset.name] = dataset_status

        return status
