"""Main library for evaluation."""
from dataclasses import dataclass, field
from typing import Optional, List

import dataclasses
import logging
import pathlib
import yaml


DEFAULT_FLAG_NAMES = [
    "stereoVIOEuroc",
    "Mesher",
    "VioBackend",
    "RegularVioBackend",
    "Visualizer3D",
]


def _normalize_path(path):
    return pathlib.Path(path).expanduser().absolute()


def read_config(cls, *args, **kwargs):
    """Dataclass factory from yaml input."""
    valid_args = [x.name for x in dataclasses.fields(cls)]
    return cls(*args, **{k: v for k, v in kwargs.items() if k in valid_args})


def read_named_config(cls, *args, **kwargs):
    """Read a dataclass where name is the first argument."""
    if "name" not in kwargs:
        raise ValueError(f"name required for loading {cls}")

    valid_args = [x.name for x in dataclasses.fields(cls) if x.name != "name"]
    return cls(
        kwargs["name"], *args, **{k: v for k, v in kwargs.items() if k in valid_args}
    )


@dataclass
class PipelineConfig:
    """Configuration for pipeline."""

    name: str
    param_path: pathlib.Path
    param_name: Optional[str]
    extra_flags_path: Optional[pathlib.Path] = None
    use_visualizer: bool = False

    def __post_init__(self):
        """Normalize paths."""
        self.param_path = _normalize_path(self.param_path)
        if self.extra_flags_path:
            self.extra_flags_path = _normalize_path(self.extra_flags_path)

    @property
    def flag_files(self):
        """Get list of full paths to flag files."""
        flag_path = self.param_path / "flags"
        full_paths = [flag_path / f"{name}.flags" for name in DEFAULT_FLAG_NAMES]
        if self.extra_flags_path:
            full_paths.append(self.extra_flags_path)

        if not self.use_visualizer:
            config_path = pathlib.Path(__file__).absolute().parent / "config"
            full_paths.append(config_path / "NoVisualizer.flag")

        return full_paths

    @property
    def args(self):
        """Get pipeline arguments."""
        return [f"--flagfile={flag_path}" for flag_path in self.flag_files] + [
            f"--params_folder_path={self.param_path / self.param_name}"
        ]


@dataclass
class AnalysisConfig:
    """Configuration for analysis."""

    discard_n_start_poses: int = 0
    discard_n_end_poses: int = 0
    segments: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class SequenceConfig:
    """Configuration for dataset."""

    name: str
    analysis: AnalysisConfig
    initial_frame: int = 0
    final_frame: Optional[int] = None
    use_lcd: bool = False

    @property
    def args(self):
        """Get dataset arguments."""
        args = []
        if self.initial_frame is not None:
            args.append(f"--initial_k={self.initial_frame}")

        if self.final_frame is not None:
            args.append(f"--final_k={self.final_frame}")

        args.append(f"--use_lcd={str(self.use_lcd).lower()}")
        return args


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    executable_path: pathlib.Path
    param_path: pathlib.Path
    dataset_path: pathlib.Path
    sequences: List[SequenceConfig]
    pipelines: List[PipelineConfig]
    vocabulary_path: Optional[pathlib.Path] = None
    extra_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Normalize paths."""
        self.dataset_path = _normalize_path(self.dataset_path)
        self.param_path = _normalize_path(self.param_path)
        self.executable_path = _normalize_path(self.executable_path)
        if self.vocabulary_path:
            self.vocabulary_path = _normalize_path(self.vocabulary_path)
        else:
            self.vocabulary_path = self.param_path / "vocabulary" / "ORBvoc.yml"

    @property
    def args(self):
        """Get base args for config."""
        return [str(self.executable_path), f"--vocabulary_path={self.vocabulary_path}"]

    @classmethod
    def load(cls, config_path, **kwargs):
        """Load from yaml file."""
        with config_path.open("r") as fin:
            params = yaml.safe_load(fin.read())

        def _error_str(name):
            return f"missing '{name}' when parsing config from '{config_path}'"

        exec_path = _read_path_param(params, kwargs, "executable_path")
        if not exec_path:
            logging.error(_error_str("executable_path"))
            return None

        param_path = _read_path_param(params, kwargs, "param_path")
        if not param_path:
            logging.error(_error_str("param_path"))
            return None

        data_path = _read_path_param(params, kwargs, "dataset_path")
        if not data_path:
            logging.error(_error_str("dataset_path"))
            return None

        sequences = [
            read_named_config(
                SequenceConfig,
                read_config(AnalysisConfig, **x.get("analysis", {})),
                **{k: v for k, v in x.items() if k != "analysis"},
            )
            for x in params["datasets"]
        ]
        pipelines = [
            read_named_config(PipelineConfig, param_path, **x)
            for x in params["pipelines"]
        ]

        vocab_path = _read_path_param(params, kwargs, "vocabulary_path")
        extra_args = params.get("extra_args", [])
        return cls(
            exec_path,
            param_path,
            data_path,
            sequences,
            pipelines,
            vocabulary_path=vocab_path,
            extra_args=extra_args,
        )


def _read_path_param(yaml_config, overrides, name):
    import os

    if name in overrides:
        return _normalize_path(overrides[name])

    if name in yaml_config:
        expanded_path = os.path.expandvars(yaml_config[name])
        return _normalize_path(expanded_path)

    return None
