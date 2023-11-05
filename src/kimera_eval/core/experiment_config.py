"""Main library for evaluation."""
from dataclasses import dataclass, field
from typing import Optional, List

import dataclasses
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


def _read_named_config(cls, *args, **kwargs):
    if not kwargs["name"]:
        raise ValueError("name required for dataset")

    valid_args = [x.name for x in dataclasses.fields(cls)]
    return cls(
        kwargs["name"], *args, {k: v for k, v in kwargs.items() if k in valid_args}
    )


@dataclass
class PipelineConfig:
    """Configuration for pipeline."""

    name: str
    param_name: Optional[str]
    extra_flags_path: Optional[pathlib.Path] = None

    def __post_init__(self):
        """Normalize paths."""
        if self.extra_flags_path:
            self.extra_flags_path = _normalize_path(self.extra_flags_path)

    @property
    def flag_files(self):
        """Get list of full paths to flag files."""
        flag_path = self.params_path / "flags"
        full_paths = [flag_path / f"{name}.flags" for name in DEFAULT_FLAG_NAMES]
        if self.extra_flags_path:
            full_paths.append(self.extra_flags_path)

        return full_paths


@dataclass
class SequenceConfig:
    """Configuration for dataset."""

    name: str
    initial_frame: int = 0
    final_frame: Optional[int] = None
    use_lcd: bool = False

    @property
    def args(self):
        """Get dataset arguments."""
        return [
            f"--initial_k={self.initial_frame}",
            f"--final_k={self.final_frame}",
            f"--use_lcd={self.use_lcd}",
        ]


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    executable_path: pathlib.Path
    param_path: pathlib.Path
    dataset_path: pathlib.Path
    pipelines: List[PipelineConfig]
    sequences: List[SequenceConfig]
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

    def get_args(self, config: PipelineConfig):
        """Get base args for config."""
        args = [str(self.executable_path)]
        args += [f"--flagfile={flag_path}" for flag_path in config.flag_files]
        args.append(f"--vocabulary_path={self.vocabulary_path}")
        args.append(f"--params_folder_path={self.params_path / config.param_name}")
        return args

    @classmethod
    def load(cls, config_path, **kwargs):
        """Load from yaml file."""
        with config_path.open("r") as fin:
            params = yaml.safe_load(fin.read())

        sequences = [
            _read_named_config(SequenceConfig, **x) for x in params["datasets"]
        ]
        pipelines = [
            _read_named_config(PipelineConfig, **x) for x in params["pipelines"]
        ]

        extra_args = params.get("extra_args", [])
        return cls(
            _read_path_param(params, kwargs, "executable_path"),
            _read_path_param(params, kwargs, "param_path"),
            _read_path_param(params, kwargs, "dataset_path"),
            sequences,
            pipelines,
            extra_args,
            _read_path_param(params, kwargs, "vocabulary_path", required=False),
        )


def _read_path_param(yaml_config, overrides, name, required=True):
    import os

    if name in overrides:
        return _normalize_path(overrides[name])

    if name in yaml_config:
        expanded_path = os.path.expandvars(yaml_config[name])
        return _normalize_path(expanded_path)

    if not required:
        return None

    raise ValueError(f"missing required parameter '{name}'")
