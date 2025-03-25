from dataclasses import dataclass, field
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from dacite import Config as DaciteConfig
from dacite import from_dict

from ocean_emulators.constants import LoaderVersion


@dataclass
class WandBConfig:
    mode: str = "disabled"  # online, disabled
    project: str = "3D_ocean_emu_CM4"
    entity: str = "suryadheeshjith"
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class TimeConfig:
    start_time: str
    end_time: str

    def __post_init__(self):
        self.time_slice = slice(self.start_time, self.end_time)


@dataclass
class DataConfig:
    data_path: str = "CM4_5daily_v0.4.0"
    data_means_path: str = "CM4_5daily_v0.4.0_means"
    data_stds_path: str = "CM4_5daily_v0.4.0_stds"
    scaling_residuals_file: Optional[str] = None
    time_delta: int = 5
    num_workers: int = 4
    hist: int = 1
    loader_version: LoaderVersion = LoaderVersion.OM4_EAGER


@dataclass
class BlockConfig:
    block_type: str = "conv_next_block"  # conv_next_block, conv_block
    kernel_size: int = 3
    activation: str = "capped_gelu"  # relu, gelu, capped_gelu
    upscale_factor: int = 4
    norm: str = "batch"  # batch, instance, layer


@dataclass
class CorrectorConfig:
    non_negative_corrector_names: Optional[List[str]] = None


@dataclass
class SamudraConfig:
    ch_width: List[int] = field(default_factory=lambda: [157, 200, 250, 300, 400])
    n_out: int = 77
    dilation: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layers: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    pred_residuals: bool = False
    last_kernel_size: int = 3
    pad: str = "circular"
    wet: Optional[Any] = None

    # Block configurations
    core_block: BlockConfig = field(default_factory=BlockConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    down_sampling_block: str = "avg_pool"  # avg_pool, max_pool
    up_sampling_block: str = "bilinear_upsample"  # bilinear_upsample, transposed_conv


@dataclass
class DistributedConfig:
    dist_url: Optional[str] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    gpu: Optional[int] = None
    dist_backend: Optional[str] = None


@dataclass
class ExperimentConfig:
    base_name: str = "train"
    sub_name: str = "cm4_samudra"
    rand_seed: int = 1
    base_output_dir: str = "train"
    gantry: bool = False
    cluster_data_dir: str = "/"
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Model configuration
    network: str = "Samudra"
    prognostic_vars_key: str = (
        "thermo_dynamic_all"  # all means all levels and _$num means $num levels
    )
    boundary_vars_key: str = "tau_hfds"

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.name = f"{timestamp}-{self.base_name}"
        self.output_dir = Path(self.base_output_dir) / f"{self.name}-{self.sub_name}"
        self.nets_dir = self.output_dir / "saved_nets"
        if self.gantry:
            self.data_dir = Path("/")
        else:
            self.data_dir = Path(self.cluster_data_dir)


# See backend.py for how these are turned into concrete devices
TrainBackendConfig = Literal["cpu", "cuda", "nccl", "auto"]


@dataclass
class TrainConfig:
    # Training parameters
    disk_mode: bool = True
    pin_mem: bool = True
    save_freq: int = 5
    epochs: int = 120
    batch_size: int = 2
    learning_rate: float = 2e-4
    scheduler: bool = False
    loss: str = "mse"
    finetune: bool = False
    resume_ckpt_path: Optional[str] = None
    debug: bool = False
    backend: TrainBackendConfig = "auto"

    # Data parameters at root level
    data_percent: float = 1.0
    data_stride: List[int] = field(default_factory=lambda: [1])
    steps: List[int] = field(default_factory=lambda: [4])
    step_transition: List[int] = field(default_factory=lambda: [])
    inference_epochs: List[int] = field(default_factory=lambda: [-1])
    train: TimeConfig = field(
        default_factory=lambda: TimeConfig("151-01-06", "306-01-01")
    )
    val: TimeConfig = field(
        default_factory=lambda: TimeConfig("306-01-01", "311-01-01")
    )
    inference: List[TimeConfig] = field(default_factory=list)

    # Config components
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    samudra: SamudraConfig = field(default_factory=SamudraConfig)

    @classmethod
    def from_yaml(
        cls, yaml_path: str | PathLike, overrides: Optional[Dict[str, Any]] = None
    ) -> "TrainConfig":
        """Load config from YAML with strict validation using dacite."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # TODO: This is a hack to allow for overrides of the sub_name
        if overrides and "sub_name" in overrides.keys():
            config_dict["experiment"]["sub_name"] = overrides["sub_name"]

        return from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(strict=True, check_types=True, cast=[Path]),
        )

    def save_yaml(self, save_path: str):
        """Save config to YAML file."""
        config_dict = {
            "debug": self.debug,
            "disk_mode": self.disk_mode,
            "pin_mem": self.pin_mem,
            "save_freq": self.save_freq,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,
            "loss": self.loss,
            "finetune": self.finetune,
            "resume_ckpt_path": self.resume_ckpt_path,
            "backend": self.backend,
            "data_percent": self.data_percent,
            "data_stride": self.data_stride,
            "steps": self.steps,
            "step_transition": self.step_transition,
            "inference_epochs": self.inference_epochs,
            "train": self.train.__dict__,
            "val": self.val.__dict__,
            "inference": [t.__dict__ for t in self.inference],
            "experiment": self.experiment.__dict__,
            "data": self.data.__dict__,
            "samudra": {
                **self.samudra.__dict__,
                "core_block": self.samudra.core_block.__dict__,
                "corrector": self.samudra.corrector.__dict__,
            },
        }

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


# See backend.py for how these are turned into concrete devices
EvalBackendConfig = Literal["cpu", "cuda", "auto"]


@dataclass
class EvalConfig:
    # Basic parameters
    debug: bool = False
    save_zarr: bool = False
    disk_mode: bool = True
    ckpt_path: str = ""
    num_model_steps_forward: int = 200
    record_every: int = 10
    backend: EvalBackendConfig = "auto"

    # Config components
    inference: TimeConfig = field(
        default_factory=lambda: TimeConfig("311-01-01", "351-01-01")
    )
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    samudra: SamudraConfig = field(default_factory=SamudraConfig)

    @classmethod
    def from_yaml(
        cls, yaml_path: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "EvalConfig":
        """Load config from YAML with strict validation using dacite."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle sub_name override if provided
        if overrides:
            if "sub_name" in overrides.keys():
                config_dict["experiment"]["sub_name"] = overrides["sub_name"]
            if "ckpt_path" in overrides.keys():
                config_dict["ckpt_path"] = overrides["ckpt_path"]
            if "save_zarr" in overrides.keys():
                config_dict["save_zarr"] = overrides["save_zarr"]

        return from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(strict=True, check_types=True, cast=[Path]),
        )

    def save_yaml(self, save_path: str):
        """Save config to YAML file."""
        config_dict = {
            "debug": self.debug,
            "save_zarr": self.save_zarr,
            "disk_mode": self.disk_mode,
            "ckpt_path": self.ckpt_path,
            "num_model_steps_forward": self.num_model_steps_forward,
            "record_every": self.record_every,
            "inference": self.inference.__dict__,
            "experiment": self.experiment.__dict__,
            "data": self.data.__dict__,
            "samudra": {
                **self.samudra.__dict__,
                "core_block": self.samudra.core_block.__dict__,
                "corrector": self.samudra.corrector.__dict__,
            },
        }

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)
