from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, List, Literal, Optional

import pydantic

from ocean_emulators.base_config import BaseConfig
from ocean_emulators.constants import LoaderVersion


class WandBConfig(pydantic.BaseModel):
    mode: str = "disabled"  # online, disabled
    project: str = "3D_ocean_emu_CM4"
    entity: str = "suryadheeshjith"
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class TimeConfig(pydantic.BaseModel):
    start: str
    end: str

    @property
    def time_slice(self) -> slice:
        return slice(self.start, self.end)


class DataConfig(pydantic.BaseModel):
    data_path: str = "CM4_5daily_v0.4.0"
    data_means_path: str = "CM4_5daily_v0.4.0_means"
    data_stds_path: str = "CM4_5daily_v0.4.0_stds"
    scaling_residuals_file: Optional[str] = None
    time_delta: int = 5
    num_workers: int = 4
    hist: int = 1
    loader_version: str = str(LoaderVersion.OM4_EAGER.value)


BlockType = Literal["conv_next_block", "conv_block"]
ActivationType = Literal["relu", "gelu", "capped_gelu"]
NormType = Literal["batch", "instance", "layer"]


class BlockConfig(pydantic.BaseModel):
    block_type: BlockType = "conv_next_block"
    kernel_size: int = 3
    activation: ActivationType = "capped_gelu"
    upscale_factor: int = 4
    norm: NormType = "batch"


class CorrectorConfig(pydantic.BaseModel):
    non_negative_corrector_names: Optional[List[str]] = None


DownSamplingBlocks = Literal["avg_pool", "max_pool"]
UpSamplingBlocks = Literal["bilinear_upsample", "transposed_conv"]


class SamudraConfig(pydantic.BaseModel):
    ch_width: List[int] = [157, 200, 250, 300, 400]
    n_out: int = 77
    dilation: List[int] = [1, 2, 4, 8]
    n_layers: List[int] = [1, 1, 1, 1]
    pred_residuals: bool = False
    last_kernel_size: int = 3
    pad: str = "circular"
    wet: Optional[Any] = None

    # Block configurations
    core_block: BlockConfig
    corrector: CorrectorConfig
    down_sampling_block: DownSamplingBlocks = "avg_pool"
    up_sampling_block: UpSamplingBlocks = "bilinear_upsample"


class DistributedConfig(pydantic.BaseModel):
    dist_url: Optional[str] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    gpu: Optional[int] = None
    dist_backend: Optional[str] = None


class ExperimentConfig(pydantic.BaseModel):
    base_name: str = "train"
    sub_name: str = "cm4_samudra"
    rand_seed: int = 1
    base_output_dir: str = "train"
    gantry: bool = False
    cluster_data_dir: str = "/"
    wandb: WandBConfig

    # Model configuration
    network: str = "Samudra"
    prognostic_vars_key: str = (
        "thermo_dynamic_all"  # all means all levels and _$num means $num levels
    )
    boundary_vars_key: str = "tau_hfds"

    @cached_property
    def name(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f"{timestamp}-{self.base_name}"

    @cached_property
    def output_dir(self) -> Path:
        return Path(self.base_output_dir) / f"{self.name}-{self.sub_name}"

    @cached_property
    def nets_dir(self) -> Path:
        return self.output_dir / "saved_nets"

    @cached_property
    def data_dir(self) -> Path:
        return Path("/") if self.gantry else Path(self.cluster_data_dir)


# See backend.py for how these are turned into concrete devices
TrainBackendConfig = Literal["cpu", "cuda", "nccl", "auto"]
LossType = Literal[
    "mse", "mse_diff_weighted", "mse_cos_weighted", "mse_residual_scaled", "mse_mae"
]


class TrainConfig(BaseConfig):
    # Training parameters
    disk_mode: bool = True
    pin_mem: bool = True
    save_freq: int = 5
    epochs: int = 120
    batch_size: int = 2
    learning_rate: float = 2e-4
    scheduler: bool = False
    loss: LossType = "mse"
    finetune: bool = False
    resume_ckpt_path: Optional[str] = None
    debug: bool = False
    test_using_ema: bool = True
    ema_decay: float = 0.999
    faster_decay_at_start: bool = True
    backend: TrainBackendConfig = "auto"

    # Data parameters at root level
    data_percent: float = 1.0
    data_stride: List[int] = [1]
    steps: List[int] = [4]
    step_transition: List[int] = []
    inference_epochs: List[int] = [-1]
    train_time: TimeConfig = TimeConfig(start="151-01-06", end="306-01-01")
    val_time: TimeConfig = TimeConfig(start="306-01-01", end="311-01-01")
    inference_times: List[TimeConfig] = []

    # Config components
    experiment: ExperimentConfig
    data: DataConfig
    samudra: SamudraConfig

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


# See backend.py for how these are turned into concrete devices
EvalBackendConfig = Literal["cpu", "cuda", "auto"]


class EvalConfig(BaseConfig):
    # Basic parameters
    debug: bool = False
    save_zarr: bool = False
    disk_mode: bool = True
    ckpt_path: str = ""
    num_model_steps_forward: int = 200
    backend: EvalBackendConfig = "auto"

    # Config components
    inference_time: TimeConfig = TimeConfig(start="311-01-01", end="351-01-01")
    experiment: ExperimentConfig
    data: DataConfig
    samudra: SamudraConfig

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)
