from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ocean_emulators.config_base import BaseConfig, TopLevelConfig
from ocean_emulators.constants import LoaderVersion


class WandBConfig(BaseConfig):
    mode: str = "disabled"  # online, disabled
    project: str = "3D_ocean_emu_CM4"
    entity: str = "suryadheeshjith"
    group: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


class TimeConfig(BaseConfig):
    start: str
    end: str

    @property
    def time_slice(self) -> slice:
        return slice(self.start, self.end)


class DataConfig(BaseConfig):
    data_path: str = "CM4_5daily_v0.4.0"
    data_means_path: str = "CM4_5daily_v0.4.0_means"
    data_stds_path: str = "CM4_5daily_v0.4.0_stds"
    scaling_residuals_file: str | None = None
    static_data_vars: list[str] | None = None
    num_workers: int = 4
    hist: int = 1
    loader_version: str = str(LoaderVersion.OM4_EAGER.value)
    normalize_before_mask: bool = True
    masked_fill_value: float = 0.0


BlockType = Literal["conv_next_block", "conv_block"]
ActivationType = Literal["relu", "gelu", "capped_gelu"]
NormType = Literal["batch", "instance", "layer"]


class BlockConfig(BaseConfig):
    block_type: BlockType = "conv_next_block"
    kernel_size: int = 3
    activation: ActivationType = "capped_gelu"
    upscale_factor: int = 4
    norm: NormType = "batch"


class CorrectorConfig(BaseConfig):
    non_negative_corrector_names: list[str] | None = None
    ocean_heat_corrector: bool = False


DownSamplingBlocks = Literal["avg_pool", "max_pool"]
UpSamplingBlocks = Literal["bilinear_upsample", "transposed_conv"]
Checkpointing = Literal["blocks", "simple"]


class SamudraConfig(BaseConfig):
    ch_width: list[int] = [157, 200, 250, 300, 400]
    n_out: int = 77
    dilation: list[int] = [1, 2, 4, 8]
    n_layers: list[int] = [1, 1, 1, 1]
    pred_residuals: bool = False
    last_kernel_size: int = 3
    pad: str = "circular"
    wet: Any | None = None

    # Block configurations
    core_block: BlockConfig = BlockConfig()
    corrector: CorrectorConfig = CorrectorConfig()
    down_sampling_block: DownSamplingBlocks = "avg_pool"
    up_sampling_block: UpSamplingBlocks = "bilinear_upsample"

    checkpointing: Checkpointing | None = Field(
        default=None,
        description="Checkpointing strategy for the model; "
        "'blocks' for recomputing each CoreBlock, "
        "'simple' for recomputing only cheap layers like scales and nonlinearities",
    )


class DistributedConfig(BaseConfig):
    dist_url: str | None = None
    world_size: int | None = None
    rank: int | None = None
    gpu: int | None = None
    dist_backend: str | None = None


class ExperimentConfig(BaseConfig):
    name: str = "cm4_samudra"
    rand_seed: int = 1
    base_output_dir: str = "train"
    gantry: bool = False
    # we require this to be set by the user but have optional here
    # so we can leave it out of config files
    cluster_data_dir: str | None = None
    wandb: WandBConfig

    # Model configuration
    network: str = "Samudra"
    prognostic_vars_key: str = (
        "thermo_dynamic_all"  # all means all levels and _$num means $num levels
    )
    boundary_vars_key: str = "tau_hfds"

    @cached_property
    def output_dir(self) -> Path:
        return Path(self.base_output_dir) / f"{self.name}"

    @cached_property
    def nets_dir(self) -> Path:
        return self.output_dir / "saved_nets"

    @cached_property
    def data_dir(self) -> Path:
        if self.gantry:
            return Path("/")
        else:
            if self.cluster_data_dir is None:
                raise ValueError(
                    "cluster_data_dir must be set, try"
                    " --experiment.cluster_data_dir=path/to/data"
                )
            else:
                return Path(self.cluster_data_dir)


# See backend.py for how these are turned into concrete devices
TrainBackendConfig = Literal["cpu", "cuda", "nccl", "auto"]
LossType = Literal[
    "mse", "mse_diff_weighted", "mse_cos_weighted", "mse_residual_scaled", "mse_mae"
]


class TrainConfig(TopLevelConfig):
    # Training parameters
    disk_mode: bool = True
    pin_mem: bool = True
    save_freq: int = 5
    epochs: int = 120
    preemptible: bool = True
    batch_size: int = 2
    learning_rate: float = 2e-4
    scheduler: bool = False
    loss: LossType = "mse"
    finetune: bool = False
    resume_ckpt_path: str | None = None
    debug: bool = False
    test_using_ema: bool = True
    ema_decay: float = 0.999
    faster_decay_at_start: bool = True
    backend: TrainBackendConfig = "auto"

    # Data parameters at root level
    data_percent: float = 1.0
    data_stride: list[int] = [1]
    steps: list[int] = [4]
    step_transition: list[int] = []
    inference_epochs: list[int] = [-1]
    train_time: TimeConfig = TimeConfig(start="151-01-06", end="306-01-01")
    val_time: TimeConfig = TimeConfig(start="306-01-01", end="311-01-01")
    inference_times: list[TimeConfig] = []

    # Config components
    experiment: ExperimentConfig
    data: DataConfig
    samudra: SamudraConfig

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


# See backend.py for how these are turned into concrete devices
EvalBackendConfig = Literal["cpu", "cuda", "auto"]


class EvalConfig(TopLevelConfig):
    # Basic parameters
    debug: bool = False
    save_zarr: bool = False
    disk_mode: bool = True
    # we require this to be set by the user but have optional here
    # so we can leave it out of config files
    ckpt_path: str | None = None
    num_model_steps_forward: int = 200
    backend: EvalBackendConfig = "auto"

    # Config components
    inference_time: TimeConfig = TimeConfig(start="311-01-01", end="351-01-01")
    experiment: ExperimentConfig
    data: DataConfig
    samudra: SamudraConfig = SamudraConfig()

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


AnyTopLevelConfig = TrainConfig | EvalConfig
