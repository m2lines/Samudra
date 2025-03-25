from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from ocean_emulators.constants import LoaderVersion

if TYPE_CHECKING:
    from ocean_emulators.eval import Eval
    from ocean_emulators.train import Trainer


# See backend.py for how these are turned into concrete devices
class TrainBackendConfig(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NCCL = "nccl"
    AUTO = "auto"


class BlockType(Enum):
    conv_next_block = "conv_next_block"
    conv_block = "conv_block"


class BlockNorm(Enum):
    batch = "batch"
    instance = "instance"
    layer = "layer"


class BlockActivation(Enum):
    relu = "relu"
    gelu = "gelu"
    capped_gelu = "capped_gelu"


class DownSampleBlockType(Enum):
    avg_pool = "avg_pool"
    max_pool = "max_pool"


class UpSampleBlockType(Enum):
    bilinear_upsample = "bilinear_upsample"
    transposed_conv = "transposed_conv"


class EvalBackendConfig(Enum):
    cpu = "cpu"
    cuda = "cuda"
    auto = "auto"


@dataclass
class WandBConfig:
    project: str
    entity: str
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class TimeConfig:
    start_time: str
    end_time: str


@dataclass
class DataConfig:
    data_path: str
    data_means_path: str
    data_stds_path: str
    scaling_residuals_file: Optional[str] = None
    time_delta: int = 5
    num_workers: int = 4
    hist: int = 1
    loader_version: LoaderVersion = LoaderVersion.OM4_EAGER


@dataclass
class BlockConfig:
    block_type: BlockType = "conv_next_block"
    kernel_size: int = 3
    activation: BlockActivation = "capped_gelu"
    upscale_factor: int = 4
    norm: BlockNorm = "batch"


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
    down_sampling_block: DownSampleBlockType = "avg_pool"
    up_sampling_block: UpSampleBlockType = "bilinear_upsample"


@dataclass
class ExperimentConfig:
    base_name: str
    sub_name: str
    data_dir: str  # Root directory which data paths can be relative to
    rand_seed: int = 1
    base_output_dir: str = "train"
    wandb: WandBConfig | None = None  # None means disabled

    # Model configuration
    network: str = "Samudra"
    prognostic_vars_key: str = (
        "thermo_dynamic_all"  # all means all levels and _$num means $num levels
    )
    boundary_vars_key: str = "tau_hfds"

    def __post_init__(self):
        # TODO(jder): remove
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.name = f"{timestamp}-{self.base_name}"
        self.output_dir = Path(self.base_output_dir) / f"{self.name}-{self.sub_name}"
        self.nets_dir = self.output_dir / "saved_nets"


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

    def build(self) -> "Trainer":
        from ocean_emulators.train import Trainer

        Trainer(self)

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


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

    def build(self) -> "Eval":
        from ocean_emulators.eval import Eval

        Eval(self)

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)
