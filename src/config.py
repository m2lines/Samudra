from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import yaml
from dacite import Config as DaciteConfig
from dacite import from_dict


@dataclass
class WandBConfig:
    mode: str = "disabled"  # online, disabled
    project: str = "3D_ocean_emu_CM4"
    entity: str = "suryadheeshjith"
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class TrainingConfig:
    distributed: bool = True
    disk_mode: bool = True
    num_workers: int = 4
    pin_mem: bool = True
    save_freq: int = 5
    epochs: int = 70
    batch_size: int = 32
    learning_rate: float = 2e-4
    scheduler: bool = False
    loss: str = "mse"
    network: str = "convnextunet"
    exp_num_in: str = "3D_all"
    exp_num_extra: str = "3D_all"
    exp_num_out: str = "3D_all"
    finetune: bool = False
    dist_url: Optional[str] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    gpu: Optional[int] = None
    dist_backend: Optional[str] = None
    resume_ckpt_path: Optional[str] = None


@dataclass
class TimeConfig:
    start_time: str
    end_time: str


@dataclass
class DataConfig:
    wet_file: str = "CM4_5daily_v0.4.0_wetmask"
    data_path: str = "CM4_5daily_v0.4.0"
    data_means_path: str = "CM4_5daily_v0.4.0_means"
    data_stds_path: str = "CM4_5daily_v0.4.0_stds"
    scaling_residuals_file: Optional[str] = None
    depth_mode: str = "all"
    data_stride: List[int] = field(default_factory=lambda: [1])
    steps: List[int] = field(default_factory=lambda: [4])
    step_transition: List[int] = field(default_factory=lambda: [])
    hist: int = 0
    data_percent: float = 1.0
    time_delta: int = 5
    train: TimeConfig = field(
        default_factory=lambda: TimeConfig("151-01-06", "306-01-01")
    )
    val: TimeConfig = field(
        default_factory=lambda: TimeConfig("306-01-01", "311-01-01")
    )
    inference: List[TimeConfig] = field(default_factory=list)
    inference_epochs: List[int] = field(default_factory=list)


@dataclass
class BlockConfig:
    block_type: str = "conv_next_block"  # conv_next_block, conv_block
    kernel_size: int = 3
    activation: str = "capped_gelu"  # relu, gelu, capped_gelu
    upscale_factor: int = 4
    norm: str = "batch"  # batch, instance, layer


@dataclass
class UNetConfig:
    # Core architecture
    ch_width: List[int] = field(default_factory=lambda: [80, 24, 45, 90, 180])
    n_out: int = 77
    dilation: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layers: List[int] = field(default_factory=lambda: [1, 1, 1, 1])

    # Block configurations
    core_block: BlockConfig = field(default_factory=BlockConfig)
    down_sampling_block: str = "avg_pool"  # avg_pool, max_pool
    up_sampling_block: str = "bilinear_upsample"  # bilinear_upsample, transposed_conv

    # Other settings
    pred_residuals: bool = False
    last_kernel_size: int = 3
    pad: str = "circular"
    wet: Optional[Any] = None  # Will be set during training
    hist: int = 0  # Will be set during training


@dataclass
class Config:
    wandb: WandBConfig
    training: TrainingConfig
    data: DataConfig
    unet: UNetConfig
    name: str = "train"
    sub_name: str = "cm4_samudra_thermo"

    rand_seed: int = 1
    base_output_dir: str = "train_3D"
    debug: bool = False
    gantry: bool = False
    cluster_data_dir: str = "/"

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.name = f"{timestamp}-{self.name}"
        self.output_dir = Path(self.base_output_dir) / f"{self.name}-{self.sub_name}"
        self.nets_dir = self.output_dir / "saved_nets"
        if self.gantry:
            self.data_dir = Path("/")
        else:
            self.data_dir = Path(self.cluster_data_dir)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML with strict validation using dacite."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return from_dict(
            data_class=cls,
            data=config_dict,
            config=DaciteConfig(strict=True, check_types=True, cast=[Path]),
        )

    def save_yaml(self, save_path: str):
        """Save config to YAML file."""
        config_dict = {
            "wandb": self.wandb.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "unet": {**self.unet.__dict__, "core_block": self.unet.core_block.__dict__},
            "name": self.name,
            "sub_name": self.sub_name,
            "rand_seed": self.rand_seed,
            "base_output_dir": self.base_output_dir,
            "debug": self.debug,
            "gantry": self.gantry,
            "cluster_data_dir": self.cluster_data_dir,
        }

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
