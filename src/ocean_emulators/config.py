from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import cftime
import torch
from pydantic import (
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
    field_validator,
)

from ocean_emulators.config_base import BaseConfig, TopLevelConfig
from ocean_emulators.constants import BoundaryVarNames, LoaderVersion
from ocean_emulators.utils.data import DataContainer, DataSource, validate_data
from ocean_emulators.utils.dropout import StochasticDepthManager
from ocean_emulators.utils.location import LocalLocation, Location, ResolvedLocation
from ocean_emulators.utils.profiler import Profiler
from ocean_emulators.utils.schedule import SchedulerConfig


class WandBConfig(BaseConfig):
    mode: Literal["online", "disabled"] = "disabled"
    project: str = "3D_ocean_emu_CM4"
    entity: str = "suryadheeshjith"
    group: str | None = None
    tags: list[str] | None = None
    notes: str | None = None


class JulianDate:
    """Represents a Julian date as a cftime.datetime at noon on the relevant day.

    This is the format the OM4 data uses, so we match that here.
    TODO(jder): probably worth asserting the date format when opening the data.
    """

    datetime: cftime.datetime

    def __init__(self, s: str):
        datetime = cftime.datetime.strptime(s, "%Y-%m-%d", calendar="julian")
        datetime = datetime.replace(hour=12)
        self.datetime = datetime

    def __str__(self) -> str:
        return self.datetime.strftime("%Y-%m-%d")


def _julian_date_validator(value: str | JulianDate) -> JulianDate:
    """Pydantic validator which must handle strings or JulianDate objects."""
    if isinstance(value, str):
        return JulianDate(value)
    else:
        return value


"""Represents a Julian date as a string."""
DateConfig = Annotated[
    JulianDate,
    PlainValidator(_julian_date_validator),
    PlainSerializer(JulianDate.__str__),
    WithJsonSchema({"type": "string", "format": "date"}),
]


class TimeConfig(BaseConfig):
    """Represents a time slice of the data.

    Endpoints are Julian dates (not times) but cftime stores them in datetimes.
    The final endpoint is exclusive.
    """

    start: DateConfig
    end: DateConfig

    @property
    def time_slice(self) -> slice:
        return slice(self.start.datetime, self.end.datetime)

    def overlaps(self, other: Self) -> bool:
        """Check if this time range overlaps with another time range.

        Args:
            other: Another TimeConfig to check for overlap

        Returns:
            True if the time ranges overlap, False otherwise
        """
        return (
            self.start.datetime < other.end.datetime
            and self.end.datetime > other.start.datetime
        )

    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


LOCATION_DOCS = (
    "Use a string relative to the `data_root` or use a structured location "
    "see location.py for possible types."
)


class DataConfig(BaseConfig):
    data_location: Location = Field(
        description="Location of the data; " + LOCATION_DOCS
    )
    data_means_location: Location = Field(
        description="Location of the data means; " + LOCATION_DOCS
    )
    data_stds_location: Location = Field(
        description="Location of the data standard deviations; " + LOCATION_DOCS
    )
    scaling_residuals_file: Location | None = Field(
        description="Location of the scaling residuals file; " + LOCATION_DOCS,
        default=None,
    )
    static_data_vars: list[str] | None = None
    num_workers: int = 4
    hist: int = 1
    loader_version: str = str(LoaderVersion.OM4_TORCH.value)
    normalize_before_mask: bool = True
    masked_fill_value: float = 0.0
    concurrent_compute: bool = False

    def build(
        self,
        data_root: ResolvedLocation,
        boundary_var_names: BoundaryVarNames,
    ) -> DataContainer:
        loader_version = LoaderVersion(self.loader_version)
        use_dask = loader_version != LoaderVersion.OM4_TORCH

        data_location = data_root.resolve(self.data_location)
        means_location = data_root.resolve(self.data_means_location)
        stds_location = data_root.resolve(self.data_stds_location)

        source = DataSource.from_locations(
            data_location=data_location,
            means_location=means_location,
            stds_location=stds_location,
            use_dask=use_dask,
        )
        source = validate_data(source, boundary_var_names, self.static_data_vars)

        if use_dask:
            # If we're already using dask, we don't need a second source
            source_using_dask = source
        else:
            # If we're not using dask for the main source, create a separate one
            source_using_dask = DataSource.from_locations(
                data_location=data_location,
                means_location=means_location,
                stds_location=stds_location,
                use_dask=True,
            )
            source_using_dask = validate_data(
                source_using_dask, boundary_var_names, self.static_data_vars
            )

        if self.scaling_residuals_file is not None:
            scaling_residuals_location = data_root.resolve(self.scaling_residuals_file)
            scaling_residuals = scaling_residuals_location.open()
        else:
            scaling_residuals_location = None
            scaling_residuals = None

        static_data = (
            source.data[self.static_data_vars]
            if self.static_data_vars is not None
            else None
        )

        supports_fork = all(
            location is None or location.supports_fork
            for location in [
                data_location,
                means_location,
                stds_location,
                scaling_residuals_location,
            ]
        )
        return DataContainer(
            source,
            source_using_dask,
            loader_version,
            supports_fork,
            scaling_residuals,
            static_data,
        )


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


DropoutSchedule = Literal["early_only", "late_only", "constant"]


class StochasticDepthConfig(BaseConfig):
    """Configuration for early dropout and stochastic depth.

    Based on:
    - "Early Dropout: Dropping layers to reduce underfitting" (2303.01500)
    - "Deep Networks with Stochastic Depth" (1603.09382)
    - "A ConvNet for the 2020s" (2201.03545)
    """

    # Early dropout settings (2303.01500)
    drop_path_rate: float = 0.0  # Peak drop rate during early training
    early_dropout_epochs: int = 0  # Apply for first N epochs (0 = disabled)
    dropout_schedule: DropoutSchedule = "early_only"

    # Optional: Linear decay within early period (2303.01500)
    linear_decay_to_zero: bool = (
        False  # Decay from drop_path_rate to 0 over early_dropout_epochs
    )

    # Optional: per-stage rates for spatial variation (inspired by 2201.03545)
    per_stage_multipliers: list[float] | None = None  # e.g., [0.5, 1.0, 1.5, 2.0]

    @field_validator("per_stage_multipliers")
    @classmethod
    def validate_multipliers(cls, v: list[float] | None) -> list[float] | None:
        """Validate that stage multipliers are non-negative."""
        if v is not None and any(m < 0 for m in v):
            raise ValueError("Stage multipliers must be non-negative")
        return v

    def build(self):
        """Build a StochasticDepthManager from this configuration."""
        return StochasticDepthManager(self)


DownSamplingBlocks = Literal["avg_pool", "max_pool"]
UpSamplingBlocks = Literal["bilinear_upsample", "transposed_conv"]
Checkpointing = Literal["all", "simple"]


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

    # Stochastic depth configuration
    stochastic_depth: StochasticDepthConfig = StochasticDepthConfig()

    checkpointing: Checkpointing | None = Field(
        default=None,
        description="""Strategy for storing activations for the model for use in
        the backward pass. If not set, the model will store all activations in memory
        (fast but lots of memory). If set to 'all', the model will recompute each
        top-level layer (CoreBlocks, scaling layers, etc.) in the backward pass.
        If set to 'simple', the model will recompute only cheap layers like scales
        and nonlinearities.""",
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
    # we require this to be set by the user but have optional here
    # so we can leave it out of config files
    data_root: Location | None = None
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
    def resolved_data_root(self) -> ResolvedLocation:
        if self.data_root is None:
            raise ValueError(
                "data_root must be set, try --experiment.data_root=path/to/data"
            )
        default_root = LocalLocation(path=Path.cwd())
        return default_root.resolve(self.data_root)


class ProfilerConfig(BaseConfig):
    # How often (in batches processed) to take a snapshot of the CUDA memory
    # (None = no snapshots)
    cuda_snapshot_frequency: int | None = None

    def build(self, output_dir: Path, device: torch.device) -> Profiler:
        if self.cuda_snapshot_frequency is not None and device.type != "cuda":
            raise ValueError(
                "cuda_snapshot_frequency is only supported on CUDA devices, got "
                f"{device.type}"
            )
        return Profiler(output_dir, self.cuda_snapshot_frequency)


# See backend.py for how these are turned into concrete devices
TrainBackendConfig = Literal["cpu", "cuda", "nccl", "auto"]
LossType = Literal[
    "mse",
    "mse_diff_weighted",
    "mse_cos_weighted",
    "mse_residual_scaled",
    "mse_mae",
    "mse_dynamic",
    "mse_dynamic_no_limit",
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
    scheduler: SchedulerConfig | None = None
    loss: LossType = "mse"
    finetune: bool = False
    resume_ckpt_path: str | None = None
    debug: bool = False
    test_using_ema: bool = True
    ema_decay: float = 0.999
    faster_decay_at_start: bool = True
    backend: TrainBackendConfig = "auto"

    # Profiling parameters
    profiler: ProfilerConfig = ProfilerConfig()

    # Data parameters at root level
    data_percent: float = 1.0
    data_stride: list[int] = [1]
    steps: list[int] = [4]
    step_transition: list[int] = []
    inference_epochs: list[int] = [-1]
    train_time: TimeConfig = TimeConfig(
        start=JulianDate("0151-01-06"), end=JulianDate("0306-01-01")
    )
    val_time: TimeConfig = TimeConfig(
        start=JulianDate("0306-01-01"), end=JulianDate("0311-01-01")
    )
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
    inference_time: TimeConfig = TimeConfig(
        start=JulianDate("0311-01-01"), end=JulianDate("0351-01-01")
    )
    experiment: ExperimentConfig
    data: DataConfig
    samudra: SamudraConfig = SamudraConfig()

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


AnyTopLevelConfig = TrainConfig | EvalConfig
