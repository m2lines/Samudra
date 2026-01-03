import abc
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Self, assert_never

import cftime
import pydantic
import torch
import xarray as xr
from perceiver_pytorch import Perceiver as NaivePerceiver
from pydantic import Field, PlainSerializer, PlainValidator, WithJsonSchema
from torch import nn
from torch.nn import GELU

from ocean_emulators.config_base import BaseConfig, TopLevelConfig
from ocean_emulators.constants import (
    MAX_LAT,
    MAX_LON,
    BoundaryVarNames,
    Grid,
    LoaderVersion,
    PrognosticVarNames,
)
from ocean_emulators.models import FOMO, Samudra
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import (
    AvgPool,
    BilinearUpsample,
    CappedGELU,
    ConvBlock,
    ConvNeXtBlock,
    CoreBlock,
    CoreBlockBuilder,
    MaxPool,
    PerceiverEncoder,
    ReLU,
    TransposedConvUpsample,
    UNetBackbone,
)
from ocean_emulators.models.modules.augment_input import Concat3dCoordinates
from ocean_emulators.models.modules.blocks import ZonallyPeriodicBilinearUpsample
from ocean_emulators.utils.data import DataContainer, DataSource
from ocean_emulators.utils.location import LocalLocation, Location, ResolvedLocation
from ocean_emulators.utils.loss import (
    DynamicLoss,
    GradientLoss,
    LossFnWithMask,
    LossMetric,
    loss_fn_from_metric,
)
from ocean_emulators.utils.profiler import Profiler
from ocean_emulators.utils.schedule import SchedulerConfig


class WandBConfig(BaseConfig):
    mode: Literal["online", "disabled"] = "disabled"
    project: str = "default"
    entity: str = "ocean_emulators"
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


class DataSourceConfig(BaseConfig):
    data_location: Location = Field(
        description="Location of the data; " + LOCATION_DOCS
    )
    data_means_location: Location = Field(
        description="Location of the data means; " + LOCATION_DOCS
    )
    data_stds_location: Location = Field(
        description="Location of the data standard deviations; " + LOCATION_DOCS
    )


class DataConfig(BaseConfig):
    sources: list[DataSourceConfig] = Field(
        description=(
            "Data sources to include, each with explicit data/means/stds "
            "locations. These are resolved relative to data_root."
        ),
        min_length=1,
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
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
    ) -> DataContainer:
        loader_version = LoaderVersion(self.loader_version)
        use_dask = loader_version != LoaderVersion.OM4_TORCH

        def make_source(
            data_location: Location,
            means_location: Location,
            stds_location: Location,
            turn_on_dask: bool = use_dask,
        ) -> tuple[DataSource, bool]:
            resolved_data_location = data_root.resolve(data_location)
            resolved_means_location = data_root.resolve(means_location)
            resolved_stds_location = data_root.resolve(stds_location)
            data_source = DataSource.from_locations(
                data_location=resolved_data_location,
                means_location=resolved_means_location,
                stds_location=resolved_stds_location,
                prognostic_var_names=prognostic_var_names,
                boundary_var_names=boundary_var_names,
                static_data_vars=self.static_data_vars,
                use_dask=turn_on_dask,
            )

            return data_source, all(
                loc.supports_fork
                for loc in [
                    resolved_data_location,
                    resolved_means_location,
                    resolved_stds_location,
                ]
            )

        sources = []
        supports_forks = []
        for source_cfg in self.sources:
            src, fork = make_source(
                source_cfg.data_location,
                source_cfg.data_means_location,
                source_cfg.data_stds_location,
            )
            sources.append(src)
            supports_forks.append(fork)
        supports_fork = all(supports_forks)

        primary_source = sources[0]
        if use_dask:
            # If we're already using dask, we don't need a second source
            inference_source = primary_source
        else:
            # If we're not using dask for the main source, create a separate one
            primary = self.sources[0]
            inference_source, _ = make_source(
                primary.data_location,
                primary.data_means_location,
                primary.data_stds_location,
                turn_on_dask=True,
            )

        static_data = (
            primary_source.data[self.static_data_vars]
            if self.static_data_vars is not None
            else None
        )

        return DataContainer(
            sources,
            inference_source,
            loader_version,
            supports_fork,
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

    def build(self) -> CoreBlockBuilder:
        match self.activation:
            case "relu":
                activation: type[nn.Module] = ReLU
            case "capped_gelu":
                activation = CappedGELU
            case "gelu":
                activation = GELU
            case _:
                assert_never(self.activation)

        def create_block(
            in_channels: int,
            out_channels: int,
            dilation: int,
            n_layers: int,
            pad: str,
            checkpoint_simple: bool,
        ) -> CoreBlock:
            match self.block_type:
                case "conv_block":
                    return ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dilation=dilation,
                        n_layers=n_layers,
                        pad=pad,
                        checkpoint_simple=checkpoint_simple,
                        kernel_size=self.kernel_size,
                        activation=activation,
                    )
                case "conv_next_block":
                    return ConvNeXtBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dilation=dilation,
                        n_layers=n_layers,
                        pad=pad,
                        checkpoint_simple=checkpoint_simple,
                        kernel_size=self.kernel_size,
                        upscale_factor=self.upscale_factor,
                        norm=self.norm,
                        activation=activation,
                    )
                case _:
                    assert_never(self.block_type)

        return create_block


class CorrectorConfig(BaseConfig):
    non_negative_corrector_names: list[str] | None = None
    ocean_heat_corrector: bool = False

    def build(
        self, hist: int, area_weights: Grid, static_data: xr.Dataset | None
    ) -> nn.Module:
        # This prevents a circular import bug.
        from ocean_emulators.models.corrector import Correctors

        return Correctors(
            non_negative_corrector_names=self.non_negative_corrector_names,
            ocean_heat_corrector=self.ocean_heat_corrector,
            hist=hist,
            area_weights=area_weights,
            static_data=static_data,
        )


PerceiverImpl = Literal["auto", "naive", "flash"]


class PerceiverConfig(BaseConfig):
    """A standard config interface to various perceiver implementations.

    The `implementation="auto"` option will guess what is the best implementation given the runtime environment.
    """

    implementation: PerceiverImpl = "auto"
    depth: int = 6
    latent_dim: int = Field(
        default=128,
        description="The small, latent dimension of the Perceiver. This is the `N` dimension for the Perceiver's `O(M*N)` complexity",
    )
    num_latents: int = Field(
        default=512,
        description="The number of latent vectors in the Perceiver. This is the `M` dimension for the Perceiver's `O(M*N)` complexity",
    )

    def build(
        self, in_channels: int, out_channels: int, patch_size: tuple[int, int]
    ) -> nn.Module:
        # This is not really a "frequency" but a maximum of the width appears to be reasonable from looking at the code.
        max_freq = max(*patch_size)

        # TODO(alxmrs,jder): Each implementation takes the mean of the num_latents dim to produce the final output_dim.
        #  Why compute the mean? Is it better to directly project from the num_latents x latent_dim?
        if (
            self.implementation == "auto" and torch.cuda.is_available()
        ) or self.implementation == "flash":
            try:
                from flash_perceiver import Perceiver as FlashPerceiver  # type: ignore
            except ModuleNotFoundError as e:
                raise ValueError(
                    "`implementation==flash` or flash was automatically chosen for `implementation==auto`, but the flash attention dependencies could not be imported. Please run `uv sync --extra cuda` or specify the `naive` attention implementation."
                ) from e
            perceiver = FlashPerceiver(
                latent_rotary_emb_dim=max_freq,
                depth=self.depth,
                input_dim=in_channels,
                output_dim=out_channels,
                output_mode="average",
                latent_dim=self.latent_dim,
                num_latents=self.num_latents,
                use_flash_attn=True,
                weight_tie_layers=True,  # share weights of cross-attn blocks during latent iteration
                self_per_cross_attn=2,  # ratio of self-attention (latent, small) per cross-attn (input, big) blocks
            )
        elif (
            self.implementation == "auto" and not torch.cuda.is_available()
        ) or self.implementation == "naive":
            perceiver = NaivePerceiver(
                num_freq_bands=4,
                max_freq=max_freq,
                depth=self.depth,
                input_axis=2,  # Number of positional dims before token dim
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=self.latent_dim,
                num_latents=self.num_latents,
                weight_tie_layers=True,  # share weights of cross-attn blocks
                self_per_cross_attn=2,  # ratio of self-attn (latent, small) and cross-attn (input, big) blocks
            )
        else:
            raise ValueError(
                f"Unknown perceiver implementation: {self.implementation}."
            )

        return perceiver


class EncoderConfig(BaseConfig):
    # TODO(alxmrs): Remove patch_size from all configs
    patch_size: int | list[int] = Field(
        default=4,
        description="Either a square patch (int) or a rectangular patch of [height: int, width: int]. It must evenly divide the grid size.",
    )
    spatial_extent: list[float] = Field(
        default=[6.0, 10.0],
        description="Target physical extent of each patch in degrees [height_deg, width_deg]. "
        "Patch sizes will be calculated to match this extent for each grid resolution.",
    )
    perceiver: PerceiverConfig = PerceiverConfig()

    def build(self, in_channels: int, out_channels: int) -> PerceiverEncoder:
        assert len(self.spatial_extent) == 2, "spatial extent must be a pair of floats."
        extent = self.spatial_extent[0], self.spatial_extent[1]
        # TODO(alxmrs): Make DRY?
        lat_spacing = 180.0 / MAX_LAT
        lon_spacing = 360.0 / MAX_LON
        patch_h = int(round(self.spatial_extent[0] / lat_spacing))
        patch_w = int(round(self.spatial_extent[1] / lon_spacing))
        max_patch_size = (patch_h, patch_w)
        return PerceiverEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_extent=extent,
            perceiver=self.perceiver.build(in_channels, out_channels, max_patch_size),
        )


DownSamplingBlocks = Literal["avg_pool", "max_pool"]
UpSamplingBlocks = Literal[
    "bilinear_upsample", "transposed_conv", "zonally_periodic_upsample"
]
Checkpointing = Literal["all", "simple"]


class UNetBackboneConfig(BaseConfig):
    ch_width: list[int] = [200, 250, 300, 400]
    dilation: list[int] = [1, 2, 4, 8]
    n_layers: list[int] = [1, 1, 1, 1]
    core_block: BlockConfig = BlockConfig()
    down_sampling_block: DownSamplingBlocks = "avg_pool"
    up_sampling_block: UpSamplingBlocks = "zonally_periodic_upsample"

    def build(
        self,
        in_channels: int,
        pad: str,
        checkpointing: Checkpointing | None,
    ) -> UNetBackbone:
        assert len(self.ch_width) == len(self.dilation) == len(self.n_layers), (
            "`ch_width`, `dilation`, and `n_layers` must have the same length."
        )

        def create_upsampling_block(in_channels: int, out_channels: int):
            match self.up_sampling_block:
                case "bilinear_upsample":
                    return BilinearUpsample(
                        in_channels=in_channels, out_channels=out_channels
                    )
                case "transposed_conv":
                    return TransposedConvUpsample(
                        in_channels=in_channels, out_channels=out_channels
                    )
                case "zonally_periodic_upsample":
                    return ZonallyPeriodicBilinearUpsample()
                case _:
                    assert_never(self.up_sampling_block)

        match self.down_sampling_block:
            case "avg_pool":
                downsampling_block: nn.Module = AvgPool()
            case "max_pool":
                downsampling_block = MaxPool()
            case _:
                assert_never(self.down_sampling_block)

        return UNetBackbone(
            in_channels=in_channels,
            ch_width=self.ch_width,
            dilation=self.dilation,
            n_layers=self.n_layers,
            pad=pad,
            create_block=self.core_block.build(),
            downsampling_block=downsampling_block,
            create_upsampling_block=create_upsampling_block,
            checkpointing=checkpointing,
        )


class BaseModelConfig(BaseConfig, abc.ABC):
    pred_residuals: bool = False
    last_kernel_size: int = 3
    pad: str = "circular"

    checkpointing: Checkpointing | None = Field(
        default=None,
        description="""Strategy for storing activations for the model for use in
        the backward pass. If not set, the model will store all activations in memory
        (fast but lots of memory). If set to 'all', the model will recompute each
        top-level layer (CoreBlocks, scaling layers, etc.) in the backward pass.
        If set to 'simple', the model will recompute only cheap layers like scales
        and nonlinearities.""",
    )

    gradient_detach_interval: int = Field(
        default=0,
        description="""Interval for detaching gradients in autoregressive training. `0` means no detaching.""",
    )

    add_3d_coordinates: bool = Field(
        default=False,
        description="Add 3d coordinates representing position on the Earth (cartesian coordinates on a unit sphere) to the input channels.",
    )

    @abc.abstractmethod
    def build(
        self,
        in_channels: int,
        out_channels: int,
        hist: int,
        static_data: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> BaseModel:
        pass


class SamudraConfig(BaseModelConfig):
    unet: UNetBackboneConfig = UNetBackboneConfig()
    corrector: CorrectorConfig | None = None  # None turns all correctors off.
    pos_channels: int = Field(
        default=0,
        description="""Number of channels used for a learned positional embedding""",
    )
    use_bfloat16: bool = Field(
        default=False,
        description="Use bfloat16 for most layers rather than float32.",
    )

    def build(
        self,
        in_channels: int,
        out_channels: int,
        hist: int,
        static_data: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> Samudra:
        corrector = None
        src = srcs[0]
        lat, lon = src.resolution
        if self.corrector is not None:
            corrector = self.corrector.build(hist, src.area_weights, static_data)
        total_in_channels = (
            in_channels + self.pos_channels + (3 if self.add_3d_coordinates else 0)
        )
        add_3d_coordinates = (
            Concat3dCoordinates(lat, lon) if self.add_3d_coordinates else None
        )
        return Samudra(
            in_channels=total_in_channels,
            out_channels=out_channels,
            pred_residuals=self.pred_residuals,
            last_kernel_size=self.last_kernel_size,
            pad=self.pad,
            unet=self.unet.build(
                in_channels=total_in_channels,
                pad=self.pad,
                checkpointing=self.checkpointing,
            ),
            corrector=corrector,
            pos_channels=self.pos_channels,
            add_3d_coordinates=add_3d_coordinates,
            hist=hist,
            grid=(lat.shape[0], lon.shape[0]),
            static_data=static_data,
            gradient_detach_interval=self.gradient_detach_interval,
            use_bfloat16=self.use_bfloat16,
        )


class FOMOConfig(BaseModelConfig):
    encoder: EncoderConfig = EncoderConfig()
    processor: UNetBackboneConfig = UNetBackboneConfig()
    # decoder will go here.
    embedding_dim: int = 128
    use_bfloat16: bool = Field(
        default=True,
        description="Use bfloat16 for most layers rather than float32. Required for flash attention.",
    )

    def build(
        self,
        in_channels: int,
        out_channels: int,
        hist: int,
        static_data: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> FOMO:
        src = srcs[0]
        lat, lon = src.resolution
        total_in_channels = in_channels + (3 if self.add_3d_coordinates else 0)
        add_3d_coordinates = (
            Concat3dCoordinates(lat, lon) if self.add_3d_coordinates else nn.Identity()
        )
        all_grids = [(len(s.resolution[0]), len(s.resolution[1])) for s in srcs]
        return FOMO(
            in_channels=total_in_channels,
            out_channels=out_channels,
            pred_residuals=self.pred_residuals,
            last_kernel_size=self.last_kernel_size,
            pad=self.pad,
            encoder=self.encoder.build(in_channels, self.embedding_dim),
            processor=self.processor.build(
                self.embedding_dim,
                self.pad,
                self.checkpointing,
            ),
            # decoder = self.decoder.build(processor.out_channels, out_channels)  # will be something like this
            add_3d_coordinates=add_3d_coordinates,
            hist=hist,
            static_data=static_data,
            checkpointing=self.checkpointing,
            gradient_detach_interval=self.gradient_detach_interval,
            all_grids=all_grids,
            use_bfloat16=self.use_bfloat16,
        )


AnyModelConfig = SamudraConfig | FOMOConfig


class DistributedConfig(BaseConfig):
    dist_url: str | None = None
    world_size: int | None = None
    rank: int | None = None
    gpu: int | None = None
    dist_backend: str | None = None


TrainSchedule = Literal["standard", "match", "mix"]


class ExperimentConfig(BaseConfig):
    name: str = "cm4_samudra"
    rand_seed: int = 1
    base_output_dir: str = "train"
    # we require this to be set by the user but have optional here
    # so we can leave it out of config files
    data_root: Location | None = None
    # Define multi-scale dataloader example schedule. Default: single scale.
    train_schedule: TrainSchedule = "standard"
    wandb: WandBConfig

    # Model configuration
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


class DynamicLossConfig(pydantic.BaseModel):
    type: Literal["dynamic"] = "dynamic"
    metric: LossMetric = "mse"
    limit: float | None = Field(
        description="The ratio of the largest weight to the smallest weight across all channels which we'll allow. Default of None means no limit.",
        default=None,
        ge=1.0,
    )


class GradientLossConfig(pydantic.BaseModel):
    type: Literal["gradient"] = "gradient"
    # at the moment this metric is only used for the non-gradient loss
    # (and would take a bit of refactoring to make it work for the gradient loss too)
    # so we fix it to MAE for now until it's clear we what flexibility is needed here.
    # TODO(#497): support other metrics for the gradient loss
    metric: Literal["mae"] = "mae"
    alpha: float = Field(
        description="Scaling factor for the gradient penalty term (alpha in the gradient-weighted loss).",
        default=0.1,
        ge=0.0,
    )


Loss = LossMetric | DynamicLossConfig | GradientLossConfig


def build_loss_fn(
    loss_cfg: Loss,
    y_coord: xr.DataArray,
    device: torch.device,
    num_channels: int,
    pad_mode: str,
) -> LossFnWithMask:
    match loss_cfg:
        case str():
            return loss_fn_from_metric(loss_cfg, y_coord=y_coord, device=device)
        case DynamicLossConfig(metric=metric, limit=limit):
            loss_fn = loss_fn_from_metric(metric, y_coord=y_coord, device=device)
            return DynamicLoss(
                loss_fn=loss_fn,
                limit=limit,
                device=device,
                num_channels=num_channels,
            )
        case GradientLossConfig(metric=metric, alpha=alpha):
            loss_fn = loss_fn_from_metric(metric, y_coord=y_coord, device=device)
            return GradientLoss(
                loss_fn=loss_fn,
                gradient_weight=alpha,
                pad_mode=pad_mode,
            )
        case _:
            assert_never(loss_cfg)


class TrainConfig(TopLevelConfig):
    # Training parameters
    disk_mode: bool = True
    pin_mem: bool = True
    save_freq: int = 5
    epochs: int = 120
    preemptible: bool = True
    batch_size: int = 2
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1
    scheduler: SchedulerConfig | None = None
    loss: Loss = "mse"
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
    model: AnyModelConfig

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
    model: AnyModelConfig = SamudraConfig()

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


AnyTopLevelConfig = TrainConfig | EvalConfig
