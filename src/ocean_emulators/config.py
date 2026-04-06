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
    BoundaryVarNames,
    Grid,
    LoaderVersion,
    PrognosticVarNames,
)
from ocean_emulators.models import FOMO, FOMini, Samudra
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import (
    AvgPool,
    AxialAttentionBlock,
    BilinearUpsample,
    CappedGELU,
    ConvBlock,
    ConvNeXtBlock,
    CoreBlock,
    CoreBlockBuilder,
    FullAttentionBlock,
    MaxPool,
    PerceiverDecoder,
    PerceiverEncoder,
    ReLU,
    TransposedConvUpsample,
    UNetBackbone,
)
from ocean_emulators.models.modules.augment_input import Concat3dCoordinates
from ocean_emulators.models.modules.blocks import ZonallyPeriodicBilinearUpsample
from ocean_emulators.models.modules.encoder import patch_from
from ocean_emulators.utils.data import DataContainer, DataSource
from ocean_emulators.utils.location import LocalLocation, Location, ResolvedLocation
from ocean_emulators.utils.loss import (
    DynamicLoss,
    GradientLoss,
    LossFnWithContext,
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
AttentionType = Literal["axial", "full"]
BottleneckBlockType = Literal["attention"]


class AttentionBlockConfig(BaseConfig):
    """Configuration for a single attention block in the U-Net."""

    attention_type: AttentionType = Field(
        default="full",
        description="Attention implementation to use at this stage.",
    )

    num_heads: int = Field(
        default=8,
        description="Number of attention heads. Must divide the stage channel width evenly.",
    )
    attn_drop: float = Field(
        default=0.0,
        description="Dropout rate applied to attention weights.",
    )
    proj_drop: float = Field(
        default=0.0,
        description="Dropout rate applied to the output projection.",
    )

    def build(self, channels: int) -> nn.Module:
        assert channels % self.num_heads == 0, (
            f"channels {channels} must be divisible by num_heads {self.num_heads}"
        )
        if self.attention_type == "axial":
            return AxialAttentionBlock(
                channels=channels,
                num_heads=self.num_heads,
                attn_drop=self.attn_drop,
                proj_drop=self.proj_drop,
            )
        if self.attention_type == "full":
            return FullAttentionBlock(
                channels=channels,
                num_heads=self.num_heads,
                attn_drop=self.attn_drop,
                proj_drop=self.proj_drop,
            )
        assert_never(self.attention_type)


class BottleneckBlockConfig(BaseConfig):
    # TODO: Add Transformer and MaxVIT style bottlenecks as options here as well
    block_type: BottleneckBlockType = Field(
        default="attention",
        description="Bottleneck block family to insert between the U-Net middle block and decoder.",
    )
    attention: AttentionBlockConfig | None = Field(
        default=None,
        description="Attention bottleneck settings used when `block_type` is `attention`.",
    )

    def build(self, channels: int) -> nn.Module:
        if self.block_type == "attention":
            attention_config = self.attention
            if attention_config is None:
                raise ValueError(
                    "`attention.bottleneck.attention` must be set when "
                    "`attention.bottleneck.block_type` is `attention`."
                )
            return attention_config.build(channels)
        assert_never(self.block_type)


class UNetAttentionConfig(BaseConfig):
    """Optional attention blocks to insert after U-Net core blocks."""

    encoder: list[AttentionBlockConfig | None] | None = Field(
        default=None,
        description="Optional attention blocks after encoder core blocks, one entry per value in ch_width.",
    )
    bottleneck: BottleneckBlockConfig | None = Field(
        default=None,
        description="Optional bottleneck block after the bottleneck core block.",
    )
    decoder: list[AttentionBlockConfig | None] | None = Field(
        default=None,
        description="Optional attention blocks after decoder core blocks, one entry per value in ch_width.",
    )


class BlockConfig(BaseConfig):
    block_type: BlockType = "conv_next_block"
    kernel_size: int = 3
    activation: ActivationType = "capped_gelu"
    upscale_factor: int = 4
    norm: NormType = "batch"
    pointwise_linear: bool = False

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
                        pointwise_linear=self.pointwise_linear,
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

    Builds either a regular Perceiver (for the encoder, via ``build``) or a
    PerceiverIO (for the decoder, via ``build_io``).  Both respect the shared
    ``implementation`` setting from ``FOMOConfig.perceiver_implementation``.
    """

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
        self,
        in_channels: int,
        out_channels: int,
        max_patch_size: tuple[int, int],
        implementation: PerceiverImpl,
    ) -> nn.Module:
        """Build a regular Perceiver (used by the encoder)."""
        # This is not really a "frequency" but a maximum of the width appears to be reasonable from looking at the code.
        max_freq = max(*max_patch_size)

        if _use_flash(implementation):
            try:
                from flash_perceiver import Perceiver as FlashPerceiver  # type: ignore
            except ModuleNotFoundError as e:
                raise _flash_import_error() from e
            from einops.layers.torch import Rearrange

            # Flash perceiver expects (batch, seq_len, dim); naive handles
            # (batch, ph, pw, dim) natively via input_axis=2.  Bake the
            # spatial-flatten into the module so callers don't need to care.
            perceiver: nn.Module = nn.Sequential(
                Rearrange("b ph pw v -> b (ph pw) v"),
                FlashPerceiver(
                    latent_rotary_emb_dim=max_freq,
                    depth=self.depth,
                    input_dim=in_channels,
                    output_dim=out_channels,
                    output_mode="average",
                    latent_dim=self.latent_dim,
                    num_latents=self.num_latents,
                    use_flash_attn=True,
                    weight_tie_layers=True,
                    self_per_cross_attn=2,
                ),
            )
        elif _use_naive(implementation):
            perceiver = NaivePerceiver(
                num_freq_bands=4,
                max_freq=max_freq,
                depth=self.depth,
                input_axis=2,
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=self.latent_dim,
                num_latents=self.num_latents,
                weight_tie_layers=True,
                self_per_cross_attn=2,
            )
        else:
            raise ValueError(f"Unknown perceiver implementation: {implementation}.")

        return perceiver

    def build_io(
        self,
        in_channels: int,
        queries_dim: int,
        out_channels: int,
        implementation: PerceiverImpl,
    ) -> nn.Module:
        """Build a PerceiverIO (used by the decoder)."""
        if _use_flash(implementation):
            try:
                from flash_perceiver.perceiver import (  # type: ignore
                    PerceiverIO as FlashPerceiverIO,  # type: ignore
                )
            except ModuleNotFoundError as e:
                raise _flash_import_error() from e
            perceiver_io: nn.Module = FlashPerceiverIO(
                depth=self.depth,
                input_dim=in_channels,
                query_dim=queries_dim,
                proj_dim=out_channels,
                num_latents=self.num_latents,
                latent_dim=self.latent_dim,
                use_flash_attn=True,
                weight_tie_layers=True,
            )
        elif _use_naive(implementation):
            from perceiver_pytorch.perceiver_io import PerceiverIO as NaivePerceiverIO

            perceiver_io = NaivePerceiverIO(
                depth=self.depth,
                dim=in_channels,
                queries_dim=queries_dim,
                logits_dim=out_channels,
                num_latents=self.num_latents,
                latent_dim=self.latent_dim,
                weight_tie_layers=True,
                decoder_ff=True,
            )
        else:
            raise ValueError(f"Unknown perceiver implementation: {implementation}.")

        return perceiver_io


def _use_flash(implementation: PerceiverImpl) -> bool:
    return (
        implementation == "auto" and torch.cuda.is_available()
    ) or implementation == "flash"


def _use_naive(implementation: PerceiverImpl) -> bool:
    return (
        implementation == "auto" and not torch.cuda.is_available()
    ) or implementation == "naive"


def _flash_import_error() -> ValueError:
    return ValueError(
        "`implementation==flash` or flash was automatically chosen for `implementation==auto`, "
        "but the flash attention dependencies could not be imported. "
        "Please run `uv sync --extra cuda` or specify the `naive` attention implementation."
    )


class EncoderConfig(BaseConfig):
    perceiver: PerceiverConfig = PerceiverConfig()

    def build(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        max_lat_size: int,
        max_lon_size: int,
        implementation: PerceiverImpl,
    ) -> PerceiverEncoder:
        max_patch_size = patch_from(patch_extent, max_lat_size, max_lon_size)
        return PerceiverEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_extent=patch_extent,
            perceiver=self.perceiver.build(
                in_channels, out_channels, max_patch_size, implementation
            ),
        )


class DecoderConfig(BaseConfig):
    """A PerceiverIO-based decoder configuration.

    Uses PerceiverIO (with an explicit query mechanism) rather than a regular
    Perceiver.  Output pixel positions are encoded as queries, so the output
    size is determined by the query count — not by ``num_latents``.

    When ``window_patches`` is set, the decoder tiles the output grid into
    spatial blocks of that many patches per side.  Each block's PerceiverIO
    call receives only the overlapping latent tokens plus ``context_patches``
    extra rings of neighbors, keeping cost bounded even when the latent grid
    is large (i.e. fine ``patch_extent``).
    """

    perceiver: PerceiverConfig = PerceiverConfig()
    queries_dim: int = Field(
        default=64,
        description="Embedding dimension for pixel-position queries in the PerceiverIO decoder head.",
    )
    window_patches: int | None = Field(
        default=4096,
        description="Side length (in patches) of each spatial decode window. "
        "None = decode all patches at once (global attention). "
        "E.g. window_patches=8 means each PerceiverIO call covers an 8x8 block of patches.",
    )
    context_patches: int | None = Field(
        default=1,
        description="Number of extra patch rings around each window to include as data context. "
        "Only used when window_patches is set. None = full context (every window sees all latent tokens).",
    )

    def build(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        implementation: PerceiverImpl,
    ) -> PerceiverDecoder:
        return PerceiverDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_extent=patch_extent,
            queries_dim=self.queries_dim,
            perceiver_io=self.perceiver.build_io(
                in_channels, self.queries_dim, out_channels, implementation
            ),
            window_patches=self.window_patches,
            context_patches=self.context_patches,
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
    attention: UNetAttentionConfig | None = Field(
        default=None,
        description="Optional attention blocks to insert after encoder, bottleneck, and decoder core blocks.",
    )

    def build(
        self,
        in_channels: int,
        pad: str,
        checkpointing: Checkpointing | None,
    ) -> UNetBackbone:
        assert len(self.ch_width) == len(self.dilation) == len(self.n_layers), (
            "`ch_width`, `dilation`, and `n_layers` must have the same length."
        )

        encoder_attention_configs: list[AttentionBlockConfig | None] | None = None
        decoder_attention_configs: list[AttentionBlockConfig | None] | None = None
        bottleneck_config: BottleneckBlockConfig | None = None

        if self.attention is not None:
            encoder_attention_configs = self.attention.encoder
            decoder_attention_configs = self.attention.decoder

            bottleneck_config = self.attention.bottleneck

            if encoder_attention_configs is not None:
                assert len(encoder_attention_configs) == len(self.ch_width), (
                    "`attention.encoder` must have the same length as `ch_width`."
                )
            if decoder_attention_configs is not None:
                assert len(decoder_attention_configs) == len(self.ch_width), (
                    "`attention.decoder` must have the same length as `ch_width`."
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

        encoder_attention_blocks: list[nn.Module | None] | None = None
        if encoder_attention_configs is not None:
            encoder_attention_blocks = []
            for cfg, channels in zip(
                encoder_attention_configs, self.ch_width, strict=True
            ):
                if cfg is None:
                    encoder_attention_blocks.append(None)
                else:
                    encoder_attention_blocks.append(cfg.build(channels))

        bottleneck_block = (
            bottleneck_config.build(self.ch_width[-1])
            if bottleneck_config is not None
            else None
        )
        decoder_attention_blocks: list[nn.Module | None] | None = None
        if decoder_attention_configs is not None:
            # Skip the bottleneck width and repeat the final decoder width (as done in core blocks)
            decoder_attention_channels = self.ch_width[-2::-1] + [self.ch_width[0]]
            decoder_attention_blocks = []
            for cfg, channels in zip(
                decoder_attention_configs,
                decoder_attention_channels,
                strict=True,
            ):
                if cfg is None:
                    decoder_attention_blocks.append(None)
                else:
                    decoder_attention_blocks.append(cfg.build(channels))

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
            encoder_attention_blocks=encoder_attention_blocks,
            bottleneck_block=bottleneck_block,
            decoder_attention_blocks=decoder_attention_blocks,
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
        static_data_for_corrector: xr.Dataset | None,
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
        static_data_for_corrector: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> Samudra:
        corrector = None
        if len(srcs) != 1:
            raise ValueError(
                'Samudra only supports training at a single scale! Please set `training_schedule="standard"`.'
            )
        src = srcs[0]
        if self.corrector is not None:
            corrector = self.corrector.build(
                hist, src.spherical_area_weights, static_data_for_corrector
            )
        total_in_channels = (
            in_channels + self.pos_channels + (3 if self.add_3d_coordinates else 0)
        )
        add_3d_coordinates = Concat3dCoordinates() if self.add_3d_coordinates else None
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
            grid_size=src.grid_size,
            gradient_detach_interval=self.gradient_detach_interval,
            use_bfloat16=self.use_bfloat16,
        )


class FOMOConfig(BaseModelConfig):
    encoder: EncoderConfig = EncoderConfig()
    processor: UNetBackboneConfig = UNetBackboneConfig()
    decoder: DecoderConfig = DecoderConfig()
    perceiver_implementation: PerceiverImpl = Field(
        default="auto",
        description="Perceiver attention implementation shared by the encoder and decoder. "
        "'auto' selects flash attention when CUDA is available, otherwise naive.",
    )
    patch_extent: list[float] = Field(
        default=[6.0, 10.0],
        description="Target physical extent of each patch in degrees [height_deg, width_deg]. "
        "Shared by the encoder and decoder for consistent spatial semantics.",
    )
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
        static_data_for_corrector: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> FOMO:
        assert len(self.patch_extent) == 2, "patch_extent must be a pair of floats."
        extent = self.patch_extent[0], self.patch_extent[1]

        all_grid_sizes = [s.grid_size for s in srcs]
        max_lat_size, max_lon_size = (
            max(g[0] for g in all_grid_sizes),
            max(g[1] for g in all_grid_sizes),
        )

        impl = self.perceiver_implementation
        if _use_flash(impl) and not self.use_bfloat16:
            raise ValueError(
                "Perceiver implementation resolves to flash attention. "
                "Please set `use_bfloat16=True` or `perceiver_implementation='naive'`."
            )

        encoder = self.encoder.build(
            in_channels, self.embedding_dim, extent, max_lat_size, max_lon_size, impl
        )
        processor = self.processor.build(
            self.embedding_dim,
            self.pad,
            self.checkpointing,
        )
        decoder = self.decoder.build(
            processor.out_channels,
            out_channels,
            extent,
            impl,
        )

        total_in_channels = in_channels + (3 if self.add_3d_coordinates else 0)
        add_3d_coordinates = Concat3dCoordinates() if self.add_3d_coordinates else None
        return FOMO(
            in_channels=total_in_channels,
            out_channels=out_channels,
            pred_residuals=self.pred_residuals,
            last_kernel_size=self.last_kernel_size,
            pad=self.pad,
            encoder=encoder,
            processor=processor,
            decoder=decoder,
            add_3d_coordinates=add_3d_coordinates,
            hist=hist,
            checkpointing=self.checkpointing,
            gradient_detach_interval=self.gradient_detach_interval,
            use_bfloat16=self.use_bfloat16,
        )


class FOMiniConfig(BaseModelConfig):
    perceiver: PerceiverConfig = PerceiverConfig()
    perceiver_implementation: PerceiverImpl = Field(
        default="auto",
        description="Perceiver attention implementation for the single PerceiverIO model. "
        "'auto' selects flash attention when CUDA is available, otherwise naive.",
    )
    embedding_dim: int = Field(
        default=128,
        description="Dimension of data-token embeddings before PerceiverIO.",
    )
    queries_dim: int = Field(
        default=128,
        description="Dimension of PerceiverIO output queries.",
    )
    coordinate_embedding_dim: int = Field(
        default=64,
        description="Hidden dimension used by learned 3D Cartesian coordinate embeddings.",
    )
    query_chunk_size: int | None = Field(
        default=None,
        description="Optional chunk size for query decoding. If set, PerceiverIO is called "
        "over query chunks to reduce memory use.",
    )
    use_bfloat16: bool = Field(
        default=True,
        description="Use bfloat16 for most layers rather than float32. Required for flash attention.",
    )

    def build(
        self,
        in_channels: int,
        out_channels: int,
        hist: int,
        static_data_for_corrector: xr.Dataset | None,
        srcs: list[DataSource],
    ) -> FOMini:
        if self.add_3d_coordinates:
            raise ValueError(
                "FOMini always uses learned Cartesian coordinate embeddings. "
                "Please set `add_3d_coordinates=False`."
            )

        impl = self.perceiver_implementation
        if _use_flash(impl) and not self.use_bfloat16:
            raise ValueError(
                "Perceiver implementation resolves to flash attention. "
                "Please set `use_bfloat16=True` or `perceiver_implementation='naive'`."
            )

        perceiver_io = self.perceiver.build_io(
            self.embedding_dim,
            self.queries_dim,
            out_channels,
            impl,
        )
        return FOMini(
            in_channels=in_channels,
            out_channels=out_channels,
            pred_residuals=self.pred_residuals,
            last_kernel_size=self.last_kernel_size,
            pad=self.pad,
            input_embedding_dim=self.embedding_dim,
            coordinate_embedding_dim=self.coordinate_embedding_dim,
            queries_dim=self.queries_dim,
            query_chunk_size=self.query_chunk_size,
            perceiver_io=perceiver_io,
            hist=hist,
            checkpointing=self.checkpointing,
            gradient_detach_interval=self.gradient_detach_interval,
            use_bfloat16=self.use_bfloat16,
        )


AnyModelConfig = SamudraConfig | FOMOConfig | FOMiniConfig


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
    device: torch.device,
    num_channels: int,
    pad_mode: str,
) -> LossFnWithContext:
    match loss_cfg:
        case str():
            return loss_fn_from_metric(loss_cfg)
        case DynamicLossConfig(metric=metric, limit=limit):
            loss_fn = loss_fn_from_metric(metric)
            return DynamicLoss(
                loss_fn=loss_fn,
                limit=limit,
                device=device,
                num_channels=num_channels,
            )
        case GradientLossConfig(metric=metric, alpha=alpha):
            loss_fn = loss_fn_from_metric(metric)
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
    validation_image_log_freq: int = Field(
        default=10,
        ge=1,
        description=(
            "How often to log expensive validation images. Epochs are 1-based, so "
            "a value of 10 logs on epochs 1, 11, 21, ..."
        ),
    )
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
    delayed_loss_estimate: bool = False
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
