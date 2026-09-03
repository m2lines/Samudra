# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import abc
import datetime
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Self, assert_never

import cftime
import numpy as np
import pandas as pd
import pydantic
import torch
import xarray as xr
from pydantic import (
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
    model_validator,
)
from torch import nn
from torch.nn import GELU

from samudra.config_base import BaseConfig, TopLevelConfig
from samudra.constants import (
    DataLayout,
    GridSize,
    GridType,
    LoaderVersion,
    build_om4_layout,
)
from samudra.models import Samudra, SamudraMini, SamudraMulti
from samudra.models.base import BaseModel
from samudra.models.modules import (
    AvgPool,
    BilinearUpsample,
    CappedGELU,
    ConvBlock,
    ConvNeXtBlock,
    CoreBlock,
    CoreBlockBuilder,
    DCTDetailDecoder,
    DCTDetailEncoder,
    DirectCrossAttentionIO,
    MaxPool,
    PatchMomentEncoder,
    Perceiver,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    ReLU,
    SpatialLatentGridEncoder,
    SpatialQueryPerceiver,
    StructuredLocalDecoder,
    TransposedConvUpsample,
    UNetBackbone,
)
from samudra.models.modules.augment_input import (
    Concat3dCoordinates,
    FourierFeatures2D,
    fourier_features_2d_dim,
)
from samudra.models.modules.blocks import ZonallyPeriodicBilinearUpsample
from samudra.models.modules.encoder import patch_from
from samudra.utils.data import (
    CanonicalSource,
    DataBundle,
    SourceSplits,
    compute_anomalies,
    flatten_masks,
    get_anomalies_vars,
    with_lat_lon_coords,
    with_level_index_vars,
)
from samudra.utils.llc import canonicalize_llc_datasets
from samudra.utils.location import (
    LocalLocation,
    Location,
    ResolvedLocation,
    UnresolvedLocation,
)
from samudra.utils.loss import (
    DynamicLoss,
    GradientLoss,
    LossFnWithContext,
    LossMetric,
    loss_fn_from_metric,
)
from samudra.utils.profiler import Profiler
from samudra.utils.schedule import SchedulerConfig


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

    def __init__(self, value: str):
        parsed = cftime.datetime.strptime(value, "%Y-%m-%d", calendar="julian")
        self.datetime = parsed.replace(hour=12)

    def __str__(self) -> str:
        return self.datetime.strftime("%Y-%m-%d")


def _julian_date_validator(value: str | JulianDate) -> JulianDate:
    return JulianDate(value) if isinstance(value, str) else value


JulianDateConfig = Annotated[
    JulianDate,
    PlainValidator(_julian_date_validator),
    PlainSerializer(JulianDate.__str__),
    WithJsonSchema({"type": "string", "format": "date"}),
]


# We reuse pydantic's AwareDatetime to match JSONSchema's expected RFC3339 format
_aware_datetime_adapter = pydantic.TypeAdapter(pydantic.AwareDatetime)


def _datetime64_validator(value: str | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    parsed = _aware_datetime_adapter.validate_python(value)
    utc = parsed.astimezone(datetime.UTC).replace(tzinfo=None)
    return np.datetime64(utc, "ns")


def _serialize_datetime64(value: np.datetime64) -> str:
    return str(np.datetime_as_string(value, unit="ns", timezone="UTC"))


LlcDatetimeConfig = Annotated[
    np.datetime64,
    PlainValidator(_datetime64_validator),
    PlainSerializer(_serialize_datetime64),
    WithJsonSchema({"type": "string", "format": "date-time"}),
]


class Om4TimeConfig(BaseConfig):
    model_config = pydantic.ConfigDict(frozen=True)

    start: JulianDateConfig
    end: JulianDateConfig

    @property
    def time_slice(self) -> slice:
        return slice(self.start.datetime, self.end.datetime)

    def overlaps(self, other: Self) -> bool:
        return (
            self.start.datetime < other.end.datetime
            and self.end.datetime > other.start.datetime
        )

    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


class LlcTimeConfig(BaseConfig):
    model_config = pydantic.ConfigDict(frozen=True)

    start: LlcDatetimeConfig
    end: LlcDatetimeConfig

    @property
    def time_slice(self) -> slice:
        return slice(self.start, self.end)

    def overlaps(self, other: Self) -> bool:
        return bool(self.start < other.end and self.end > other.start)

    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


TimeConfig = Om4TimeConfig | LlcTimeConfig


LOCATION_DOCS = (
    "Use a string relative to the `data_root` or use a structured location "
    "see location.py for possible types."
)


class BaseDataSourceConfig[SourceTimeConfigT: TimeConfig](BaseConfig, abc.ABC):
    train_time: SourceTimeConfigT = Field(frozen=True)
    val_time: SourceTimeConfigT = Field(frozen=True)
    inference_times: tuple[SourceTimeConfigT, ...] = Field(default=(), frozen=True)
    data_location: Location = Field(
        description="Location of the data; " + LOCATION_DOCS
    )
    data_means_location: Location = Field(
        description="Location of the data means; " + LOCATION_DOCS
    )
    data_stds_location: Location = Field(
        description="Location of the data standard deviations; " + LOCATION_DOCS
    )

    @abc.abstractmethod
    def canonicalize_datasets(
        self,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, DataLayout]:
        raise NotImplementedError

    def build(
        self,
        data_root: ResolvedLocation,
        *,
        use_dask: bool,
        is_primary: bool,
    ) -> SourceSplits:
        source = self._build_source(
            data_root,
            turn_on_dask=use_dask,
        )
        inference_source = None
        if is_primary and self.inference_times:
            if use_dask:
                full_inference_source = source
            else:
                full_inference_source = self._build_source(
                    data_root,
                    turn_on_dask=True,
                )
            # TODO: remove multiple inference time ranges altogether (see #813)
            assert len(self.inference_times) == 1, (
                "multiple inference time ranges have been deprecated"
            )
            inference_source = full_inference_source.slice_time(self.inference_times[0])

        return SourceSplits(
            train=source.slice_time(self.train_time),
            val=source.slice_time(self.val_time),
            inference=inference_source,
        )

    def _build_source(
        self,
        data_root: ResolvedLocation,
        *,
        turn_on_dask: bool,
    ) -> CanonicalSource:
        resolved_data_location = data_root.resolve(self.data_location)
        resolved_means_location = data_root.resolve(self.data_means_location)
        resolved_stds_location = data_root.resolve(self.data_stds_location)

        chunks: dict[str, int] | None = {} if turn_on_dask else None
        data = resolved_data_location.open(chunks)
        means = resolved_means_location.open(chunks)
        stds = resolved_stds_location.open(chunks)
        data, means, stds, data_layout = self.canonicalize_datasets(
            data,
            means,
            stds,
        )
        source = CanonicalSource.from_datasets(
            data,
            means,
            stds,
            data_layout=data_layout,
            prognostic_var_names=data_layout.prognostic_var_names,
            boundary_var_names=data_layout.boundary_var_names,
            name=f"{resolved_data_location}-{turn_on_dask}",
        )
        return source


class BaseDataLoadingConfig(BaseConfig):
    def num_pytorch_workers(self) -> int:
        raise NotImplementedError

    def persistent_pytorch_workers(self) -> bool:
        raise NotImplementedError


class CpuDataLoadingConfig(BaseDataLoadingConfig):
    type: Literal["cpu"] = "cpu"
    num_workers: int = Field(default=4, ge=0)
    persistent_workers: bool = True

    def num_pytorch_workers(self) -> int:
        return self.num_workers

    def persistent_pytorch_workers(self) -> bool:
        return self.persistent_workers


class GpuDataLoadingConfig(BaseDataLoadingConfig):
    type: Literal["gpu"] = "gpu"
    kvikio_task_size: int = Field(default=64 * 1024 * 1024, gt=0)
    kvikio_num_threads: int = Field(default=8, gt=0)

    def num_pytorch_workers(self) -> int:
        # When loading data direct to GPU, we don't want worker processes.
        # 0 means "load in the main process"
        return 0

    def persistent_pytorch_workers(self) -> bool:
        return False


DataLoadingConfig = Annotated[
    CpuDataLoadingConfig | GpuDataLoadingConfig,
    Field(discriminator="type"),
]


class Om4DataSourceConfig(BaseDataSourceConfig[Om4TimeConfig]):
    type: Literal["om4"] = "om4"
    prognostic_vars_key: str = "thermo_dynamic_all"
    boundary_vars_key: str = "tau_hfds"
    grid_type: GridType = "gaussian"

    @pydantic.model_validator(mode="after")
    def validate_time_splits(self) -> Self:
        if self.train_time.overlaps(self.val_time):
            raise ValueError(
                f"Training time range {self.train_time} overlaps "
                f"with validation time range {self.val_time}"
            )
        return self

    def canonicalize_datasets(
        self,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, DataLayout]:
        """Convert raw flat or compact OM4 xarray inputs to canonical channels."""
        data_layout = build_om4_layout(
            self.prognostic_vars_key,
            self.boundary_vars_key,
            grid_type=self.grid_type,
        )
        data = data.copy()
        means = means.copy()
        stds = stds.copy()

        data = with_lat_lon_coords(data)
        data = with_level_index_vars(data, depth_levels=data_layout.depth_levels)
        means = with_level_index_vars(means, depth_levels=data_layout.depth_levels)
        stds = with_level_index_vars(stds, depth_levels=data_layout.depth_levels)
        data = flatten_masks(data)

        anomalies_vars = get_anomalies_vars(data_layout.boundary_var_names)
        if anomalies_vars:
            data, means, stds = compute_anomalies(data, means, stds, anomalies_vars)

        def expand_levels(dataset: xr.Dataset) -> xr.Dataset:
            canonical = xr.Dataset(attrs=dataset.attrs)
            for name, coordinate in dataset.coords.items():
                if "lev" not in coordinate.dims:
                    canonical = canonical.assign_coords({name: coordinate})
            for name, variable in dataset.data_vars.items():
                if "lev" not in variable.dims:
                    canonical[str(name)] = variable
                    continue
                for level in range(variable.sizes["lev"]):
                    canonical[f"{name}_{level}"] = variable.isel(lev=level, drop=True)
            return canonical

        canonical_data = expand_levels(data)
        canonical_means = expand_levels(means)
        canonical_stds = expand_levels(stds)
        return canonical_data, canonical_means, canonical_stds, data_layout


class LlcDataSourceConfig(BaseDataSourceConfig[LlcTimeConfig]):
    type: Literal["llc"] = "llc"
    prognostic_vars_key: str = "single_1"
    boundary_vars_key: str = "single_1"
    face: int = Field(default=1, ge=0)
    i_start: int = Field(default=0, ge=0)
    i_end: int = Field(default=720, gt=0)
    j_start: int = Field(default=0, ge=0)
    j_end: int = Field(default=720, gt=0)

    @pydantic.model_validator(mode="after")
    def validate_time_splits(self) -> Self:
        if self.train_time.overlaps(self.val_time):
            raise ValueError(
                f"Training time range {self.train_time} overlaps "
                f"with validation time range {self.val_time}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_crop_bounds(self) -> Self:
        if self.i_end <= self.i_start:
            raise ValueError("LLC crop bounds must satisfy i_start < i_end")
        if self.j_end <= self.j_start:
            raise ValueError("LLC crop bounds must satisfy j_start < j_end")
        return self

    def canonicalize_datasets(
        self,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, DataLayout]:
        return canonicalize_llc_datasets(
            data,
            means,
            stds,
            face=self.face,
            i_start=self.i_start,
            i_end=self.i_end,
            j_start=self.j_start,
            j_end=self.j_end,
            prognostic_vars_key=self.prognostic_vars_key,
            boundary_vars_key=self.boundary_vars_key,
        )


DataSourceConfig = Annotated[
    Om4DataSourceConfig | LlcDataSourceConfig,
    Field(discriminator="type"),
]


class DataConfig(BaseConfig):
    sources: list[DataSourceConfig] = Field(
        description=(
            "Data sources to include, each with explicit data/means/stds "
            "locations. These are resolved relative to data_root."
        ),
        min_length=1,
    )
    loading: DataLoadingConfig = Field(default_factory=CpuDataLoadingConfig)
    hist: int = 1
    loader_version: str = str(LoaderVersion.OM4_TORCH.value)
    normalize_before_mask: bool = True
    masked_fill_value: float = 0.0
    concurrent_compute: bool = False

    def build(
        self,
        data_root: ResolvedLocation,
    ) -> DataBundle:
        loader_version = LoaderVersion(self.loader_version)
        use_dask = loader_version != LoaderVersion.OM4_TORCH

        source_splits = [
            source_cfg.build(
                data_root,
                use_dask=use_dask,
                is_primary=index == 0,
            )
            for index, source_cfg in enumerate(self.sources)
        ]
        train_sources = [splits.train for splits in source_splits]
        val_sources = [splits.val for splits in source_splits]
        primary_source = train_sources[0]
        data_layout = primary_source.data_layout
        if any(source.data_layout != data_layout for source in train_sources[1:]):
            raise ValueError("All data sources must use the same data layout")

        return DataBundle(
            train_sources=train_sources,
            val_sources=val_sources,
            inference_source=source_splits[0].inference,
            loader_version=loader_version,
            data_layout=data_layout,
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


PerceiverImpl = Literal["auto", "naive", "flash"]


class PerceiverConfig(BaseConfig):
    """A standard config interface to various perceiver implementations.

    Builds either a regular Perceiver (for the encoder, via ``build``) or a
    PerceiverIO (for the decoder, via ``build_io``).  Both respect the shared
    ``implementation`` setting from ``SamudraMultiConfig.perceiver_implementation``.
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
    cross_heads: int = Field(
        default=1,
        ge=1,
        description="Number of heads in Perceiver input/output cross-attention.",
    )
    latent_heads: int = Field(
        default=8,
        ge=1,
        description="Number of heads in latent self-attention.",
    )
    cross_dim_head: int = Field(
        default=64,
        ge=1,
        description="Width of each cross-attention head. The transported value width is cross_heads * cross_dim_head.",
    )
    latent_dim_head: int = Field(
        default=64,
        ge=1,
        description="Width of each latent self-attention head.",
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

        num_freq_bands = 4
        fourier_dim = fourier_features_2d_dim(num_freq_bands)
        # Use the same explicit 2D Fourier features in both implementations so
        # intra-patch positions are encoded equivalently.
        return nn.Sequential(
            FourierFeatures2D(num_freq_bands=num_freq_bands, max_freq=max_freq),
            Perceiver(
                # Retained for compatibility with the original constructor;
                # positional features are added explicitly above.
                num_freq_bands=num_freq_bands,
                max_freq=max_freq,
                depth=self.depth,
                input_axis=2,
                input_channels=in_channels + fourier_dim,
                num_classes=out_channels,
                latent_dim=self.latent_dim,
                num_latents=self.num_latents,
                cross_heads=self.cross_heads,
                latent_heads=self.latent_heads,
                cross_dim_head=self.cross_dim_head,
                latent_dim_head=self.latent_dim_head,
                weight_tie_layers=True,
                fourier_encode_data=False,
                self_per_cross_attn=2,
                attention_backend=_attention_backend(implementation),
            ),
        )

    def build_io(
        self,
        in_channels: int,
        queries_dim: int,
        out_channels: int,
        implementation: PerceiverImpl,
    ) -> nn.Module:
        """Build a PerceiverIO (used by the decoder)."""
        return PerceiverIO(
            depth=self.depth,
            dim=in_channels,
            queries_dim=queries_dim,
            logits_dim=out_channels,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=self.cross_heads,
            latent_heads=self.latent_heads,
            cross_dim_head=self.cross_dim_head,
            latent_dim_head=self.latent_dim_head,
            weight_tie_layers=True,
            decoder_ff=True,
            attention_backend=_attention_backend(implementation),
        )


def _attention_backend(
    implementation: PerceiverImpl,
) -> Literal["auto", "math", "flash"]:
    match implementation:
        case "auto":
            return "auto"
        case "naive":
            return "math"
        case "flash":
            return "flash"
        case _:
            assert_never(implementation)


EncoderArchitecture = Literal[
    "perceiver", "spatial_query", "spatial_grid", "patch_moment", "dct_detail"
]


class EncoderConfig(BaseConfig):
    architecture: EncoderArchitecture = "perceiver"
    perceiver: PerceiverConfig = PerceiverConfig()
    spatial_query_shape: tuple[int, int] = (2, 2)
    spatial_query_channels: int = Field(default=32, ge=1)
    queries_dim: int = Field(default=64, ge=1)
    moment_count: int = Field(default=4, ge=1)
    mean_channels: int = Field(default=32, ge=1)
    moment_add_geometry: bool = Field(default=True)
    detail_count: int = Field(default=4, ge=0)

    def build(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        max_lat_size: int,
        max_lon_size: int,
        implementation: PerceiverImpl,
    ) -> nn.Module:
        max_patch_size = patch_from(patch_extent, max_lat_size, max_lon_size)
        if self.architecture == "patch_moment":
            return PatchMomentEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                patch_extent=patch_extent,
                moment_count=self.moment_count,
                mean_channels=self.mean_channels,
                add_geometry=self.moment_add_geometry,
            )
        if self.architecture == "dct_detail":
            if self.detail_count >= max_patch_size[0] * max_patch_size[1]:
                raise ValueError(
                    "encoder.detail_count must be smaller than the number of "
                    f"cells in the largest patch ({max_patch_size})."
                )
            return DCTDetailEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                patch_extent=patch_extent,
                detail_count=self.detail_count,
            )
        if self.architecture in ("spatial_query", "spatial_grid"):
            query_count = self.spatial_query_shape[0] * self.spatial_query_shape[1]
            query_channels = (
                out_channels
                if self.architecture == "spatial_grid"
                else self.spatial_query_channels
            )
            if (
                self.architecture == "spatial_query"
                and query_count * query_channels != out_channels
            ):
                raise ValueError(
                    "spatial_query_shape product times spatial_query_channels "
                    f"must equal embedding_dim={out_channels}."
                )
            num_freq_bands = 4
            spatial_perceiver = SpatialQueryPerceiver(
                query_shape=self.spatial_query_shape,
                queries_dim=self.queries_dim,
                channels_per_query=query_channels,
                perceiver_io=self.perceiver.build_io(
                    in_channels + fourier_features_2d_dim(num_freq_bands),
                    self.queries_dim,
                    query_channels,
                    implementation,
                ),
                num_freq_bands=num_freq_bands,
                max_freq=max(*max_patch_size),
            )
            if self.architecture == "spatial_grid":
                return SpatialLatentGridEncoder(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    patch_extent=patch_extent,
                    spatial_perceiver=spatial_perceiver,
                )
            return PerceiverEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                patch_extent=patch_extent,
                perceiver=spatial_perceiver,
            )
        return PerceiverEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_extent=patch_extent,
            perceiver=self.perceiver.build(
                in_channels, out_channels, max_patch_size, implementation
            ),
        )


DecoderArchitecture = Literal[
    "perceiver_io", "direct_cross_attention", "dct_detail", "structured_local"
]


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
    architecture: DecoderArchitecture = Field(
        default="perceiver_io",
        description="Decoder core. 'direct_cross_attention' applies only the Perceiver IO decode stage to processor tokens, avoiding a second latent bottleneck.",
    )
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
    output_overlap_patches: int = Field(
        default=0,
        ge=0,
        description="Number of output-query patch rings decoded on each side of a window and cosine-blended with neighboring windows. Zero preserves hard assembly.",
    )
    processor_conditioning: bool = Field(
        default=False,
        description="Add a smoothly upsampled processor-grid conditioning field to globally assembled decoder features.",
    )
    pixel_refinement: bool = Field(
        default=False,
        description="Apply an identity-initialized full-resolution depthwise residual block after output assembly.",
    )
    detail_count: int = Field(
        default=4,
        ge=0,
        description="Number of non-DC patch modes synthesized by the dct_detail decoder.",
    )
    residual_hidden_dim: int = Field(default=128, ge=1)
    residual_heads: int = Field(default=4, ge=1)
    residual_dim_head: int = Field(default=32, ge=1)
    residual_neighborhood_radius: int = Field(default=1, ge=0)
    residual_position_bias_strength: float = Field(default=8.0, gt=0)
    residual_query_chunk_size: int = Field(default=4096, ge=1)

    def build(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        implementation: PerceiverImpl,
    ) -> nn.Module:
        if (
            self.processor_conditioning
            and self.architecture != "direct_cross_attention"
        ):
            raise ValueError(
                "processor_conditioning requires architecture='direct_cross_attention'."
            )
        if self.pixel_refinement and self.architecture == "perceiver_io":
            raise ValueError(
                "pixel_refinement requires architecture='direct_cross_attention' "
                "or architecture='dct_detail'."
            )
        if self.architecture == "dct_detail":
            return DCTDetailDecoder(
                in_channels=in_channels,
                out_channels=out_channels,
                patch_extent=patch_extent,
                detail_count=self.detail_count,
                pixel_refinement=self.pixel_refinement,
            )
        if self.architecture == "structured_local":
            return StructuredLocalDecoder(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=self.residual_hidden_dim,
                heads=self.residual_heads,
                dim_head=self.residual_dim_head,
                neighborhood_radius=self.residual_neighborhood_radius,
                position_bias_strength=self.residual_position_bias_strength,
                query_chunk_size=self.residual_query_chunk_size,
            )
        if self.architecture == "perceiver_io":
            decoder_core = self.perceiver.build_io(
                in_channels, self.queries_dim, out_channels, implementation
            )
        else:
            decoder_core = DirectCrossAttentionIO(
                input_dim=in_channels,
                queries_dim=self.queries_dim,
                output_dim=out_channels,
                heads=self.perceiver.cross_heads,
                dim_head=self.perceiver.cross_dim_head,
            )

        return PerceiverDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_extent=patch_extent,
            queries_dim=self.queries_dim,
            perceiver_io=decoder_core,
            window_patches=self.window_patches,
            context_patches=self.context_patches,
            output_overlap_patches=self.output_overlap_patches,
            processor_conditioning=self.processor_conditioning,
            pixel_refinement=self.pixel_refinement,
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
    drop_path_rate: float = Field(
        default=0.0,
        description="Shortcut dropout rate. The chance we turn off skip connections in the UNet. Reasonable values are 0.1-0.3. Use 0.0 to disable.",
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
            drop_path_rate=self.drop_path_rate,
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
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        hist: int,
        grid_sizes: list[GridSize],
    ) -> BaseModel:
        pass


class SamudraConfig(BaseModelConfig):
    unet: UNetBackboneConfig = UNetBackboneConfig()
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
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        hist: int,
        grid_sizes: list[GridSize],
    ) -> Samudra:
        if len(grid_sizes) != 1:
            raise ValueError(
                "Samudra only supports training at a single scale! Please configure exactly one data source."
            )
        in_channels = prog_channels + boundary_channels
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
            pos_channels=self.pos_channels,
            add_3d_coordinates=add_3d_coordinates,
            hist=hist,
            grid_size=grid_sizes[0],
            gradient_detach_interval=self.gradient_detach_interval,
            use_bfloat16=self.use_bfloat16,
        )


class SamudraMultiConfig(BaseModelConfig):
    encoder: EncoderConfig = EncoderConfig()
    processor: UNetBackboneConfig = UNetBackboneConfig()
    decoder: DecoderConfig = DecoderConfig()
    perceiver_implementation: PerceiverImpl = Field(
        default="auto",
        description="Perceiver attention implementation shared by the encoder and decoder. "
        "'auto' lets PyTorch select the best SDPA kernel; 'naive' "
        "forces math attention and 'flash' forces PyTorch FlashAttention.",
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
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        hist: int,
        grid_sizes: list[GridSize],
    ) -> SamudraMulti:
        assert len(self.patch_extent) == 2, "patch_extent must be a pair of floats."
        extent = self.patch_extent[0], self.patch_extent[1]

        max_lat_size, max_lon_size = (
            max(g[0] for g in grid_sizes),
            max(g[1] for g in grid_sizes),
        )

        dct_encoder = self.encoder.architecture == "dct_detail"
        dct_decoder = self.decoder.architecture == "dct_detail"
        if dct_encoder != dct_decoder:
            raise ValueError(
                "The dct_detail representation requires paired encoder and decoder "
                "architectures."
            )
        if dct_encoder and self.encoder.detail_count != self.decoder.detail_count:
            raise ValueError(
                "Paired dct_detail encoder and decoder detail_count values must match."
            )

        impl = self.perceiver_implementation
        if impl == "flash" and not self.use_bfloat16:
            raise ValueError(
                "Perceiver implementation resolves to flash attention. "
                "Please set `use_bfloat16=True` or `perceiver_implementation='naive'`."
            )

        in_channels = prog_channels + boundary_channels
        total_in_channels = in_channels + (3 if self.add_3d_coordinates else 0)

        encoder = self.encoder.build(
            total_in_channels,
            self.embedding_dim,
            extent,
            max_lat_size,
            max_lon_size,
            impl,
        )
        processor = self.processor.build(
            self.embedding_dim,
            self.pad,
            self.checkpointing,
        )
        decoder = self.decoder.build(
            processor.out_channels,
            out_channels,
            getattr(encoder, "output_patch_extent", extent),
            impl,
        )

        add_3d_coordinates = Concat3dCoordinates() if self.add_3d_coordinates else None
        return SamudraMulti(
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


class SamudraMiniConfig(BaseModelConfig):
    perceiver: PerceiverConfig = PerceiverConfig()
    perceiver_implementation: PerceiverImpl = Field(
        default="auto",
        description="Perceiver attention implementation for the single PerceiverIO model. "
        "'auto' lets PyTorch select the best SDPA kernel; 'naive' "
        "forces math attention and 'flash' forces PyTorch FlashAttention.",
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
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        hist: int,
        grid_sizes: list[GridSize],
    ) -> SamudraMini:
        if self.add_3d_coordinates:
            raise ValueError(
                "SamudraMini always uses learned Cartesian coordinate embeddings. "
                "Please set `add_3d_coordinates=False`."
            )

        impl = self.perceiver_implementation
        if impl == "flash" and not self.use_bfloat16:
            raise ValueError(
                "Perceiver implementation resolves to flash attention. "
                "Please set `use_bfloat16=True` or `perceiver_implementation='naive'`."
            )

        in_channels = prog_channels + boundary_channels
        perceiver_io = self.perceiver.build_io(
            self.embedding_dim,
            self.queries_dim,
            out_channels,
            impl,
        )
        return SamudraMini(
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


AnyModelConfig = SamudraConfig | SamudraMultiConfig | SamudraMiniConfig


class DistributedConfig(BaseConfig):
    dist_url: str | None = None
    world_size: int | None = None
    rank: int | None = None
    gpu: int | None = None
    dist_backend: str | None = None


class SearchRunConfig(BaseConfig):
    """Identity of a training run managed by an architecture search."""

    name: str
    run_id: str | None = None
    candidate: str
    rung: int = Field(ge=0)
    target_epochs: int = Field(ge=1)
    objective: str
    executor: str
    code_commit: str | None = None
    code_layer_sha256: str | None = None
    container_image_ref: str | None = None
    container_git_commit: str | None = None
    artifacts_uri: str | None = None
    job_id: str | None = None
    parent_checkpoint: str | None = None
    world_size: int = Field(default=1, ge=1)
    local_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    effective_global_batch_size: int = Field(default=1, ge=1)
    adaptive_data_parallel: bool = False


class ExperimentConfig(BaseConfig):
    name: str = "cm4_samudra"
    rand_seed: int = 1
    base_output_dir: str = "train"
    # we require this to be set by the user but have optional here
    # so we can leave it out of config files
    data_root: Location | None = None
    wandb: WandBConfig
    search: SearchRunConfig | None = None

    @cached_property
    def output_dir(self) -> Path:
        return Path(self.base_output_dir) / f"{self.name}"

    @cached_property
    def nets_dir(self) -> Path:
        return self.output_dir / "saved_nets"

    @cached_property
    def resolved_data_root(self) -> ResolvedLocation:
        # Default to the current directory when no data_root is given. Absolute
        # data locations (e.g. an s3:// demo source) ignore the root entirely, so
        # they run flagless; relative locations resolve against cwd and still
        # fail loudly at open() if the data isn't there.
        default_root = LocalLocation(path=Path.cwd())
        if self.data_root is None:
            return default_root
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


class RolloutValidationConfig(BaseConfig):
    """Configuration for expensive autoregressive validation during training."""

    model_steps: int = Field(
        default=-1,
        ge=-1,
        description=(
            "Number of autoregressive model steps to run over val_time for "
            "raw-field rollout validation. Use -1 to use the full validation "
            "period. Set rollout_validation to null to disable this path. "
            "Ignored when days is set."
        ),
    )
    days: list[int] = Field(
        default_factory=list,
        description=(
            "Forecast-day horizons to run over val_time for raw-field rollout "
            "validation, e.g. [30, 90, 180, 360]. Takes precedence over "
            "model_steps."
        ),
    )
    steps_forward: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of autoregressive model steps to run per rollout validation "
            "chunk before recording metrics. Higher values reduce CPU/GPU "
            "synchronization overhead but increase target materialization memory."
        ),
    )
    frequency: int = Field(
        default=1,
        ge=1,
        description=(
            "How often to run rollout validation, in epochs. Epochs are 1-based, "
            "so a value of 10 runs on epochs 1, 11, 21, ..."
        ),
    )

    @pydantic.field_validator("model_steps")
    @classmethod
    def _model_steps_must_enable_rollout(cls, model_steps: int) -> int:
        if model_steps == 0:
            raise ValueError(
                "rollout_validation.model_steps must be positive or -1; set "
                "rollout_validation to null to disable rollout validation"
            )
        return model_steps

    @pydantic.field_validator("days")
    @classmethod
    def _days_must_be_positive(cls, days: list[int]) -> list[int]:
        invalid_days = [day for day in days if day <= 0]
        if invalid_days:
            raise ValueError(
                f"rollout_validation.days must be positive, got {invalid_days}"
            )
        return days


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
    rollout_validation: RolloutValidationConfig | None = None
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
    data_stride: list[int] = [1]
    steps: list[int] = [4]
    step_transition: list[int] = []
    inference_epochs: list[int] = [-1]

    # Config components
    experiment: ExperimentConfig
    data: DataConfig
    model: AnyModelConfig

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


class ObsMetricsConfig(BaseConfig):
    """Where the observation products live, and over what period to score.

    Supplying this block is what turns observation metrics on: an eval job then
    compares its rollout against DUACS, OISST and ARGO-IAP once the rollout is
    on disk. Omit it (or pass `--observations=null`) to skip that phase.
    """

    duacs_location: Location = Field(
        default=UnresolvedLocation(path="obs/duacs.zarr"),
        description="DUACS surface geostrophic velocity, on its native grid.",
    )
    oisst_location: Location = Field(
        default=UnresolvedLocation(path="obs/oisst.zarr"),
        description="OISST sea-surface temperature, on its native grid.",
    )
    argo_iap_location: Location = Field(
        default=UnresolvedLocation(path="obs/argo-iap.zarr"),
        description="ARGO-IAP gridded temperature, on its native grid.",
    )
    rmse_start: str = Field(
        default="2015-01-01",
        description=(
            "Start of the primary scoring window. Must begin a complete calendar "
            "year: the score and its bootstrap both use equal-year blocks."
        ),
    )
    rmse_end: str = Field(
        default="2022-12-31",
        description="End of the primary scoring window; must end a complete calendar year.",
    )
    bootstrap_samples: int = Field(
        default=10_000,
        ge=0,
        description="Calendar-year block-bootstrap draws for the 95% CI; 0 disables.",
    )
    velocity_kind: Literal["absolute", "anomaly"] = Field(
        default="absolute",
        description=(
            "Which DUACS geostrophic velocity to compare against. 'absolute' "
            "matches the model velocity derived from zos."
        ),
    )
    baselines: list[str] = Field(
        default_factory=lambda: ["om4"],
        description=(
            "Reference rollouts scored alongside the model. 'om4' scores the "
            "ground-truth OM4 data already staged for the eval, which is what "
            "makes the model's own numbers interpretable."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "ObsMetricsConfig":
        unknown = set(self.baselines) - {"om4"}
        if unknown:
            raise ValueError(f"Unknown baselines {sorted(unknown)}; supported: ['om4']")
        if self.window[0] > self.window[1]:
            raise ValueError(
                f"rmse_start ({self.rmse_start}) must not be after rmse_end "
                f"({self.rmse_end})"
            )
        return self

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The scoring window, with a date-only end bound taken to end-of-day.

        OM4 samples are stamped at 12:00, so a bare `2022-12-31` end bound would
        exclude that day's sample. Currently masked because the observation
        products end earlier and the window gets trimmed to the shared span, but
        it becomes a silent one-sample loss the moment coverage extends past
        `rmse_end` -- and the 7.5-day tolerance in the complete-calendar-year
        check is far too loose to notice.
        """
        end = pd.Timestamp(self.rmse_end)
        if end == end.normalize():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
        return pd.Timestamp(self.rmse_start), end


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
    experiment: ExperimentConfig
    data: DataConfig
    model: AnyModelConfig
    observations: ObsMetricsConfig | None = Field(
        default=None,
        description=(
            "Observation-based metrics computed after the rollout finishes. "
            "Present means enabled; omit to skip the phase."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _observations_need_a_saved_rollout(self) -> Self:
        # Caught here rather than after the rollout: observation metrics score a
        # rollout read back from disk, and discovering the misconfiguration at
        # the end would waste the whole job.
        if self.observations is not None and not self.save_zarr:
            raise ValueError(
                "observations requires save_zarr=true: the metrics are computed "
                "from the predictions.zarr the rollout writes."
            )
        return self

    def prepare_output_dirs(self) -> None:
        self.experiment.output_dir.mkdir(parents=True, exist_ok=True)


AnyTopLevelConfig = TrainConfig | EvalConfig
