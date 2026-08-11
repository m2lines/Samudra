# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from samudra.constants import Boundary, Prognostic
from samudra.models.base import BaseModel
from samudra.models.modules import (
    BoundaryEncoder,
    NativeProjectionEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
    ProcessorGeometryConditioner,
    ResampleProjectionDecoder,
)
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.utils.ctx import GridContext
from samudra.utils.device import autocast

if TYPE_CHECKING:
    from samudra.config import Checkpointing

_checkpoint_types: tuple[type, ...] = (
    nn.LayerNorm,
    FeedForward,
    nn.Linear,
    Perceiver,
    PerceiverDecoder,
    PerceiverEncoder,
    NativeProjectionEncoder,
    ResampleProjectionDecoder,
    UNetBackbone,
    Attention,
)

try:
    from flash_attn.modules.block import (
        Block as FlashBlock,  # type: ignore[import-not-found]
    )
    from flash_perceiver.perceiver import (
        PerceiverBase as FlashPerceiverBase,  # type: ignore[import-not-found]
    )

    _checkpoint_types = _checkpoint_types + (FlashPerceiverBase, FlashBlock)
except ImportError:
    pass


class SamudraMulti(BaseModel):
    """Multi-resolution encoder-processor-decoder model.

    Currently, this model is used only as a physical ocean emulator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        add_3d_coordinates: nn.Module | None,
        encoder: PerceiverEncoder | NativeProjectionEncoder,
        processor: nn.Module,
        decoder: PerceiverDecoder | ResampleProjectionDecoder,
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
        processor_geometry: ProcessorGeometryConditioner | None = None,
        boundary_encoder: BoundaryEncoder | None = None,
        processor_residual: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )

        self.maybe_add_3d_coordinates = add_3d_coordinates
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.use_bfloat16 = use_bfloat16
        self.processor_geometry = processor_geometry
        self.boundary_encoder = boundary_encoder
        processor_out_channels = getattr(processor, "out_channels", None)
        if processor_residual:
            if processor_out_channels is None:
                raise ValueError(
                    "A residual processor requires an explicit output channel width."
                )
            encoder_out_channels = getattr(encoder, "out_channels", None)
            if encoder_out_channels != processor_out_channels:
                raise ValueError(
                    "A residual processor requires equal state and processor widths; "
                    f"got {encoder_out_channels} and {processor_out_channels}."
                )
            self.processor_residual_scale = nn.Parameter(
                torch.zeros(1, processor_out_channels, 1, 1)
            )
        else:
            self.register_parameter("processor_residual_scale", None)

        if checkpointing == "all":
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(m, _checkpoint_types),
            )
        elif checkpointing == "selective":
            # The processor applies checkpointing to its individual layers itself.
            # Checkpoint only the expensive representation heads here so the
            # processor is not wrapped a second time.
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(
                    m,
                    (
                        PerceiverEncoder,
                        PerceiverDecoder,
                        NativeProjectionEncoder,
                        ResampleProjectionDecoder,
                    ),
                ),
            )

    def encode(
        self, prognostic: Prognostic, boundary: Boundary | None, ctx: GridContext
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode physical inputs and return content plus its canonical grid."""
        # Boundary forcing has its own encoder and is injected once per latent
        # physical-time transition. It must not contaminate the state that the
        # decoder learns to invert at depth zero.
        del boundary
        fts = prognostic
        if self.maybe_add_3d_coordinates is not None:
            fts = self.maybe_add_3d_coordinates(fts, ctx.input_resolution_cpu)
        fts = self.encoder(fts, ctx.input_resolution_cpu)
        latent_resolution = self.encoder.output_resolution(ctx.input_resolution_cpu)
        return fts, latent_resolution

    def process(
        self,
        fts: torch.Tensor,
        latent_resolution: tuple[torch.Tensor, torch.Tensor],
        boundary: Boundary | None = None,
        boundary_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Advance the latent state by one physical step.

        Forcing is added through its own encoder rather than mixed into the
        state encoder, so the representation the decoder inverts is a function
        of ocean state alone.
        """
        latent_state = fts
        if self.boundary_encoder is not None:
            if boundary is None:
                raise ValueError(
                    "Boundary-conditioned processor calls require the forcing "
                    "for that physical time step."
                )
            if boundary_resolution is None:
                raise ValueError(
                    "Boundary-conditioned processor calls require the boundary "
                    "grid coordinates."
                )
            boundary_state = boundary[:, -self.boundary_encoder.boundary_channels :]
            encoded_boundary = self.boundary_encoder(
                boundary_state,
                boundary_resolution,
                latent_resolution,
            ).to(dtype=fts.dtype)
            if encoded_boundary.shape != fts.shape:
                raise ValueError(
                    "Encoded boundary forcing and latent state must share shape; "
                    f"got {tuple(encoded_boundary.shape)} and {tuple(fts.shape)}."
                )
            fts = fts + encoded_boundary
        if self.processor_geometry is not None:
            fts = self.processor_geometry(fts, latent_resolution)
        fts = self.processor(fts)
        if self.processor_residual_scale is not None:
            # Zero-initialized per-channel residual: the transition starts as
            # latent persistence and learns away from it.
            scale = self.processor_residual_scale.to(dtype=fts.dtype)
            fts = latent_state + scale * fts
        return fts

    def decode(
        self,
        fts: torch.Tensor,
        latent_resolution: tuple[torch.Tensor, torch.Tensor],
        ctx: GridContext,
    ) -> Prognostic:
        """Render latent content on the requested output grid."""
        source_valid_mask = ctx.input_mask
        if source_valid_mask is None and getattr(
            self.decoder, "requires_source_mask", False
        ):
            source_lat, source_lon = latent_resolution
            output_lat, output_lon = ctx.output_resolution_cpu
            transports = not (
                torch.equal(source_lat, output_lat)
                and torch.equal(source_lon, output_lon)
            )
            # Same-grid decoding reduces exactly to the learned 1x1 channel map,
            # so no interpolation weights need renormalizing. Any other route
            # does interpolate, and doing that unmasked would quietly mix land
            # into ocean cells.
            if transports:
                raise ValueError(
                    "This decoder renders prognostic channels before spatial "
                    "transport and needs the per-channel source wet mask, but "
                    "GridContext.input_mask is unset. Cross-grid decoding "
                    "without it would interpolate across land."
                )
        if source_valid_mask is not None:
            if source_valid_mask.shape[0] < self.decoder.out_channels:
                raise ValueError(
                    "Input validity mask has fewer channels than the decoder "
                    f"output: {source_valid_mask.shape[0]} < "
                    f"{self.decoder.out_channels}."
                )
            source_valid_mask = source_valid_mask[-self.decoder.out_channels :]
        fts = self.decoder(
            fts,
            ctx.output_resolution_cpu,
            source_resolution=latent_resolution,
            valid_mask=source_valid_mask,
        )
        fts = fts.to(torch.float32)
        return torch.where(ctx.label_mask, fts, 0.0)

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, latent_resolution = self.encode(prognostic, boundary, ctx)
            fts = self.process(
                fts,
                latent_resolution,
                boundary=boundary,
                boundary_resolution=ctx.input_resolution_cpu,
            )

            # TODO(alxmrs): When the output resolution differs from the input (i.e. in a "mix" schedule), we cannot use
            #  residual predictions (`self.pred_residuals` must be `False`).
            return self.decode(fts, latent_resolution, ctx)
