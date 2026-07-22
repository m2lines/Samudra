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
    CanonicalResampleEncoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
    ProcessorGeometryConditioner,
    ResampleAttentionResidualDecoder,
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
    CanonicalResampleEncoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    ResampleProjectionDecoder,
    ResampleAttentionResidualDecoder,
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
        encoder: PerceiverEncoder | DirectPatchEncoder | CanonicalResampleEncoder,
        processor: nn.Module,
        decoder: (
            PerceiverDecoder
            | DirectPatchDecoder
            | ResampleProjectionDecoder
            | ResampleAttentionResidualDecoder
        ),
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
        processor_iterations: int = 1,
        processor_geometry: ProcessorGeometryConditioner | None = None,
        zero_depth_reconstruction_weight: float = 0.0,
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
        if processor_iterations < 0:
            raise ValueError("processor_iterations must be non-negative.")
        self.processor_iterations = processor_iterations
        self.processor_geometry = processor_geometry
        if zero_depth_reconstruction_weight < 0:
            raise ValueError("zero_depth_reconstruction_weight must be non-negative.")
        self.zero_depth_reconstruction_weight = zero_depth_reconstruction_weight

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
                        CanonicalResampleEncoder,
                        PerceiverDecoder,
                        DirectPatchEncoder,
                        DirectPatchDecoder,
                        ResampleProjectionDecoder,
                        ResampleAttentionResidualDecoder,
                    ),
                ),
            )

    def encode(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode physical inputs and return content plus its canonical grid."""
        # Prognostic and boundary are carried as separate tensors through the
        # data pipeline, but this encoder still expects a single concatenated
        # input.  The dual-perceiver encoder that fuses them at the token level
        # (enabling cross-resolution) lands in a follow-up PR.
        fts = torch.cat((prognostic, boundary), dim=1)
        if self.maybe_add_3d_coordinates is not None:
            fts = self.maybe_add_3d_coordinates(fts, ctx.input_resolution_cpu)
        fts = self.encoder(fts, ctx.input_resolution_cpu)
        latent_resolution = self.encoder.output_resolution(ctx.input_resolution_cpu)
        return fts, latent_resolution

    def process(
        self,
        fts: torch.Tensor,
        latent_resolution: tuple[torch.Tensor, torch.Tensor],
        iterations: int | None = None,
    ) -> torch.Tensor:
        """Apply the shared processor zero or more times in latent space."""
        count = self.processor_iterations if iterations is None else iterations
        if count < 0:
            raise ValueError("Processor iteration count must be non-negative.")
        for _ in range(count):
            if self.processor_geometry is not None:
                fts = self.processor_geometry(fts, latent_resolution)
            fts = self.processor(fts)
        return fts

    def decode(
        self,
        fts: torch.Tensor,
        latent_resolution: tuple[torch.Tensor, torch.Tensor],
        ctx: GridContext,
    ) -> Prognostic:
        """Render latent content on the requested output grid."""
        fts = self.decoder(
            fts,
            ctx.output_resolution_cpu,
            source_resolution=latent_resolution,
        )
        fts = fts.to(torch.float32)
        return torch.where(ctx.label_mask, fts, 0.0)

    def reconstruct_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        """Decode the learned representation without applying the processor."""
        input_lat, input_lon = ctx.input_resolution_cpu
        output_lat, output_lon = ctx.output_resolution_cpu
        same_grid = torch.equal(input_lat, output_lat) and torch.equal(
            input_lon, output_lon
        )
        if not same_grid or prognostic.shape[-2:] != ctx.label_mask.shape[-2:]:
            raise ValueError(
                "Zero-depth reconstruction loss currently requires identical input "
                "and output grids so the source mask and target are unambiguous."
            )
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, latent_resolution = self.encode(prognostic, boundary, ctx)
            return self.decode(fts, latent_resolution, ctx)

    def training_auxiliary_loss(self, train_data, loss_fn):
        """Apply the optional zero-depth inverse objective once per batch."""
        if self.zero_depth_reconstruction_weight == 0:
            return None
        prognostic, boundary = train_data.get_initial_input()
        reconstruction = self.reconstruct_once(prognostic, boundary, train_data.ctx)
        return self.zero_depth_reconstruction_weight * loss_fn(
            reconstruction, prognostic
        )

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, latent_resolution = self.encode(prognostic, boundary, ctx)
            fts = self.process(fts, latent_resolution)

            # TODO(alxmrs): When the output resolution differs from the input (i.e. in a "mix" schedule), we cannot use
            #  residual predictions (`self.pred_residuals` must be `False`).
            return self.decode(fts, latent_resolution, ctx)
