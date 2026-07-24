# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
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
    CanonicalResampleEncoder,
    ContinuousResampleAttentionResidualDecoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    PatchMomentEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
    ProcessorGeometryConditioner,
    ResampleAttentionResidualDecoder,
    ResampleProjectionDecoder,
)
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.utils.ctx import GridContext
from samudra.utils.device import autocast, get_device
from samudra.utils.output import ModelInferenceOutput

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from samudra.config import Checkpointing
    from samudra.datasets import InferenceDataset, TrainData

_checkpoint_types: tuple[type, ...] = (
    nn.LayerNorm,
    FeedForward,
    nn.Linear,
    Perceiver,
    PerceiverDecoder,
    PerceiverEncoder,
    CanonicalResampleEncoder,
    PatchMomentEncoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    ResampleProjectionDecoder,
    ResampleAttentionResidualDecoder,
    ContinuousResampleAttentionResidualDecoder,
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
        encoder: (
            PerceiverEncoder
            | DirectPatchEncoder
            | CanonicalResampleEncoder
            | PatchMomentEncoder
        ),
        processor: nn.Module,
        decoder: (
            PerceiverDecoder
            | DirectPatchDecoder
            | ResampleProjectionDecoder
            | ResampleAttentionResidualDecoder
            | ContinuousResampleAttentionResidualDecoder
        ),
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
        processor_iterations: int = 1,
        processor_geometry: ProcessorGeometryConditioner | None = None,
        boundary_encoder: BoundaryEncoder | None = None,
        zero_depth_reconstruction_weight: float = 0.0,
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
        if processor_iterations < 0:
            raise ValueError("processor_iterations must be non-negative.")
        self.processor_iterations = processor_iterations
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
                        PatchMomentEncoder,
                        PerceiverDecoder,
                        DirectPatchEncoder,
                        DirectPatchDecoder,
                        ResampleProjectionDecoder,
                        ResampleAttentionResidualDecoder,
                        ContinuousResampleAttentionResidualDecoder,
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
        iterations: int | None = None,
        boundary: Boundary | None = None,
        boundary_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the shared processor zero or more times in latent space."""
        count = self.processor_iterations if iterations is None else iterations
        if count < 0:
            raise ValueError("Processor iteration count must be non-negative.")
        for _ in range(count):
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
                scale = self.processor_residual_scale.to(dtype=fts.dtype)
                fts = latent_state + scale * fts
        return fts

    def latent_forecast(
        self, train_data: "TrainData", depths: list[int]
    ) -> dict[int, Prognostic]:
        """Encode once, then decode physical forecasts after selected depths.

        Depth ``N`` consumes the boundary forcing for steps ``0..N-1`` and is
        paired with ``TrainData`` label ``N-1``, i.e. the physical target at
        ``t + N * dt``. Intermediate processor states remain latent and are not
        decoded/re-encoded.
        """
        if not depths or any(depth <= 0 for depth in depths):
            raise ValueError("Latent forecast depths must be positive and non-empty.")
        selected = set(depths)
        maximum = max(selected)
        if len(train_data) < maximum:
            raise ValueError(
                f"Depth {maximum} requires {maximum} physical steps, but the "
                f"batch contains {len(train_data)}."
            )

        prognostic, initial_boundary = train_data.get_initial_input()
        outputs: dict[int, Prognostic] = {}
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, latent_resolution = self.encode(
                prognostic, initial_boundary, train_data.ctx
            )
            for step in range(maximum):
                _, boundary = train_data.get_input(step)
                fts = self.process(
                    fts,
                    latent_resolution,
                    iterations=1,
                    boundary=boundary,
                    boundary_resolution=train_data.ctx.input_resolution_cpu,
                )
                depth = step + 1
                if depth in selected:
                    outputs[depth] = self.decode(fts, latent_resolution, train_data.ctx)
        return outputs

    def initialize_rollout(
        self, initial_prognostic: Prognostic, ctx: GridContext
    ) -> torch.Tensor:
        """Encode the initial physical state once for a latent rollout."""
        if self.boundary_encoder is None:
            return super().initialize_rollout(initial_prognostic, ctx)
        if self.pred_residuals:
            raise ValueError(
                "Latent autoregression requires absolute decoder outputs; "
                "pred_residuals must be false."
            )
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, _ = self.encode(initial_prognostic.to(get_device()), None, ctx)
        return fts

    def inference(
        self,
        dataset: "InferenceDataset",
        rollout_state: torch.Tensor,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> ModelInferenceOutput:
        """Advance a latent state without decoding/re-encoding between steps."""
        if self.boundary_encoder is None:
            return super().inference(
                dataset,
                rollout_state,
                steps_completed=steps_completed,
                num_steps=num_steps,
                epoch=epoch,
            )
        if num_steps is None or num_steps <= 0:
            raise ValueError("Latent inference requires a positive num_steps.")
        if self.processor_iterations != 1:
            raise ValueError(
                "Latent physical-time inference requires processor_iterations=1."
            )

        out_shape = (num_steps, *dataset[0][-1].shape[1:])
        pred_tensor = torch.zeros(out_shape, device=get_device())
        fts = rollout_state.to(get_device())
        latent_resolution = self.encoder.output_resolution(
            dataset.ctx.input_resolution_cpu
        )

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            for step in range(num_steps):
                physical_step = steps_completed + step
                logger.info(
                    f"Inference [epoch {epoch}]: latent rollout step "
                    f"{physical_step} of {steps_completed + num_steps - 1}."
                )
                boundary = dataset.get_boundary(physical_step).to(device=fts.device)
                fts = self.process(
                    fts,
                    latent_resolution,
                    iterations=1,
                    boundary=boundary,
                    boundary_resolution=dataset.ctx.input_resolution_cpu,
                )
                pred_tensor[step] = self.decode(fts, latent_resolution, dataset.ctx)[0]

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)
        return ModelInferenceOutput(
            pred_tensor,
            target_tensor,
            target_time,
            rollout_state=fts,
        )

    def forward(
        self,
        train_data: "TrainData",
        loss_fn=None,
        processor_depth: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Use true latent lead-time training when a processor depth is selected."""
        if processor_depth is None:
            return super().forward(train_data, loss_fn=loss_fn)

        prediction = self.latent_forecast(train_data, [processor_depth])[
            processor_depth
        ]
        if loss_fn is None:
            return [prediction]
        loss = loss_fn(prediction, train_data.get_label(processor_depth - 1))
        auxiliary_loss = self.training_auxiliary_loss(train_data, loss_fn)
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        return loss

    def decode(
        self,
        fts: torch.Tensor,
        latent_resolution: tuple[torch.Tensor, torch.Tensor],
        ctx: GridContext,
    ) -> Prognostic:
        """Render latent content on the requested output grid."""
        source_valid_mask = ctx.input_mask
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

    def reconstruction_context(
        self, prognostic: Prognostic, ctx: GridContext
    ) -> GridContext:
        """Build the source-grid context used by zero-depth reconstruction."""
        if ctx.input_mask is None:
            input_lat, input_lon = ctx.input_resolution_cpu
            output_lat, output_lon = ctx.output_resolution_cpu
            same_grid = torch.equal(input_lat, output_lat) and torch.equal(
                input_lon, output_lon
            )
            if not same_grid or prognostic.shape[-2:] != ctx.label_mask.shape[-2:]:
                raise ValueError(
                    "Cross-grid zero-depth reconstruction requires an input mask "
                    "so the source-grid objective is unambiguous."
                )
            input_mask = ctx.label_mask
        else:
            input_mask = ctx.input_mask
        if prognostic.shape[-2:] != input_mask.shape[-2:]:
            raise ValueError(
                "The zero-depth reconstruction mask must match the prognostic "
                f"source grid; got {tuple(input_mask.shape[-2:])} and "
                f"{tuple(prognostic.shape[-2:])}."
            )
        return GridContext(
            label_mask=input_mask,
            input_resolution_cpu=ctx.input_resolution_cpu,
            output_resolution_cpu=ctx.input_resolution_cpu,
            input_mask=input_mask,
        )

    def reconstruct_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        """Decode the learned representation without applying the processor."""
        source_ctx = self.reconstruction_context(prognostic, ctx)
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            fts, latent_resolution = self.encode(prognostic, boundary, source_ctx)
            return self.decode(fts, latent_resolution, source_ctx)

    def training_auxiliary_loss(self, train_data, loss_fn):
        """Apply a source-grid zero-depth inverse MSE once per batch."""
        if self.zero_depth_reconstruction_weight == 0:
            return None
        del loss_fn
        prognostic, boundary = train_data.get_initial_input()
        reconstruction = self.reconstruct_once(prognostic, boundary, train_data.ctx)
        return self.zero_depth_reconstruction_weight * torch.nn.functional.mse_loss(
            reconstruction,
            prognostic,
            reduction="none",
        ).mean(dim=(0, 2, 3))

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.boundary_encoder is not None and self.processor_iterations != 1:
                raise ValueError(
                    "A single forward_once call has only one forcing tensor. "
                    "Use latent_forecast with a boundary sequence for zero-to-N "
                    "physical-time processor rollout."
                )
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
