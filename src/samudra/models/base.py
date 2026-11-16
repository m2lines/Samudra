# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Need to return step-wise losses for logging

import logging

import torch

from samudra.constants import Boundary, Prognostic
from samudra.utils.ctx import BatchGrid

logger = logging.getLogger(__name__)

from samudra.datasets import InferenceDataset, ModelBatch
from samudra.utils.device import get_device
from samudra.utils.output import ModelInferenceOutput


class BaseModel(torch.nn.Module):
    """Abstract base model for neural ocean emulators.

    Provides shared functionality for all ocean emulator models, including
    residual prediction, ocean masking, and gradient detaching for multi-step
    autoregressive rollouts.
    """

    residual_scale: torch.Tensor

    def __init__(
        self,
        in_channels,
        out_channels,
        input_steps,
        pred_residuals,
        last_kernel_size,
        pad,
        gradient_detach_interval: int,
        scale_residuals: bool = False,
        residual_scale_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad: str = pad
        self.pred_residuals = pred_residuals
        if input_steps < 1:
            raise ValueError(f"input_steps must be positive, got {input_steps}")
        self.input_steps = input_steps
        self.gradient_detach_interval = gradient_detach_interval
        if scale_residuals and not pred_residuals:
            raise ValueError("scale_residuals requires pred_residuals")
        self.scale_residuals = scale_residuals
        if scale_residuals:
            # NaN until set_residual_scale() or a checkpoint fills it, so a
            # model that was never given its scale produces NaN, not garbage.
            shape = (out_channels, *(residual_scale_grid or ()))
            self.register_buffer("residual_scale", torch.full(shape, torch.nan))
        elif residual_scale_grid is not None:
            raise ValueError("residual_scale_grid requires scale_residuals")

    def set_residual_scale(self, scale_per_var: torch.Tensor) -> None:
        """Set the fixed per-channel residual scale (normalized units).

        ``scale_per_var`` has one positive value per prognostic variable; it is
        repeated for every output step, since channels are time-major.
        """
        if not self.scale_residuals:
            raise ValueError("Model was built without scale_residuals")
        if scale_per_var.ndim != 1 or self.out_channels % scale_per_var.numel():
            raise ValueError(
                f"Expected one scale per variable dividing {self.out_channels} "
                f"output channels, got shape {tuple(scale_per_var.shape)}"
            )
        if not torch.isfinite(scale_per_var).all() or (scale_per_var <= 0).any():
            raise ValueError("Residual scale must be positive and finite")
        repeats = self.out_channels // scale_per_var.numel()
        self.residual_scale.copy_(scale_per_var.float().repeat(repeats))

    def set_residual_scale_map(self, scale_per_var: torch.Tensor) -> None:
        """Set a fixed per-cell residual scale, shape (variables, lat, lon)."""
        if not self.scale_residuals or self.residual_scale.ndim != 3:
            raise ValueError("Model was built without a per-cell residual scale")
        n_var = scale_per_var.shape[0]
        if (
            scale_per_var.ndim != 3
            or scale_per_var.shape[1:] != self.residual_scale.shape[1:]
            or self.out_channels % n_var
        ):
            raise ValueError(
                f"Expected (variables, {tuple(self.residual_scale.shape[1:])}) "
                f"scales, got {tuple(scale_per_var.shape)}"
            )
        if not torch.isfinite(scale_per_var).all() or (scale_per_var <= 0).any():
            raise ValueError("Residual scale must be positive and finite")
        repeats = self.out_channels // n_var
        self.residual_scale.copy_(scale_per_var.float().repeat(repeats, 1, 1))

    def _assemble_prediction(
        self, prognostic: Prognostic, decoding: Prognostic
    ) -> Prognostic:
        if not self.pred_residuals:
            return decoding
        # Anchor each predicted timestep to the corresponding most-recent input
        # timestep. For 2-in/1-out this is x[t]; for the legacy 2-in/2-out case
        # this remains the original elementwise, same-slot residual assembly.
        residual_base = prognostic[:, -self.out_channels :]
        if self.scale_residuals:
            scale = self.residual_scale
            if scale.ndim == 1:
                scale = scale.view(-1, 1, 1)
            decoding = decoding * scale.unsqueeze(0)
        return residual_base + decoding

    def _advance_prognostic_history(
        self, prognostic: Prognostic, prediction: Prognostic
    ) -> Prognostic:
        """Drop the oldest predicted span and append the new prediction block."""
        return torch.cat((prognostic[:, self.out_channels :], prediction), dim=1)

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> Prognostic:
        raise NotImplementedError()

    def forward(
        self,
        batch: ModelBatch,
        loss_fn=None,
    ) -> torch.Tensor | list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        prog_tensor, _ = batch.get_initial_input()
        for step in range(len(batch)):
            _, boundary_tensor = batch.get_input(step)
            if step > 0:
                if (
                    self.gradient_detach_interval > 0
                    and step % self.gradient_detach_interval == 0
                ):
                    prog_tensor = prog_tensor.detach()

            decodings = self.forward_once(prog_tensor, boundary_tensor, batch.ctx)
            pred = self._assemble_prediction(prog_tensor, decodings)

            if loss_fn is not None:
                if step == 0:
                    loss = loss_fn(
                        pred,
                        batch.get_label(step),
                    )
                else:
                    loss += loss_fn(
                        pred,
                        batch.get_label(step),
                    )

            outputs.append(pred)
            prog_tensor = self._advance_prognostic_history(prog_tensor, pred)

        if loss_fn is None:
            return outputs
        else:
            return loss

    def inference(
        self,
        dataset: InferenceDataset,
        initial_prognostic: torch.Tensor,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> ModelInferenceOutput:
        # `dataset[idx]` returns `(prog, boundary, label)`.
        out_shape = (num_steps, *dataset[0][-1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        initial_prognostic = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)

        prog_tensor = initial_prognostic
        for step in range(num_steps):
            logger.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            boundary_tensor = dataset.get_boundary(
                steps_completed + step,
            ).to(device=prog_tensor.device)

            decodings = self.forward_once(prog_tensor, boundary_tensor, dataset.ctx)
            pred = self._assemble_prediction(prog_tensor, decodings)

            pred_tensor[step] = pred[0]
            prog_tensor = self._advance_prognostic_history(prog_tensor, pred)

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        inference_output = ModelInferenceOutput(
            pred_tensor,
            target_tensor,
            target_time,
            final_prognostic=prog_tensor,
        )
        return inference_output
