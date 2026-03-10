# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Need to return step-wise losses for logging

import logging

import torch

from samudra.constants import Boundary, Prognostic
from samudra.utils.ctx import GridContext

logger = logging.getLogger(__name__)

from samudra.datasets import InferenceDataset, TrainData
from samudra.utils.device import get_device
from samudra.utils.output import ModelInferenceOutput


class BaseModel(torch.nn.Module):
    """Abstract base model for neural ocean emulators.

    Provides shared functionality for all ocean emulator models, including
    residual prediction, ocean masking, and gradient detaching for multi-step
    autoregressive rollouts.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hist,
        pred_residuals,
        last_kernel_size,
        pad,
        gradient_detach_interval: int,
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad: str = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.gradient_detach_interval = gradient_detach_interval

    def set_epoch(self, epoch: int) -> None:
        """Per-epoch hook for scheduling (e.g. early stochastic depth).

        Subclasses should override to delegate to child modules.
        """

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        raise NotImplementedError()

    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ) -> torch.Tensor | list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        for step in range(len(train_data)):
            if step == 0:
                prog_tensor, boundary_tensor = train_data.get_initial_input()
            else:
                prev_output = outputs[-1]
                if (
                    self.gradient_detach_interval > 0
                    and step % self.gradient_detach_interval == 0
                ):
                    prev_output = prev_output.detach()
                _, boundary_tensor = train_data.get_input(step)
                prog_tensor = prev_output

            decodings = self.forward_once(prog_tensor, boundary_tensor, train_data.ctx)
            if self.pred_residuals:
                pred = prog_tensor + decodings  # Residual prediction
            else:
                pred = decodings  # Absolute prediction

            if loss_fn is not None:
                if step == 0:
                    loss = loss_fn(
                        pred,
                        train_data.get_label(step),
                    )
                else:
                    loss += loss_fn(
                        pred,
                        train_data.get_label(step),
                    )

            outputs.append(pred)

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

        for step in range(num_steps):
            logger.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0:
                prog_tensor = initial_prognostic
                boundary_tensor = dataset.get_boundary(steps_completed).to(
                    device=prog_tensor.device
                )
            else:
                prog_tensor = pred_tensor[step - 1].unsqueeze(0)
                boundary_tensor = dataset.get_boundary(
                    steps_completed + step,
                ).to(device=prog_tensor.device)

            decodings = self.forward_once(prog_tensor, boundary_tensor, dataset.ctx)
            if self.pred_residuals:
                pred = prog_tensor[0].to(device=get_device()) + decodings
            else:
                pred = decodings

            pred_tensor[step] = pred

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        IO = ModelInferenceOutput(pred_tensor, target_tensor, target_time)
        return IO
