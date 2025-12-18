# TODO: Need to return step-wise losses for logging

import logging

import torch

logger = logging.getLogger(__name__)

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.output import ModelInferenceOutput


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        wet,
        num_input_states: int,
        num_output_states: int,
        pred_residuals,
        last_kernel_size,
        pad,
        static_data,
        gradient_detach_interval: int,
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        if out_channels % num_output_states != 0:
            raise ValueError(
                "out_channels must be divisible by num_output_states to support "
                "per-state buffering."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wet = wet.bool()
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.num_input_states = num_input_states
        self.num_output_states = num_output_states
        self.static_data = static_data
        self.gradient_detach_interval = gradient_detach_interval

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ) -> torch.Tensor | list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        prog_channels_per_state = self.out_channels // self.num_output_states
        total_prognostic_channels = prog_channels_per_state * self.num_input_states

        prognostic_buffer = train_data.get_initial_input()[
            :, :total_prognostic_channels
        ]

        for step in range(len(train_data)):
            if (
                step > 0
                and self.gradient_detach_interval > 0
                and step % self.gradient_detach_interval == 0
            ):
                prognostic_buffer = prognostic_buffer.detach()

            input_tensor = train_data.merge_prognostic_and_boundary(
                prognostic=prognostic_buffer, step=step
            )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                base = prognostic_buffer[:, -self.out_channels :]
                pred = (
                    base + decodings
                )  # Residual prediction on the most recent input states
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
            if step + 1 < len(train_data):
                prognostic_buffer = torch.cat(
                    [
                        prognostic_buffer[
                            :,
                            prog_channels_per_state * self.num_output_states :,
                        ],
                        pred,
                    ],
                    dim=1,
                )

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
        out_shape = (num_steps, *dataset[0][1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        prognostic_buffer = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)
        prog_channels_per_state = self.out_channels // self.num_output_states

        for step in range(num_steps):
            logger.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if (
                step > 0
                and self.gradient_detach_interval > 0
                and step % self.gradient_detach_interval == 0
            ):
                prognostic_buffer = prognostic_buffer.detach()

            input_tensor = dataset.merge_prognostic_and_boundary(
                prognostic=prognostic_buffer,
                step=steps_completed + step,
            )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                pred = (
                    prognostic_buffer[:, -self.out_channels :].to(device=get_device())
                    + decodings
                )
            else:
                pred = decodings

            pred_tensor[step] = pred.squeeze(0)
            if step + 1 < num_steps:
                prognostic_buffer = torch.cat(
                    [
                        prognostic_buffer[
                            :,
                            prog_channels_per_state * self.num_output_states :,
                        ],
                        pred,
                    ],
                    dim=1,
                )

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        IO = ModelInferenceOutput(pred_tensor, target_tensor, target_time)
        return IO
