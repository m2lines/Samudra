# TODO: Need to return step-wise losses for logging

import logging
from typing import Union

import torch

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device


class BaseModel(torch.nn.Module):
    def __init__(
        self, ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet.bool()
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.input_channels = ch_width[0]
        self.output_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        for step in range(len(train_data)):
            if step == 0:
                input_tensor = train_data.get_initial_input()
            else:
                # TODO(jder): this function seems to be unused, resolve
                # as part of https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
                input_tensor = train_data.merge_prognostic_and_boundary(  # type: ignore[attr-defined]
                    prognostic=outputs[-1], step=step
                )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                pred = (
                    input_tensor[
                        :,
                        : self.output_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
            else:
                pred = decodings  # Absolute prediction

            if loss_fn is not None:
                if torch.isnan(loss).all():
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
        initial_prognostic=None,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for step in range(num_steps):
            logging.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0 and steps_completed == 0:
                input_tensor = dataset.get_initial_input().to(device=get_device())

            elif step == 0 and steps_completed > 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=outputs[-1],
                    step=steps_completed + step,
                )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                pred = (
                    input_tensor[
                        0,
                        : self.output_channels,
                    ].to(  # Residuals on last state in input
                        device=get_device()
                    )
                    + decodings
                )
            else:
                pred = decodings

            outputs.append(pred)

        return outputs
