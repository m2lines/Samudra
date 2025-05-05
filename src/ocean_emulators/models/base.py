# TODO: Need to return step-wise losses for logging

import logging
from typing import Union

import torch

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.model import InfOutput


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        ch_width,
        n_out,
        wet,
        hist,
        pred_residuals,
        last_kernel_size,
        pad,
        static_data,
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
        self.num_prognostic_channels = n_out
        self.static_data = static_data

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        train_data: TrainData,
        extra_batched: torch.Tensor = None,  # JRSv2
        loss_fn=None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        for step in range(len(train_data)):
            if step == 0:
                input_tensor = train_data.get_initial_input()
            else:
                input_tensor = train_data.merge_prognostic_and_boundary(
                    prognostic=outputs[-1], step=step
                )
            
            extra_inputs = extra_batched[:, step, 0] if extra_batched is not None else None
            #print(f"InBase step: {step}") # JRSv2
            #print(f"InBase input_tensor shape: {input_tensor.size()}") # JRSv2 ; torch.Size([3, 162, 180, 360])
            #print(f"InBase extra_batched shape: {extra_inputs.shape}") # JRSv2; torch.Size([batch=3, time=4, var=3, 180, 360])
            
            decodings = self.forward_once(input_tensor, extra_inputs)  # JRSv2, this use submodel samudra
            if self.pred_residuals:    # JRS, where the residuals are predicted, turn on pred_residuals
                pred = (
                    input_tensor[
                        :,
                        : self.num_prognostic_channels,
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
        initial_prognostic: torch.Tensor,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> InfOutput:
        out_shape = (num_steps, *dataset[0][1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        initial_prognostic = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)

        for step in range(num_steps):
            logging.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
                extra_batched_new = dataset.get_full_boundary(step=steps_completed) # JRSv2
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=pred_tensor[step - 1].unsqueeze(0),
                    step=steps_completed + step,
                )
                extra_batched_new = dataset.get_full_boundary(step=steps_completed + step) # JRSv2

            #print(f"Inference InBase extra_batched shape: {extra_batched_new.size()}") # JRSv2 Size([batch=1, time=4, var=3, 180, 360])
            #print(f"Inference InBase input_tensor shape: {input_tensor.size()}") # JRSv2; torch.Size([1, 160, 180, 360])
            #extra_inputs = extra_batched[:, step, 0]
            #print(f"Inference InBase step: {step}")
            #print(f"Inference InBase extra_inputs shape: {extra_inputs.size()}")

            decodings = self.forward_once(input_tensor, extra_batched_new)  # JRSv2
            if self.pred_residuals:
                pred = (
                    input_tensor[
                        0,
                        : self.num_prognostic_channels,
                    ].to(  # Residuals on last state in input
                        device=get_device()
                    )
                    + decodings
                )
            else:
                pred = decodings

            pred_tensor[step] = pred

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        IO = InfOutput(pred_tensor, target_tensor, target_time)
        return IO
