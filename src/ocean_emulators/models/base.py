# TODO: Need to return step-wise losses for logging

import logging

import torch

from ocean_emulators.utils.ctx import GridContext

logger = logging.getLogger(__name__)

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.output import ModelInferenceOutput


class BaseModel(torch.nn.Module):
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

    def forward_once(self, fts, ctx: GridContext):
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
                input_tensor = train_data.get_initial_input()
            else:
                prev_output = outputs[-1]
                if (
                    self.gradient_detach_interval > 0
                    and step % self.gradient_detach_interval == 0
                ):
                    prev_output = prev_output.detach()
                input_tensor = train_data.merge_prognostic_and_boundary(
                    prognostic=prev_output, step=step
                )

            decodings = self.forward_once(input_tensor, train_data.ctx)
            if self.pred_residuals:
                pred = (
                    input_tensor[
                        :,
                        : self.out_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
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
        out_shape = (num_steps, *dataset[0][1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        initial_prognostic = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)

        for step in range(num_steps):
            logger.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=pred_tensor[step - 1].unsqueeze(0),
                    step=steps_completed + step,
                )
            decodings = self.forward_once(input_tensor, dataset.ctx)
            if self.pred_residuals:
                pred = (
                    input_tensor[
                        0,
                        : self.out_channels,
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

        IO = ModelInferenceOutput(pred_tensor, target_tensor, target_time)
        return IO
