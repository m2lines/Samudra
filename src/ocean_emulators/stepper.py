import logging
from collections.abc import Callable
from os import PathLike

from ocean_emulators.utils.loss import LossFn

logger = logging.getLogger(__name__)

import torch

from ocean_emulators.aggregator import InferenceEvaluatorAggregator
from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.output import (
    ModelInferenceOutput,
    TrainBatchOutput,
    ValBatchOutput,
)
from ocean_emulators.utils.wandb import get_record_to_wandb
from ocean_emulators.utils.writer import ZarrWriter


class Stepper:
    @staticmethod
    def train_batch(
        model: torch.nn.Module,
        batch: TrainData,
        loss_fn: LossFn,
        gradient_detach_interval: int,
        pred_residuals: bool,
        out_channels: int,
    ) -> TrainBatchOutput:
        outputs: list[torch.Tensor] = []
        loss_per_channel: torch.Tensor = torch.zeros(out_channels)
        for step in range(len(batch)):
            if step == 0:
                input_tensor = batch.get_initial_input()
            else:
                prev_output = outputs[-1]
                if (
                    gradient_detach_interval > 0
                    and step % gradient_detach_interval == 0
                ):
                    prev_output = prev_output.detach()
                input_tensor = batch.merge_prognostic_and_boundary(
                    prognostic=prev_output, step=step
                )

            decodings = model(input_tensor)

            if pred_residuals:
                pred = (
                    input_tensor[
                        :,
                        :out_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
            else:
                pred = decodings  # Absolute prediction

            loss_per_channel += loss_fn(
                pred,
                batch.get_label(step),
            )

            outputs.append(pred)

        loss = torch.mean(loss_per_channel)
        return TrainBatchOutput(loss, loss_per_channel)

    @staticmethod
    @torch.no_grad()
    def validate_batch(
        model: torch.nn.Module,
        batch: TrainData,
        loss_fn: Callable,
    ) -> ValBatchOutput:
        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)
        outs = model(input)
        loss_per_channel = loss_fn(outs, label)
        loss = torch.mean(loss_per_channel)
        return ValBatchOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inference(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
        epoch: int,
        pred_residuals: bool,
        out_channels: int,
        output_dir: str | PathLike | None = None,
        model_path: str | PathLike | None = None,
        num_model_steps_forward: int = 200,
        save_zarr: bool = False,
    ) -> None:
        if save_zarr:
            if output_dir is None or model_path is None:
                raise ValueError(
                    "output_dir and model_path must be provided if save_zarr is True"
                )
            coords = dataset.get_coords_dict()
            if num_model_steps_forward > 0:
                chunk_size = num_model_steps_forward
            else:
                chunk_size = 20
            writer = ZarrWriter(
                output_dir,
                coords=coords,
                hist=inf_aggregator.hist,
                model_path=model_path,
                time_chunk_size=chunk_size,
            )
        else:
            writer = None
        record_logs = get_record_to_wandb(label="inference")
        logger.info(f"Inference [epoch {epoch}]: processing initial prognostic.")
        logs = inf_aggregator.record_initial_prognostic(
            initial_prognostic=dataset.initial_prognostic.to(get_device()),
        )
        record_logs(logs)
        num_model_steps = len(dataset)
        num_steps_list = []

        # If num_model_steps_forward is -1, then we are doing a full forward pass
        if num_model_steps_forward == -1:
            num_steps_list = [num_model_steps]
        else:
            # Windows of partial forward passes
            num_loops = num_model_steps // num_model_steps_forward
            if num_loops > 0:
                num_steps_list = [num_model_steps_forward] * num_loops
                last_model_steps_forward = num_model_steps % num_model_steps_forward
                if last_model_steps_forward > 0:
                    num_steps_list = num_steps_list + [last_model_steps_forward]
            else:
                num_steps_list = [num_model_steps]

        num_loops = len(num_steps_list)
        initial_prognostic = dataset.initial_prognostic
        step = 0
        for loop, num_steps in enumerate(num_steps_list):
            logger.info(
                f"Inference [epoch {epoch}]: loop {loop} of {num_loops - 1}. "
                f"Stepping {num_steps} steps forward."
            )
            IO: ModelInferenceOutput = Stepper.inference_steps(
                model,
                dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=step,
                num_steps=num_steps,
                pred_residuals=pred_residuals,
                out_channels=out_channels,
            )
            # Setting initial prognostic for next loop
            initial_prognostic = IO.prediction[-1].unsqueeze(0).clone()
            if writer:
                logger.info(f"Writing to zarr...")
                writer.record_batch(IO)
                writer.write()

            logger.info(f"Recording logs...")
            logs = inf_aggregator.record_batch(IO)
            logger.info(f"Logging to wandb...")
            record_logs(logs)
            step += num_steps

    @staticmethod
    def inference_steps(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        initial_prognostic: torch.Tensor,
        steps_completed,
        num_steps,
        pred_residuals: bool,
        out_channels: int,
    ) -> ModelInferenceOutput:
        out_shape = (num_steps, *dataset[0][1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        initial_prognostic = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)

        for step in range(num_steps):
            logger.info(
                f"Inference: Rollout step {steps_completed + step} "
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

            decodings = model(input_tensor)
            if pred_residuals:
                pred = (
                    input_tensor[
                        0,
                        :out_channels,
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
