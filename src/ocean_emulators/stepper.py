import logging
from os import PathLike
from typing import Callable, Optional

import torch

from ocean_emulators.aggregator import InferenceEvaluatorAggregator
from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.models.base import BaseModel
from ocean_emulators.utils.model import TrainOutput, ValOutput
from ocean_emulators.utils.wandb import get_record_to_wandb
from ocean_emulators.utils.writer import ZarrWriter


class Stepper:
    def __init__(self):
        pass

    @staticmethod
    def train_step(
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> TrainOutput:
        loss_per_channel = model(batch, loss_fn=loss_fn)
        loss = torch.mean(loss_per_channel)
        return TrainOutput(loss, loss_per_channel)

    @staticmethod
    @torch.no_grad()
    def validate_step(
        model: BaseModel | torch.nn.parallel.DistributedDataParallel,
        batch: TrainData,
        loss_fn: Callable,
    ) -> ValOutput:
        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)
        # TODO(jder): we need the underlying model so we can use forward_once;
        # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
        model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        outs = model.forward_once(input)
        loss_per_channel = loss_fn(outs, label)
        loss = torch.mean(loss_per_channel)
        return ValOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inference(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
        epoch: int,
        output_dir: Optional[str | PathLike] = None,
        model_path: Optional[str | PathLike] = None,
        num_model_steps_forward: int = 200,
        save_zarr: bool = False,
    ) -> None:
        if save_zarr:
            if output_dir is None or model_path is None:
                raise ValueError(
                    "output_dir and model_path must be provided if save_zarr is True"
                )
            coords = dataset.get_coords_dict()
            writer = ZarrWriter(
                output_dir,
                coords=coords,
                hist=inf_aggregator.hist,
                model_path=model_path,
            )
        else:
            writer = None
        record_logs = get_record_to_wandb(label="inference")
        logging.info(f"Inference [epoch {epoch}]: processing initial prognostic.")
        logs = inf_aggregator.record_initial_prognostic(
            initial_prognostic=dataset.initial_prognostic,
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
        initial_prognostic = None
        step = 0
        for loop, num_steps in enumerate(num_steps_list):
            logging.info(
                f"Inference [epoch {epoch}]: loop {loop} of {num_loops - 1}. "
                f"Stepping {num_steps} steps forward."
            )
            IO = model.inference(
                dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=step,
                num_steps=num_steps,
                epoch=epoch,
            )
            # Setting initial prognostic for next loop
            initial_prognostic = IO.prediction[-1].unsqueeze(0).clone()
            if writer:
                writer.record_batch(IO)
                writer.write()

            logs = inf_aggregator.record_batch(IO)
            record_logs(logs)
            step += num_steps
