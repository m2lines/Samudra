import logging
from collections.abc import Callable
from functools import partial
from os import PathLike

logger = logging.getLogger(__name__)

import torch

from ocean_emulators.aggregator import InferenceEvaluatorAggregator
from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.models.base import BaseModel
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.output import (
    ModelInferenceOutput,
    TrainBatchOutput,
    ValBatchOutput,
)
from ocean_emulators.utils.wandb import get_record_to_wandb
from ocean_emulators.utils.writer import ZarrWriter


class Stepper:
    def __init__(self):
        pass

    @staticmethod
    def train_batch(
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> TrainBatchOutput:
        loss_per_channel = model(batch, loss_fn=partial(loss_fn, wet=batch.label_mask))
        loss = torch.mean(loss_per_channel)
        return TrainBatchOutput(loss, loss_per_channel)

    @staticmethod
    @torch.no_grad()
    def validate_batch(
        model: BaseModel | torch.nn.parallel.DistributedDataParallel,
        batch: TrainData,
        loss_fn: Callable,
    ) -> ValBatchOutput:
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
        outs = model.forward_once(
            input,
            wet=batch.label_mask.to(device=input.device),
            resolution=batch.input_res,
        )
        loss_per_channel = loss_fn(outs, label, wet=batch.label_mask)
        loss = torch.mean(loss_per_channel)
        return ValBatchOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inference(
        model: BaseModel,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
        epoch: int,
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
            IO: ModelInferenceOutput = model.inference(
                dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=step,
                num_steps=num_steps,
                epoch=epoch,
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
