# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Time-stepping primitives for training, validation, and inference.

Provides module-level functions that handle single-step forward passes
(``train_batch``, ``validate_batch``) and multi-step autoregressive
rollouts (``run_rollout``).
"""

import logging
from collections.abc import Callable, Mapping
from functools import partial
from os import PathLike

import torch

from samudra.aggregator import InferenceEvaluatorAggregator
from samudra.aggregator.validate.rollout import RolloutValidationAggregator
from samudra.constants import TensorMap
from samudra.datasets import InferenceDataset, TrainData
from samudra.models.base import BaseModel
from samudra.utils.data import Normalize
from samudra.utils.device import get_device
from samudra.utils.output import ModelInferenceOutput, TrainBatchOutput, ValBatchOutput
from samudra.utils.wandb import get_record_to_wandb
from samudra.utils.writer import ZarrWriter

logger = logging.getLogger(__name__)


def train_batch(
    model: torch.nn.Module, batch: TrainData, loss_fn: Callable
) -> TrainBatchOutput:
    loss_per_channel = model(batch, loss_fn=partial(loss_fn, ctx=batch.ctx))
    loss = torch.mean(loss_per_channel)
    return TrainBatchOutput(loss, loss_per_channel)


@torch.no_grad()
def validate_batch(
    model: BaseModel | torch.nn.parallel.DistributedDataParallel,
    batch: TrainData,
    loss_fn: Callable,
) -> ValBatchOutput:
    assert len(batch) == 1  # Assert we are using one step of input and output
    prognostic, boundary = batch.get_input(0)
    label = batch.get_label(0)

    outs = model(batch)[0]
    loss_per_channel = loss_fn(outs, label, batch.ctx)
    loss = torch.mean(loss_per_channel)
    # `input_data` in ValBatchOutput is used for val visualization; pass the
    # channel-concatenated tensor for continuity with existing consumers.
    # We don't mix scales across boundary and prognostic during training.
    input_data = torch.cat((prognostic, boundary), dim=1)
    return ValBatchOutput(loss, loss_per_channel, input_data, label, outs, batch.ctx)


def _get_rollout_step_chunks(
    *,
    total_steps: int,
    num_model_steps_forward: int,
    boundaries: tuple[int, ...] = (),
) -> list[int]:
    if total_steps <= 0:
        return []
    if num_model_steps_forward <= 0:
        num_model_steps_forward = total_steps

    boundary_steps = sorted(
        boundary for boundary in set(boundaries) if 0 < boundary < total_steps
    )
    chunks = []
    step = 0
    while step < total_steps:
        next_step = min(step + num_model_steps_forward, total_steps)
        for boundary in boundary_steps:
            if step < boundary < next_step:
                next_step = boundary
                break
        chunks.append(next_step - step)
        step = next_step
    return chunks


def _rollout_horizon_boundaries(
    aggregators_by_step: Mapping[int, RolloutValidationAggregator],
    total_steps: int,
) -> tuple[int, ...]:
    return tuple(step for step in aggregators_by_step if 0 < step < total_steps)


def _record_rollout_horizon_batch(
    *,
    aggregators_by_step: Mapping[int, RolloutValidationAggregator],
    step: int,
    num_steps: int,
    output: ModelInferenceOutput,
) -> None:
    chunk_end = step + num_steps
    for horizon_steps, aggregator in aggregators_by_step.items():
        if chunk_end <= horizon_steps:
            aggregator.record_batch(output)


@torch.no_grad()
def validate_rollout(
    model: BaseModel,
    dataset: InferenceDataset,
    aggregators_by_step: Mapping[int, RolloutValidationAggregator],
    epoch: int,
    *,
    num_model_steps: int,
    num_model_steps_forward: int,
) -> None:
    """Run a bounded autoregressive validation rollout and record RMSE metrics.

    ``aggregators_by_step`` maps a rollout horizon, in model steps, to the
    aggregator that should record metrics for that horizon.
    """
    if not aggregators_by_step:
        raise ValueError("At least one rollout validation aggregator is required")

    dataset.to(get_device())
    num_model_steps = min(num_model_steps, len(dataset))
    horizon_boundaries = _rollout_horizon_boundaries(
        aggregators_by_step,
        num_model_steps,
    )
    chunks = _get_rollout_step_chunks(
        total_steps=num_model_steps,
        num_model_steps_forward=num_model_steps_forward,
        boundaries=horizon_boundaries,
    )

    initial_prognostic = dataset.initial_prognostic
    step = 0
    for loop, num_steps in enumerate(chunks):
        logger.info(
            f"Rollout validation [epoch {epoch}]: loop {loop} of "
            f"{len(chunks) - 1}. Stepping {num_steps} steps forward."
        )
        output = model.inference(
            dataset,
            initial_prognostic=initial_prognostic,
            steps_completed=step,
            num_steps=num_steps,
            epoch=epoch,
        )
        initial_prognostic = output.prediction[-1].unsqueeze(0).clone()
        _record_rollout_horizon_batch(
            aggregators_by_step=aggregators_by_step,
            step=step,
            num_steps=num_steps,
            output=output,
        )
        step += num_steps


@torch.no_grad()
def run_rollout(
    model: BaseModel,
    dataset: InferenceDataset,
    inf_aggregator: InferenceEvaluatorAggregator,
    epoch: int,
    output_dir: str | PathLike | None = None,
    model_path: str | PathLike | None = None,
    num_model_steps_forward: int = 200,
    save_zarr: bool = False,
    tensor_map: TensorMap | None = None,
    normalize: Normalize | None = None,
) -> None:
    """Performs inference, which is an auto-regressive rollout."""
    if save_zarr:
        if output_dir is None or model_path is None:
            raise ValueError(
                "output_dir and model_path must be provided if save_zarr is True"
            )
        if tensor_map is None or normalize is None:
            raise ValueError(
                "tensor_map and normalize must be provided if save_zarr is True"
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
            normalize=normalize,
            tensor_map=tensor_map,
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
    num_steps_list = _get_rollout_step_chunks(
        total_steps=num_model_steps,
        num_model_steps_forward=num_model_steps_forward,
    )

    num_loops = len(num_steps_list)
    initial_prognostic = dataset.initial_prognostic
    step = 0
    for loop, num_steps in enumerate(num_steps_list):
        logger.info(
            f"Inference [epoch {epoch}]: loop {loop} of {num_loops - 1}. "
            f"Stepping {num_steps} steps forward."
        )
        dataset.to(get_device())
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
            logger.info("Writing to zarr...")
            writer.record_batch(IO)
            writer.write()

        logger.info("Recording logs...")
        logs = inf_aggregator.record_batch(IO)
        logger.info("Logging to wandb...")
        record_logs(logs)
        step += num_steps
