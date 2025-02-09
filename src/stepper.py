import logging
from typing import Callable

import torch

from aggregator import InferenceEvaluatorAggregator
from datasets import InferenceDataset, TrainData
from utils.device import using_gpu
from utils.model import InfOutput, TrainOutput, ValOutput
from utils.wandb import get_record_to_wandb


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
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> ValOutput:
        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)
        model = model.module if using_gpu() else model
        outs = model.forward_once(input)
        loss_per_channel = loss_fn(outs, label)
        loss = torch.mean(loss_per_channel)
        return ValOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inline_inference(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
    ) -> None:
        record_logs = get_record_to_wandb(label="inference")
        logging.info(f"Inference: processing initial prognostic.")
        logs = inf_aggregator.record_initial_prognostic(
            initial_prognostic=dataset.initial_prognostic,
        )
        record_logs(logs)
        num_model_steps = len(dataset)
        outs = model.inference(
            dataset,
            initial_prognostic=None,
            num_steps=num_model_steps,  # Here we consider history
        )

        for i in range(num_model_steps):
            logging.info(
                f"Inference: processing output window {i} of {num_model_steps - 1}."
            )
            IO = InfOutput(
                prediction=outs[i].cpu(),
                target=dataset.inference_target(i),  # TODO: Pack with input
                time=dataset.inputs.time[i],
            )  # time-dependent aggs dont work, time is incorrect as well
            logs = inf_aggregator.record_batch(IO)
            record_logs(logs)
