import logging
from typing import Callable, Dict

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
    def inference(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
        epoch: int,
        record_every: int = 10,
    ) -> None:
        record_logs = get_record_to_wandb(label="inference")
        logging.info(f"Inference [epoch {epoch}]: processing initial prognostic.")
        logs = inf_aggregator.record_initial_prognostic(
            initial_prognostic=dataset.initial_prognostic,
        )
        record_logs(logs)
        num_model_steps = len(dataset)
        outs = model.inference(
            dataset,
            initial_prognostic=None,
            num_steps=num_model_steps,  # Here we consider history
            epoch=epoch,
        )

        all_logs: list[Dict[str, float | int | str]] = []
        for i in range(num_model_steps):
            logging.info(
                f"Inference [epoch {epoch}]: recording output window {i} of "
                f"{num_model_steps - 1}."
            )
            IO = InfOutput(
                prediction=outs[i].cpu(),
                target=dataset.inference_target(i),  # TODO: Pack with input
                time=dataset.inputs.time[i],
            )  # time-dependent aggs dont work, time is incorrect as well
            logs = inf_aggregator.record_batch(IO)
            all_logs = all_logs + logs
            if (i + 1) % record_every == 0:
                logging.info(f"Inference [epoch {epoch}]: wandb logging...")
                record_logs(all_logs)
                all_logs = []

        if len(all_logs) > 0:
            logging.info(f"Inference [epoch {epoch}]: wandb logging...")
            record_logs(all_logs)
