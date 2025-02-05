from typing import Callable

import numpy as np
import torch

from aggregator.inference import InferenceAggregator
from datasets import Data_CNN_Disk
from utils.device import using_gpu


class TrainOutput:
    def __init__(self, loss: torch.Tensor, loss_per_channel: torch.Tensor):
        self.loss = loss
        self.loss_per_channel = loss_per_channel


class ValOutput(TrainOutput):
    def __init__(
        self,
        loss: torch.Tensor,
        loss_per_channel: torch.Tensor,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        gen_data: torch.Tensor,
    ):
        super().__init__(loss, loss_per_channel)
        self.input_data = input_data
        self.target_data = target_data
        self.gen_data = gen_data


class InfOutput:
    def __init__(
        self,
        model_pred: np.ndarray,
        target_data: np.ndarray,
    ):
        self.model_pred = model_pred
        self.target_data = target_data


class Stepper:
    def __init__(self):
        pass

    @staticmethod
    def train_step(
        model: torch.nn.Module, batch: torch.Tensor, loss_fn: Callable
    ) -> TrainOutput:
        loss_per_channel = model(batch, loss_fn=loss_fn)
        loss = torch.mean(loss_per_channel)
        return TrainOutput(loss, loss_per_channel)

    @staticmethod
    @torch.no_grad()
    def validate_step(
        model: torch.nn.Module, batch: torch.Tensor, loss_fn: Callable
    ) -> ValOutput:
        assert len(batch) == 2  # Assert we are using one step of input and output
        model = model.module if using_gpu() else model
        outs = model.forward_once(batch[0])
        loss_per_channel = loss_fn(outs, batch[1])
        loss = torch.mean(loss_per_channel)
        return ValOutput(loss, loss_per_channel, batch[0], batch[1], outs)

    @staticmethod
    @torch.no_grad()
    def inline_inference(
        model: torch.nn.Module,
        data_loader: Data_CNN_Disk,
        target_data: torch.Tensor,
        n_steps: int,
        hist: int,
        inf_aggregator: InferenceAggregator,
    ) -> None:
        model_pred = np.zeros((n_steps, *data_loader[0][0].shape[1:]))

        with torch.no_grad():
            outs = model.inference(
                data_loader,
                initial_input=None,
                num_steps=n_steps,
            )

        for i in range(n_steps):
            pred_temp = outs[i]
            model_pred[i * (hist + 1) : (i + 1) * (hist + 1)] = pred_temp.cpu()

        assert model_pred.shape == target_data.shape
        inf_aggregator.record_batch(InfOutput(model_pred, target_data.cpu().numpy()))
