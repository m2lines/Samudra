import logging

import numpy as np
import torch
import xarray as xr
from torchinfo import summary


def get_model_summary(model: torch.nn.Module, num_input_channels: int):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Number of parameters: {params}")
    logging.info(summary(model))


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
        prediction: torch.Tensor,
        target: torch.Tensor,
        time: xr.DataArray,
    ):
        self.prediction = prediction
        self.target = target
        self.time = time


class SingleTimeseriesOutput:
    def __init__(self, data: torch.Tensor, time: torch.Tensor):
        self.data = data
        self.time = time
