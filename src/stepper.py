from typing import Callable

import torch

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
    def __init__(self):
        pass


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
    def inference_step(batch: torch.Tensor) -> InfOutput:
        return InfOutput()
