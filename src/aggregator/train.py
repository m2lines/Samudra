# Adapted from https://github.com/ai2cm/ace/tree/main/fme/fme/ace/aggregator

from typing import Dict

import torch

from aggregator.loss import LossAggregator
from utils.device import get_device
from utils.distributed import all_reduce_mean


class TrainAggregator:
    """Aggregates train statistics for an epoch."""

    def __init__(self, num_output_channels: int):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._loss_per_channel = torch.zeros(num_output_channels, device=get_device())

    @torch.no_grad()
    def log_loss(self, loss: torch.Tensor, loss_per_channel: torch.Tensor):
        self._loss += loss
        self._loss_per_channel += loss_per_channel
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Dict[str, torch.Tensor]:
        loss = self._loss / self._n_batches

        loss_aggregator = LossAggregator.get_instance()
        loss_per_channel = self._loss_per_channel / self._n_batches
        depth_loss_dict = loss_aggregator.get_depth_loss_dict(label, loss_per_channel)
        var_loss_dict = loss_aggregator.get_variable_loss_dict(label, loss_per_channel)
        channel_loss_dict = loss_aggregator.get_channel_loss_dict(
            label, loss_per_channel
        )
        logs = {
            f"{label}/mean/loss": loss,
            **depth_loss_dict,
            **var_loss_dict,
            **channel_loss_dict,
        }
        for key in sorted(logs.keys()):
            logs[key] = float(all_reduce_mean(logs[key].detach()).cpu().numpy())

        return logs
