from typing import Dict

import torch

from aggregator.loss import LossAggregator
from stepper import TrainOutput
from utils.distributed import all_reduce_mean


class TrainAggregator:
    """Aggregates train statistics for an epoch."""

    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(torch.nan)
        self._loss_per_channel = torch.tensor(torch.nan)

    @torch.no_grad()
    def record_batch(self, batch: TrainOutput):
        if torch.isnan(self._loss):
            self._loss = batch.loss
        else:
            self._loss += batch.loss
        if torch.isnan(self._loss_per_channel).all():
            self._loss_per_channel = batch.loss_per_channel
        else:
            self._loss_per_channel += batch.loss_per_channel
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
