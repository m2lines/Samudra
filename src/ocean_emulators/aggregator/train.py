from typing import Generic, TypeVar

import torch

from ocean_emulators.aggregator.loss import LossAggregator
from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.model import TrainOutput

T = TypeVar("T", bound=TrainOutput)
Logs = dict[str, float]


class TrainAggregator(Generic[T]):
    """Aggregates train statistics for an epoch."""

    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(torch.nan)
        self._loss_per_channel = torch.tensor(torch.nan)

    @torch.no_grad()
    def record_batch(self, batch: T):
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
    def get_logs(self, label: str = "train") -> Logs:
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
        meaned_logs = {}
        for key in sorted(logs.keys()):
            meaned_logs[key] = float(all_reduce_mean(logs[key].detach()).cpu().numpy())

        return meaned_logs
