from collections import defaultdict

import torch

from ocean_emulators.aggregator.loss import (
    get_channel_loss_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.output import TrainBatchOutput
from ocean_emulators.utils.wandb import Metrics


class TrainAggregator:
    """Aggregates train statistics for an epoch."""

    def __init__(self):
        self._n_batches = 0
        self._loss_per_grid = defaultdict(lambda: torch.tensor(torch.nan))
        self._loss_per_channel_per_grid = defaultdict(lambda: torch.tensor(torch.nan))

    @torch.no_grad()
    def record_batch(self, batch: TrainBatchOutput):
        if self._n_batches == 0:
            self._loss_per_grid[batch.grid] = batch.loss
            self._loss_per_channel_per_grid[batch.grid] = batch.loss_per_channel
        else:
            self._loss_per_grid[batch.grid] += batch.loss
            self._loss_per_channel_per_grid[batch.grid] += batch.loss_per_channel
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Metrics:
        meaned_logs = {}
        for grid in self._loss_per_grid.keys():
            label_by_grid = f"{label}/{grid[0] - grid[1]}"

            loss = self._loss_per_grid[grid] / self._n_batches
            loss_per_channel = self._loss_per_channel_per_grid[grid] / self._n_batches
            depth_loss_dict = get_depth_loss_dict(label_by_grid, loss_per_channel)
            var_loss_dict = get_variable_loss_dict(label_by_grid, loss_per_channel)
            channel_loss_dict = get_channel_loss_dict(label_by_grid, loss_per_channel)
            logs = {
                f"{label_by_grid}/mean/loss": loss,
                **depth_loss_dict,
                **var_loss_dict,
                **channel_loss_dict,
            }
            for key in sorted(logs.keys()):
                meaned_logs[key] = float(
                    all_reduce_mean(logs[key].detach()).cpu().numpy()
                )

        return meaned_logs
