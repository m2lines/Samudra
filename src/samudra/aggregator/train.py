# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.aggregator.loss import (
    get_channel_loss_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
from samudra.constants import TensorMap
from samudra.utils.distributed import all_reduce_mean
from samudra.utils.output import TrainBatchOutput
from samudra.utils.wandb import Metrics


class TrainAggregator:
    """Aggregates train statistics for an epoch."""

    def __init__(self, tensor_map: TensorMap):
        self.tensor_map = tensor_map
        self._n_batches = 0
        self._loss = torch.tensor(torch.nan)
        self._loss_per_channel = torch.tensor(torch.nan)

    @torch.no_grad()
    def record_batch(self, batch: TrainBatchOutput):
        if self._n_batches == 0:
            self._loss = batch.loss
            self._loss_per_channel = batch.loss_per_channel
        else:
            self._loss += batch.loss
            self._loss_per_channel += batch.loss_per_channel
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Metrics:
        loss = self._loss / self._n_batches

        loss_per_channel = self._loss_per_channel / self._n_batches
        depth_loss_dict = get_depth_loss_dict(
            label, loss_per_channel, tensor_map=self.tensor_map
        )
        var_loss_dict = get_variable_loss_dict(
            label, loss_per_channel, tensor_map=self.tensor_map
        )
        channel_loss_dict = get_channel_loss_dict(
            label, loss_per_channel, tensor_map=self.tensor_map
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
