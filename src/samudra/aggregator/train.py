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
from samudra.utils.ctx import GridContext
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
            # Aggregators may record the same batch in parallel (for example,
            # the overall and resolution-specific validation aggregators).  Own
            # the accumulation tensors so a later in-place addition in one
            # aggregator cannot mutate another aggregator's state.
            self._loss = batch.loss.detach().clone()
            self._loss_per_channel = batch.loss_per_channel.detach().clone()
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


class RouteTrainAggregator:
    """Aggregate losses overall and by physical input/output grid route."""

    def __init__(self, tensor_map: TensorMap):
        self.tensor_map = tensor_map
        self._overall = TrainAggregator(tensor_map)
        self._routes: dict[
            tuple[tuple[int, int], tuple[int, int]], TrainAggregator
        ] = {}

    @staticmethod
    def route(ctx: GridContext) -> tuple[tuple[int, int], tuple[int, int]]:
        input_grid = tuple(len(axis) for axis in ctx.input_resolution_cpu)
        output_grid = tuple(len(axis) for axis in ctx.output_resolution_cpu)
        if len(input_grid) != 2 or len(output_grid) != 2:
            raise ValueError("Grid routes require latitude/longitude coordinate pairs.")
        return (input_grid[0], input_grid[1]), (output_grid[0], output_grid[1])

    @torch.no_grad()
    def record_batch(self, batch: TrainBatchOutput, ctx: GridContext) -> None:
        self._overall.record_batch(batch)
        route = self.route(ctx)
        if route not in self._routes:
            self._routes[route] = TrainAggregator(self.tensor_map)
        self._routes[route].record_batch(batch)

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        logs = dict(self._overall.get_logs(label))
        for (input_grid, output_grid), aggregator in sorted(self._routes.items()):
            route_label = (
                f"{input_grid[0]}x{input_grid[1]}_to_{output_grid[0]}x{output_grid[1]}"
            )
            route_logs = aggregator.get_logs(label=f"{label}/route/{route_label}")
            overlap = logs.keys() & route_logs.keys()
            if overlap:
                raise ValueError(f"Duplicate route log keys: {sorted(overlap)}")
            logs.update(route_logs)
        return logs
