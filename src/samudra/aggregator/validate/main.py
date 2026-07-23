# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.aggregator.train import RouteTrainAggregator, TrainAggregator
from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.constants import TensorMap
from samudra.models.modules.decoder import coordinate_bilinear_resample
from samudra.utils.data import Normalize, get_aggregator_dicts
from samudra.utils.loss import loss_fn_from_metric
from samudra.utils.output import TrainBatchOutput, ValBatchOutput
from samudra.utils.wandb import Metrics, MetricsDict


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        aggregators: dict[str, ValidateSubAggregator],
        hist: int,
        num_prognostic_channels: int,
        *,
        tensor_map: TensorMap,
        normalize: Normalize,
        record_baselines: bool = False,
    ):
        super().__init__(tensor_map)
        self._aggregators = aggregators
        self.hist = hist
        self.num_prognostic_channels = num_prognostic_channels
        self.normalize = normalize
        self._unweighted_mse = TrainAggregator(tensor_map) if record_baselines else None
        self._persistence_mse = (
            TrainAggregator(tensor_map) if record_baselines else None
        )
        self._mse = loss_fn_from_metric("mse")

    # TODO(jder): we could remove this by moving from inheritance
    # to composition with the TrainAggregator functionality.
    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
        super().record_batch(batch)  # Record losses

        # Persistence lives on the input grid. Render it on the target grid before
        # computing baselines or passing it to map/reduced diagnostics. This is a
        # deterministic normalized-space baseline; it avoids pretending that two
        # differently shaped physical tensors are directly comparable.
        persistence = coordinate_bilinear_resample(
            batch.input_data[:, : self.num_prognostic_channels],
            batch.ctx.input_resolution_cpu,
            batch.ctx.output_resolution_cpu,
            valid_mask=batch.ctx.input_mask,
        )

        if self._unweighted_mse is not None and self._persistence_mse is not None:
            forecast_loss_per_channel = self._mse(
                batch.gen_data, batch.target_data, batch.ctx
            )
            persistence_loss_per_channel = self._mse(
                persistence,
                batch.target_data,
                batch.ctx,
            )
            self._unweighted_mse.record_batch(
                TrainBatchOutput(
                    forecast_loss_per_channel.mean(),
                    forecast_loss_per_channel,
                )
            )
            self._persistence_mse.record_batch(
                TrainBatchOutput(
                    persistence_loss_per_channel.mean(),
                    persistence_loss_per_channel,
                )
            )

        # If there are no log aggregators, omit doing any extra work.
        if not self._aggregators:
            return

        # Translate the GridContext mask by removing history.
        target_data = batch.target_data  # [B, C*(hist+1), H, W]
        wet = batch.ctx.label_mask  # [C*(hist+1), H, W]
        assert wet.shape == target_data.shape[1:], (
            "The wetmask must match the target data shape excluding batch."
        )
        assert wet.shape[0] % (self.hist + 1) == 0, (
            "The wetmask channel count must be divisible by history size."
        )
        first_wetmask_chunk = wet.shape[0] // (self.hist + 1)
        wet = wet[:first_wetmask_chunk]  # [C, H, W]

        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        assert target_data.shape[1] == self.num_prognostic_channels
        target_data_dict, target_data_unnorm_dict = get_aggregator_dicts(
            target_data,
            normalize=self.normalize,
            tensor_map=self.tensor_map,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        gen_data_dict, gen_data_unnorm_dict = get_aggregator_dicts(
            batch.gen_data,
            normalize=self.normalize,
            tensor_map=self.tensor_map,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        input_data_dict, input_data_unnorm_dict = get_aggregator_dicts(
            persistence,
            normalize=self.normalize,
            tensor_map=self.tensor_map,
            wet=wet,
            long_rollout=False,
            input_type="input",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        for agg in self._aggregators.values():
            agg.record_batch(
                loss=batch.loss,
                target_data=target_data_unnorm_dict,
                gen_data=gen_data_unnorm_dict,
                input_data=input_data_unnorm_dict,
                target_data_norm=target_data_dict,
                gen_data_norm=gen_data_dict,
                input_data_norm=input_data_dict,
            )

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Metrics:
        logs: MetricsDict = dict(super().get_logs(label))
        if self._unweighted_mse is not None and self._persistence_mse is not None:
            logs.update(
                self._unweighted_mse.get_logs(
                    label=f"{label}/unweighted_normalized_mse"
                )
            )
            logs.update(
                self._persistence_mse.get_logs(
                    label=f"{label}/persistence_normalized_mse"
                )
            )
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v

        return logs


class MultiScaleValidateAggregator:
    """Route validation batches to a scale-specific aggregator.

    Multi-scale loaders keep every batch on one homogeneous output grid.  This
    wrapper preserves that contract while giving each grid independent loss,
    reduced-metric, and image state.  Pre-registering every grid also makes an
    unexpected output resolution fail loudly instead of silently pooling it.
    """

    def __init__(
        self,
        aggregators: dict[tuple[int, int], tuple[str, ValidateAggregator]],
        route_aggregators: dict[
            tuple[tuple[int, int], tuple[int, int]], tuple[str, ValidateAggregator]
        ]
        | None = None,
    ) -> None:
        if not aggregators:
            raise ValueError("At least one validation grid must be registered.")
        self._aggregators = aggregators
        self._route_aggregators = route_aggregators
        first_aggregator = next(iter(aggregators.values()))[1]
        self._overall = TrainAggregator(first_aggregator.tensor_map)
        self._route_losses = (
            RouteTrainAggregator(first_aggregator.tensor_map)
            if route_aggregators is None
            else None
        )
        self._recorded_grids: set[tuple[int, int]] = set()
        self._recorded_routes: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput) -> None:
        grid = (batch.target_data.shape[-2], batch.target_data.shape[-1])
        if grid not in self._aggregators:
            raise ValueError(
                f"Validation batch uses unregistered output grid {grid}; "
                f"expected one of {sorted(self._aggregators)}."
            )
        _, aggregator = self._aggregators[grid]
        self._overall.record_batch(batch)
        route = RouteTrainAggregator.route(batch.ctx)
        if self._route_aggregators is None:
            assert self._route_losses is not None
            self._route_losses.record_batch(batch, batch.ctx)
        else:
            if route not in self._route_aggregators:
                raise ValueError(
                    f"Validation batch uses unregistered route {route}; "
                    f"expected one of {sorted(self._route_aggregators)}."
                )
            _, route_aggregator = self._route_aggregators[route]
            route_aggregator.record_validation_batch(batch)
            self._recorded_routes.add(route)
        aggregator.record_validation_batch(batch)
        self._recorded_grids.add(grid)

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Metrics:
        logs: MetricsDict = dict(self._overall.get_logs(label))
        if self._route_aggregators is None:
            assert self._route_losses is not None
            route_logs = dict(self._route_losses.get_logs(label))
            route_logs.pop(f"{label}/mean/loss")
            route_logs = {
                key: value for key, value in route_logs.items() if "/route/" in key
            }
            logs.update(route_logs)
        else:
            for route in sorted(self._recorded_routes):
                route_label, route_aggregator = self._route_aggregators[route]
                full_route_logs = route_aggregator.get_logs(
                    label=f"{label}/route/{route_label}"
                )
                overlap = logs.keys() & full_route_logs.keys()
                if overlap:
                    raise ValueError(
                        f"Duplicate multi-scale route log keys: {sorted(overlap)}"
                    )
                logs.update(full_route_logs)
        for grid in sorted(self._recorded_grids):
            scale_label, aggregator = self._aggregators[grid]
            scale_logs = aggregator.get_logs(label=f"{label}/resolution/{scale_label}")
            overlap = logs.keys() & scale_logs.keys()
            if overlap:
                raise ValueError(
                    f"Duplicate multi-scale validation log keys: {sorted(overlap)}"
                )
            logs.update(scale_logs)
        return logs
