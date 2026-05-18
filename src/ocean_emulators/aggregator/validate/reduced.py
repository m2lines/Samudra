# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_gradient_magnitude_percent_diff,
    area_weighted_mean,
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.distributed import all_reduce_mean, all_reduce_sum


class AreaWeightedReducedMetric:
    """
    A wrapper around an area-weighted metric function.
    """

    def __init__(
        self,
        device: torch.device,
        compute_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self._compute_metric = compute_metric
        self._total: torch.Tensor | None = None
        self._device = device

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
        """
        new_value = self._compute_metric(target, gen).mean(dim=0)
        if self._total is None:
            self._total = torch.zeros_like(new_value, device=self._device)
        self._total += new_value

    def get(self) -> torch.Tensor:
        """Returns the metric."""
        assert self._total is not None
        return self._total


class MeanAggregator(ValidateSubAggregator):
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(self, area_weights: torch.Tensor, target_time: int):
        self._n_batches = 0
        self._variable_metrics: dict | None = None
        self._target_time = target_time
        self._area_weights = area_weights

    def _get_variable_metrics(self, gen_data):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "weighted_rmse": {},
                "weighted_bias": {},
                "weighted_grad_mag_percent_diff": {},
            }
            device = get_device()
            for key in gen_data:
                self._variable_metrics["weighted_rmse"][key] = (
                    AreaWeightedReducedMetric(
                        device=device,
                        compute_metric=partial(
                            area_weighted_rmse, area_weights=self._area_weights
                        ),
                    )
                )
                self._variable_metrics["weighted_bias"][key] = (
                    AreaWeightedReducedMetric(
                        device=device,
                        compute_metric=partial(
                            area_weighted_mean_bias, area_weights=self._area_weights
                        ),
                    )
                )
                self._variable_metrics["weighted_grad_mag_percent_diff"][key] = (
                    AreaWeightedReducedMetric(
                        device=device,
                        compute_metric=partial(
                            area_weighted_gradient_magnitude_percent_diff,
                            area_weights=self._area_weights,
                        ),  # noqa: E501
                    )
                )

        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss: torch.Tensor = torch.tensor(np.nan),
        target_data,
        gen_data,
        target_data_norm,
        gen_data_norm,
        input_data: dict[str, torch.Tensor] | None = None,
        input_data_norm: dict[str, torch.Tensor] | None = None,
        i_time_start: int = 0,
    ):
        if input_data is None:
            input_data = {}
        if input_data_norm is None:
            input_data_norm = {}

        variable_metrics = self._get_variable_metrics(gen_data)
        time_dim = 1
        time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim]
        target_time = self._target_time - i_time_start
        if target_time >= 0 and time_len > target_time:
            for name in gen_data.keys():
                target = target_data[name].select(dim=time_dim, index=target_time)
                gen = gen_data[name].select(dim=time_dim, index=target_time)
                for metric in variable_metrics:
                    variable_metrics[metric][name].record(
                        target=target,
                        gen=gen,
                    )
            # only increment n_batches if we actually recorded a batch
            self._n_batches += 1

    def _get_data(self):
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: dict[str, torch.Tensor] = {}
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                data[f"{metric}/{key}"] = (
                    self._variable_metrics[metric][key].get() / self._n_batches
                )
        meaned_data: dict[str, float] = {}
        for key in sorted(data.keys()):
            meaned_data[key] = float(all_reduce_mean(data[key].detach()).cpu().numpy())
        return meaned_data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return {
            f"{label}/{key}": data for key, data in sorted(self._get_data().items())
        }


class AreaWeightedTemporalStdMetric:
    """Per-pixel running stats for one variable, used to compute temporal std
    across the validation set.

    `get()` returns the area-weighted spatial mean of std(pred) / std(target),
    where each per-pixel std is computed over all recorded validation samples.
    Diagnoses mode collapse: the ratio is near 0 when the model output has
    no temporal variability, and near 1 when it tracks the target distribution.
    """

    def __init__(self, device: torch.device, area_weights: torch.Tensor):
        self._device = device
        self._area_weights = area_weights
        self._sum_gen: torch.Tensor | None = None
        self._sum_sq_gen: torch.Tensor | None = None
        self._sum_target: torch.Tensor | None = None
        self._sum_sq_target: torch.Tensor | None = None
        self._count: int = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        if self._sum_gen is None:
            shape = gen.shape[1:]
            self._sum_gen = torch.zeros(shape, device=self._device, dtype=torch.float64)
            self._sum_sq_gen = torch.zeros(
                shape, device=self._device, dtype=torch.float64
            )
            self._sum_target = torch.zeros(
                shape, device=self._device, dtype=torch.float64
            )
            self._sum_sq_target = torch.zeros(
                shape, device=self._device, dtype=torch.float64
            )
        assert self._sum_sq_gen is not None
        assert self._sum_target is not None
        assert self._sum_sq_target is not None
        g = gen.to(torch.float64)
        t = target.to(torch.float64)
        self._sum_gen += g.nansum(dim=0)
        self._sum_sq_gen += (g * g).nansum(dim=0)
        self._sum_target += t.nansum(dim=0)
        self._sum_sq_target += (t * t).nansum(dim=0)
        self._count += gen.shape[0]

    def get(self) -> torch.Tensor:
        assert self._sum_gen is not None
        assert self._sum_sq_gen is not None
        assert self._sum_target is not None
        assert self._sum_sq_target is not None
        sum_gen = all_reduce_sum(self._sum_gen.clone())
        sum_sq_gen = all_reduce_sum(self._sum_sq_gen.clone())
        sum_target = all_reduce_sum(self._sum_target.clone())
        sum_sq_target = all_reduce_sum(self._sum_sq_target.clone())
        count_t = torch.tensor([self._count], device=self._device, dtype=torch.float64)
        count = float(all_reduce_sum(count_t).item())
        if count < 2:
            return torch.tensor(float("nan"), device=self._device)

        mean_gen = sum_gen / count
        mean_target = sum_target / count
        var_gen = (sum_sq_gen / count - mean_gen * mean_gen).clamp_min(0.0)
        var_target = (sum_sq_target / count - mean_target * mean_target).clamp_min(0.0)
        std_gen = var_gen.sqrt()
        std_target = var_target.sqrt()

        # Pixels with zero target variability (e.g. static land masked to a
        # constant or near-constant value) are excluded from the spatial mean
        # by NaN-ing the ratio; area_weighted_mean treats NaN as land.
        ratio = torch.where(
            std_target > 1e-12,
            std_gen / std_target,
            torch.full_like(std_gen, float("nan")),
        )
        return area_weighted_mean(ratio, self._area_weights.to(self._device))


class StdRatioAggregator(ValidateSubAggregator):
    """Per-variable temporal std(pred)/std(target) over the validation set.

    Designed to catch mode collapse (e.g. v48 climatology output) early:
    a healthy run holds this ratio near 1; a collapsed run drops toward 0.
    """

    def __init__(self, area_weights: torch.Tensor, target_time: int):
        self._area_weights = area_weights
        self._target_time = target_time
        self._metrics: dict[str, AreaWeightedTemporalStdMetric] | None = None

    def _get_metrics(self, gen_data):
        if self._metrics is None:
            device = get_device()
            self._metrics = {
                key: AreaWeightedTemporalStdMetric(
                    device=device, area_weights=self._area_weights
                )
                for key in gen_data
            }
        return self._metrics

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss: torch.Tensor = torch.tensor(np.nan),
        target_data,
        gen_data,
        target_data_norm,
        gen_data_norm,
        input_data: dict[str, torch.Tensor] | None = None,
        input_data_norm: dict[str, torch.Tensor] | None = None,
        i_time_start: int = 0,
    ):
        metrics = self._get_metrics(gen_data)
        time_dim = 1
        time_len = gen_data[next(iter(gen_data))].shape[time_dim]
        target_time = self._target_time - i_time_start
        if target_time >= 0 and time_len > target_time:
            for name in gen_data:
                target = target_data[name].select(dim=time_dim, index=target_time)
                gen = gen_data[name].select(dim=time_dim, index=target_time)
                metrics[name].record(target=target, gen=gen)

    @torch.no_grad()
    def get_logs(self, label: str):
        if self._metrics is None:
            return {}
        return {
            f"{label}/{key}": float(metric.get().detach().cpu().numpy())
            for key, metric in sorted(self._metrics.items())
        }
