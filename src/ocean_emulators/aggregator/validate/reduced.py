from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_gradient_magnitude_percent_diff,
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.aggregator.validate.main import ValidateSubAggregator
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.distributed import all_reduce_mean


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
        self._total = torch.tensor(torch.nan)
        self._device = device

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
        """
        new_value = self._compute_metric(target, gen).mean(dim=0)
        if torch.isnan(self._total).all():
            self._total = torch.zeros_like(new_value, device=self._device)
        self._total += new_value

    def get(self) -> torch.Tensor:
        """Returns the metric."""
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
        self._variable_metrics: Optional[Dict] = None
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
        input_data: Dict[str, torch.Tensor] = {},
        input_data_norm: Dict[str, torch.Tensor] = {},
        i_time_start: int = 0,
    ):
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
        data: Dict[str, torch.Tensor] = {}
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                data[f"{metric}/{key}"] = (
                    self._variable_metrics[metric][key].get() / self._n_batches
                )
        meaned_data: Dict[str, float] = {}
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
