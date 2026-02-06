from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_gradient_magnitude_percent_diff,
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.data import DataSource, gridstr
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

    def __init__(self, srcs: list[DataSource], target_time: int):
        self.srcs = srcs
        self._n_batches = 0
        self._variable_metrics: dict | None = None
        self._target_time = target_time

    def _get_variable_metrics(self, gen_data, src: DataSource):
        area_weights = src.spherical_area_weights
        rmse_key = f"weighted_rmse/{gridstr(src)}"
        bias_key = f"weighted_bias/{gridstr(src)}"
        grad_diff_key = f"weighted_grad_mag_percent_diff/{gridstr(src)}"
        if self._variable_metrics is None:
            self._variable_metrics = {
                rmse_key: {},
                bias_key: {},
                grad_diff_key: {},
            }
            device = get_device()
            for key in gen_data:
                self._variable_metrics[rmse_key][key] = AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=partial(
                        area_weighted_rmse, area_weights=area_weights
                    ),
                )
                self._variable_metrics[bias_key][key] = AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=partial(
                        area_weighted_mean_bias, area_weights=area_weights
                    ),
                )
                self._variable_metrics[grad_diff_key][key] = AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=partial(
                        area_weighted_gradient_magnitude_percent_diff,
                        area_weights=area_weights,
                    ),  # noqa: E501
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
        # The label data shape will vary batch to batch during multi-scale training.
        # thus, we look up the relevant are weights for this batch.
        gen_date_grid = next(iter(gen_data.values())).shape[-2:]
        src = next(s for s in self.srcs if s.grid_size == gen_date_grid)

        variable_metrics = self._get_variable_metrics(gen_data, src)
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
