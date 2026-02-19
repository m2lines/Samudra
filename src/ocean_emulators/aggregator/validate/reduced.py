from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_gradient_magnitude_percent_diff,
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.wandb import Metrics


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


@dataclass
class ReducedState:
    area_weights: torch.Tensor
    target_time: int
    n_batches: int = 0
    variable_metrics: dict | None = None


def init_reduced_state(area_weights: torch.Tensor, target_time: int) -> ReducedState:
    return ReducedState(area_weights=area_weights, target_time=target_time)


def _get_variable_metrics(state: ReducedState, gen_data):
    if state.variable_metrics is None:
        state.variable_metrics = {
            "weighted_rmse": {},
            "weighted_bias": {},
            "weighted_grad_mag_percent_diff": {},
        }
        device = get_device()
        for key in gen_data:
            state.variable_metrics["weighted_rmse"][key] = AreaWeightedReducedMetric(
                device=device,
                compute_metric=partial(
                    area_weighted_rmse, area_weights=state.area_weights
                ),
            )
            state.variable_metrics["weighted_bias"][key] = AreaWeightedReducedMetric(
                device=device,
                compute_metric=partial(
                    area_weighted_mean_bias, area_weights=state.area_weights
                ),
            )
            state.variable_metrics["weighted_grad_mag_percent_diff"][key] = (
                AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=partial(
                        area_weighted_gradient_magnitude_percent_diff,
                        area_weights=state.area_weights,
                    ),
                )
            )

    return state.variable_metrics


@torch.no_grad()
def record_reduced_batch(
    state: ReducedState,
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
    del loss, target_data_norm, gen_data_norm
    if input_data is None:
        input_data = {}
    if input_data_norm is None:
        input_data_norm = {}
    del input_data, input_data_norm

    variable_metrics = _get_variable_metrics(state, gen_data)
    time_dim = 1
    time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim]
    target_time = state.target_time - i_time_start
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
        state.n_batches += 1


def _get_reduced_data(state: ReducedState):
    if state.variable_metrics is None or state.n_batches == 0:
        raise ValueError("No batches have been recorded.")
    data: dict[str, torch.Tensor] = {}
    for metric in state.variable_metrics:
        for key in state.variable_metrics[metric]:
            data[f"{metric}/{key}"] = (
                state.variable_metrics[metric][key].get() / state.n_batches
            )
    meaned_data: dict[str, float] = {}
    for key in sorted(data.keys()):
        meaned_data[key] = float(all_reduce_mean(data[key].detach()).cpu().numpy())
    return meaned_data


@torch.no_grad()
def get_reduced_logs(state: ReducedState, label: str) -> Metrics:
    return {
        f"{label}/{key}": data for key, data in sorted(_get_reduced_data(state).items())
    }


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(self, area_weights: torch.Tensor, target_time: int):
        self._state = init_reduced_state(
            area_weights=area_weights, target_time=target_time
        )

    def _get_variable_metrics(self, gen_data):
        return _get_variable_metrics(self._state, gen_data)

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
        record_reduced_batch(
            self._state,
            loss=loss,
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm=target_data_norm,
            gen_data_norm=gen_data_norm,
            input_data=input_data,
            input_data_norm=input_data_norm,
            i_time_start=i_time_start,
        )

    def _get_data(self):
        return _get_reduced_data(self._state)

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return get_reduced_logs(self._state, label=label)
