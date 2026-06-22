import re

import torch

from samudra.aggregator.metrics import area_weighted_rmse
from samudra.constants import TensorMap
from samudra.utils.data import Normalize, get_aggregator_dicts
from samudra.utils.device import get_device
from samudra.utils.distributed import all_reduce_mean
from samudra.utils.output import ModelInferenceOutput
from samudra.utils.wandb import Metrics

_DEPTH_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<depth>[0-9]+)$")


class _MeanStepAreaWeightedRmse:
    """Accumulates the mean of per-step area-weighted RMSE values."""

    def __init__(self, area_weights: torch.Tensor, *, distributed_reduce: bool = True):
        self._area_weights = area_weights
        self._distributed_reduce = distributed_reduce
        self._rmse_sum = torch.tensor(0.0, device=get_device())
        self._n_steps = torch.tensor(0.0, device=get_device())

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        if target.shape != gen.shape:
            raise RuntimeError(
                f"target and gen must have the same shape, got {target.shape} "
                f"and {gen.shape}"
            )
        rmse_by_step = area_weighted_rmse(target, gen, self._area_weights)
        finite_rmse = rmse_by_step[torch.isfinite(rmse_by_step)]

        self._rmse_sum = self._rmse_sum.to(rmse_by_step.device)
        self._n_steps = self._n_steps.to(rmse_by_step.device)
        self._rmse_sum += finite_rmse.sum()
        self._n_steps += finite_rmse.numel()

    def rmse(self) -> torch.Tensor:
        rmse_sum = self._rmse_sum.detach()
        n_steps = self._n_steps.detach()
        if self._distributed_reduce:
            rmse_sum = all_reduce_mean(rmse_sum)
            n_steps = all_reduce_mean(n_steps)
        if n_steps == 0:
            return torch.tensor(float("nan"), device=rmse_sum.device)
        return rmse_sum / n_steps


def _split_depth_name(name: str) -> tuple[str, str | None]:
    match = _DEPTH_SUFFIX.match(name)
    if match is None:
        return name, None
    return match.group("base"), match.group("depth")


def _depth_band(depth_index: int | None, tensor_map: TensorMap) -> str:
    if depth_index is None:
        return "surface"
    depth = tensor_map.dataset_spec.depth_levels[depth_index]
    if depth < 700:
        return "upper"
    if depth < 2000:
        return "middle"
    return "deep"


def _depth_weight(depth_index: int | None, tensor_map: TensorMap) -> float:
    if depth_index is None:
        return 1.0
    return float(tensor_map.dataset_spec.depth_thickness[depth_index])


class RolloutValidationAggregator:
    """Aggregates raw-field RMSE over an autoregressive validation rollout."""

    def __init__(
        self,
        *,
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        normalize: Normalize,
        tensor_map: TensorMap,
        distributed_reduce: bool = True,
    ):
        self.hist = hist
        self._wet = wet
        self._num_prognostic_channels = num_prognostic_channels
        self._normalize = normalize
        self._tensor_map = tensor_map
        self._raw_field_names = tuple(tensor_map.prognostic_var_names)
        self._field_metrics = {
            name: _MeanStepAreaWeightedRmse(
                area_weights, distributed_reduce=distributed_reduce
            )
            for name in self._raw_field_names
        }

    def record_batch(self, data: ModelInferenceOutput):
        if len(data.prediction) == 0:
            raise ValueError("No prediction values in data")
        if len(data.target) == 0:
            raise ValueError("No target values in data")

        _, target_unnorm = get_aggregator_dicts(
            data.target,
            normalize=self._normalize,
            tensor_map=self._tensor_map,
            wet=self._wet,
            long_rollout=True,
            input_type="prognostic",
            num_prognostic_channels=self._num_prognostic_channels,
            hist=self.hist,
        )
        _, gen_unnorm = get_aggregator_dicts(
            data.prediction,
            normalize=self._normalize,
            tensor_map=self._tensor_map,
            wet=self._wet,
            long_rollout=True,
            input_type="prognostic",
            num_prognostic_channels=self._num_prognostic_channels,
            hist=self.hist,
        )

        for name in self._raw_field_names:
            self._field_metrics[name].record(
                target=target_unnorm[name],
                gen=gen_unnorm[name],
            )

    def get_logs(self, label: str) -> Metrics:
        logs: dict[str, float] = {}
        values_by_base_var: dict[str, list[torch.Tensor]] = {}
        values_by_depth_band: dict[
            str, dict[str, list[tuple[torch.Tensor, float]]]
        ] = {}
        for name, metric in sorted(self._field_metrics.items()):
            rmse = metric.rmse()
            base_var, depth = _split_depth_name(name)
            depth_index = int(depth) if depth is not None else None
            band = _depth_band(depth_index, self._tensor_map)
            weight = _depth_weight(depth_index, self._tensor_map)
            values_by_base_var.setdefault(base_var, []).append(rmse)
            values_by_depth_band.setdefault(base_var, {}).setdefault(band, []).append(
                (rmse, weight)
            )
            logs[f"{label}/weighted_rmse/{name}"] = float(rmse.cpu().item())
            if depth is not None:
                logs[f"{label}/weighted_rmse/{base_var}/depth_{depth}"] = float(
                    rmse.cpu().item()
                )

        for base_var, values in sorted(values_by_base_var.items()):
            if len(values) > 1:
                mean_rmse = torch.stack(values).mean()
                logs[f"{label}/weighted_rmse/{base_var}/mean_depths"] = float(
                    mean_rmse.cpu().item()
                )

        for base_var, bands in sorted(values_by_depth_band.items()):
            for band, weighted_values in sorted(bands.items()):
                band_values = torch.stack([value for value, _ in weighted_values])
                weights = torch.tensor(
                    [weight for _, weight in weighted_values],
                    dtype=band_values.dtype,
                    device=band_values.device,
                )
                band_rmse = (band_values * weights).sum() / weights.sum()
                logs[f"{label}/weighted_rmse/{base_var}/depth_band/{band}"] = float(
                    band_rmse.cpu().item()
                )

        return logs
