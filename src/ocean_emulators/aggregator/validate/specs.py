from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from wandb.data_types import WBValue

from ocean_emulators.aggregator.metrics import (
    area_weighted_gradient_magnitude_percent_diff,
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.aggregator.plotting import plot_paneled_data
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.wandb import Metrics, MetricsDict

_MAP_CAPTIONS = {
    "full-field": (
        "{name} one step mean full field; (top) generated and (bottom) target [{units}]"
    ),
    "error": "{name} one step mean full field error (generated - target) [{units}]",
}

_SNAPSHOT_CAPTIONS = {
    "full-field": (
        "{name} one step full field for last sample; "
        "(top) generated and (bottom) target [{units}]"
    ),
    "residual": (
        "{name} one step residual (prediction - previous time) for last sample; "
        "(top) generated and (bottom) target [{units}]"
    ),
    "error": (
        "{name} one step full field error (generated - target) "
        "for last sample [{units}]"
    ),
}

MetricData = (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


@dataclass
class ValidationBatchMetricsInput:
    loss: torch.Tensor
    loss_per_channel: torch.Tensor
    target_data: dict[str, torch.Tensor]
    gen_data: dict[str, torch.Tensor]
    input_data: dict[str, torch.Tensor]
    target_data_norm: dict[str, torch.Tensor]
    gen_data_norm: dict[str, torch.Tensor]
    input_data_norm: dict[str, torch.Tensor]


@dataclass
class MetricSpec:
    path: str
    reduce: Literal["mean", "last"]
    compute: Callable[[ValidationBatchMetricsInput], MetricData]
    render: Callable[[MetricData], float | torch.Tensor | WBValue]
    distributed_mean: bool = False
    total: MetricData | None = None
    count: int = 0
    last: MetricData | None = None


def _tree_add(left: MetricData, right: MetricData) -> MetricData:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return left + right
    if len(left) == 2:
        right_pair = right
        assert isinstance(right_pair, tuple) and len(right_pair) == 2
        return (left[0] + right_pair[0], left[1] + right_pair[1])
    right_triplet = right
    assert isinstance(right_triplet, tuple) and len(right_triplet) == 3
    return (
        left[0] + right_triplet[0],
        left[1] + right_triplet[1],
        left[2] + right_triplet[2],
    )


def _tree_div(value: MetricData, divisor: int) -> MetricData:
    if isinstance(value, torch.Tensor):
        return value / divisor
    if len(value) == 2:
        return (value[0] / divisor, value[1] / divisor)
    return (value[0] / divisor, value[1] / divisor, value[2] / divisor)


def _tree_all_reduce_mean(value: MetricData) -> MetricData:
    if isinstance(value, torch.Tensor):
        return all_reduce_mean(value)
    if len(value) == 2:
        return (all_reduce_mean(value[0]), all_reduce_mean(value[1]))
    return (
        all_reduce_mean(value[0]),
        all_reduce_mean(value[1]),
        all_reduce_mean(value[2]),
    )


def _to_scalar(value: MetricData) -> float:
    assert isinstance(value, torch.Tensor)
    return float(value.detach().cpu().numpy())


def _get_caption(
    caption_templates: dict[str, str],
    metadata: dict[str, dict[str, str]],
    key: str,
    name: str,
) -> str:
    if name in metadata:
        caption_name = metadata[name]["long_name"]
        units = metadata[name]["units"]
    else:
        caption_name, units = name, "unknown_units"
    return caption_templates[key].format(name=caption_name, units=units)


def _image_snapshot_error(
    value: MetricData,
    metadata: dict[str, dict[str, str]],
    name: str,
):
    assert isinstance(value, tuple) and len(value) == 3
    gen, target, _ = value
    return plot_paneled_data(
        [[(gen - target).cpu().numpy()]],
        diverging=True,
        caption=_get_caption(_SNAPSHOT_CAPTIONS, metadata, "error", name),
    )


def _image_snapshot_full(
    value: MetricData,
    metadata: dict[str, dict[str, str]],
    name: str,
):
    assert isinstance(value, tuple) and len(value) == 3
    gen, target, _ = value
    return plot_paneled_data(
        [[gen.cpu().numpy()], [target.cpu().numpy()]],
        diverging=False,
        caption=_get_caption(_SNAPSHOT_CAPTIONS, metadata, "full-field", name),
    )


def _image_snapshot_residual(
    value: MetricData,
    metadata: dict[str, dict[str, str]],
    name: str,
):
    assert isinstance(value, tuple) and len(value) == 3
    gen, target, input_data = value
    return plot_paneled_data(
        [[(gen - input_data).cpu().numpy()], [(target - input_data).cpu().numpy()]],
        diverging=True,
        caption=_get_caption(_SNAPSHOT_CAPTIONS, metadata, "residual", name),
    )


def _image_mean_map_error(
    value: MetricData,
    metadata: dict[str, dict[str, str]],
    name: str,
):
    assert isinstance(value, tuple) and len(value) == 2
    gen, target = value
    return plot_paneled_data(
        [[(gen - target).cpu().numpy()]],
        diverging=True,
        caption=_get_caption(_MAP_CAPTIONS, metadata, "error", name),
    )


def _image_mean_map_full(
    value: MetricData,
    metadata: dict[str, dict[str, str]],
    name: str,
):
    assert isinstance(value, tuple) and len(value) == 2
    gen, target = value
    return plot_paneled_data(
        [[gen.cpu().numpy()], [target.cpu().numpy()]],
        diverging=False,
        caption=_get_caption(_MAP_CAPTIONS, metadata, "full-field", name),
    )


def _mean_path(path: str, compute, render) -> MetricSpec:
    return MetricSpec(
        path=path,
        reduce="mean",
        compute=compute,
        render=render,
        distributed_mean=True,
    )


def _last_path(path: str, compute, render) -> MetricSpec:
    return MetricSpec(
        path=path,
        reduce="last",
        compute=compute,
        render=render,
        distributed_mean=False,
    )


def build_validation_metric_specs(
    *,
    metadata: dict[str, dict[str, str]],
    hist: int,
    area_weights: torch.Tensor,
    var_names: list[str],
) -> list[MetricSpec]:
    tensor_map = TensorMap.get_instance()
    specs: list[MetricSpec] = []

    def define_mean(path, compute, render):
        specs.append(_mean_path(path, compute, render))

    def define_last(path, compute, render):
        specs.append(_last_path(path, compute, render))

    define_mean("mean/loss", lambda batch: batch.loss, _to_scalar)
    for depth in tensor_map.DEPTH_SET:
        idx = tensor_map.DP_3D_IDX[depth]
        define_mean(
            f"loss/depth/depth_{depth}_loss",
            lambda batch, idx=idx: batch.loss_per_channel[idx].mean(),
            _to_scalar,
        )
    for variable in tensor_map.VAR_SET:
        idx = tensor_map.VAR_3D_IDX[variable]
        define_mean(
            f"loss/variable/{variable}_loss",
            lambda batch, idx=idx: batch.loss_per_channel[idx].mean(),
            _to_scalar,
        )
    for i, channel in enumerate(tensor_map.prognostic_var_names):
        define_mean(
            f"loss/channel/{channel}_loss",
            lambda batch, i=i: batch.loss_per_channel[i],
            _to_scalar,
        )

    for name in var_names:
        define_mean(
            f"reduced/weighted_rmse/{name}",
            lambda batch, name=name: area_weighted_rmse(
                target=batch.target_data[name].select(dim=1, index=hist),
                gen=batch.gen_data[name].select(dim=1, index=hist),
                area_weights=area_weights,
            ).mean(dim=0),
            _to_scalar,
        )
        define_mean(
            f"reduced/weighted_bias/{name}",
            lambda batch, name=name: area_weighted_mean_bias(
                target=batch.target_data[name].select(dim=1, index=hist),
                gen=batch.gen_data[name].select(dim=1, index=hist),
                area_weights=area_weights,
            ).mean(dim=0),
            _to_scalar,
        )
        define_mean(
            f"reduced/weighted_grad_mag_percent_diff/{name}",
            lambda batch, name=name: area_weighted_gradient_magnitude_percent_diff(
                target=batch.target_data[name].select(dim=1, index=hist),
                gen=batch.gen_data[name].select(dim=1, index=hist),
                area_weights=area_weights,
            ).mean(dim=0),
            _to_scalar,
        )

    for name in sorted(var_names):
        define_mean(
            f"mean_map/image-error/{name}",
            lambda batch, name=name: (
                batch.gen_data[name].mean(dim=0).select(dim=0, index=hist),
                batch.target_data[name].mean(dim=0).select(dim=0, index=hist),
            ),
            lambda value, name=name: _image_mean_map_error(value, metadata, name),
        )
        define_mean(
            f"mean_map/image-full-field/{name}",
            lambda batch, name=name: (
                batch.gen_data[name].mean(dim=0).select(dim=0, index=hist),
                batch.target_data[name].mean(dim=0).select(dim=0, index=hist),
            ),
            lambda value, name=name: _image_mean_map_full(value, metadata, name),
        )

    for name in var_names:
        define_last(
            f"snapshot/image-error/{name}",
            lambda batch, name=name: (
                batch.gen_data[name].select(dim=1, index=0)[0],
                batch.target_data[name].select(dim=1, index=0)[0],
                batch.input_data[name].select(dim=1, index=hist)[0],
            ),
            lambda value, name=name: _image_snapshot_error(value, metadata, name),
        )
        define_last(
            f"snapshot/image-full-field/{name}",
            lambda batch, name=name: (
                batch.gen_data[name].select(dim=1, index=0)[0],
                batch.target_data[name].select(dim=1, index=0)[0],
                batch.input_data[name].select(dim=1, index=hist)[0],
            ),
            lambda value, name=name: _image_snapshot_full(value, metadata, name),
        )
        define_last(
            f"snapshot/image-residual/{name}",
            lambda batch, name=name: (
                batch.gen_data[name].select(dim=1, index=0)[0],
                batch.target_data[name].select(dim=1, index=0)[0],
                batch.input_data[name].select(dim=1, index=hist)[0],
            ),
            lambda value, name=name: _image_snapshot_residual(value, metadata, name),
        )

    return specs


class ValidationMetricEngine:
    def __init__(self, specs: list[MetricSpec]):
        self._specs = specs

    def record(self, batch: ValidationBatchMetricsInput):
        for spec in self._specs:
            value = spec.compute(batch)
            if spec.reduce == "last":
                spec.last = value
                continue
            if spec.total is None:
                spec.total = value
            else:
                spec.total = _tree_add(spec.total, value)
            spec.count += 1

    def get_logs(self) -> Metrics:
        logs: MetricsDict = {}
        for spec in self._specs:
            if spec.reduce == "last":
                if spec.last is None:
                    raise ValueError(
                        f"No values recorded for metric path '{spec.path}'."
                    )
                value = spec.last
            else:
                if spec.total is None or spec.count == 0:
                    raise ValueError(
                        f"No values recorded for metric path '{spec.path}'."
                    )
                value = _tree_div(spec.total, spec.count)
                if spec.distributed_mean:
                    value = _tree_all_reduce_mean(value)
            logs[spec.path] = spec.render(value)
        return logs
