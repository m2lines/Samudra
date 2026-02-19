from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from wandb.data_types import WBValue

from ocean_emulators.utils.distributed import all_reduce_mean
from ocean_emulators.utils.wandb import Metrics, MetricsDict

MetricData: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
RenderedMetric: TypeAlias = float | torch.Tensor | WBValue


@dataclass
class MetricSpec:
    path: str
    reduce: Literal["mean", "last"]
    compute: Callable[[Any], MetricData]
    render: Callable[[MetricData], RenderedMetric]
    distributed_mean: bool = False
    total: MetricData | None = None
    count: int = 0
    last: MetricData | None = None


def mean_metric(path: str, compute, render) -> MetricSpec:
    return MetricSpec(
        path=path,
        reduce="mean",
        compute=compute,
        render=render,
        distributed_mean=True,
    )


def last_metric(path: str, compute, render) -> MetricSpec:
    return MetricSpec(
        path=path,
        reduce="last",
        compute=compute,
        render=render,
        distributed_mean=False,
    )


def _tree_binary(
    left: MetricData,
    right: MetricData,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> MetricData:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return op(left, right)
    assert isinstance(right, tuple)
    assert len(left) == len(right)
    return tuple(
        op(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left, right, strict=True)
    )


def _tree_unary(
    value: MetricData,
    op: Callable[[torch.Tensor], torch.Tensor],
) -> MetricData:
    if isinstance(value, torch.Tensor):
        return op(value)
    return tuple(op(item) for item in value)


def _tree_add(left: MetricData, right: MetricData) -> MetricData:
    return _tree_binary(
        left, right, lambda left_tensor, right_tensor: left_tensor + right_tensor
    )


def _tree_div(value: MetricData, divisor: int) -> MetricData:
    return _tree_unary(value, lambda tensor: tensor / divisor)


def _tree_all_reduce_mean(value: MetricData) -> MetricData:
    return _tree_unary(value, all_reduce_mean)


class MetricEngine:
    def __init__(self, specs: list[MetricSpec]):
        self._specs = specs

    def record(self, batch: Any):
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
