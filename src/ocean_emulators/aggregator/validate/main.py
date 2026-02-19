from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.map import (
    get_map_logs,
    init_map_state,
    record_map_batch,
)
from ocean_emulators.aggregator.validate.reduced import (
    get_reduced_logs,
    init_reduced_state,
    record_reduced_batch,
)
from ocean_emulators.aggregator.validate.snapshot import (
    get_snapshot_logs,
    init_snapshot_state,
    record_snapshot_batch,
)
from ocean_emulators.utils.data import Normalize, get_aggregator_dicts
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict


@dataclass
class _ValidateAggregation:
    state: Any
    record_batch: Callable[..., None]
    get_logs: Callable[..., Metrics]


class ValidateAggregator:
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
    ):
        self._train_aggregator = TrainAggregator()
        self._aggregations: dict[str, _ValidateAggregation] = {
            "snapshot": _ValidateAggregation(
                state=init_snapshot_state(metadata, hist),
                record_batch=record_snapshot_batch,
                get_logs=get_snapshot_logs,
            ),
            "mean_map": _ValidateAggregation(
                state=init_map_state(metadata, hist),
                record_batch=record_map_batch,
                get_logs=get_map_logs,
            ),
            "reduced": _ValidateAggregation(
                state=init_reduced_state(area_weights, hist),
                record_batch=record_reduced_batch,
                get_logs=get_reduced_logs,
            ),
        }
        self.normalize = Normalize.get_instance()
        self.hist = hist
        self.num_prognostic_channels = num_prognostic_channels
        self.wet = wet

    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
        self._train_aggregator.record_batch(batch)  # Record losses

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        assert batch.target_data.shape[1] == self.num_prognostic_channels
        target_data_dict, target_data_unnorm_dict = get_aggregator_dicts(
            batch.target_data,
            wet=self.wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        gen_data_dict, gen_data_unnorm_dict = get_aggregator_dicts(
            batch.gen_data,
            wet=self.wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        input_data_dict, input_data_unnorm_dict = get_aggregator_dicts(
            batch.input_data,
            wet=self.wet,
            long_rollout=False,
            input_type="input",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        for agg in self._aggregations.values():
            agg.record_batch(
                agg.state,
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
        logs: MetricsDict = dict(self._train_aggregator.get_logs(label))
        for agg_label, agg in self._aggregations.items():
            for k, v in agg.get_logs(agg.state, label=agg_label).items():
                logs[f"{label}/{k}"] = v

        return logs
