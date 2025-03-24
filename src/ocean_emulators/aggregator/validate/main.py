from typing import Dict

import torch

from ocean_emulators.aggregator.loss import LossAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.map import MapAggregator
from ocean_emulators.aggregator.validate.reduced import MeanAggregator
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.data import Normalize, get_norm_unnorm_dicts
from ocean_emulators.utils.model import ValOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        metadata: Dict[str, Dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        num_prognostic_channels: int,
    ):
        super().__init__()

        val_aggregators: dict[str, ValidateSubAggregator] = {
            "snapshot": SnapshotAggregator(metadata, hist),
            "mean_map": MapAggregator(metadata, hist),
            "reduced": MeanAggregator(area_weights, hist),
        }
        self._aggregators = val_aggregators
        self.normalize = Normalize.get_instance()
        self._loss_scaling = LossAggregator.get_instance().loss_scale
        self.hist = hist
        self.num_prognostic_channels = num_prognostic_channels

    # TODO(jder): we could remove this by moving from inheritance
    # to composition with the TrainAggregator functionality.
    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValOutput):
        super().record_batch(batch)  # Record losses

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        target_data_dict, target_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.target_data,
            input_type="target",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        gen_data_dict, gen_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.gen_data,
            input_type="gen",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        input_data_dict, input_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.input_data,
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
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
        # TODO(jder): we have an implicit assumption here that
        # the superclass actually only returns float values;
        # would be nice move that to a separate, maybe standalone, function.
        logs.update(
            self._get_loss_scaled_mse_components(
                validation_metrics=logs,  # type: ignore[arg-type]
                label=label,
            )
        )
        return logs

    def _get_loss_scaled_mse_components(
        self,
        validation_metrics: Dict[str, float],
        label: str,
    ):
        """
        Account for different scales and units between variables using a
        custom loss scaling dictionary. Then compute the fractional contributions
        of each variable to the total loss.
        """
        scaled_squared_errors = {}

        for var in self._loss_scaling:
            rmse_key = f"{label}/mean/weighted_rmse/{var}"
            if rmse_key in validation_metrics:
                scaled_squared_errors[var] = (
                    validation_metrics[rmse_key] / self._loss_scaling[var]
                ) ** 2
        scaled_squared_errors_sum = sum(scaled_squared_errors.values())
        fractional_contribs = {
            f"{label}/mean/mse_fractional_components/{k}": v / scaled_squared_errors_sum
            for k, v in scaled_squared_errors.items()
        }
        return fractional_contribs
