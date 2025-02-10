from typing import Dict

import torch

from aggregator.loss import LossAggregator
from aggregator.train import TrainAggregator
from aggregator.validate.map import MapAggregator
from aggregator.validate.reduced import MeanAggregator
from aggregator.validate.snapshot import SnapshotAggregator
from utils.data import Normalize, get_norm_unnorm_dicts
from utils.model import ValOutput


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        metadata: Dict[str, Dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        output_channels: int,
    ):
        super().__init__()

        val_aggregators = {
            "snapshot": SnapshotAggregator(metadata, hist),
            "mean_map": MapAggregator(metadata, hist),
            "reduced": MeanAggregator(area_weights, hist),
        }
        self._aggregators = val_aggregators
        self.normalize = Normalize.get_instance()
        self._loss_scaling = LossAggregator.get_instance().loss_scale
        self.hist = hist
        self.output_channels = output_channels

    @torch.no_grad()
    def record_batch(self, batch: ValOutput):
        super().record_batch(batch)  # Record losses

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        target_data_dict, target_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.target_data,
            input_type="target",
            output_channels=self.output_channels,
            hist=self.hist,
        )

        gen_data_dict, gen_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.gen_data,
            input_type="gen",
            output_channels=self.output_channels,
            hist=self.hist,
        )
        input_data_dict, input_data_unnorm_dict = get_norm_unnorm_dicts(
            batch.input_data,
            input_type="input",
            output_channels=self.output_channels,
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
    def get_logs(self, label: str):
        logs = super().get_logs(label)
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
        logs.update(
            self._get_loss_scaled_mse_components(
                validation_metrics=logs,
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
                    validation_metrics[rmse_key] / self._loss_scaling[var].item()
                ) ** 2
        scaled_squared_errors_sum = sum(scaled_squared_errors.values())
        fractional_contribs = {
            f"{label}/mean/mse_fractional_components/{k}": v / scaled_squared_errors_sum
            for k, v in scaled_squared_errors.items()
        }
        return fractional_contribs
