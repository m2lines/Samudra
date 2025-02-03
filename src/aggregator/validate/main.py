"""
* This file includes code from ACE (https://github.com/ai2cm/ace).

* Licensed under the Apache License, Version 2.0
*
* Modified by Surya Dheeshjith
"""

from typing import Dict

import torch
from einops import rearrange

from aggregator.loss import LossAggregator
from aggregator.train import TrainAggregator
from aggregator.validate.map import MapAggregator
from aggregator.validate.snapshot import SnapshotAggregator
from stepper import ValOutput
from utils.data import Normalize, convert_tensor_out_to_dict


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(self, metadata: Dict[str, Dict[str, str]], hist: int):
        super().__init__()

        val_aggregators = {
            "snapshot": SnapshotAggregator(metadata, hist),
            "mean_map": MapAggregator(metadata, hist),
        }
        self._aggregators = val_aggregators
        self.normalize = Normalize.get_instance()
        self._loss_scaling = LossAggregator.get_instance().loss_scale
        self.hist = hist

    @torch.no_grad()
    def record_batch(self, batch: ValOutput):
        super().record_batch(batch)  # Record losses

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        target_data_dict, target_data_unnorm_dict = self.get_norm_unnorm_dicts(
            batch.target_data
        )

        gen_data_dict, gen_data_unnorm_dict = self.get_norm_unnorm_dicts(batch.gen_data)
        num_output_channels = len(target_data_dict.keys())
        input_data_dict, input_data_unnorm_dict = self.get_norm_unnorm_dicts(
            batch.input_data,
            input_type="input",
            num_output_channels=num_output_channels,
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
    def get_norm_unnorm_dicts(
        self,
        data: torch.Tensor,
        input_type: str = "target",
        num_output_channels: int = 0,
    ):
        # Remove boundary data if input
        if input_type == "input":
            data = data[:, : num_output_channels * (self.hist + 1)]

        # Separate history from channels
        data_ = rearrange(data, "n (hi c) h w -> n hi c h w", hi=self.hist + 1)
        # Get normalized dict
        data_dict = convert_tensor_out_to_dict(data_)
        # Unnormalize
        data_unnorm = self.normalize.unnormalize_tensor_outputs(data_)
        # Get unnormalized dict
        data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm)
        return data_dict, data_unnorm_dict

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
