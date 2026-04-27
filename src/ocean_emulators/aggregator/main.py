# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
The code in this directory is directly inspired by the ACE project by AI2.

See the original repository at: https://github.com/ai2cm/ace/tree/39133c18524cda85965486ecdc8cb64aac06f4c3/fme/fme/ace/aggregator
"""

import torch

from ocean_emulators.aggregator.inference import InferenceEvaluatorAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate import ValidateAggregator
from ocean_emulators.aggregator.validate.map import MapAggregator
from ocean_emulators.aggregator.validate.reduced import MeanAggregator
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.data import Normalize


class Aggregator:
    @staticmethod
    def get_train_aggregator(tensor_map: TensorMap) -> TrainAggregator:
        return TrainAggregator(tensor_map)

    @staticmethod
    def get_validation_aggregator(
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        *,
        include_image_aggregators: bool = True,
    ) -> ValidateAggregator:
        val_aggregators: dict[str, ValidateSubAggregator] = {
            "reduced": MeanAggregator(area_weights, hist),
        }
        if include_image_aggregators:
            val_aggregators.update(
                {
                    "snapshot": SnapshotAggregator(metadata, hist),
                    "mean_map": MapAggregator(metadata, hist),
                }
            )

        return ValidateAggregator(
            val_aggregators,
            hist=hist,
            num_prognostic_channels=num_prognostic_channels,
            tensor_map=tensor_map,
            normalize=normalize,
        )

    @staticmethod
    def get_inline_inference_aggregator(
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
            normalize=normalize,
            tensor_map=tensor_map,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=False,
            log_global_mean_norm_time_series=False,
            channel_mean_names=channel_mean_names,
        )

    @staticmethod
    def get_standalone_inference_aggregator(
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
            normalize=normalize,
            tensor_map=tensor_map,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=True,
            log_global_mean_norm_time_series=True,
            channel_mean_names=channel_mean_names,
        )
