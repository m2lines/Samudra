"""
The code in this directory is directly inspired by the ACE project by AI2.

See the original repository at: https://github.com/ai2cm/ace/tree/39133c18524cda85965486ecdc8cb64aac06f4c3/fme/fme/ace/aggregator
"""

import torch

from ocean_emulators.aggregator.inference import InferenceEvaluatorAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate import ValidateAggregator


class Aggregator:
    @staticmethod
    def get_train_aggregator() -> TrainAggregator:
        return TrainAggregator()

    @staticmethod
    def get_validation_aggregator(
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
    ) -> ValidateAggregator:
        return ValidateAggregator(
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
        )

    @staticmethod
    def get_inline_inference_aggregator(
        n_timesteps: int,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        num_prognostic_channels: int,
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
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
        channel_mean_names: list[str] | None = None,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            wet=wet,
            num_prognostic_channels=num_prognostic_channels,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=True,
            log_global_mean_norm_time_series=True,
            channel_mean_names=channel_mean_names,
        )
