from typing import Dict

import torch

from aggregator.inference import InferenceEvaluatorAggregator
from aggregator.train import TrainAggregator
from aggregator.validate import ValidateAggregator


class Aggregator:
    @staticmethod
    def get_train_aggregator() -> TrainAggregator:
        return TrainAggregator()

    @staticmethod
    def get_validation_aggregator(
        metadata: Dict[str, Dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        output_channels: int,
    ) -> ValidateAggregator:
        return ValidateAggregator(
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            output_channels=output_channels,
        )

    @staticmethod
    def get_inline_inference_aggregator(
        n_timesteps: int,
        metadata: Dict[str, Dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        output_channels: int,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            output_channels=output_channels,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=False,
            log_global_mean_norm_time_series=False,
        )

    @staticmethod
    def get_standalone_inference_aggregator(
        n_timesteps: int,
        metadata: Dict[str, Dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        output_channels: int,
    ) -> InferenceEvaluatorAggregator:
        return InferenceEvaluatorAggregator(
            n_timesteps=n_timesteps,
            metadata=metadata,
            hist=hist,
            area_weights=area_weights,
            output_channels=output_channels,
            record_step_20=(n_timesteps > 20),
            log_global_mean_time_series=True,
            log_global_mean_norm_time_series=True,
        )
