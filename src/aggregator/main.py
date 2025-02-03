from typing import Dict

import torch

from aggregator.inference import InferenceAggregator
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
    ) -> ValidateAggregator:
        return ValidateAggregator(metadata, hist, area_weights)

    @staticmethod
    def get_inference_aggregator() -> InferenceAggregator:
        return InferenceAggregator()
