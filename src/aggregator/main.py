"""
* This file includes code from ACE (https://github.com/ai2cm/ace).

* Licensed under the Apache License, Version 2.0
*
* Modified by Surya Dheeshjith
"""

from typing import Dict

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
    ) -> ValidateAggregator:
        return ValidateAggregator(metadata, hist)

    @staticmethod
    def get_inference_aggregator() -> InferenceAggregator:
        return InferenceAggregator()
