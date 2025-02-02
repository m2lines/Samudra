# Adapted from https://github.com/ai2cm/ace/tree/main/fme/fme/ace/aggregator


from aggregator.inference import InferenceAggregator
from aggregator.train import TrainAggregator
from aggregator.validate import ValidateAggregator


class Aggregator:
    @staticmethod
    def get_train_aggregator(num_output_channels: int) -> TrainAggregator:
        return TrainAggregator(num_output_channels)

    @staticmethod
    def get_validation_aggregator(num_output_channels: int) -> ValidateAggregator:
        return ValidateAggregator(num_output_channels)

    @staticmethod
    def get_inference_aggregator() -> InferenceAggregator:
        return InferenceAggregator()
