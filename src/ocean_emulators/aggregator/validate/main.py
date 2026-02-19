import torch

from ocean_emulators.aggregator.spec_engine import MetricEngine
from ocean_emulators.aggregator.validate.specs import (
    ValidationBatchMetricsInput,
    build_validation_metric_specs,
)
from ocean_emulators.utils.data import Normalize, get_aggregator_dicts
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics


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
        self._metric_engine: MetricEngine | None = None
        self.normalize = Normalize.get_instance()
        self.metadata = metadata
        self.hist = hist
        self.area_weights = area_weights
        self.num_prognostic_channels = num_prognostic_channels
        self.wet = wet

    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
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

        if self._metric_engine is None:
            self._metric_engine = MetricEngine(
                build_validation_metric_specs(
                    metadata=self.metadata,
                    hist=self.hist,
                    area_weights=self.area_weights,
                    var_names=list(gen_data_unnorm_dict.keys()),
                )
            )
        self._metric_engine.record(
            ValidationBatchMetricsInput(
                loss=batch.loss,
                loss_per_channel=batch.loss_per_channel,
                target_data=target_data_unnorm_dict,
                gen_data=gen_data_unnorm_dict,
                input_data=input_data_unnorm_dict,
                target_data_norm=target_data_dict,
                gen_data_norm=gen_data_dict,
                input_data_norm=input_data_dict,
            )
        )

    @torch.no_grad()
    def get_logs(self, label: str = "train") -> Metrics:
        if self._metric_engine is None:
            raise ValueError("No batches have been recorded.")
        return {f"{label}/{k}": v for k, v in self._metric_engine.get_logs().items()}
