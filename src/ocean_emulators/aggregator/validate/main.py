import torch

from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.map import MapAggregator
from ocean_emulators.aggregator.validate.reduced import MeanAggregator
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.data import Normalize, get_aggregator_dicts
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        metadata: dict[str, dict[str, str]],
        hist: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
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
        self.hist = hist
        self.num_prognostic_channels = num_prognostic_channels
        self.wet = wet

    def record_batch(self, batch):
        """Override to handle both TrainBatchOutput and ValBatchOutput."""
        if isinstance(batch, ValBatchOutput):
            # If it's a ValBatchOutput, use the validation-specific method
            self.record_validation_batch(batch)
        else:
            # For TrainBatchOutput, just record the losses
            super().record_batch(batch)

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
        super().record_batch(batch)  # Record losses

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
        # Only get logs from sub-aggregators if they have been initialized with data
        for agg_label, aggregator in self._aggregators.items():
            # Check if the aggregator has been initialized with data
            if hasattr(aggregator, "_gen_data") and aggregator._gen_data is not None:
                for k, v in aggregator.get_logs(label=agg_label).items():
                    logs[f"{label}/{k}"] = v

        return logs

    def get_sub_aggregator(self, name: str) -> ValidateSubAggregator | None:
        """Get a specific sub-aggregator by name. For testing purposes."""
        return self._aggregators.get(name)
