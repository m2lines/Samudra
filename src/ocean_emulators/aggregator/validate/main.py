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
        num_input_states: int,
        num_target_states: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        target_prognostic_channels: int,
    ):
        super().__init__()

        val_aggregators: dict[str, ValidateSubAggregator] = {
            "snapshot": SnapshotAggregator(
                metadata, input_time_index=num_input_states - 1
            ),
            "mean_map": MapAggregator(
                metadata, target_time_index=num_target_states - 1
            ),
            "reduced": MeanAggregator(area_weights, target_time=num_target_states - 1),
        }
        self._aggregators = val_aggregators
        self.normalize = Normalize.get_instance()
        self.num_target_states = num_target_states
        self.num_input_states = num_input_states
        self.num_target_channels = target_prognostic_channels
        if target_prognostic_channels % num_target_states != 0:
            raise ValueError(
                "Target prognostic channels must be divisible by the number "
                "of target states."
            )
        self.num_prognostic_vars = target_prognostic_channels // num_target_states
        self.num_input_channels = self.num_prognostic_vars * num_input_states
        self.wet = wet

    # TODO(jder): we could remove this by moving from inheritance
    # to composition with the TrainAggregator functionality.
    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
        super().record_batch(batch)  # Record losses

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        assert batch.target_data.shape[1] == self.num_target_channels
        target_data_dict, target_data_unnorm_dict = get_aggregator_dicts(
            batch.target_data,
            wet=self.wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_target_channels,
            num_states=self.num_target_states,
        )

        gen_data_dict, gen_data_unnorm_dict = get_aggregator_dicts(
            batch.gen_data,
            wet=self.wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_target_channels,
            num_states=self.num_target_states,
        )
        input_data_dict, input_data_unnorm_dict = get_aggregator_dicts(
            batch.input_data,
            wet=self.wet,
            long_rollout=False,
            input_type="input",
            num_prognostic_channels=self.num_input_channels,
            num_states=self.num_input_states,
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
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v

        return logs
