import torch

from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.data import Normalize, get_aggregator_dicts
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict


class ValidateAggregator(TrainAggregator):
    """Aggregates Validation Statistics."""

    def __init__(
        self,
        aggregators: dict[str, ValidateSubAggregator],
        hist: int,
        num_prognostic_channels: int,
    ):
        super().__init__()
        self._aggregators = aggregators
        self.normalize = Normalize.get_instance()
        self.hist = hist
        self.num_prognostic_channels = num_prognostic_channels

    # TODO(jder): we could remove this by moving from inheritance
    # to composition with the TrainAggregator functionality.
    def record_batch(self, batch):
        raise NotImplementedError(
            "Call record_validation_batch instead of record_batch"
        )

    @torch.no_grad()
    def record_validation_batch(self, batch: ValBatchOutput):
        super().record_batch(batch)  # Record losses

        # If there are no log aggregators, omit doing any extra work.
        if not self._aggregators:
            return

        # Translate the GridContext mask by removing history.
        wet = batch.ctx.label_mask[: (self.hist + 1)]

        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        assert batch.target_data.shape[1] == self.num_prognostic_channels
        target_data_dict, target_data_unnorm_dict = get_aggregator_dicts(
            batch.target_data,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )

        gen_data_dict, gen_data_unnorm_dict = get_aggregator_dicts(
            batch.gen_data,
            wet=wet,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=self.num_prognostic_channels,
            hist=self.hist,
        )
        input_data_dict, input_data_unnorm_dict = get_aggregator_dicts(
            batch.input_data,
            wet=wet,
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
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v

        return logs
