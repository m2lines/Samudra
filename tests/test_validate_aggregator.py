import pytest
import torch

from ocean_emulators.aggregator import ValidateAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource
from ocean_emulators.utils.output import TrainBatchOutput, ValBatchOutput
from ocean_emulators.utils.wandb import Metrics


@pytest.fixture
def val_batch() -> ValBatchOutput:
    n_channels = 1
    loss_per_channel = torch.ones(n_channels) * 1.5
    loss = loss_per_channel.sum()
    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(1, 2, 1, 4, 8),  # prog=1, boundary=1
        target_data=torch.randn(1, 1, 1, 4, 8),
        gen_data=torch.randn(1, 1, 1, 4, 8),
        ctx=GridContext(
            label_mask=torch.ones(1, 1, 1, 4, 8),
            input_resolution_cpu=(
                torch.linspace(-90, 90, steps=4),
                torch.linspace(-180, 180, steps=8),
            ),
        ),
    )
    return batch


def val_to_train_batch(val_batch: ValBatchOutput) -> TrainBatchOutput:
    return TrainBatchOutput(
        loss=val_batch.loss,
        loss_per_channel=val_batch.loss_per_channel,
    )


class FakeSubAggregator(ValidateSubAggregator):
    def __init__(self):
        self.num_recordings = 0

    def get_logs(self, label: str) -> Metrics:
        return {f"{label}/num_recordings": float(self.num_recordings)}

    def record_batch(
        self,
        *,
        loss: torch.Tensor,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ):
        self.num_recordings += 1


def test_val_aggregator__no_op__is_same_as_train_aggregator(val_batch: ValBatchOutput):
    val_agg = ValidateAggregator({}, hist=1, num_prognostic_channels=1)
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator()
    train_agg.record_batch(val_to_train_batch(val_batch))
    train_agg.record_batch(val_to_train_batch(val_batch))

    val_logs = val_agg.get_logs(label="test")
    train_logs = train_agg.get_logs(label="test")

    assert val_logs == train_logs


def test_train_val_aggregator__with_fake_subagg__is_added_to_logs(
    val_batch: ValBatchOutput, dummy_src: DataSource
):
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()}, hist=1, num_prognostic_channels=1
    )
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator()
    train_agg.record_batch(val_to_train_batch(val_batch))
    train_agg.record_batch(val_to_train_batch(val_batch))

    val_logs = val_agg.get_logs(label="test")
    train_logs = train_agg.get_logs(label="test")

    assert set(train_logs).difference(set(val_logs)) == set(), (
        "The validation logs should include all the train logs"
    )
    assert "test/fake/num_recordings" in val_logs, (
        "The val logs should include logs from the sub aggregator."
    )
    assert val_logs["test/fake/num_recordings"] == 2.0, (
        "All the sub aggregations should be reflected in the logs."
    )
