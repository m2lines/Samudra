import torch

from ocean_emulators.aggregator import ValidateAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource
from ocean_emulators.utils.output import TrainBatchOutput, ValBatchOutput
from ocean_emulators.utils.wandb import Metrics


def val_batch_of(h: int, w: int) -> ValBatchOutput:
    """Create a dummy Validation Batch loss / data from a DataSource."""
    # If we can consume a DataSource, then TensorMap has to be initialized.
    tm = TensorMap.get_instance()
    n_prog = len(tm.prognostic_var_names)
    n_boundary = len(tm.boundary_var_names)

    loss_per_channel = torch.ones(n_prog) * 1.5
    loss = loss_per_channel.sum()

    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(1, n_prog + n_boundary, 1, h, w),
        target_data=torch.randn(1, n_prog, 1, h, w),
        gen_data=torch.randn(1, n_prog, 1, h, w),
        ctx=GridContext(
            label_mask=torch.ones(1, n_prog, 1, h, w),
            input_resolution_cpu=(
                torch.linspace(-90, 90, steps=h),
                torch.linspace(-180, 180, steps=w),
            ),
        ),
    )
    return batch


def val_to_train_batch(val_batch: ValBatchOutput) -> TrainBatchOutput:
    """Ensures the TrainBatch loss is the same as the ValBatch loss."""
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


def test_val_aggregator__no_op__is_same_as_train_aggregator(dummy_src: DataSource):
    val_batch = val_batch_of(*dummy_src.grid_size)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator({}, hist=0, num_prognostic_channels=num_prog_channels)
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator()
    train_agg.record_batch(val_to_train_batch(val_batch))
    train_agg.record_batch(val_to_train_batch(val_batch))

    val_logs = val_agg.get_logs(label="test")
    train_logs = train_agg.get_logs(label="test")

    assert val_logs == train_logs


def test_train_val_aggregator__with_fake_subagg__is_added_to_logs(
    dummy_src: DataSource,
):
    val_batch = val_batch_of(*dummy_src.grid_size)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()}, hist=0, num_prognostic_channels=num_prog_channels
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
