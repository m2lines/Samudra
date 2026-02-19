import pytest
import torch

from ocean_emulators.aggregator import ValidateAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.output import TrainBatchOutput, ValBatchOutput


@pytest.fixture
def val_batch() -> ValBatchOutput:
    TensorMap.init_instance("thermo_dynamic_5", "hfds")
    n_channels = (
        21  # TensorMap "thermo_dynamic_5" has 21 channels (4 vars x 5 depths + zos)
    )
    loss_per_channel = torch.ones(n_channels) * 1.5
    loss = loss_per_channel.sum()
    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(1, 2, 2, 4, 8),  # prog=1, boundary=1
        target_data=torch.randn(1, 1, 2, 4, 8),
        gen_data=torch.randn(1, 1, 2, 4, 8),
        ctx=GridContext(
            label_mask=torch.ones(1, 1, 2, 4, 8),
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
