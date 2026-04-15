import pytest
import torch

from ocean_emulators.aggregator import Aggregator, ValidateAggregator
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.map import MapAggregator
from ocean_emulators.aggregator.validate.snapshot import SnapshotAggregator
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.output import ValBatchOutput
from ocean_emulators.utils.wandb import Metrics


def val_batch_of(
    h: int,
    w: int,
    *,
    tensor_map: TensorMap,
    hist: int = 0,
    batch_size: int = 1,
) -> ValBatchOutput:
    """Create a dummy Validation Batch loss / data from a DataSource."""
    n_prog_base = len(tensor_map.prognostic_var_names)
    n_boundary_base = len(tensor_map.boundary_var_names)
    n_prog = (hist + 1) * n_prog_base
    n_boundary = (hist + 1) * n_boundary_base

    loss_per_channel = torch.ones(n_prog) * 1.5
    loss = loss_per_channel.sum()

    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(batch_size, n_prog + n_boundary, h, w),
        target_data=torch.randn(batch_size, n_prog, h, w),
        gen_data=torch.randn(batch_size, n_prog, h, w),
        ctx=GridContext(
            label_mask=torch.ones(n_prog, h, w),
            input_resolution_cpu=(
                torch.linspace(-90, 90, steps=h),
                torch.linspace(-180, 180, steps=w),
            ),
            output_resolution_cpu=(
                torch.linspace(-90, 90, steps=h),
                torch.linspace(-180, 180, steps=w),
            ),
        ),
    )
    return batch


def tensor_map_for(src: DataSource) -> TensorMap:
    return TensorMap("thetao_1", "hfds", dataset_spec=src.dataset_spec)


def normalize_for(src: DataSource, tensor_map: TensorMap) -> Normalize:
    return Normalize(
        src,
        tensor_map.prognostic_var_names,
        tensor_map.boundary_var_names,
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
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    val_batch = val_batch_of(*dummy_src.grid_size, tensor_map=tensor_map)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {},
        hist=0,
        num_prognostic_channels=num_prog_channels,
        tensor_map=tensor_map,
        normalize=normalize,
    )
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator(tensor_map)
    train_agg.record_batch(val_batch)
    train_agg.record_batch(val_batch)

    val_logs = val_agg.get_logs(label="test")
    train_logs = train_agg.get_logs(label="test")

    assert val_logs == train_logs


def test_train_val_aggregator__with_fake_subagg__is_added_to_logs(
    dummy_src: DataSource,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    val_batch = val_batch_of(*dummy_src.grid_size, tensor_map=tensor_map)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()},
        hist=0,
        num_prognostic_channels=num_prog_channels,
        tensor_map=tensor_map,
        normalize=normalize,
    )
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator(tensor_map)
    train_agg.record_batch(val_batch)
    train_agg.record_batch(val_batch)

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


def test_val_aggregator__hist_gt_0__does_not_require_wetmask_target_shape_match(
    dummy_src: DataSource,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    val_batch = val_batch_of(
        *dummy_src.grid_size, tensor_map=tensor_map, hist=1, batch_size=2
    )
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()},
        hist=1,
        num_prognostic_channels=num_prog_channels,
        tensor_map=tensor_map,
        normalize=normalize,
    )
    val_agg.record_validation_batch(val_batch)
    val_logs = val_agg.get_logs(label="test")
    assert val_logs["test/fake/num_recordings"] == 1.0


def test_validation_aggregator__reduced_only__omits_image_logs(
    dummy_src: DataSource,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    val_batch = val_batch_of(*dummy_src.grid_size, tensor_map=tensor_map)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = Aggregator.get_validation_aggregator(
        dummy_src.metadata,
        hist=0,
        area_weights=dummy_src.spherical_area_weights,
        num_prognostic_channels=num_prog_channels,
        tensor_map=tensor_map,
        normalize=normalize,
        include_image_aggregators=False,
    )

    val_agg.record_validation_batch(val_batch)
    val_logs = val_agg.get_logs(label="val")

    assert any(key.startswith("val/reduced/") for key in val_logs)
    assert not any("/snapshot/" in key for key in val_logs)
    assert not any("/mean_map/" in key for key in val_logs)


def test_snapshot_aggregator__non_main_rank__skips_plot_rendering(
    dummy_src: DataSource, monkeypatch: pytest.MonkeyPatch
):
    val_batch = val_batch_of(*dummy_src.grid_size)
    aggregator = SnapshotAggregator(dummy_src.metadata, hist=0)

    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.snapshot.is_main_process",
        lambda: False,
    )

    aggregator.record_batch(
        loss=val_batch.loss,
        target_data={"foo": val_batch.target_data.unsqueeze(1)[:, :, 0]},
        gen_data={"foo": val_batch.gen_data.unsqueeze(1)[:, :, 0]},
        input_data={"foo": val_batch.input_data.unsqueeze(1)[:, :, 0]},
        target_data_norm={},
        gen_data_norm={},
        input_data_norm={},
    )

    assert aggregator.get_logs("snapshot") == {}


def test_map_aggregator__non_main_rank__still_reduces_but_skips_plot_rendering(
    dummy_src: DataSource, monkeypatch: pytest.MonkeyPatch
):
    aggregator = MapAggregator(dummy_src.metadata, hist=0)
    reduce_calls: list[torch.Tensor] = []

    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.map.is_main_process",
        lambda: False,
    )

    def fake_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        reduce_calls.append(tensor.clone())
        return tensor

    monkeypatch.setattr(
        "ocean_emulators.aggregator.validate.map.all_reduce_mean",
        fake_all_reduce_mean,
    )

    data = torch.ones(2, 1, *dummy_src.grid_size)
    aggregator.record_batch(
        loss=torch.tensor(1.0),
        target_data={"foo": data},
        gen_data={"foo": data},
        input_data={},
        target_data_norm={},
        gen_data_norm={},
        input_data_norm={},
    )

    assert aggregator.get_logs("mean_map") == {}
    assert len(reduce_calls) == 2
