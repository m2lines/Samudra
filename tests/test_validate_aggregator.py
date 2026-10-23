# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.aggregator import Aggregator, ValidateAggregator
from samudra.aggregator.train import TrainAggregator
from samudra.aggregator.validate.map import MapAggregator
from samudra.aggregator.validate.snapshot import SnapshotAggregator
from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.constants import DataLayout
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import BatchPreprocessor, CanonicalSource
from samudra.utils.output import ValBatchOutput
from samudra.utils.wandb import Metrics


def val_batch_of(
    h: int,
    w: int,
    *,
    data_layout: DataLayout,
    input_steps: int = 1,
    output_steps: int = 1,
    batch_size: int = 1,
) -> ValBatchOutput:
    """Create a dummy Validation Batch loss / data from a CanonicalSource."""
    n_prog_base = len(data_layout.prognostic_var_names)
    n_boundary_base = len(data_layout.boundary_var_names)
    n_prog_in = input_steps * n_prog_base
    n_boundary = input_steps * n_boundary_base
    n_prog_out = output_steps * n_prog_base

    loss_per_channel = torch.ones(n_prog_out) * 1.5
    loss = loss_per_channel.sum()

    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(batch_size, n_prog_in + n_boundary, h, w),
        target_data=torch.randn(batch_size, n_prog_out, h, w),
        gen_data=torch.randn(batch_size, n_prog_out, h, w),
        ctx=BatchGrid(
            label_mask=torch.ones(n_prog_out, h, w),
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


def data_layout_for(source: CanonicalSource) -> DataLayout:
    return source.data_layout


def preprocessor_for(
    source: CanonicalSource, data_layout: DataLayout
) -> BatchPreprocessor:
    return BatchPreprocessor(
        source,
        data_layout.prognostic_var_names,
        data_layout.boundary_var_names,
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


def test_val_aggregator__no_op__is_same_as_train_aggregator(
    dummy_source: CanonicalSource,
):
    data_layout = data_layout_for(dummy_source)
    preprocessor = preprocessor_for(dummy_source, data_layout)
    val_batch = val_batch_of(*dummy_source.grid_size, data_layout=data_layout)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {},
        input_steps=1,
        output_steps=1,
        num_prognostic_channels=num_prog_channels,
        data_layout=data_layout,
        preprocessor=preprocessor,
    )
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator(data_layout)
    train_agg.record_batch(val_batch)
    train_agg.record_batch(val_batch)

    val_logs = val_agg.get_logs(label="test")
    train_logs = train_agg.get_logs(label="test")

    assert val_logs == train_logs


def test_train_val_aggregator__with_fake_subagg__is_added_to_logs(
    dummy_source: CanonicalSource,
):
    data_layout = data_layout_for(dummy_source)
    preprocessor = preprocessor_for(dummy_source, data_layout)
    val_batch = val_batch_of(*dummy_source.grid_size, data_layout=data_layout)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()},
        input_steps=1,
        output_steps=1,
        num_prognostic_channels=num_prog_channels,
        data_layout=data_layout,
        preprocessor=preprocessor,
    )
    val_agg.record_validation_batch(val_batch)
    val_agg.record_validation_batch(val_batch)

    train_agg = TrainAggregator(data_layout)
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


def test_val_aggregator_supports_different_input_and_output_steps(
    dummy_source: CanonicalSource,
):
    data_layout = data_layout_for(dummy_source)
    preprocessor = preprocessor_for(dummy_source, data_layout)
    val_batch = val_batch_of(
        *dummy_source.grid_size,
        data_layout=data_layout,
        input_steps=2,
        output_steps=1,
        batch_size=2,
    )
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = ValidateAggregator(
        {"fake": FakeSubAggregator()},
        input_steps=2,
        output_steps=1,
        num_prognostic_channels=num_prog_channels,
        data_layout=data_layout,
        preprocessor=preprocessor,
    )
    val_agg.record_validation_batch(val_batch)
    val_logs = val_agg.get_logs(label="test")
    assert val_logs["test/fake/num_recordings"] == 1.0


def test_validation_aggregator__reduced_only__omits_image_logs(
    dummy_source: CanonicalSource,
):
    data_layout = data_layout_for(dummy_source)
    preprocessor = preprocessor_for(dummy_source, data_layout)
    val_batch = val_batch_of(*dummy_source.grid_size, data_layout=data_layout)
    num_prog_channels = val_batch.loss_per_channel.shape[0]
    val_agg = Aggregator.get_validation_aggregator(
        dummy_source.metadata,
        input_steps=1,
        output_steps=1,
        area_weights=dummy_source.area_weights,
        num_prognostic_channels=num_prog_channels,
        data_layout=data_layout,
        preprocessor=preprocessor,
        include_image_aggregators=False,
    )

    val_agg.record_validation_batch(val_batch)
    val_logs = val_agg.get_logs(label="val")

    assert any(key.startswith("val/reduced/") for key in val_logs)
    assert not any("/snapshot/" in key for key in val_logs)
    assert not any("/mean_map/" in key for key in val_logs)


def test_snapshot_aggregator__non_main_rank__skips_plot_rendering(
    dummy_source: CanonicalSource, monkeypatch: pytest.MonkeyPatch
):
    data_layout = data_layout_for(dummy_source)
    val_batch = val_batch_of(*dummy_source.grid_size, data_layout=data_layout)
    aggregator = SnapshotAggregator(dummy_source.metadata, input_steps=1)

    monkeypatch.setattr(
        "samudra.aggregator.validate.snapshot.is_main_process",
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
    dummy_source: CanonicalSource, monkeypatch: pytest.MonkeyPatch
):
    aggregator = MapAggregator(dummy_source.metadata, output_steps=1)
    reduce_calls: list[torch.Tensor] = []

    monkeypatch.setattr(
        "samudra.aggregator.validate.map.is_main_process",
        lambda: False,
    )

    def fake_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        reduce_calls.append(tensor.clone())
        return tensor

    monkeypatch.setattr(
        "samudra.aggregator.validate.map.all_reduce_mean",
        fake_all_reduce_mean,
    )

    data = torch.ones(2, 1, *dummy_source.grid_size)
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


@pytest.mark.parametrize("invalid_prediction", [False, True])
def test_normalized_one_step_rmse_weights_samples_and_channels(invalid_prediction):
    from samudra.aggregator.validate.reduced import MeanAggregator

    agg = MeanAggregator(torch.tensor([[1.0, 3.0], [1.0, 3.0]]), target_time=0)
    # One land cell and spatially varying errors distinguish area weighting
    # from uniform weighting; unequal batches distinguish sample averaging.
    target = torch.zeros(1, 1, 2, 2)
    target[..., 0, 0] = float("nan")
    prediction = torch.tensor([[[[float("nan"), 2.0], [4.0, 2.0]]]])
    if invalid_prediction:
        prediction[..., 1, 1] = float("nan")
        with pytest.raises(ValueError, match="Non-finite prediction"):
            agg.record_batch(
                target_data={"a": target},
                gen_data={"a": prediction},
                target_data_norm={"a": target},
                gen_data_norm={"a": prediction},
            )
        return
    for batch_size, scale in [(2, 1.0), (1, 2.0)]:
        agg.record_batch(
            target_data={"a": target.expand(batch_size, -1, -1, -1)},
            gen_data={"a": prediction.expand(batch_size, -1, -1, -1)},
            target_data_norm={
                name: target.expand(batch_size, -1, -1, -1) for name in ("a", "b")
            },
            gen_data_norm={
                name: prediction.expand(batch_size, -1, -1, -1) * scale * factor
                for name, factor in [("a", 1.0), ("b", 2.0)]
            },
        )
    score = agg.get_logs("reduced")["reduced/normalized_rmse/channel_mean"]
    expected = (40 / 7) ** 0.5 * 4 / 3 * 1.5
    assert score == pytest.approx(expected)
