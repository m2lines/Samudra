# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.aggregator import (
    Aggregator,
    MultiScaleValidateAggregator,
    ValidateAggregator,
)
from samudra.aggregator.train import RouteTrainAggregator, TrainAggregator
from samudra.aggregator.validate.map import MapAggregator
from samudra.aggregator.validate.snapshot import SnapshotAggregator
from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.constants import TensorMap
from samudra.utils.ctx import GridContext
from samudra.utils.data import CanonicalDataset, Normalize
from samudra.utils.output import ValBatchOutput
from samudra.utils.wandb import Metrics


def val_batch_of(
    h: int,
    w: int,
    *,
    tensor_map: TensorMap,
    hist: int = 0,
    batch_size: int = 1,
    input_grid: tuple[int, int] | None = None,
) -> ValBatchOutput:
    """Create a dummy Validation Batch loss / data from a CanonicalDataset."""
    n_prog_base = len(tensor_map.prognostic_var_names)
    n_boundary_base = len(tensor_map.boundary_var_names)
    n_prog = (hist + 1) * n_prog_base
    n_boundary = (hist + 1) * n_boundary_base

    loss_per_channel = torch.ones(n_prog) * 1.5
    loss = loss_per_channel.sum()

    input_h, input_w = input_grid or (h, w)
    batch = ValBatchOutput(
        loss=loss,
        loss_per_channel=loss_per_channel,
        input_data=torch.randn(batch_size, n_prog + n_boundary, input_h, input_w),
        target_data=torch.randn(batch_size, n_prog, h, w),
        gen_data=torch.randn(batch_size, n_prog, h, w),
        ctx=GridContext(
            label_mask=torch.ones(n_prog, h, w),
            input_resolution_cpu=(
                torch.linspace(-90, 90, steps=input_h),
                torch.arange(input_w) * (360.0 / input_w) - 180.0,
            ),
            output_resolution_cpu=(
                torch.linspace(-90, 90, steps=h),
                torch.arange(w) * (360.0 / w) - 180.0,
            ),
        ),
    )
    return batch


def tensor_map_for(src: CanonicalDataset) -> TensorMap:
    return TensorMap(dataset_spec=src.dataset_spec)


def normalize_for(src: CanonicalDataset, tensor_map: TensorMap) -> Normalize:
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


def test_val_aggregator__no_op__is_same_as_train_aggregator(
    dummy_src: CanonicalDataset,
):
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
    dummy_src: CanonicalDataset,
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
    dummy_src: CanonicalDataset,
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
    dummy_src: CanonicalDataset,
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


def test_validation_aggregator__records_plain_mse_and_persistence_baseline(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    batch = val_batch_of(*dummy_src.grid_size, tensor_map=tensor_map)
    batch.target_data.fill_(2.0)
    batch.gen_data.fill_(1.0)
    batch.input_data[:, : batch.target_data.shape[1]].fill_(0.0)
    aggregator = ValidateAggregator(
        {},
        hist=0,
        num_prognostic_channels=batch.target_data.shape[1],
        tensor_map=tensor_map,
        normalize=normalize,
        record_baselines=True,
    )

    aggregator.record_validation_batch(batch)
    logs = aggregator.get_logs(label="val")

    assert logs["val/unweighted_normalized_mse/mean/loss"] == pytest.approx(1.0)
    assert logs["val/persistence_normalized_mse/mean/loss"] == pytest.approx(4.0)


def test_validation_aggregator_resamples_cross_grid_persistence(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    batch = val_batch_of(4, 8, tensor_map=tensor_map)
    num_prog = batch.target_data.shape[1]
    num_boundary = batch.input_data.shape[1] - num_prog
    batch.input_data = torch.zeros(1, num_prog + num_boundary, 2, 4)
    batch.target_data.fill_(2.0)
    batch.gen_data.fill_(1.0)
    batch.ctx = GridContext(
        label_mask=torch.ones(num_prog, 4, 8, dtype=torch.bool),
        input_resolution_cpu=(
            torch.tensor([-45.0, 45.0]),
            torch.tensor([45.0, 135.0, 225.0, 315.0]),
        ),
        output_resolution_cpu=(
            torch.tensor([-67.5, -22.5, 22.5, 67.5]),
            torch.arange(8) * 45.0 + 22.5,
        ),
        input_mask=torch.ones(num_prog, 2, 4, dtype=torch.bool),
    )
    aggregator = ValidateAggregator(
        {"fake": FakeSubAggregator()},
        hist=0,
        num_prognostic_channels=num_prog,
        tensor_map=tensor_map,
        normalize=normalize,
        record_baselines=True,
    )

    aggregator.record_validation_batch(batch)
    logs = aggregator.get_logs(label="val")

    assert logs["val/unweighted_normalized_mse/mean/loss"] == pytest.approx(1.0)
    assert logs["val/persistence_normalized_mse/mean/loss"] == pytest.approx(4.0)
    assert logs["val/fake/num_recordings"] == 1.0


def test_multiscale_validation_aggregator__routes_and_prefixes_by_grid(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    small = ValidateAggregator(
        {},
        hist=0,
        num_prognostic_channels=len(tensor_map.prognostic_var_names),
        tensor_map=tensor_map,
        normalize=normalize,
    )
    large = ValidateAggregator(
        {},
        hist=0,
        num_prognostic_channels=len(tensor_map.prognostic_var_names),
        tensor_map=tensor_map,
        normalize=normalize,
    )
    aggregator = MultiScaleValidateAggregator(
        {
            (4, 8): ("4x8", small),
            (8, 16): ("8x16", large),
        }
    )

    aggregator.record_validation_batch(val_batch_of(4, 8, tensor_map=tensor_map))
    aggregator.record_validation_batch(val_batch_of(8, 16, tensor_map=tensor_map))
    logs = aggregator.get_logs(label="val")

    assert "val/resolution/4x8/mean/loss" in logs
    assert "val/resolution/8x16/mean/loss" in logs
    assert "val/route/4x8_to_4x8/mean/loss" in logs
    assert "val/route/8x16_to_8x16/mean/loss" in logs
    assert "val/mean/loss" in logs


def test_route_train_aggregator__separates_shared_output_grid(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    aggregator = RouteTrainAggregator(tensor_map)
    same_grid = val_batch_of(4, 8, tensor_map=tensor_map)
    cross_grid = val_batch_of(4, 8, tensor_map=tensor_map, input_grid=(8, 16))
    same_grid.loss.fill_(1.0)
    same_grid.loss_per_channel.fill_(1.0)
    cross_grid.loss.fill_(3.0)
    cross_grid.loss_per_channel.fill_(3.0)

    aggregator.record_batch(same_grid, same_grid.ctx)
    aggregator.record_batch(cross_grid, cross_grid.ctx)
    logs = aggregator.get_logs("val/physical_lead_1")

    assert logs["val/physical_lead_1/mean/loss"] == pytest.approx(2.0)
    assert logs["val/physical_lead_1/route/4x8_to_4x8/mean/loss"] == pytest.approx(1.0)
    assert logs["val/physical_lead_1/route/8x16_to_4x8/mean/loss"] == pytest.approx(3.0)


def test_multiscale_validation_aggregator__does_not_alias_overall_and_scale_state(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    scale = ValidateAggregator(
        {},
        hist=0,
        num_prognostic_channels=len(tensor_map.prognostic_var_names),
        tensor_map=tensor_map,
        normalize=normalize,
    )
    aggregator = MultiScaleValidateAggregator({(4, 8): ("4x8", scale)})
    first = val_batch_of(4, 8, tensor_map=tensor_map)
    first.loss.fill_(1.0)
    first.loss_per_channel.fill_(1.0)
    second = val_batch_of(4, 8, tensor_map=tensor_map)
    second.loss.fill_(3.0)
    second.loss_per_channel.fill_(3.0)

    aggregator.record_validation_batch(first)
    aggregator.record_validation_batch(second)
    logs = aggregator.get_logs(label="val")

    assert logs["val/mean/loss"] == pytest.approx(2.0)
    assert logs["val/resolution/4x8/mean/loss"] == pytest.approx(2.0)


def test_multiscale_validation_aggregator__uses_full_route_aggregators(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)

    def new_aggregator() -> ValidateAggregator:
        return ValidateAggregator(
            {},
            hist=0,
            num_prognostic_channels=len(tensor_map.prognostic_var_names),
            tensor_map=tensor_map,
            normalize=normalize,
            record_baselines=True,
        )

    aggregator = MultiScaleValidateAggregator(
        {(4, 8): ("4x8", new_aggregator())},
        {
            ((4, 8), (4, 8)): ("4x8_to_4x8", new_aggregator()),
            ((8, 16), (4, 8)): ("8x16_to_4x8", new_aggregator()),
        },
    )
    aggregator.record_validation_batch(val_batch_of(4, 8, tensor_map=tensor_map))
    aggregator.record_validation_batch(
        val_batch_of(4, 8, tensor_map=tensor_map, input_grid=(8, 16))
    )
    logs = aggregator.get_logs("val")

    assert "val/route/4x8_to_4x8/mean/loss" in logs
    assert "val/route/8x16_to_4x8/mean/loss" in logs
    assert "val/route/4x8_to_4x8/persistence_normalized_mse/mean/loss" in logs
    assert "val/route/8x16_to_4x8/persistence_normalized_mse/mean/loss" in logs


def test_multiscale_validation_aggregator__rejects_unknown_grid(
    dummy_src: CanonicalDataset,
):
    tensor_map = tensor_map_for(dummy_src)
    normalize = normalize_for(dummy_src, tensor_map)
    aggregator = MultiScaleValidateAggregator(
        {
            (4, 8): (
                "4x8",
                ValidateAggregator(
                    {},
                    hist=0,
                    num_prognostic_channels=len(tensor_map.prognostic_var_names),
                    tensor_map=tensor_map,
                    normalize=normalize,
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="unregistered output grid"):
        aggregator.record_validation_batch(val_batch_of(8, 16, tensor_map=tensor_map))


def test_snapshot_aggregator__non_main_rank__skips_plot_rendering(
    dummy_src: CanonicalDataset, monkeypatch: pytest.MonkeyPatch
):
    tensor_map = tensor_map_for(dummy_src)
    val_batch = val_batch_of(*dummy_src.grid_size, tensor_map=tensor_map)
    aggregator = SnapshotAggregator(dummy_src.metadata, hist=0)

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
    dummy_src: CanonicalDataset, monkeypatch: pytest.MonkeyPatch
):
    aggregator = MapAggregator(dummy_src.metadata, hist=0)
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
