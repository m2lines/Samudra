# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import samudra.identity as identity_module
from samudra.config import SamudraMultiConfig
from samudra.datasets import TrainData
from samudra.identity import (
    IdentityConfig,
    _fixed_batches,
    _identity_routes,
    _masked_physical_resampler_reference,
    _processor_depth,
    evaluate_identity_routes,
    set_identity_target,
    train_identity,
)
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext


def make_train_data(
    prognostic_channels=3,
    label_channels=3,
    steps=1,
    input_grid=(4, 8),
    output_grid=(4, 8),
):
    context = GridContext(
        label_mask=torch.ones(label_channels, *output_grid, dtype=torch.bool),
        input_resolution_cpu=(
            torch.arange(input_grid[0]),
            torch.arange(input_grid[1]),
        ),
        output_resolution_cpu=(
            torch.arange(output_grid[0]),
            torch.arange(output_grid[1]),
        ),
    )
    data = TrainData(prognostic_channels, 2, context)
    for _ in range(steps):
        data.append(
            torch.randn(2, prognostic_channels, *input_grid),
            torch.randn(2, 2, *input_grid),
            torch.randn(2, label_channels, *output_grid),
        )
    return data


def test_set_identity_target_reuses_prognostic_input():
    data = make_train_data()
    prognostic, boundary, _ = data[0]

    target = set_identity_target(data)

    new_prognostic, new_boundary, new_target = data[0]
    assert target is prognostic
    assert new_prognostic is prognostic
    assert new_boundary is boundary
    assert new_target is prognostic


def test_masked_resampler_uses_output_climatology_without_source_support():
    input_physical = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[7.0, 8.0], [9.0, 10.0]]]]
    )
    input_wet = torch.tensor(
        [[[False, False], [False, False]], [[True, False], [False, False]]]
    )

    output = _masked_physical_resampler_reference(
        input_physical,
        input_wet,
        (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 180.0])),
        (torch.tensor([0.5]), torch.tensor([90.0])),
        torch.tensor([3.0, 5.0]),
    )

    torch.testing.assert_close(output, torch.tensor([[[[3.0]], [[7.0]]]]))


def test_set_identity_target_rejects_multiple_steps():
    with pytest.raises(ValueError, match="exactly one model step"):
        set_identity_target(make_train_data(steps=2))


def test_set_identity_target_rejects_shape_change():
    with pytest.raises(ValueError, match="matching prognostic and output shapes"):
        set_identity_target(make_train_data(prognostic_channels=3, label_channels=2))


class _FixedLoader:
    def __init__(self, batches):
        self.batches = batches
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.batches)


class _FixedTrainer:
    def __init__(self, batches):
        self.val_loader = _FixedLoader(batches)


def test_fixed_batches_selects_disjoint_exact_ranges():
    batches = [make_train_data() for _ in range(4)]
    trainer = _FixedTrainer(batches)

    selected = list(
        _fixed_batches(cast(Any, trainer), requested_samples=4, sample_offset=4)
    )

    assert trainer.val_loader.epoch == 0
    assert selected == batches[2:4]


def test_fixed_batches_rejects_unaligned_range():
    trainer = _FixedTrainer([make_train_data() for _ in range(2)])

    with pytest.raises(ValueError, match="offsets must align"):
        list(_fixed_batches(cast(Any, trainer), requested_samples=2, sample_offset=1))


def test_fixed_batches_balance_cross_resolution_routes():
    route_a = [make_train_data() for _ in range(3)]
    route_b = [
        make_train_data(input_grid=(4, 8), output_grid=(8, 16)) for _ in range(3)
    ]
    trainer = _FixedTrainer([route_a[0], route_b[0], route_a[1], route_b[1]])

    selected = list(
        _fixed_batches(
            cast(Any, trainer),
            requested_samples=4,
            sample_offset=4,
            route_count=2,
        )
    )

    assert selected == [route_a[1], route_b[1]]


def test_identity_routes_expand_mix_schedule_in_stable_order():
    sources = [
        SimpleNamespace(grid_size=(180, 360)),
        SimpleNamespace(grid_size=(360, 720)),
    ]
    trainer = SimpleNamespace(
        data_container=SimpleNamespace(sources=sources), train_schedule="mix"
    )

    assert _identity_routes(cast(Any, trainer)) == [
        ("180x360_to_180x360", (180, 360)),
        ("180x360_to_360x720", (360, 720)),
        ("360x720_to_180x360", (180, 360)),
        ("360x720_to_360x720", (360, 720)),
    ]


def test_route_evaluation_forms_equal_route_means(monkeypatch):
    sources = [
        SimpleNamespace(grid_size=(180, 360)),
        SimpleNamespace(grid_size=(360, 720)),
    ]
    trainer = SimpleNamespace(
        data_container=SimpleNamespace(sources=sources), train_schedule="mix"
    )

    def fake_evaluate(
        trainer,
        requested_samples,
        patch_size,
        *,
        sample_offset,
        prefix,
        target_time_mode,
        route_filter,
    ):
        del trainer, patch_size, sample_offset, target_time_mode
        route_index = {
            "180x360_to_180x360": 0,
            "180x360_to_360x720": 1,
            "360x720_to_180x360": 1,
            "360x720_to_360x720": 2,
        }[route_filter]
        return {
            f"{prefix}/mean/mse": float(route_index),
            f"{prefix}/actual_samples": float(requested_samples),
        }, {"target_zonal_power": torch.tensor([route_index])}

    monkeypatch.setattr(identity_module, "evaluate_identity", fake_evaluate)

    logs, spectra = evaluate_identity_routes(
        cast(Any, trainer),
        requested_samples=32,
        patch_extent=(1.0, 1.0),
        sample_offset=32,
        prefix="identity/heldout",
        target_time_mode="current",
    )

    assert logs["identity/heldout/mean/mse"] == pytest.approx(1.0)
    assert logs["identity/heldout/actual_samples"] == 32
    assert len(spectra) == 4


def test_identity_config_uses_disjoint_sample_ranges():
    fields = IdentityConfig.model_fields

    assert fields["identity_train_samples"].default == 32
    assert fields["identity_eval_samples"].default == 32
    assert fields["identity_train_offset"].default == 0
    assert fields["identity_eval_offset"].default == 32
    assert fields["identity_eval_only"].default is False
    assert fields["identity_eval_processor_depths"].default is None


def test_identity_eval_only_requires_explicit_finetune_checkpoint(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_1deg_decoder_wide.yaml"
    )
    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    ).model_copy(update={"identity_eval_only": True, "epochs": 1})

    with pytest.raises(ValueError, match="explicit model checkpoint"):
        train_identity(config)


def test_identity_eval_only_is_one_pass(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_1deg_decoder_wide.yaml"
    )
    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    ).model_copy(
        update={
            "identity_eval_only": True,
            "finetune": True,
            "resume_ckpt_path": "checkpoint.pt",
            "epochs": 2,
        }
    )

    with pytest.raises(ValueError, match="epochs: 1"):
        train_identity(config)


def test_processor_depth_context_restores_configured_depth():
    model = SamudraMulti.__new__(SamudraMulti)
    torch.nn.Module.__init__(model)
    model.processor_iterations = 1
    trainer = SimpleNamespace(model=model)

    with _processor_depth(cast(Any, trainer), 4):
        assert model.processor_iterations == 4

    assert model.processor_iterations == 1


def test_processor_depth_context_rejects_negative_depth():
    model = SamudraMulti.__new__(SamudraMulti)
    torch.nn.Module.__init__(model)
    model.processor_iterations = 1
    trainer = SimpleNamespace(model=model)

    with pytest.raises(ValueError, match="non-negative"):
        with _processor_depth(cast(Any, trainer), -1):
            pass


def test_wide_decoder_identity_config_exposes_true_attention_width(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_1deg_decoder_wide.yaml"
    )

    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert isinstance(config.model, SamudraMultiConfig)
    assert config.identity_train_samples == 32
    assert config.identity_eval_samples == 32
    assert config.identity_eval_frequency == 2
    assert config.model.bypass_processor
    assert config.model.decoder.perceiver.cross_heads == 2
    assert config.model.decoder.perceiver.cross_dim_head == 64


def test_cross_resolution_identity_config_uses_current_paired_targets(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_cross_1_halfdeg.yaml"
    )

    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert config.target_time_mode == "current"
    assert config.experiment.train_schedule == "mix"
    assert len(config.data.sources) == 2
    assert config.identity_train_samples == 32
    assert config.identity_eval_samples == 32


def test_all_resolution_identity_config_balances_nine_routes(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_cross_1_half_quarter.yaml"
    )

    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert config.target_time_mode == "current"
    assert config.experiment.train_schedule == "mix"
    assert len(config.data.sources) == 3
    assert config.identity_train_samples == 36
    assert config.identity_eval_samples == 36
    assert config.epochs * config.identity_train_samples == 2880
    assert isinstance(config.model, SamudraMultiConfig)
    assert config.model.encoder.native_projection
    assert not config.model.encoder.canonical_resampling
    assert config.model.embedding_dim == 160


def test_common_stats_identity_config_keeps_native_grids_in_one_basis(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_cross_1_halfdeg_common_stats.yaml"
    )

    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert len(config.data.sources) == 2
    assert config.data.sources[0].data_location != config.data.sources[1].data_location
    assert (
        config.data.sources[0].data_means_location
        == config.data.sources[1].data_means_location
    )
    assert (
        config.data.sources[0].data_stds_location
        == config.data.sources[1].data_stds_location
    )
    assert isinstance(config.model, SamudraMultiConfig)
    assert config.model.encoder.native_projection
    assert not config.model.encoder.canonical_resampling


def test_processor_identity_config_evaluates_zero_to_four_calls(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "identity_cross_1_halfdeg_processor.yaml"
    )

    config = IdentityConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert config.identity_eval_processor_depths == [0, 1, 2, 4]
    assert isinstance(config.model, SamudraMultiConfig)
    assert not config.model.bypass_processor
    assert config.model.processor_iterations == 1
    assert config.model.encoder.native_projection
    assert not config.model.encoder.canonical_resampling
    assert config.model.encoder.geometry_mode == "sidecar"
    assert config.model.embedding_dim == 160
    assert config.model.zero_depth_reconstruction_weight == pytest.approx(0.05)
