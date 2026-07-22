# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from samudra.config import SamudraMultiConfig
from samudra.datasets import TrainData
from samudra.identity import (
    IdentityConfig,
    _fixed_batches,
    _processor_depth,
    set_identity_target,
)
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext


def make_train_data(prognostic_channels=3, label_channels=3, steps=1):
    context = GridContext(
        label_mask=torch.ones(label_channels, 4, 8, dtype=torch.bool),
        input_resolution_cpu=(torch.arange(4), torch.arange(8)),
        output_resolution_cpu=(torch.arange(4), torch.arange(8)),
    )
    data = TrainData(prognostic_channels, 2, context)
    for _ in range(steps):
        data.append(
            torch.randn(2, prognostic_channels, 4, 8),
            torch.randn(2, 2, 4, 8),
            torch.randn(2, label_channels, 4, 8),
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


def test_identity_config_uses_disjoint_sample_ranges():
    fields = IdentityConfig.model_fields

    assert fields["identity_train_samples"].default == 32
    assert fields["identity_eval_samples"].default == 32
    assert fields["identity_train_offset"].default == 0
    assert fields["identity_eval_offset"].default == 32
    assert fields["identity_eval_processor_depths"].default is None


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
