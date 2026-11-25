# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Spike-diagnosis tooling: a read-only probe, a loss guard, and exact replay forks."""

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch

from samudra.observations.config import LossGuardConfig, ObservationTrainConfig
from samudra.observations.pipeline import train
from samudra.observations.probes import LossGuard
from samudra.observations.snapshot import LLC_STATE_NAMES
from samudra.observations.snapshot_data import SnapshotSources, snapshot_manifest
from samudra.observations.state import TrainingState
from samudra.utils.train import CheckpointPaths
from tests.test_snapshot_runtime import archive

FORK = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "experiments/llc-obs-spike-diagnosis/fork_replay.py"
    )
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configure(base, **changes):
    """Two examples per update, so per-example records cover both slots."""
    values = {
        **base.model_dump(),
        "gradient_accumulation_steps": 2,
        "fresh_examples": 2,
        **changes,
    }
    return ObservationTrainConfig.model_validate(values)


def run(cfg, output, *, stop_after=None, restore=None, patch=None):
    torch.manual_seed(73)
    sources = SnapshotSources(cfg, torch.device("cpu"))
    model = cfg.model.build(LLC_STATE_NAMES)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas
    )
    state = TrainingState(model, optimizer, sources.names, config=cfg)
    manifest = snapshot_manifest(cfg)
    if restore is not None:
        state.restore(restore, manifest)
    output.mkdir(exist_ok=True)
    train(
        cfg,
        state,
        sources,
        output,
        manifest,
        lambda _: {cfg.selection_metric: 1.0},
        lambda _: None,
        stop_after=stop_after,
    )
    return state


def assert_same_weights(a, b):
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)
    assert a.ema == b.ema


def test_probe_records_without_changing_training(tmp_path):
    plain_cfg = configure(archive(tmp_path))
    probed_cfg = ObservationTrainConfig.model_validate(
        {**plain_cfg.model_dump(), "probe": {"weights_every": 2}}
    )
    plain = run(plain_cfg, tmp_path / "plain")
    probed = run(probed_cfg, tmp_path / "probed")
    assert_same_weights(plain, probed)
    records = [
        json.loads(line)
        for line in (tmp_path / "probed/probe.jsonl").read_text().splitlines()
    ]
    assert [r["updates"] for r in records] == [1, 2, 3, 4]
    assert [r["source"] for r in records] == ["llc", "obs", "llc", "obs"]
    for record in records:
        assert {"grad_norm/evolution.net.0", "grad_norm/initializer.net.0"} <= set(
            record
        )
        assert [e["index"] for e in record["examples"]] == [0, 1]
        assert "mean/diag/day_range/sst/3" in record
    assert "mean/diag/true_day_range/ssh/1" in records[0]
    assert sorted(p.name for p in (tmp_path / "probed/probe-weights").iterdir()) == [
        "000002.pt",
        "000004.pt",
    ]


def test_loss_guard_skips_outliers_and_fails_on_persistent_ones():
    guard = LossGuard(LossGuardConfig(factor=4, window=2, max_consecutive=2), {})
    assert not guard.skip("obs", 100.0)  # Inactive until the window is full.
    assert not guard.skip("obs", 1.0)
    assert not guard.skip("llc", 1000.0)  # Each source has its own median.
    assert guard.skip("obs", 300.0)  # median(100, 1) = 50.5; 300 > 4 x 50.5.
    assert guard.state["history"]["obs"] == [100.0, 1.0]  # Skips are not recorded.
    with pytest.raises(FloatingPointError, match="consecutive obs"):
        guard.skip("obs", 300.0)


def test_skipped_update_keeps_weights_and_data_plan(tmp_path, monkeypatch):
    base = archive(tmp_path)
    guarded_cfg = configure(base, loss_guard={"factor": 2, "window": 1})
    calls = []

    def skip_third(self, source, loss):
        calls.append(source)
        return len(calls) == 3

    monkeypatch.setattr(LossGuard, "skip", skip_third)
    before = run(guarded_cfg, tmp_path / "before", stop_after=2)
    guarded = run(
        guarded_cfg,
        tmp_path / "before",
        restore=CheckpointPaths(tmp_path / "before").latest_checkpoint_path,
        stop_after=3,
    )
    assert_same_weights(before, guarded)
    assert guarded.updates == 3
    plain = run(configure(base), tmp_path / "plain", stop_after=3)
    assert guarded.rng.bit_generator.state == plain.rng.bit_generator.state


def test_control_fork_reproduces_uninterrupted_training(tmp_path):
    cfg = configure(archive(tmp_path))
    full = run(cfg, tmp_path / "full")
    run(cfg, tmp_path / "parent", stop_after=2)
    replay_cfg = ObservationTrainConfig.model_validate(
        {**cfg.model_dump(), "stop_after": 4, "probe": {}}
    )
    saved = torch.load(
        CheckpointPaths(tmp_path / "parent").latest_checkpoint_path,
        weights_only=False,
    )
    forked, changes = FORK["fork"](
        saved, snapshot_manifest(replay_cfg), replay_cfg, None
    )
    assert changes == {}
    path = tmp_path / "forked.pt"
    torch.save(forked, path)
    replay = run(replay_cfg, tmp_path / "replay", restore=path)
    assert_same_weights(full, replay)


def test_fork_permits_only_listed_differences(tmp_path):
    cfg = configure(archive(tmp_path))
    parent = snapshot_manifest(cfg)
    del parent["config"]["adam_betas"], parent["config"]["loss_guard"]
    child = snapshot_manifest(
        ObservationTrainConfig.model_validate(
            {**cfg.model_dump(), "gradient_clip": 0.2, "adam_betas": (0.9, 0.95)}
        )
    )
    assert FORK["validate_fork"](parent, child) == {
        "gradient_clip": {"parent": 1.0, "child": 0.2},
        "adam_betas": {"parent": [0.9, 0.999], "child": [0.9, 0.95]},
    }
    for key, value in (("weight_decay", 0.5), ("ema_decay", 0.9)):
        changed = copy.deepcopy(child)
        changed["config"][key] = value
        with pytest.raises(ValueError, match=f"may not change {key}"):
            FORK["validate_fork"](parent, changed)
    longer = copy.deepcopy(child)
    longer["config"]["main_updates"] = 10
    with pytest.raises(ValueError, match="may not change main_updates"):
        FORK["validate_fork"](parent, longer, stop_after=11)  # Reaches past main.
    assert "main_updates" in FORK["validate_fork"](parent, longer, stop_after=10)
    changed = copy.deepcopy(child)
    changed["code_sha256"]["models/surface_initialized.py"] = "edited"
    with pytest.raises(ValueError, match="Code differs beyond the probe"):
        FORK["validate_fork"](parent, changed)
