# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU archive, source planning, validation and exact-resume checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.pipeline import train
from samudra.observations.snapshot import LLC_STATE_NAMES, snapshot_objective
from samudra.observations.snapshot_data import SnapshotSources, snapshot_manifest
from samudra.observations.state import TrainingState
from samudra.utils.train import CheckpointPaths


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(root):
    rng = np.random.default_rng(31)
    root.mkdir(exist_ok=True)
    daily, prepared, llc = root / "daily", root / "prepared", root / "llc.zarr"
    prepared.mkdir()
    times = np.arange(
        np.datetime64("2011-01-01T00"),
        np.datetime64("2011-02-15T00"),
        np.timedelta64(6, "h"),
    )
    fields = rng.normal(size=(len(times), 51, 4, 8)).astype("f4")
    lat, lon = np.linspace(-60, 60, 4), np.arange(8) * 45
    xr.Dataset(
        {
            **{
                name: (("time", "k", "y", "x"), fields + i)
                for i, name in enumerate(("U", "V", "Theta", "Salt"))
            },
            "Eta": (("time", "y", "x"), fields[:, 0]),
        },
        coords={"time": times, "lat": ("y", lat), "lon": ("x", lon)},
    ).to_zarr(llc, consolidated=True)
    wet = np.ones((205, 4, 8), dtype="f4")
    wet[:, 0, 0] = 0
    np.savez(
        prepared / "grid.npz",
        names=np.array(LLC_STATE_NAMES),
        lat=lat,
        lon=lon,
        wet_area=wet,
    )
    np.savez(
        prepared / "statistics.npz",
        state_mean=np.zeros(205, dtype="f4"),
        state_std=np.ones(205, dtype="f4"),
        atmosphere_mean=np.zeros(8, dtype="f4"),
        atmosphere_std=np.ones(8, dtype="f4"),
    )
    proof = dict(
        version="llc-obs-six-hour-v1",
        daily_root=str(daily),
        llc_metadata_sha256=checksum(llc / ".zmetadata"),
        time_start="2011-01-01",
        time_end="2011-02-15",
        splits={
            s: {
                "train": ["2011-01-01", "2011-01-31"],
                "validation": ["2011-01-31", "2011-02-15"],
            }
            for s in ("llc", "obs")
        },
        excluded=[],
        sha256={
            name: checksum(prepared / name) for name in ("grid.npz", "statistics.npz")
        },
        surface_shards={},
        era5_shards={},
    )
    for day in np.arange(np.datetime64("2011-01-01"), np.datetime64("2011-02-15")):
        for product in ("sst", "adt", "era5"):
            relative = f"{product}/2011/{day}.npz"
            path = (prepared if product == "era5" else daily) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            values = rng.normal(
                size=(4, 8, 4, 8) if product == "era5" else (1, 4, 8)
            ).astype("f4")
            if product != "era5":
                values[..., 0, 0] = np.nan
            np.savez(path, values=values)
            proof["era5_shards" if product == "era5" else "surface_shards"][
                relative
            ] = checksum(path)
    (prepared / "COMPLETE.json").write_text(json.dumps(proof))
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llc-observation-six-hour/train.yaml"
        ).read_text()
    )
    cfg["experiment"].update(
        data_root=str(root), base_output_dir=str(root), name="tiny"
    )
    cfg["experiment"]["wandb"]["mode"] = "disabled"
    cfg["data"] = dict(llc="llc.zarr", snapshots="prepared")
    cfg.update(
        main_updates=4,
        gradient_accumulation_steps=1,
        fresh_examples=1,
        validation_freq=4,
        validation_examples=1,
        save_freq=2,
        milestone_updates=[],
        prefetch_enabled=True,
    )
    for key in ("backbone", "initializer_backbone", "evolution_backbone"):
        cfg["model"][key].update(ch_width=[4], dilation=[1], n_layers=[1])
    cfg["model"]["activation_checkpointing"] = False
    return ObservationTrainConfig.model_validate(cfg)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_source_contract_sparse_history_and_compact_fields(tmp_path):
    cfg = archive(tmp_path)
    sources = SnapshotSources(cfg, torch.device("cpu"))
    for source in sources.names:
        origin = sources.fresh_origin(source, np.random.default_rng(3))
        sample = sources.load(source, origin)
        assert sample["surface"].shape == (1, 41, 2, 4, 8)
        if source == "obs":
            assert sample["validity"][:, ::4].any()
            assert not sample["validity"][:, 1::4].any()
            assert not sample["validity"][:, 2::4].any()
            assert not sample["validity"][:, 3::4].any()
        else:
            assert sample["target_fields"].shape == (1, 12, 205, 4, 8)
            assert sample["initial_fields"].shape == (1, 2, 205, 4, 8)
        model = cfg.model.build(LLC_STATE_NAMES)
        loss, _, _ = snapshot_objective(
            model,
            sample,
            sources.obs.area,
            "obs" if source == "obs" else "llc",
            9,
            cfg.snapshot,
        )
        loss.backward()
        assert torch.isfinite(loss)
    metrics = sources.validate(model, 1)
    assert cfg.selection_metric in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert "snapshot_val/obs/persistence/24h/zos/rmse" in metrics


@pytest.mark.parametrize("mix", ["llc-obs", "obs"])
def test_snapshot_training_resume_reproduces_uninterrupted_updates(
    tmp_path, mix, monkeypatch
):
    cfg = archive(tmp_path)
    cfg = ObservationTrainConfig.model_validate(
        {**cfg.model_dump(), "pretrain_mix": mix}
    )
    manifest = snapshot_manifest(cfg)

    def state_and_sources():
        torch.manual_seed(73)
        sources = SnapshotSources(cfg, torch.device("cpu"))
        if mix == "obs":
            original_load = sources.load

            def obs_only_load(source, *args, **kwargs):
                assert source == "obs", "OBS-only training read an LLC sample"
                return original_load(source, *args, **kwargs)

            monkeypatch.setattr(sources, "load", obs_only_load)
        model = cfg.model.build(LLC_STATE_NAMES)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        state = TrainingState(model, optimizer, sources.names, config=cfg)
        return state, sources

    full, full_sources = state_and_sources()
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    train(
        cfg,
        full,
        full_sources,
        full_dir,
        manifest,
        lambda _: {cfg.selection_metric: 1.0},
        lambda _: None,
    )
    stopped, stopped_sources = state_and_sources()
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    train(
        cfg,
        stopped,
        stopped_sources,
        resumed_dir,
        manifest,
        lambda _: {cfg.selection_metric: 1.0},
        lambda _: None,
        stop_after=2,
    )
    resumed, resumed_sources = state_and_sources()
    resumed.restore(CheckpointPaths(resumed_dir).latest_checkpoint_path, manifest)
    train(
        cfg,
        resumed,
        resumed_sources,
        resumed_dir,
        manifest,
        lambda _: {cfg.selection_metric: 1.0},
        lambda _: None,
    )
    assert (
        full.exposure
        == resumed.exposure
        == {
            "llc": {
                "updates": 2 if mix == "llc-obs" else 0,
                "fresh": 2 if mix == "llc-obs" else 0,
                "replay": 0,
            },
            "obs": {
                "updates": 2 if mix == "llc-obs" else 4,
                "fresh": 2 if mix == "llc-obs" else 4,
                "replay": 0,
            },
        }
    )
    assert full.updates == resumed.updates == 4
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )
    assert full.pending_plan is resumed.pending_plan is None


def test_completed_resume_restores_completion_receipt(tmp_path):
    cfg = archive(tmp_path)
    torch.manual_seed(3)
    sources = SnapshotSources(cfg, torch.device("cpu"))
    model = cfg.model.build(LLC_STATE_NAMES)
    state = TrainingState(
        model, torch.optim.AdamW(model.parameters()), sources.names, config=cfg
    )
    state.updates = cfg.total_updates
    output = tmp_path / "receipt"
    output.mkdir()
    train(
        cfg,
        state,
        sources,
        output,
        snapshot_manifest(cfg),
        lambda _: {},
        lambda _: None,
    )
    assert (
        json.loads((output / "COMPLETE.json").read_text())["updates"]
        == cfg.total_updates
    )


def test_obs_only_preset_matches_mixed_scientific_settings():
    root = Path(__file__).resolve().parents[1] / "experiments/llc-observation-six-hour"
    mixed = ObservationTrainConfig.model_validate(
        yaml.safe_load((root / "train.yaml").read_text())
    )
    obs = ObservationTrainConfig.model_validate(
        yaml.safe_load((root / "obs-only.yaml").read_text())
    )
    expected = mixed.scientific_config()
    expected["pretrain_mix"] = "obs"
    assert obs.scientific_config() == expected
    assert {obs.source_at(i) for i in range(obs.total_updates)} == {"obs"}
    assert obs.experiment.rand_seed == mixed.experiment.rand_seed


def test_snapshot_cooldown_continuation_and_boundary_resume(tmp_path):
    import runpy

    base = archive(tmp_path)
    extended = ObservationTrainConfig.model_validate(
        {
            **base.model_dump(),
            "main_updates": 8,
            "post_training_updates": 4,
            "post_training_scheduler": {"type": "linear_cooldown", "final_lr": 1e-6},
        }
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments/llc-observation-six-hour/extend_checkpoint.py"
    )
    validate_extension = runpy.run_path(str(script))["validate_extension"]
    current = snapshot_manifest(extended)
    parent = snapshot_manifest(base)
    config_path = (
        Path(__file__).resolve().parents[1] / "src/samudra/observations/config.py"
    )
    old_code = (
        config_path.read_text()
        .replace(
            "                or self.loss.monthly_ts != 0",
            "                or self.post_training_updates != 0\n                or self.loss.monthly_ts != 0",
        )
        .replace("no monthly loss,", "no monthly loss/cooldown,")
    )
    old_source = tmp_path / "old_config.py"
    old_source.write_text(old_code)
    parent["code_sha256"]["observations/config.py"] = checksum(old_source)

    def make(cfg):
        torch.manual_seed(73)
        sources = SnapshotSources(cfg, torch.device("cpu"))
        model = cfg.model.build(LLC_STATE_NAMES)
        return TrainingState(
            model,
            torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate),
            sources.names,
            config=cfg,
        ), sources

    def run(cfg, state, sources, output, manifest, stop=None):
        output.mkdir(exist_ok=True)
        rows: list[dict] = []
        train(
            cfg,
            state,
            sources,
            output,
            manifest,
            lambda _: {cfg.selection_metric: 1.0},
            rows.append,
            stop_after=stop,
        )
        return rows

    full, sources = make(extended)
    rows = run(extended, full, sources, tmp_path / "full", current)
    prefix, sources = make(base)
    run(base, prefix, sources, tmp_path / "prefix", parent)
    saved = torch.load(tmp_path / "prefix/ckpt.pt", weights_only=False)
    changes = validate_extension(saved, current, old_source)
    assert changes["main_updates"] == {"parent": 4, "child": 8}
    import copy

    bad = copy.deepcopy(current)
    bad["config"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="recipe differs"):
        validate_extension(saved, bad, old_source)
    bad = copy.deepcopy(current)
    bad["code_sha256"]["observations/pipeline.py"] = "changed"
    with pytest.raises(ValueError, match="recipe differs"):
        validate_extension(saved, bad, old_source)
    saved["manifest"] = current
    saved["best_score"] = float("inf")
    migrated = tmp_path / "migrated.pt"
    torch.save(saved, migrated)
    child, sources = make(extended)
    child.restore(migrated, current)
    run(extended, child, sources, tmp_path / "child", current, stop=8)
    boundary, sources = make(extended)
    boundary.restore(tmp_path / "child/ckpt.pt", current)
    assert not boundary.in_post_training
    run(extended, boundary, sources, tmp_path / "child", current, stop=10)
    resumed, sources = make(extended)
    resumed.restore(tmp_path / "child/ckpt.pt", current)
    assert resumed.in_post_training
    run(extended, resumed, sources, tmp_path / "child", current)
    assert (
        resumed.exposure
        == full.exposure
        == {
            "llc": {"updates": 4, "fresh": 4, "replay": 0},
            "obs": {"updates": 8, "fresh": 8, "replay": 0},
        }
    )
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )
    assert full.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert full.ema.num_updates.item() == resumed.ema.num_updates.item() == 4
    for key, left in full.ema.get_state(True)["ema_params"].items():
        torch.testing.assert_close(
            left, resumed.ema.get_state(True)["ema_params"][key], rtol=0, atol=0
        )
    rates = [
        r["train/obs/lr"] for r in rows if "train/obs/lr" in r and r["updates"] > 8
    ]
    assert rates == pytest.approx([7.525e-5, 5.05e-5, 2.575e-5, 1e-6])
