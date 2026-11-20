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


def test_snapshot_training_resume_reproduces_uninterrupted_updates(tmp_path):
    cfg = archive(tmp_path)
    manifest = snapshot_manifest(cfg)

    def state_and_sources():
        torch.manual_seed(73)
        sources = SnapshotSources(cfg, torch.device("cpu"))
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
            "llc": {"updates": 2, "fresh": 2, "replay": 0},
            "obs": {"updates": 2, "fresh": 2, "replay": 0},
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
