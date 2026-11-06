# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar-safe ERA5 conditioning and matched experiment budgets."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from samudra.observations.campaign import screen_configs, winner_configs
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.shared_era5 import SharedEra5, digest


def cache_fixture(tmp_path):
    indices = np.arange(21)
    np.save(tmp_path / "indices.npy", indices)
    raw = np.arange(21 * 8 * 2 * 3, dtype=np.float32).reshape(21, 8, 2, 3)
    np.save(tmp_path / "values.npy", raw)
    starts = pd.date_range("2000-01-01", periods=21, freq="5D")
    bounds = np.array([[day, day + pd.Timedelta(days=5)] for day in starts])
    reader = SimpleNamespace(windows=[(19, 21, None, 10)], bounds=bounds)
    data = SimpleNamespace(
        grid={"mask": np.ones((77, 2, 3))},
        stats={"atmosphere_mean": np.arange(8), "atmosphere_std": np.arange(8) + 1},
    )
    proof = {
        "windows": 1,
        "all_intervals_readback_exact": True,
        "contracts": {},
        "bounds": [[t.isoformat() for t in pair] for pair in bounds],
        "files": {
            name: digest(tmp_path / name) for name in ("indices.npy", "values.npy")
        },
    }
    path = tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(proof))
    return reader, data, proof, path, raw


def test_era5_uses_all_fields_and_shared_observation_statistics(tmp_path):
    reader, data, _, path, raw = cache_fixture(tmp_path)
    adapter = SharedEra5(tmp_path, reader, data, digest(path))
    expected = (raw[17:21] - np.arange(8)[None, :, None, None]) / (np.arange(8) + 1)[
        None, :, None, None
    ]
    np.testing.assert_allclose(adapter.load(17, 21), expected)
    # A native gap stays a gap: indexing does not interpolate across it.
    reader.bounds[20] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="bounds differ"):
        SharedEra5(tmp_path, reader, data, digest(path))


def test_era5_rejects_changed_values_or_window_pool(tmp_path):
    reader, data, _, path, raw = cache_fixture(tmp_path)
    reader.windows.append((19, 21, None, 10))
    with pytest.raises(ValueError, match="sample pool"):
        SharedEra5(tmp_path, reader, data, digest(path))
    reader.windows.pop()
    raw[0, 7, 0, 0] += 1
    np.save(tmp_path / "values.npy", raw)
    with pytest.raises(ValueError, match="content changed"):
        SharedEra5(tmp_path, reader, data, digest(path))


def test_five_arm_screen_and_winner_budget():
    cfg = ObservationTrainConfig.from_yaml(
        Path("src/samudra/configs/observations/deterministic.yaml")
    )
    values = cfg.model_dump(mode="json")
    values["data"].update(era5_cache="era5-cache", era5_proof_sha256="test-proof")
    cfg = ObservationTrainConfig.model_validate(values)
    arms = screen_configs(cfg)
    assert len(arms) == 5
    assert sum(c.total_updates for c in arms.values()) == 25000
    assert arms["E-mixed-large-no-replay"].fresh_examples == 8
    for c in arms.values():
        assert c.model.forcing_mode == "shared_era5"
        assert c.model.evolution_backbone is not None
        assert c.model.evolution_backbone.ch_width == [280, 380, 480, 520]
    winner = arms["D-mixed-large"]
    extension = winner_configs(winner)
    main, *branches = extension.values()
    assert main.main_updates == 16000
    assert main.fork_ckpt_path == str(winner.experiment.nets_dir / "main.pt")
    assert (
        main.total_updates - 4000 + sum(c.post_training_updates for c in branches)
        == 15000
    )
    assert [c.main_updates for c in branches] == [8000, 12000]
    assert all(c.post_training_updates == 1000 for c in branches)


def test_runtime_estimate_counts_validation_checkpoint_once():
    from samudra.observations.runtime_estimate import estimate

    receipt = {
        "training_records": [
            {"updates": 2, "train/om4/seconds": 20, "train/om4/data_seconds": 2},
            {"updates": 3, "train/obs/seconds": 10, "train/obs/data_seconds": 1},
            {"updates": 6, "timing/validation_seconds": 100},
        ],
        "exposure": {"obs": {}, "om4": {}},
        "gpu": "test",
        "checkpoint_seconds": 5,
    }
    screen = estimate(receipt)
    continuation = estimate(receipt, 16000, 3000, main_start=4000)
    assert screen["source_updates"] == {"obs": 3000, "om4": 2000}
    assert continuation["source_updates"] == {"obs": 9000, "om4": 6000}
    assert screen["hours_observed_range"][0] == pytest.approx(
        (70000 + 200 * 5 + 20 * 100) / 3600
    )


def test_distributed_estimate_excludes_sync_controls_and_counts_gpu_allocation():
    from samudra.observations.runtime_estimate import estimate

    receipt = {
        "world_size": 8,
        "gpu": "test",
        "exposure": {"obs": {}, "om4": {}},
        "training_records": [
            {"stage": "sync", "updates": 18, "train/om4/seconds": 10000},
            {
                "stage": "steady",
                "updates": 25,
                "train/obs/seconds": 2,
                "train/obs/data_seconds": 0.1,
            },
            {
                "stage": "steady",
                "updates": 26,
                "train/om4/seconds": 3,
                "train/om4/data_seconds": 0.2,
            },
            {"stage": "steady", "updates": 32, "timing/checkpoint_seconds": 5},
            {"stage": "steady", "updates": 128, "timing/checkpoint_seconds": 20},
            {"stage": "steady", "updates": 128, "timing/validation_seconds": 100},
        ],
    }
    result = estimate(receipt)
    assert result["seconds_per_update"]["om4"]["max"] == 3
    assert result["checkpoint_seconds"] == 5
    assert result["hours_observed_range"][0] == pytest.approx(
        (12000 + 223 * 5 + 20 * 100) / 3600
    )
    assert result["allocated_gpu_hours_with_30_percent_execution_allowance"][
        0
    ] == pytest.approx(8 * result["hours_with_30_percent_execution_allowance"][0])


@pytest.mark.parametrize("prefetch", [False, True])
def test_qualification_keeps_prefetched_sources_on_cpu(monkeypatch, tmp_path, prefetch):
    """Stop at the data boundary without allocating a GPU or opening real archives."""
    import torch

    from samudra.observations import qualify

    cfg = qualify.QualificationConfig.from_yaml("observations/train.yaml").model_copy(
        update={
            "main_updates": 4,
            "post_training_updates": 2,
            "update_prefetch": prefetch,
        }
    )
    cfg.experiment.base_output_dir = str(tmp_path)
    monkeypatch.setattr(qualify, "set_seed", lambda _: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    devices = []

    class DataBoundary(Exception):
        pass

    def build_sources(config, device):
        devices.append(device.type)
        data = SimpleNamespace(to=lambda target: SimpleNamespace(device=target.type))
        return SimpleNamespace(obs=data, models={})

    def manifest(config, data, sources):
        if prefetch:
            assert data.device == "cuda"
            assert data is not sources.obs
        raise DataBoundary

    monkeypatch.setattr(qualify, "build_sources", build_sources)
    monkeypatch.setattr(qualify.baseline, "run_manifest", manifest)
    with pytest.raises(DataBoundary):
        qualify.run(cfg)
    assert devices == ["cpu" if prefetch else "cuda"]
