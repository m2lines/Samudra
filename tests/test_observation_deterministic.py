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
        assert c.model.evolution_backbone.ch_width == [280, 380, 480, 520]
    winner = arms["D-mixed-large"]
    extension = winner_configs(winner)
    main, *branches = extension.values()
    assert main.main_updates == 16000
    assert main.fork_ckpt_path == str(winner.experiment.nets_dir / "main.pt")
    assert (
        main.total_updates - 4000 + sum(c.cooldown_updates for c in branches) == 15000
    )
    assert [c.main_updates for c in branches] == [8000, 12000]
    assert all(c.cooldown_updates == 1000 for c in branches)
