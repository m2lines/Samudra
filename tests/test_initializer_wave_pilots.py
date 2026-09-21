# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import pytest

SELECT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/select_initializer_wave_pilots.py")
)["select"]


def populate(root):
    for arm in "ABCDEF":
        for tag, rate, score in [("lr1", 1e-4, 0.2), ("lr3", 3e-4, 0.1)]:
            path = root / f"{arm}-{tag}"
            path.mkdir()
            args = dict(
                arm=arm,
                learning_rate=rate,
                phase="reconstruction",
                max_steps=0,
                seed=1729,
                batch_size=2,
                accumulate=2,
                hours=0.5,
                data_root="data",
                val_origins=12,
            )
            (path / "manifest.json").write_text(
                json.dumps(
                    dict(
                        arguments=args,
                        world_size=2,
                        device="matched GPU",
                        code_commit="producer",
                    )
                )
            )
            (path / "TRAIN_COMPLETE.json").write_text(
                json.dumps(
                    dict(
                        state=dict(complete=True, step=10),
                        selected_validation=dict(ts_mse=score),
                    )
                )
            )
            (path / "progress.jsonl").write_text(
                json.dumps(dict(event="baseline", ts_mse=1.0)) + "\n"
            )


def test_pilots_select_one_rate_per_architecture(tmp_path):
    populate(tmp_path)
    result = SELECT(tmp_path)
    assert all(item["learning_rate"] == 3e-4 for item in result["selected"].values())


def test_missing_pilot_prevents_selection(tmp_path):
    populate(tmp_path)
    (tmp_path / "F-lr3" / "TRAIN_COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        SELECT(tmp_path)


def test_hardware_mismatch_prevents_comparison(tmp_path):
    populate(tmp_path)
    path = tmp_path / "F-lr3" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["device"] = "different GPU"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol differs"):
        SELECT(tmp_path)
