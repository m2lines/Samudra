# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

WATCHER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/watch_diffusion_beta.py")
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source(tmp_path):
    for name, score in (("scratch-main", 2.0), ("transfer-main", 1.0)):
        run = tmp_path / name
        run.mkdir()
        (run / "best.pt").write_bytes(name.encode())
        checksum = hashlib.sha256(name.encode()).hexdigest()
        write(run / "best.json", {"score": score, "checkpoint_sha256": checksum})
        write(
            run / "TRAIN_COMPLETE.json",
            {"best_score": score, "best_checkpoint_sha256": checksum},
        )
        write(
            run / "manifest.json",
            {
                "arguments": {"normalization": "instance"},
                "code_commit": "producer",
                "data_manifest_sha256": "same-data",
                "statistics_sha256": "same-stats",
            },
        )
        write(run / "selection-reference.json", {"same": "selection"})
        write(
            tmp_path / (name + "-monthly") / "COMPLETE.json",
            {"selected_sha256": checksum, "origins": 96},
        )
        write(
            tmp_path / (name + "-annual") / "COMPLETE.json",
            {"inputs": {"checkpoint_sha256": checksum, "split": "test"}},
        )
    return tmp_path


def probe(root):
    code = WATCHER["PROBE"].replace(WATCHER["SOURCE"], str(root))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_final_validation_selects_only_after_both_reporting_paths_complete(tmp_path):
    root = source(tmp_path)
    result = probe(root)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected"] == "transfer-main"
    (root / "transfer-main-annual/COMPLETE.json").unlink()
    result = probe(root)
    assert result.returncode == 0
    assert json.loads(result.stdout)["ready"] is False


@pytest.mark.parametrize("corruption", ["weights", "completion", "statistics"])
def test_changed_checkpoint_or_comparison_contract_cannot_launch(tmp_path, corruption):
    root = source(tmp_path)
    run = root / "transfer-main"
    if corruption == "weights":
        (run / "best.pt").write_bytes(b"changed")
    elif corruption == "completion":
        write(
            run / "TRAIN_COMPLETE.json",
            {"best_score": 0.5, "best_checkpoint_sha256": "changed"},
        )
    else:
        path = run / "manifest.json"
        data = json.loads(path.read_text())
        data["statistics_sha256"] = "different"
        write(path, data)
    assert probe(root).returncode != 0
