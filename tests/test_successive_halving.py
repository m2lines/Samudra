# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml

from scripts import successive_halving as sh


def manifest(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "name": "architecture-search",
        "rungs": [1, 3, 6],
        "promotion_fraction": 0.5,
        "minimum_promoted": 1,
        "metric": "validation_loss",
        "mode": "min",
        "runtime": {
            "output_base": str(tmp_path / "runs"),
            "train_harness": "/repo/scripts/slurm_apptainer_train.sbatch",
        },
        "slurm": {"account": "account", "partition": "gpu"},
        "candidates": [
            {
                "name": "control",
                "config": "control.yaml",
                "image_ref": "image",
                "fixed": True,
            },
            {"name": "a", "config": "a.yaml", "image_ref": "image"},
            {"name": "b", "config": "b.yaml", "image_ref": "image"},
            {"name": "c", "config": "c.yaml", "image_ref": "image"},
        ],
    }


def write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_validate_manifest_rejects_unpinned_candidate(tmp_path):
    data = manifest(tmp_path)
    del data["candidates"][0]["image_ref"]

    with pytest.raises(ValueError, match="code_layer or image_ref"):
        sh.validate_manifest(data)


def test_advance_promotes_best_half_without_fixed_anchor(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    manifest_path = write_manifest(tmp_path, data)
    state_path = tmp_path / "state.json"
    state = {
        "schema_version": 1,
        "name": data["name"],
        "manifest_sha256": sh._manifest_hash(manifest_path),
        "status": "running",
        "anchors": {"candidates": ["control"], "results": []},
        "rungs": [
            {
                "index": 0,
                "epochs": 1,
                "candidates": ["a", "b", "c"],
                "results": [],
                "promoted": [],
                "advanced": False,
            },
            {
                "index": 1,
                "epochs": 3,
                "candidates": [],
                "results": [],
                "promoted": [],
                "advanced": False,
            },
            {
                "index": 2,
                "epochs": 6,
                "candidates": [],
                "results": [],
                "promoted": [],
                "advanced": False,
            },
        ],
    }
    sh._atomic_json(state_path, state)
    for candidate, score in {"a": 0.3, "b": 0.5, "c": 0.7}.items():
        output = sh._rung_output(data, candidate, 0)
        (output / "saved_nets").mkdir(parents=True)
        (output / "saved_nets/ckpt.pt").touch()
        (output / "training_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "epoch": 1,
                    "complete": True,
                    "validation_loss": score,
                }
            ),
            encoding="utf-8",
        )
    submitted = []
    monkeypatch.setattr(
        sh,
        "_submit_rung",
        lambda manifest_path, state_path, state, rung_index, dry_run: submitted.append(
            (rung_index, list(state["rungs"][rung_index]["candidates"]))
        ),
    )

    sh.advance(manifest_path, state_path, 0, dry_run=True)

    updated = json.loads(state_path.read_text())
    assert updated["rungs"][0]["promoted"] == ["a", "b"]
    assert updated["rungs"][1]["candidates"] == ["a", "b"]
    assert submitted == [(1, ["a", "b"])]
    assert (tmp_path / "leaderboard-r0.csv").is_file()


def test_results_exclude_incomplete_or_missing_candidates(tmp_path):
    data = manifest(tmp_path)
    state = {
        "rungs": [
            {
                "candidates": ["control", "a"],
            }
        ]
    }
    output = sh._rung_output(data, "control", 0)
    output.mkdir(parents=True)
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "epoch": 1,
                "complete": False,
                "validation_loss": 0.1,
            }
        ),
        encoding="utf-8",
    )

    results = sh._read_results(data, state, 0)

    assert not results[0]["eligible"]
    assert "did not finish" in results[0]["error"]
    assert not results[1]["eligible"]


def test_promoted_task_resumes_previous_checkpoint(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    data["candidates"] = [data["candidates"][1]]
    manifest_path = write_manifest(tmp_path, data)
    state = {
        "anchors": {"candidates": []},
        "rungs": [
            {"candidates": ["a"]},
            {"candidates": ["a"]},
            {"candidates": []},
        ],
    }
    state_path = tmp_path / "state.json"
    sh._atomic_json(state_path, state)
    previous = sh._rung_output(data, "a", 0)
    (previous / "saved_nets").mkdir(parents=True)
    checkpoint = previous / "saved_nets/ckpt.pt"
    checkpoint.touch()
    calls = []

    def record_call(command, *, env, check):
        calls.append((command, env, check))

    monkeypatch.setattr(sh.subprocess, "run", record_call)

    sh.run_task(manifest_path, state_path, 1, 0, anchor=False)

    command, environment, check = calls[0]
    assert command == [data["runtime"]["train_harness"]]
    assert check
    assert environment["NAME"].endswith("--a--e3")
    assert "--epochs=3" in environment["ARGS"]
    assert f"--resume_ckpt_path={checkpoint}" in environment["ARGS"]
