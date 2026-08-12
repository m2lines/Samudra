# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from samudra import search as sh


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
        "compute": {
            "type": "slurm",
            "account": "account",
            "partition": "gpu",
            "worker_harness": "/repo/scripts/successive_halving_worker.sbatch",
        },
        "candidates": [
            {
                "name": "control",
                "config": "control.yaml",
                "image_ref": "image",
                "code_commit": "0" * 40,
                "fixed": True,
            },
            {
                "name": "a",
                "config": "a.yaml",
                "image_ref": "image",
                "code_commit": "1" * 40,
            },
            {
                "name": "b",
                "config": "b.yaml",
                "image_ref": "image",
                "code_commit": "2" * 40,
            },
            {
                "name": "c",
                "config": "c.yaml",
                "image_ref": "image",
                "code_commit": "3" * 40,
            },
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


def test_validate_manifest_rejects_unknown_compute_backend(tmp_path):
    data = manifest(tmp_path)
    data["compute"] = {"type": "warp-drive"}

    with pytest.raises(ValueError, match="Unknown compute backend 'warp-drive'"):
        sh.validate_manifest(data)


def test_compute_backend_can_load_from_entry_point(tmp_path, monkeypatch):
    class ExternalBackend:
        @classmethod
        def validate_config(cls, config, rungs):
            pass

        def snapshot(self, bundle_dir):
            return {}

        def submit_anchors(self, state, *, dry_run):
            pass

        def submit_rung(self, state, rung_index, *, dry_run):
            pass

    class EntryPoint:
        def load(self):
            return ExternalBackend

    class EntryPoints:
        def select(self, *, group, name):
            assert group == sh.COMPUTE_BACKEND_ENTRY_POINT
            return [EntryPoint()] if name == "external-test" else []

    monkeypatch.setattr(sh.importlib.metadata, "entry_points", EntryPoints)
    data = manifest(tmp_path)
    data["compute"] = {"type": "external-test"}

    assert sh._compute_backend_type(data) is ExternalBackend


def test_local_backend_dry_run_needs_no_slurm_worker(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    data["compute"] = {"type": "local", "python": "/usr/bin/python3"}
    manifest_path = write_manifest(tmp_path, data)
    monkeypatch.setattr(
        sh,
        "_orchestrator_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "git_commit": "f" * 40,
            "git_root": "/repo",
            "dirty": False,
            "package_version": "1.0",
        },
    )

    state_path = sh.start(
        manifest_path, tmp_path / "searches", dry_run=True, allow_dirty=False
    )

    state = json.loads(state_path.read_text())
    assert state["orchestrator"]["compute_backend"] == "local"
    assert state["anchors"]["job_id"] == "local-dry-run"
    assert state["rungs"][0]["job_id"] == "local-dry-run"
    assert not (state_path.parent / "successive_halving_worker.sbatch").exists()


def test_local_backend_advances_after_synchronous_tasks(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    data["compute"] = {"type": "local"}
    manifest_path = write_manifest(tmp_path, data)
    state_path = tmp_path / "state.json"
    state = {
        "status": "prepared",
        "anchors": {"candidates": []},
        "rungs": [{"candidates": ["a", "b"]}],
    }
    sh._atomic_json(state_path, state)
    backend = sh.LocalBackend(manifest_path, state_path)
    events: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        backend,
        "_run_tasks",
        lambda count, rung_index, anchor, dry_run: events.append(
            ("tasks", count, rung_index, anchor, dry_run)
        ),
    )
    monkeypatch.setattr(
        sh,
        "advance",
        lambda manifest_path, state_path, rung_index, dry_run: events.append(
            ("advance", rung_index, dry_run)
        ),
    )

    backend.submit_rung(state, 0, dry_run=False)

    assert events == [("tasks", 2, 0, False, False), ("advance", 0, False)]
    assert json.loads(state_path.read_text())["rungs"][0]["job_id"] == "local"


def test_local_backend_runs_complete_search(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    data["rungs"] = [1, 2]
    data["candidates"] = data["candidates"][:3]
    harness = tmp_path / "fake_train.py"
    harness.write_text(
        """#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

epoch = int(re.search(r"--epochs=(\\d+)", os.environ["ARGS"]).group(1))
name = os.environ["SAMUDRA_SEARCH_CANDIDATE"]
output = Path(os.environ["OUTPUT_BASE"]) / os.environ["NAME"]
(output / "saved_nets").mkdir(parents=True)
(output / "saved_nets" / "ckpt.pt").touch()
(output / "training_summary.json").write_text(json.dumps({
    "schema_version": 1,
    "epoch": epoch,
    "complete": True,
    "validation_loss": {"control": 0.05, "a": 0.1, "b": 0.2}[name],
    "provenance": {"code_commit": os.environ["SAMUDRA_SEARCH_CANDIDATE_COMMIT"]},
    "search": {
        "compute_backend": os.environ["SAMUDRA_SEARCH_COMPUTE_BACKEND"],
        "compute_job_id": os.environ["SAMUDRA_SEARCH_COMPUTE_JOB_ID"],
    },
}))
""",
        encoding="utf-8",
    )
    harness.chmod(harness.stat().st_mode | 0o100)
    data["runtime"]["train_harness"] = str(harness)
    data["compute"] = {"type": "local", "python": sys.executable}
    manifest_path = write_manifest(tmp_path, data)
    monkeypatch.setattr(
        sh,
        "_orchestrator_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "git_commit": "f" * 40,
            "git_root": "/repo",
            "dirty": False,
            "package_version": "1.0",
        },
    )

    state_path = sh.start(
        manifest_path, tmp_path / "searches", dry_run=False, allow_dirty=False
    )

    state = json.loads(state_path.read_text())
    assert state["status"] == "complete"
    assert state["rungs"][0]["promoted"] == ["a"]
    assert state["rungs"][1]["results"][0]["compute_backend"] == "local"
    assert state["anchors"]["results"][0]["eligible"]


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
                    "provenance": {
                        "code_commit": data["candidates"][
                            ["control", "a", "b", "c"].index(candidate)
                        ]["code_commit"]
                    },
                }
            ),
            encoding="utf-8",
        )
    submitted = []

    class RecordingBackend:
        def submit_rung(self, state, rung_index, *, dry_run):
            submitted.append(
                (rung_index, list(state["rungs"][rung_index]["candidates"]))
            )

    monkeypatch.setattr(
        sh, "_compute_backend", lambda manifest_path, state_path: RecordingBackend()
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
                "provenance": {"code_commit": "0" * 40},
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
        "manifest_sha256": sh._manifest_hash(manifest_path),
        "orchestrator": {"commit": "f" * 40},
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
    assert environment["SAMUDRA_SEARCH_CANDIDATE"] == "a"
    assert environment["SAMUDRA_SEARCH_RUNG"] == "1"
    assert environment["SAMUDRA_SEARCH_PARENT_CHECKPOINT"] == str(checkpoint)


def test_orchestrator_provenance_rejects_dirty_checkout(monkeypatch):
    responses = iter(["/repo\n", "a" * 40 + "\n", " M src/samudra/search.py\n"])

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(
        sh.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(responses)),
    )

    with pytest.raises(ValueError, match="tracked changes"):
        sh._orchestrator_provenance(allow_dirty=False)


def test_slurm_start_snapshots_controller_worker_and_provenance(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    worker = tmp_path / "worker.sbatch"
    worker.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    data["compute"]["worker_harness"] = str(worker)
    manifest_path = write_manifest(tmp_path, data)
    monkeypatch.setattr(
        sh,
        "_orchestrator_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "git_commit": "f" * 40,
            "git_root": "/repo",
            "dirty": False,
            "package_version": "1.0",
        },
    )

    state_path = sh.start(
        manifest_path, tmp_path / "searches", dry_run=True, allow_dirty=False
    )

    state = json.loads(state_path.read_text())
    bundle = state_path.parent
    assert state["orchestrator"]["commit"] == "f" * 40
    assert state["orchestrator"]["controller_sha256"] == sh._file_hash(
        bundle / "search_controller.py"
    )
    assert state["orchestrator"]["worker_sha256"] == sh._file_hash(
        bundle / "successive_halving_worker.sbatch"
    )
