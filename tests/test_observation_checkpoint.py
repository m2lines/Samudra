# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from samudra.experiments import observation_evaluate as evaluation
from samudra.experiments.observation_checkpoint import load_fixed_checkpoint
from samudra.experiments.observation_pilot import digest


def fixed_run(path):
    path.mkdir()
    manifest = {
        "arguments": {
            "data": "unused",
            "from_scratch": True,
            "normalization": "instance",
            "evolution_architecture": "samudra2",
        },
        "data_manifest_sha256": "test-data",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    saved = {
        "manifest": manifest,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"opaque": "not an evaluation input"},
        "task_counts": {"om4": 5, "observation": 2},
        "global_step": 7,
        "step": 2,
        "phase": "joint",
    }
    torch.save(saved, path / "joint-00002.pt")
    return saved


def test_counts_completion_and_manifest_are_required(tmp_path):
    run = tmp_path / "run"
    saved = fixed_run(run)
    with pytest.raises(ValueError, match="completed training"):
        load_fixed_checkpoint(run, "joint-00002.pt", "test", 5, 2)
    model, lineage = load_fixed_checkpoint(run, "joint-00002.pt", "validation", 5, 2)
    assert set(model) == {"weight"}
    assert lineage["task_counts"] == {"om4": 5, "observation": 2}
    (run / "TRAIN_COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="counts differ"):
        load_fixed_checkpoint(run, "joint-00002.pt", "test", 6, 2)
    with pytest.raises(ValueError, match="filename"):
        load_fixed_checkpoint(run, "best.pt", "test", 5, 2)
    saved["manifest"]["arguments"]["from_scratch"] = False
    torch.save(saved, run / "joint-00002.pt")
    with pytest.raises(ValueError, match="lineage/counts differ"):
        load_fixed_checkpoint(run, "joint-00002.pt", "test", 5, 2)


def test_fixed_evaluation_does_not_claim_validation_selection(tmp_path, monkeypatch):
    run, output = tmp_path / "run", tmp_path / "out"
    fixed_run(run)
    (run / "TRAIN_COMPLETE.json").write_text("{}")

    # Deliberately no best.json and no source checkpoint: neither belongs to this
    # evaluation and accessing either would fail the test.
    class Model:
        @classmethod
        def from_arguments(cls, names, arguments):
            return cls(
                names,
                arguments.get("normalization", "batch"),
                arguments.get("evolution_architecture", "d"),
            )

        def __init__(self, names, normalization, architecture):
            assert architecture == "samudra2"

        def cuda(self):
            return self

        def load_state_dict(self, state, strict):
            assert strict and state["weight"].item() == 1.0

    data = SimpleNamespace(
        grid={"names": np.array(["sst"])},
        use_observation_normalization=lambda: None,
        paths=lambda split: list(range(96)),
    )
    monkeypatch.setattr(evaluation, "Samples", lambda *a, **k: data)
    monkeypatch.setattr(evaluation, "ObservationTransfer", Model)
    seen = []

    def evaluate(self, paths, export, **options):
        seen.append(export.stem)
        export.write_bytes(b"test")
        return {"test": True}

    monkeypatch.setattr(evaluation.Pilot, "evaluate", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--run",
            str(run),
            "--output",
            str(output),
            "--selected-only",
            "--checkpoint",
            "joint-00002.pt",
            "--fixed-om4-updates",
            "5",
            "--fixed-observation-updates",
            "2",
        ],
    )
    evaluation.main()
    assert seen == [
        "fixed-budget-predictions",
        "fixed-budget-inferred-persistence",
        "fixed-budget-inferred-anomaly-persistence",
        "seasonal-climatology",
    ]
    record = json.loads((output / "fixed-budget.json").read_text())
    assert record["selected_checkpoint"] is None
    assert record["validation_selection_score"] is None
    assert record["fixed_budget_lineage"]["global_step"] == 7
    evaluation.main()
    assert len(seen) == 4


def quarter_run(path, arm, world_size=4):
    from samudra.experiments.quarter_joint import ARMS, exposure_counts, pilot_arguments
    from samudra.experiments.task_schedule import TaskSchedule

    path.mkdir()
    arguments = vars(
        pilot_arguments(
            SimpleNamespace(
                arm=arm,
                mode="train",
                data=str(path / "data"),
                output=str(path),
                name="quarter-test",
                qualification="qualification.json",
                selection_reference="selection-reference.json",
            )
        )
    )
    contract: dict[str, Any] = {
        "version": 1,
        "arm": arm,
        "schedule": list(ARMS[arm]),
        "code_commit": "a" * 40,
        "model_options": {
            key: arguments[key]
            for key in (
                "initializer_architecture",
                "surface_policy",
                "task_conditioning",
                "latent_channels",
            )
        },
        "learning_rate": 1e-4,
        "global_batch": 8,
        "world_size": world_size,
        **{
            key: arguments[key]
            for key in (
                "normalization",
                "evolution_architecture",
                "surface_fill",
                "completion_weight",
                "reconstruction_weight",
                "global_observations",
                "seed",
            )
        },
        **{
            key: "b" * 64
            for key in (
                "runtime_extension_sha256",
                "runner_sha256",
                "pilot_sha256",
                "distributed_helper_sha256",
                "selection_reference_sha256",
            )
        },
        "data": {
            key: "c" * 64
            for key in ("SHA256SUMS", "grid.npz", "statistics.npz", "iap-grid.npz")
        },
    }
    manifest = {
        "arguments": arguments,
        "normalization_mode": "observation-only",
        "code_commit": contract["code_commit"],
        "runtime_extension_sha256": contract["runtime_extension_sha256"],
        **{
            key: "c" * 64
            for key in (
                "data_manifest_sha256",
                "grid_sha256",
                "statistics_sha256",
                "iap_grid_sha256",
            )
        },
    }
    counts = exposure_counts(arm, TaskSchedule(*ARMS[arm]), 6000)
    saved = {
        "model": {"weight": torch.tensor([2.0])},
        "optimizer": {"opaque": "never returned for evaluation"},
        "completed": 6000,
        "last_validated": 6000,
        "task_counts": counts,
        "quarter_contract": contract,
        "diagnostic_only": False,
    }
    complete = {
        "updates": 6000,
        "global_step": 6000,
        "task_counts": counts,
        "quarter_contract": contract,
    }
    return manifest, saved, complete


def write_quarter_run(path, manifest, saved, complete):
    (path / "manifest.json").write_text(json.dumps(manifest))
    torch.save(saved, path / "quarter-last.pt")
    complete["checkpoint_sha256"] = digest(path / "quarter-last.pt")
    (path / "TRAIN_COMPLETE.json").write_text(json.dumps(complete))


@pytest.mark.parametrize(
    "arm",
    ["u-global", "observation-only", "u-multitask-early", "u-multitask-early-latent"],
)
@pytest.mark.parametrize("split", ["test", "validation"])
@pytest.mark.parametrize("world_size", [4, 8])
def test_quarter_endpoint_supports_all_approved_arms_without_conversion(
    tmp_path, monkeypatch, arm, split, world_size
):
    path = tmp_path / arm
    manifest, saved, complete = quarter_run(path, arm, world_size)
    write_quarter_run(path, manifest, saved, complete)
    # The evaluation producer can differ from immutable training provenance.
    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "d" * 40)
    counts = saved["task_counts"]
    model, lineage = load_fixed_checkpoint(
        path, "quarter-last.pt", split, counts["om4"], counts["observation"]
    )
    assert model["weight"].item() == 2
    assert set(model) == {"weight"}
    assert lineage["global_step"] == 6000
    assert lineage["task_counts"] == counts
    assert lineage["training_producer"] == "a" * 40
    assert lineage["checkpoint_sha256"] == complete["checkpoint_sha256"]
    assert lineage["training_manifest_sha256"] == digest(path / "manifest.json")
    assert lineage["quarter_contract"]["world_size"] == world_size
    assert not list(path.glob("joint-*.pt"))


@pytest.mark.parametrize(
    "corrupt",
    [
        "incomplete",
        "unvalidated",
        "diagnostic",
        "early_counts",
        "contract",
        "producer",
        "runtime",
        "schedule",
        "latent",
        "science",
        "manifest_data",
        "manifest_model",
        "manifest_geometry",
        "terminal_step",
        "terminal_counts",
        "hash",
        "missing_completion",
        "requested_counts",
        "unsupported_world",
        "terminal_world",
    ],
)
def test_quarter_endpoint_rejects_unfinished_or_mismatched_lineage(tmp_path, corrupt):
    path = tmp_path / "quarter"
    manifest, saved, complete = quarter_run(path, "u-multitask-early-latent")
    if corrupt == "incomplete":
        saved["completed"] = 5999
    elif corrupt == "unvalidated":
        saved["last_validated"] = 5900
    elif corrupt == "diagnostic":
        saved["diagnostic_only"] = True
    elif corrupt == "early_counts":
        saved["task_counts"]["om4_early"] = 1499
    elif corrupt == "contract":
        complete["quarter_contract"] = {}
    elif corrupt == "producer":
        manifest["code_commit"] = "e" * 40
    elif corrupt == "runtime":
        manifest["runtime_extension_sha256"] = "e" * 64
    elif corrupt == "schedule":
        saved["quarter_contract"]["schedule"][2] = "scratch"
    elif corrupt == "latent":
        saved["quarter_contract"]["model_options"]["latent_channels"] = 0
    elif corrupt == "science":
        saved["quarter_contract"]["learning_rate"] = 1e-3
    elif corrupt == "manifest_data":
        manifest["statistics_sha256"] = "f" * 64
    elif corrupt == "manifest_model":
        manifest["arguments"]["surface_policy"] = "legacy-copy"
    elif corrupt == "manifest_geometry":
        del manifest["arguments"]["iap_grid"]
    elif corrupt == "terminal_step":
        complete["global_step"] = 5999
    elif corrupt == "terminal_counts":
        complete["task_counts"] = {"om4": 3000, "observation": 3000}
    elif corrupt == "unsupported_world":
        saved["quarter_contract"]["world_size"] = 16
    elif corrupt == "terminal_world":
        complete["quarter_contract"] = {**saved["quarter_contract"], "world_size": 8}
    write_quarter_run(path, manifest, saved, complete)
    if corrupt == "hash":
        with (path / "quarter-last.pt").open("ab") as stream:
            stream.write(b"changed")
    elif corrupt == "missing_completion":
        (path / "TRAIN_COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="Quarter endpoint"):
        load_fixed_checkpoint(
            path,
            "quarter-last.pt",
            "validation",
            3000,
            2999 if corrupt == "requested_counts" else 3000,
        )


def test_screen_endpoint_preserves_weight_choice_and_rejects_tampering(tmp_path):
    from samudra.experiments.observation_checkpoint import load_screen_endpoint

    run = tmp_path / "screen"
    saved = fixed_run(run)
    manifest = saved["manifest"]
    manifest["code_commit"] = "producer"
    (run / "manifest.json").write_text(json.dumps(manifest))
    saved.update(
        global_step=5000,
        step=3000,
        task_counts={"om4": 2000, "observation": 3000},
        screen={
            "policy": "D",
            "start": 4000,
            "end": 5000,
            "producer": "producer",
            "ema_half_life": 100,
        },
        ema={"weight": torch.tensor([2.0])},
        ema_updates=2000,
    )
    checkpoint = run / "joint-last.pt"
    torch.save(saved, checkpoint)
    complete = dict(
        global_step=5000,
        task_counts=saved["task_counts"],
        branch="D",
        ema_updates=2000,
        sha256=digest(checkpoint),
    )
    (run / "COMPLETE.json").write_text(json.dumps(complete))
    raw, _ = load_screen_endpoint(run, "raw")
    ema, lineage = load_screen_endpoint(run, "ema")
    assert raw["weight"].item() == 1
    assert ema["weight"].item() == 2
    assert lineage["weights"] == "ema"
    complete["ema_updates"] = 1000
    (run / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="lineage"):
        load_screen_endpoint(run, "ema")
    complete["sha256"] = "wrong"
    (run / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="hash"):
        load_screen_endpoint(run, "raw")
