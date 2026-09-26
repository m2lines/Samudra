# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samudra.experiments import observation_evaluate as evaluation
from samudra.experiments.observation_checkpoint import load_fixed_checkpoint


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
    monkeypatch.setattr(evaluation, "Samples", lambda *a: data)
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
