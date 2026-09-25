# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Selected-state controls must preserve weights and scratch normalization."""

import json
import sys

import numpy as np
import pytest
import torch

from samudra.experiments import observation_evaluate as evaluation


@pytest.mark.parametrize("scratch", [False, True])
def test_control_weights_normalization_and_resume(tmp_path, monkeypatch, scratch):
    run, output = tmp_path / "run", tmp_path / "evaluation"
    run.mkdir()
    (run / "TRAIN_COMPLETE.json").write_text("{}")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": {
                    "data": "data",
                    "checkpoint": "source",
                    "from_scratch": scratch,
                },
                "data_manifest_sha256": "data-digest",
            }
        )
    )
    (run / "best.json").write_text('{"checkpoint_sha256":"digest","score":1.0}')

    class Data:
        def __init__(self, *args):
            self.normalization = "source"
            self.grid = {"names": np.array(["sst"])}

        def use_observation_normalization(self):
            self.normalization = "observations"

        def paths(self, split):
            return list(range(96))

    class Model:
        def __init__(self, names, normalization="batch"):
            assert normalization in ("batch", "instance")
            self.weights = None
            self.adapter = [torch.nn.Conv2d(1, 1, 1)]

        def cuda(self):
            return self

        def load_state_dict(self, weights, strict):
            assert strict
            self.weights = weights

        def load_core(self, weights):
            self.weights = weights

    calls = []

    def evaluate(self, paths, export, **options):
        calls.append(
            (export.stem, self.model.weights, self.data.normalization, options)
        )
        export.write_bytes(b"export")
        return {"control": export.stem}

    monkeypatch.setattr(evaluation, "Samples", Data)
    monkeypatch.setattr(evaluation, "ObservationTransfer", Model)
    monkeypatch.setattr(evaluation.Pilot, "evaluate", evaluate)
    monkeypatch.setattr(evaluation, "digest", lambda path: "digest")
    monkeypatch.setattr(
        evaluation.torch,
        "load",
        lambda path, **kwargs: {
            "model": "source" if str(path) == "source" else "selected"
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--run", str(run), "--output", str(output)]
    )
    evaluation.main()
    selected_norm = "observations" if scratch else "source"
    assert calls[:3] == [
        ("selected-predictions", "selected", selected_norm, {}),
        (
            "selected-inferred-persistence",
            "selected",
            selected_norm,
            {"persistence": True},
        ),
        (
            "selected-inferred-anomaly-persistence",
            "selected",
            selected_norm,
            {"anomaly": True},
        ),
    ]
    assert len(calls) == 7
    assert all(
        weights == "source" and norm == "source" for _, weights, norm, _ in calls[3:]
    )
    evaluation.main()
    assert len(calls) == 7
    signature = output / "evaluation-input.json"
    record = json.loads(signature.read_text())
    del record["evaluation_protocol"]
    signature.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="resume input differs"):
        evaluation.main()
