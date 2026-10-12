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
@pytest.mark.parametrize("selected_only", [False, True])
@pytest.mark.parametrize("native_iap", [False, True])
@pytest.mark.parametrize("common", [False, True])
def test_control_weights_normalization_and_resume(
    tmp_path, monkeypatch, scratch, selected_only, native_iap, common
):
    run, output = tmp_path / "run", tmp_path / "evaluation"
    data_root = tmp_path / "data"
    if common:
        (data_root / "test").mkdir(parents=True)
        for year in range(2015, 2023):
            for month in range(1, 13):
                (data_root / "test" / f"{year}-{month:02d}.npz").touch()
        np.savez(
            tmp_path / "legacy-grid.npz",
            lat=[-45, 45],
            lon=[0, 90, 180, 270],
            mask=np.ones((77, 2, 4), dtype=bool),
        )
    run.mkdir()
    (run / "TRAIN_COMPLETE.json").write_text("{}")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": {
                    "data": str(data_root) if common else "data",
                    "checkpoint": "source",
                    "from_scratch": scratch,
                    **({"iap_grid": "iap-grid.npz"} if native_iap else {}),
                },
                "data_manifest_sha256": "data-digest",
                **({"iap_grid_sha256": "digest"} if native_iap else {}),
            }
        )
    )
    (run / "best.json").write_text('{"checkpoint_sha256":"digest","score":1.0}')

    class Data:
        def __init__(self, *args, **kwargs):
            self.normalization = "source"
            self.grid = {"names": np.array(["sst"])}
            if common:
                self.grid.update(dict(np.load(tmp_path / "legacy-grid.npz")))

        def use_observation_normalization(self):
            self.normalization = "observations"

        def paths(self, split):
            return (
                sorted((data_root / split).glob("*.npz")) if common else list(range(96))
            )

    class Model:
        @classmethod
        def from_arguments(cls, names, arguments):
            return cls(
                names,
                arguments.get("normalization", "batch"),
                arguments.get("evolution_architecture", "d"),
            )

        def __init__(self, names, normalization="batch", evolution_architecture="d"):
            assert evolution_architecture == "d"
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
        if "common_export" in options:
            directory = options.pop("common_export")
            options.pop("common_grid")
            directory.mkdir(parents=True, exist_ok=True)
            for path in paths:
                (directory / path.name).write_bytes(b"sidecar")
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
        sys,
        "argv",
        ["evaluate", "--run", str(run), "--output", str(output)]
        + (["--selected-only"] if selected_only else [])
        + (["--common-grid", str(tmp_path / "legacy-grid.npz")] if common else []),
    )
    evaluation.main()
    selected_norm = "observations" if scratch else "source"
    expected_controls = [
        ("selected-predictions", "selected", selected_norm, {}),
        (
            "selected-inferred-persistence",
            "selected",
            selected_norm,
            {"persistence": True},
        ),
    ]
    if not native_iap:
        expected_controls.append(
            (
                "selected-inferred-anomaly-persistence",
                "selected",
                selected_norm,
                {"anomaly": True},
            )
        )
    count = len(expected_controls)
    assert calls[:count] == expected_controls
    expected_count = (4 if selected_only else 7) - int(native_iap)
    assert len(calls) == expected_count
    if selected_only:
        assert calls[count] == (
            "seasonal-climatology",
            "selected",
            selected_norm,
            {"climatology": True},
        )
        assert (
            json.loads((output / "COMPLETE.json").read_text())["source_sha256"] is None
        )
    else:
        assert all(
            weights == "source" and norm == "source"
            for _, weights, norm, _ in calls[count:]
        )
    evaluation.main()
    assert len(calls) == expected_count
    complete = output / "COMPLETE.json"
    completed = json.loads(complete.read_text())
    assert ("common_grid_sidecars" in completed) == common
    if common:
        assert len(completed["common_grid_sidecars"]) == 96 * (2 if native_iap else 3)
        missing = output / "common-grid" / "selected" / "2015-01.npz"
        missing.unlink()
        with pytest.raises(ValueError, match="sidecar inventory"):
            evaluation.main()
        # An interrupted evaluation repairs just the affected label even when
        # its native JSON and NPZ were already written.
        complete.unlink()
        evaluation.main()
        assert len(calls) == expected_count + 1
        assert calls[-1][0] == "selected-predictions"
        assert missing.read_bytes() == b"sidecar"
        missing.write_bytes(b"changed")
        with pytest.raises(ValueError, match="sidecar hashes differ"):
            evaluation.main()
        missing.write_bytes(b"sidecar")
    signature = output / "evaluation-input.json"
    record = json.loads(signature.read_text())
    assert ("omitted_controls" in record) == native_iap
    if native_iap:
        assert "inferred-anomaly-persistence" in record["omitted_controls"]
    del record["evaluation_protocol"]
    signature.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="resume input differs"):
        evaluation.main()
