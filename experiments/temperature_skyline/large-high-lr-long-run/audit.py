# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify run state, exact initialization, retained checkpoints and finite metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    name = "qualification" if args.qualification else "all-fields-60k"
    root = args.root / name
    reference = Path("/scratch/jr7309/skyline-large-classifier-20261112/all-fields-25k")
    receipt = json.loads((root / "provenance-restart-0.json").read_text())
    old = json.loads((reference / "provenance-restart-0.json").read_text())
    assert receipt["initial_parameter_sha256"] == old["initial_parameter_sha256"]
    assert receipt["source"] == old["source"]
    assert receipt["parameters"] == 216370176
    final = 16 if args.qualification else 60000
    complete = json.loads((root / "COMPLETE.json").read_text())
    assert complete["step"] == complete["ema_updates"] == final
    assert complete["effective_batch"] == 256 and complete["replicas_agree"]
    expected = [0, 16] if args.qualification else [0, 10000, 25000, 50000, 55000, 60000]
    for step in expected:
        state = torch.load(
            root / f"checkpoint-{step:07d}.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        assert state["step"] == step
        assert len(state["rank_states"]) == 4
        assert {
            "model",
            "optimizer",
            "ema",
            "source",
            "config",
            "rank_states",
        } <= state.keys()
        if step == 25000:
            historical = torch.load(
                reference / "latest.pt",
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            for actual, original in zip(
                state["rank_states"], historical["rank_states"], strict=True
            ):
                for key in ("rng", "stream"):
                    np.testing.assert_equal(actual[key], original[key])
            del historical
        del state
    validations = list(root.glob("validation-*.json"))
    assert validations
    for path in validations:
        value = json.loads(path.read_text())
        assert len(value["per_channel"]) == 77
        assert all(
            np.isfinite(row[k])
            for row in value["per_channel"]
            for k in ("full_grid_nmse", "full_grid_nll")
        )
    logs = [
        json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()
    ]
    updates = [v for v in logs if "loss" in v]
    assert updates and all(
        np.isfinite(v[k]) for v in updates for k in ("loss", "grad_norm")
    )
    if args.qualification:
        assert all(v["lr"] == 0.0008 for v in updates)
    result = dict(
        complete=complete,
        initial_parameters_match_historical=True,
        source_matches=True,
        retained_steps=expected,
        finite_validations=len(validations),
        finite_training=True,
        historical_25k_sampling_match=None if args.qualification else True,
    )
    (args.root / f"audit-{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
