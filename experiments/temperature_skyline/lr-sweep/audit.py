# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify branch initialization, final sampler agreement, and finite receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(parameters):
    result = hashlib.sha256()
    for name, tensor in parameters.items():
        result.update(name.encode())
        result.update(tensor.cpu().numpy().tobytes())
    return result.hexdigest()


def equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suffix", choices=["qualification", "train"], required=True)
    args = parser.parse_args()
    first = json.loads((args.root / "1x-train.json").read_text())
    parent_path = Path(first["branch_checkpoint"]["path"])
    parent = torch.load(parent_path, map_location="cpu", mmap=True, weights_only=False)
    parent_digest = digest(parent["model"])
    records = []
    baseline_states = None
    expected = 25004 if args.suffix == "qualification" else 28000
    for arm, rate in [("1x", 2e-4), ("2x", 4e-4), ("4x", 8e-4)]:
        root = args.root / f"{arm}-{args.suffix}"
        provenance = json.loads((root / "provenance-restart-0.json").read_text())
        assert provenance["restored_parameter_sha256"] == parent_digest
        assert provenance["resumed_step"] == 25000
        state = torch.load(
            root / "latest.pt", map_location="cpu", mmap=True, weights_only=False
        )
        complete = json.loads((root / "COMPLETE.json").read_text())
        assert complete["step"] == complete["ema_updates"] == expected
        assert complete["branch_examples"] == (expected - 25000) * 256
        assert complete["replicas_agree"]
        assert all(group["lr"] == rate for group in state["optimizer"]["param_groups"])
        if baseline_states is None:
            baseline_states = state["rank_states"]
        assert equal(baseline_states, state["rank_states"]), (
            "Sampling or RNG states differ between arms"
        )
        initial_differences = {}
        for weights in ("raw", "ema"):
            initial = json.loads(
                (root / f"validation-{weights}-0025000.json").read_text()
            )
            reference = json.loads(
                (parent_path.parent / f"validation-{weights}-0025000.json").read_text()
            )
            assert initial["origins"] == reference["origins"]
            initial_differences[weights] = {
                key: max(
                    abs(a[key] - b[key])
                    for a, b in zip(
                        initial["per_channel"], reference["per_channel"], strict=True
                    )
                )
                for key in ("full_grid_nmse", "full_grid_nll")
            }
            for a, b in zip(
                initial["per_channel"], reference["per_channel"], strict=True
            ):
                assert a["channel"] == b["channel"]
                assert (
                    a["persistence_full_grid_nmse"] == b["persistence_full_grid_nmse"]
                )
        for path in root.glob("validation-*.json"):
            value = json.loads(path.read_text())
            assert len(value["per_channel"]) == 77
            for row in value["per_channel"]:
                assert all(
                    np.isfinite(row[k]) for k in ("full_grid_nmse", "full_grid_nll")
                )
        records.append(
            dict(
                arm=arm,
                complete=complete,
                initial_validation_max_abs_difference=initial_differences,
            )
        )
    result = dict(
        parent_parameter_sha256=parent_digest,
        final_rank_states_identical=True,
        arms=records,
    )
    (args.root / f"audit-{args.suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
