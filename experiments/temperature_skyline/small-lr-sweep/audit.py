# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit fresh initialization, retained full checkpoints, and shared exposure."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suffix", choices=["qualification", "train"], required=True)
    parser.add_argument("--historical-checkpoint", type=Path)
    parser.add_argument(
        "--helpers",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "lr-sweep/audit.py",
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("lr_audit", args.helpers)
    assert spec is not None and spec.loader is not None
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    expected = 12 if args.suffix == "qualification" else 30000
    retained = (
        [0, 4, 12] if args.suffix == "qualification" else [0, 10000, 25000, 30000]
    )
    hashes, rank_states, configs, initial_values, records = [], [], [], [], []
    for arm in ("1x", "2x", "4x"):
        root = args.root / f"{arm}-{args.suffix}"
        initial = torch.load(
            root / "checkpoint-0000000.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        assert initial["step"] == 0
        assert not initial["optimizer"]["state"]
        hashes.append(h.digest(initial["model"]))
        config = dict(initial["config"])
        config.pop("learning_rate")
        configs.append(config)
        assert all((root / f"checkpoint-{step:07d}.pt").is_file() for step in retained)
        for step in retained:
            checkpoint = torch.load(
                root / f"checkpoint-{step:07d}.pt",
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            assert checkpoint["step"] == step
            assert all(
                k in checkpoint
                for k in (
                    "optimizer",
                    "ema",
                    "rank_states",
                    "model",
                    "config",
                    "source",
                )
            )
        final = torch.load(
            root / "latest.pt", map_location="cpu", mmap=True, weights_only=False
        )
        rank_states.append(final["rank_states"])
        complete = json.loads((root / "COMPLETE.json").read_text())
        assert complete["step"] == complete["ema_updates"] == expected
        assert complete["accepted_examples"] == expected * 256
        assert complete["replicas_agree"]
        values = {}
        for weights in ("raw", "ema"):
            initial_validation = json.loads(
                (root / f"validation-{weights}-0000000.json").read_text()
            )
            values[weights] = [
                [r["full_grid_nmse"], r["full_grid_nll"]]
                for r in initial_validation["per_channel"]
            ]
        initial_values.append(values)
        for path in root.glob("validation-*.json"):
            value = json.loads(path.read_text())
            assert len(value["per_channel"]) == 77
            assert all(
                np.isfinite(row[key])
                for row in value["per_channel"]
                for key in ("full_grid_nmse", "full_grid_nll")
            )
        records.append(dict(arm=arm, complete=complete, retained_full_updates=retained))
    assert len(set(hashes)) == 1
    assert all(config == configs[0] for config in configs)
    assert all(h.equal(states, rank_states[0]) for states in rank_states)
    result = dict(
        initial_parameter_sha256=hashes[0],
        configuration_equal_except_learning_rate=True,
        final_rank_states_identical=True,
        initial_validation_max_abs_difference={
            weights: max(
                float(
                    np.max(
                        np.abs(
                            np.array(v[weights]) - np.array(initial_values[0][weights])
                        )
                    )
                )
                for v in initial_values
            )
            for weights in ("raw", "ema")
        },
        arms=records,
    )
    if args.historical_checkpoint:
        historical = torch.load(
            args.historical_checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        before_cooldown = torch.load(
            args.root / "1x-train/checkpoint-0025000.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        result["historical_25k_sampling_states_equal"] = all(
            h.equal(a[key], b[key])
            for a, b in zip(
                historical["rank_states"], before_cooldown["rank_states"], strict=True
            )
            for key in ("rng", "stream")
        )
    (args.root / f"audit-{args.suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
