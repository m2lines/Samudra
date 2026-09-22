#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Lock architecture-specific learning rates using complete validation-only pilots."""

import argparse
import json
from pathlib import Path

ARCHITECTURES = {"unet": ("A", "B"), "wide": ("C", "D"), "swin": ("E", "F")}
RATES = {"lr1": 1e-4, "lr3": 3e-4}


def select(root):
    root = Path(root)
    records = {}
    protocol = None
    for arms in ARCHITECTURES.values():
        for arm in arms:
            for tag, rate in RATES.items():
                path = root / f"{arm}-{tag}"
                completion = json.loads((path / "TRAIN_COMPLETE.json").read_text())
                manifest = json.loads((path / "manifest.json").read_text())
                args = manifest["arguments"]
                assert args["arm"] == arm and args["learning_rate"] == rate
                assert args["phase"] == "reconstruction" and args["max_steps"] == 0
                assert completion["state"]["complete"]
                selected = completion["selected_validation"]["ts_mse"]
                events = [
                    json.loads(line)
                    for line in (path / "progress.jsonl").read_text().splitlines()
                ]
                initial = next(
                    e["ts_mse"] for e in events if e.get("event") == "baseline"
                )
                if not (0 < initial < float("inf") and 0 <= selected < float("inf")):
                    raise ValueError("Nonfinite or invalid pilot score")
                current = {
                    key: args[key]
                    for key in [
                        "seed",
                        "batch_size",
                        "accumulate",
                        "hours",
                        "data_root",
                        "val_origins",
                    ]
                }
                current.update(
                    world_size=manifest["world_size"],
                    device=manifest["device"],
                    producer=manifest["code_commit"],
                )
                if protocol is None:
                    protocol = current
                elif current != protocol:
                    raise ValueError("Pilot hardware or training protocol differs")
                records[arm, tag] = {
                    "selected_ts_mse": selected,
                    "initial_ts_mse": initial,
                    "selected_over_initial": selected / initial,
                    "steps": completion["state"]["step"],
                }
    result = {}
    for architecture, arms in ARCHITECTURES.items():
        scores = {
            tag: sum(records[arm, tag]["selected_over_initial"] for arm in arms)
            / len(arms)
            for tag in RATES
        }
        # Insertion order gives the lower rate the tie break.
        choice = min(scores, key=scores.__getitem__)
        result[architecture] = {
            "learning_rate": RATES[choice],
            "mean_selected_over_initial": scores,
            "pilots": {
                f"{arm}-{tag}": records[arm, tag] for arm in arms for tag in RATES
            },
        }
    return {
        "criterion": "mean selected/initial validation T/S reconstruction MSE across the two input conditions; lower wins",
        "protocol": protocol,
        "selected": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = select(args.raw)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
